import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Box,
  Camera,
  Crosshair,
  Eye,
  FlipHorizontal2,
  Hand,
  Images,
  Minus,
  Pause,
  Play,
  Plus,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Target
} from 'lucide-react';
import { analyzeGrip, createEmptyAnalysis } from './vision/gripAnalysis';
import { palmCenter, pointsToPixelSpace, subtract } from './vision/geometry';
import { inferObjectRegion } from './vision/objectTracking';
import { drawTrackingOverlay } from './vision/drawing';
import { createVisionEngine, type VisionEngine, type VisionModelStatus } from './vision/visionEngine';
import { TrackingStabilizer } from './vision/stabilization';
import type {
  AlgorithmVersion,
  GripAnalysis,
  GripCalibrationBaseline,
  GripCalibrationProfiles,
  GripMode,
  Landmark,
  ObjectRegion,
  Point
} from './vision/types';

const INITIAL_MODEL_STATUS: VisionModelStatus = {
  hands: 'idle',
  detector: 'idle',
  segmenter: 'idle'
};

const CALIBRATION_STORAGE_KEY = 'grip-lab-calibration-profiles-v2';
const ALGORITHM_VERSION_STORAGE_KEY = 'grip-lab-algorithm-version';
const OBJECT_PROFILES_STORAGE_KEY = 'grip-lab-object-profiles-v1';
const DESCRIPTOR_SIZE = 32;

type ObjectTrainingSample = {
  id: string;
  imageDataUrl: string;
  descriptor: number[];
  createdAt: number;
};

type ObjectProfile = {
  id: string;
  name: string;
  samples: ObjectTrainingSample[];
  descriptor: number[];
  trainedAt: number;
};

type ObjectDetection = {
  profileId: string;
  name: string;
  score: number;
  matched: boolean;
} | null;

const METRIC_INFO = {
  confidence: 'How much the app trusts the object lock and tracking signal in this frame.',
  contacts: 'How many fingertip or finger-segment contacts appear close enough to support the object.',
  closure: 'How closed the hand is around the object, normalized by hand size.',
  thumb: 'How well the thumb opposes the fingers, which is important for stable pinch and power grips.',
  enclosure: 'How much the fingers surround the object from multiple angles.',
  coupling: 'How closely the object motion follows the hand motion. Low coupling can indicate slipping.'
} as const;

const EXPLAIN = {
  lock: 'Lock keeps the app focused on one object. Click or drag on the video to place the lock over the real object.',
  shrink: 'Shrinks the locked object region when the outline is too large or includes your hand/background.',
  grow: 'Grows the locked object region when the outline is too small and misses part of the object.',
  strong: 'Records your current pose as a strong grip baseline for this grip mode. It helps personalize future scores.',
  weak: 'Records your current pose as a weak grip baseline. Similar poses can be scored lower or shown as less confident.',
  version: 'Choose V1 for the original permissive heuristic, or V2 for stricter object-first scoring with lower false positives.',
  gripQuality: 'Visual grip stability estimated from the camera. It is not real physical force.',
  state: 'The tracking state says what the app believes is happening: no hand, hand only, object uncertain, grip detected, strong hold, or slip risk.',
  mode: 'Grip mode is the type of hold the app thinks it sees, such as phone-side, pinch, power, hook, open hand, or uncertain.',
  objectLockQuality: 'How much the app trusts that the highlighted region is a real object rather than your hand or background.',
  motion: 'Motion state compares hand and object movement. Sustained mismatch raises slip risk.',
  slip: 'Slip risk rises only when the object and hand move differently across several frames.',
  gripEvidence: 'These rows show what raised or lowered the grip score.',
  objectEvidence: 'These rows describe the object tracker: shape, how long it has been locked, and whether you manually adjusted it.',
  shape: 'The object shape guessed by the tracker: phone-like, ellipse, unknown, or detector/manual fallback.',
  lockAge: 'How many video frames the current object lock has survived. A higher value usually means a more stable lock.',
  manualLock: 'Yes means you clicked or dragged the object lock yourself. Manual locks are trusted more than automatic guesses.',
  suggestedPoints: 'Suggested grip points are possible places for thumb, fingers, or support contact based on the current object outline.',
  modeFit: 'How well the current hand-object pose matches the selected grip mode.',
  contact: 'How much visible finger segment contact appears near the object boundary.',
  fingerWrap: 'How much the fingers appear to curl around or contain the object.',
  thumbSupport: 'How much the thumb appears to support or oppose the fingers.',
  motionStability: 'How stable the object-hand motion is over recent frames.',
  calibration: 'How much saved strong/weak calibration is affecting the current score.'
} as const;

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<VisionEngine | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previousObjectRef = useRef<ObjectRegion | null>(null);
  const previousPalmRef = useRef<Point | null>(null);
  const manualPointRef = useRef<Point | null>(null);
  const manualScaleRef = useRef(1);
  const draggingObjectRef = useRef(false);
  const lockObjectRef = useRef(false);
  const pausedRef = useRef(false);
  const mirroredRef = useRef(true);
  const lastDetectorRunRef = useRef(0);
  const detectorBoxRef = useRef<DOMRectReadOnly | null>(null);
  const autoRetryRef = useRef(false);
  const stabilizerRef = useRef(new TrackingStabilizer());
  const algorithmVersionRef = useRef<AlgorithmVersion>(readInitialAlgorithmVersion());
  const calibrationProfilesRef = useRef<GripCalibrationProfiles>({});
  const objectProfilesRef = useRef<ObjectProfile[]>([]);
  const lastObjectMatchRef = useRef(0);
  const calibrationCaptureRef = useRef<{
    active: boolean;
    kind: 'strong' | 'weak';
    start: number;
    samples: GripCalibrationBaseline[];
  }>({ active: false, kind: 'strong', start: 0, samples: [] });

  const [cameraState, setCameraState] = useState<'idle' | 'requesting' | 'live' | 'blocked'>('idle');
  const [modelStatus, setModelStatus] = useState<VisionModelStatus>(INITIAL_MODEL_STATUS);
  const [analysis, setAnalysis] = useState<GripAnalysis>(() => createEmptyAnalysis());
  const [mirrored, setMirrored] = useState(true);
  const [paused, setPaused] = useState(false);
  const [locked, setLocked] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [hasCalibration, setHasCalibration] = useState(false);
  const [calibrationKind, setCalibrationKind] = useState<'strong' | 'weak'>('strong');
  const [algorithmVersion, setAlgorithmVersion] = useState<AlgorithmVersion>(() => algorithmVersionRef.current);
  const [objectName, setObjectName] = useState('');
  const [trainingSamples, setTrainingSamples] = useState<ObjectTrainingSample[]>([]);
  const [objectProfiles, setObjectProfiles] = useState<ObjectProfile[]>([]);
  const [objectDetection, setObjectDetection] = useState<ObjectDetection>(null);
  const [trainingStatus, setTrainingStatus] = useState('Lock the object, then capture 2 or more masked views.');

  useEffect(() => {
    calibrationProfilesRef.current = loadCalibrationProfiles();
    setHasCalibration(hasAnyCalibration(calibrationProfilesRef.current));
    const profiles = loadObjectProfiles();
    objectProfilesRef.current = profiles;
    setObjectProfiles(profiles);
  }, []);

  useEffect(() => {
    mirroredRef.current = mirrored;
  }, [mirrored]);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    lockObjectRef.current = locked;
  }, [locked]);

  useEffect(() => {
    algorithmVersionRef.current = algorithmVersion;
  }, [algorithmVersion]);

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      engineRef.current?.dispose();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const modelSummary = useMemo(() => {
    const ready = Object.values(modelStatus).filter((state) => state === 'ready').length;
    const failed = Object.values(modelStatus).filter((state) => state === 'failed').length;
    if (modelStatus.hands === 'failed') return 'Retry hand model';
    if (modelStatus.hands === 'ready') return `${ready}/3 models ready${failed ? `, ${failed} fallback` : ''}`;
    if (modelStatus.hands === 'loading') return 'Loading hand model';
    return 'Models idle';
  }, [modelStatus]);

  const loadVisionEngine = useCallback(async (force = false) => {
    if (force || engineRef.current?.status.hands === 'failed') {
      engineRef.current?.dispose();
      engineRef.current = null;
    }

    if (!engineRef.current) {
      engineRef.current = await createVisionEngine(setModelStatus);
    }

    return engineRef.current;
  }, []);

  const startCamera = useCallback(async () => {
    if (cameraState === 'requesting' || cameraState === 'live') return;
    setCameraState('requesting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setCameraState('live');

      const engine = await loadVisionEngine();
      if (engine.status.hands === 'failed') {
        setAnalysis(createEmptyAnalysis('Hand model failed to load. Use the model status button to retry.'));
      }
      runLoop();
    } catch (error) {
      console.warn('Camera start failed', error);
      setCameraState('blocked');
      setAnalysis(createEmptyAnalysis('Camera permission is blocked or unavailable.'));
    }
  }, [cameraState, loadVisionEngine]);

  const updateCalibrationCapture = useCallback((frameAnalysis: GripAnalysis, timestamp: number) => {
    const capture = calibrationCaptureRef.current;
    if (!capture.active) return;

    if (frameAnalysis.objectLockQuality > 0.36 && frameAnalysis.guidance !== 'Object not locked') {
      capture.samples.push({
        mode: frameAnalysis.diagnostics.mode,
        gripPercentage: capture.kind === 'strong' ? Math.max(frameAnalysis.gripPercentage, 72) : Math.min(frameAnalysis.gripPercentage, 38),
        closureScore: frameAnalysis.closureScore,
        enclosureScore: frameAnalysis.enclosureScore,
        fingerCurlScore: frameAnalysis.evidence.fingerCurlScore,
        fingerSegmentContactScore: frameAnalysis.evidence.fingerSegmentContactScore,
        phoneSideGripScore: frameAnalysis.evidence.phoneSideGripScore,
        pinchScore: frameAnalysis.evidence.pinchScore,
        powerGripScore: frameAnalysis.evidence.powerGripScore,
        thumbSupportScore: frameAnalysis.evidence.thumbSupportScore,
        objectLockQuality: frameAnalysis.objectLockQuality,
        createdAt: Date.now()
      });
    }

    if (timestamp - capture.start < 1050) return;
    capture.active = false;
    setCalibrating(false);
    if (!capture.samples.length) return;
    const baseline = averageBaseline(capture.samples);
    const profiles = {
      ...calibrationProfilesRef.current,
      [baseline.mode]: {
        ...calibrationProfilesRef.current[baseline.mode],
        [capture.kind]: baseline
      }
    };
    calibrationProfilesRef.current = profiles;
    saveCalibrationProfiles(profiles);
    setHasCalibration(true);
  }, []);

  const runLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const engine = engineRef.current;
    if (!video || !canvas || !engine) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    const tick = (timestamp: number) => {
      if (!video.videoWidth || !video.videoHeight) {
        animationRef.current = requestAnimationFrame(tick);
        return;
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      let hand: Landmark[] | null = null;
      let object: ObjectRegion | null = previousObjectRef.current;
      let frameAnalysis = analysis;

      if (!pausedRef.current) {
        const hands = engine.detectHands(video, timestamp);
        const rawHand = hands[0] ? pointsToPixelSpace(hands[0], video.videoWidth, video.videoHeight) : null;
        hand = stabilizerRef.current.stabilizeHand(rawHand, timestamp);
        if (timestamp - lastDetectorRunRef.current > 650) {
          detectorBoxRef.current = engine.detectObjectBox(video, timestamp);
          lastDetectorRunRef.current = timestamp;
        }
        const rawObject = inferObjectRegion({
          video,
          hand,
          previous: previousObjectRef.current,
          manualPoint: manualPointRef.current,
          manualScale: manualScaleRef.current,
          locked: lockObjectRef.current,
          detectorBox: detectorBoxRef.current,
          algorithmVersion: algorithmVersionRef.current
        });
        object = stabilizerRef.current.stabilizeObject(rawObject, timestamp);
        const handVelocityForSlip =
          hand && previousPalmRef.current ? subtract(palmCenter(hand), previousPalmRef.current) : { x: 0, y: 0 };
        const persistentSlipScore = stabilizerRef.current.updatePersistentSlip(handVelocityForSlip, object);
        const rawFrameAnalysis = analyzeGrip(hand, object, previousPalmRef.current, {
          persistentSlipScore,
          algorithmVersion: algorithmVersionRef.current
        });
        frameAnalysis = stabilizerRef.current.stabilizeAnalysis(
          analyzeGrip(hand, object, previousPalmRef.current, {
            persistentSlipScore,
            calibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'strong'),
            weakCalibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'weak'),
            algorithmVersion: algorithmVersionRef.current
          }),
          timestamp
        );
        updateCalibrationCapture(frameAnalysis, timestamp);
        previousObjectRef.current = object;
        previousPalmRef.current = frameAnalysis.palmCenter;
        if (timestamp - lastObjectMatchRef.current > 420) {
          lastObjectMatchRef.current = timestamp;
          setObjectDetection(matchObjectProfile(video, object, objectProfilesRef.current));
        }
        setAnalysis(frameAnalysis);
      }

      drawTrackingOverlay(context, canvas.width, canvas.height, mirroredRef.current, hand, object, frameAnalysis);
      animationRef.current = requestAnimationFrame(tick);
    };

    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = requestAnimationFrame(tick);
  }, [analysis, updateCalibrationCapture]);

  const resetObject = useCallback(() => {
    manualPointRef.current = null;
    manualScaleRef.current = 1;
    draggingObjectRef.current = false;
    previousObjectRef.current = null;
    detectorBoxRef.current = null;
    stabilizerRef.current.reset();
    calibrationCaptureRef.current = { active: false, kind: calibrationKind, start: 0, samples: [] };
    setLocked(false);
    setCalibrating(false);
    setAnalysis(createEmptyAnalysis('Object reset. Place it between your thumb and fingers to relock.'));
  }, [calibrationKind]);

  const startCalibration = useCallback((kind: 'strong' | 'weak' = 'strong') => {
    calibrationCaptureRef.current = {
      active: true,
      kind,
      start: performance.now(),
      samples: []
    };
    setCalibrationKind(kind);
    setCalibrating(true);
  }, []);

  const retryModels = useCallback(async () => {
    if (modelStatus.hands === 'loading') return;
    try {
      await loadVisionEngine(true);
      if (cameraState === 'live') runLoop();
    } catch (error) {
      console.warn('Model retry failed', error);
      setModelStatus((current) => ({ ...current, hands: 'failed' }));
    }
  }, [cameraState, loadVisionEngine, modelStatus.hands, runLoop]);

  useEffect(() => {
    if (modelStatus.hands === 'ready') {
      autoRetryRef.current = false;
      return;
    }

    if (cameraState !== 'live' || modelStatus.hands !== 'failed' || autoRetryRef.current) return;
    autoRetryRef.current = true;
    const timeout = window.setTimeout(() => {
      void retryModels();
    }, 500);
    return () => window.clearTimeout(timeout);
  }, [cameraState, modelStatus.hands, retryModels]);

  const handleCanvasClick = useCallback(
    async (event: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
      const point = mirrored ? { x: canvas.width - x, y } : { x, y };
      manualPointRef.current = point;
      setLocked(true);
      await engineRef.current?.segmentAt(video, { x: point.x / canvas.width, y: point.y / canvas.height });
    },
    [mirrored]
  );

  const handleCanvasPointerDown = useCallback(
    async (event: React.PointerEvent<HTMLCanvasElement>) => {
      draggingObjectRef.current = true;
      await handleCanvasClick(event);
    },
    [handleCanvasClick]
  );

  const handleCanvasPointerMove = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!draggingObjectRef.current || !manualPointRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
      manualPointRef.current = mirrored ? { x: canvas.width - x, y } : { x, y };
      setLocked(true);
    },
    [mirrored]
  );

  const stopObjectDrag = useCallback(() => {
    draggingObjectRef.current = false;
  }, []);

  const resizeManualObject = useCallback((direction: -1 | 1) => {
    manualScaleRef.current = Math.min(1.8, Math.max(0.55, manualScaleRef.current + direction * 0.12));
    setLocked(true);
  }, []);

  const selectAlgorithmVersion = useCallback(
    (version: AlgorithmVersion) => {
      if (version === algorithmVersion) return;
      algorithmVersionRef.current = version;
      manualPointRef.current = null;
      manualScaleRef.current = 1;
      draggingObjectRef.current = false;
      previousObjectRef.current = null;
      detectorBoxRef.current = null;
      stabilizerRef.current.reset();
      setLocked(false);
      setCalibrating(false);
      setAlgorithmVersion(version);
      saveAlgorithmVersion(version);
      setAnalysis(
        createEmptyAnalysis(
          version === 'v2'
            ? 'V2 selected. It will require independent object evidence before scoring grip.'
            : 'V1 selected. It uses the original permissive grip heuristic.'
        )
      );
    },
    [algorithmVersion]
  );

  const captureObjectTrainingView = useCallback(() => {
    const video = videoRef.current;
    const object = previousObjectRef.current;
    if (!video || !video.videoWidth || !video.videoHeight || !object?.locked) {
      setTrainingStatus('Lock the object first, then capture the masked crop.');
      return;
    }

    const descriptor = describeObjectPatch(video, object);
    const imageDataUrl = createObjectThumbnail(video, object);
    if (!descriptor || !imageDataUrl) {
      setTrainingStatus('Could not capture this view. Keep the object fully visible and try again.');
      return;
    }

    const sample: ObjectTrainingSample = {
      id: crypto.randomUUID(),
      imageDataUrl,
      descriptor,
      createdAt: Date.now()
    };
    setTrainingSamples((current) => [...current, sample].slice(-8));
    setTrainingStatus('Captured masked object view. Add another angle or train the profile.');
  }, []);

  const trainObjectProfile = useCallback(() => {
    const name = objectName.trim();
    if (!name) {
      setTrainingStatus('Give the object a name before training.');
      return;
    }
    if (trainingSamples.length < 2) {
      setTrainingStatus('Capture at least 2 masked views from different angles.');
      return;
    }

    const profile: ObjectProfile = {
      id: crypto.randomUUID(),
      name,
      samples: trainingSamples,
      descriptor: averageDescriptor(trainingSamples.map((sample) => sample.descriptor)),
      trainedAt: Date.now()
    };
    const profiles = [profile, ...objectProfiles.filter((current) => current.name.toLowerCase() !== name.toLowerCase())].slice(0, 6);
    objectProfilesRef.current = profiles;
    setObjectProfiles(profiles);
    saveObjectProfiles(profiles);
    setTrainingSamples([]);
    setTrainingStatus(`${name} trained successfully. Hold it in frame to confirm detection.`);
  }, [objectName, objectProfiles, trainingSamples]);

  const clearTrainingSamples = useCallback(() => {
    setTrainingSamples([]);
    setTrainingStatus('Training views cleared. Capture new masked views.');
  }, []);

  return (
    <main className="app-shell">
      <section className="camera-workspace" aria-label="Live grip tracking workspace">
        <video ref={videoRef} className={mirrored ? 'camera-feed mirrored' : 'camera-feed'} playsInline muted />
        <canvas
          ref={canvasRef}
          className="tracking-canvas"
          aria-label="Hand, object, and grip tracking overlay"
          onPointerDown={handleCanvasPointerDown}
          onPointerMove={handleCanvasPointerMove}
          onPointerUp={stopObjectDrag}
          onPointerCancel={stopObjectDrag}
        />

        <div className="top-bar">
          <div className="brand">
            <span className="brand-mark">
              <Hand size={18} />
            </span>
            <span>GripSense RGB</span>
          </div>
          <div className="toolbar" aria-label="Camera controls">
            <div className="version-switch" aria-label="Algorithm version">
              <button
                type="button"
                className={algorithmVersion === 'v1' ? 'version-button active' : 'version-button'}
                onClick={() => selectAlgorithmVersion('v1')}
                aria-pressed={algorithmVersion === 'v1'}
              >
                V1
              </button>
              <button
                type="button"
                className={algorithmVersion === 'v2' ? 'version-button active' : 'version-button'}
                onClick={() => selectAlgorithmVersion('v2')}
                aria-pressed={algorithmVersion === 'v2'}
              >
                V2
              </button>
            </div>
            <InlineExplain label="Explain algorithm version" text={EXPLAIN.version} compact />
            <button
              className="tool-button primary"
              onClick={startCamera}
              disabled={cameraState === 'requesting'}
              aria-label={cameraState === 'live' ? 'Camera live' : 'Start camera'}
            >
              <Camera size={18} />
              <span>{cameraState === 'live' ? 'Camera live' : cameraState === 'requesting' ? 'Starting' : 'Start'}</span>
            </button>
            <button className="icon-button" onClick={() => setPaused((value) => !value)} aria-label={paused ? 'Resume tracking' : 'Pause tracking'}>
              {paused ? <Play size={18} /> : <Pause size={18} />}
            </button>
            <button className="icon-button" onClick={() => setMirrored((value) => !value)} aria-label="Toggle mirror mode">
              <FlipHorizontal2 size={18} />
            </button>
            <button
              className="tool-button"
              onClick={() => setLocked((value) => !value)}
              aria-label={locked ? 'Unlock object tracking' : 'Lock object tracking'}
            >
              <Crosshair size={17} />
              <span>{locked ? 'Unlock' : 'Lock'}</span>
            </button>
            <InlineExplain label="Explain lock" text={EXPLAIN.lock} compact />
            <button className="icon-button" onClick={resetObject} aria-label="Reset object tracking">
              <RotateCcw size={18} />
            </button>
            <button className="icon-button" onClick={() => resizeManualObject(-1)} aria-label="Shrink locked object">
              <Minus size={18} />
            </button>
            <InlineExplain label="Explain shrink object" text={EXPLAIN.shrink} compact />
            <button className="icon-button" onClick={() => resizeManualObject(1)} aria-label="Grow locked object">
              <Plus size={18} />
            </button>
            <InlineExplain label="Explain grow object" text={EXPLAIN.grow} compact />
            <button
              className="tool-button calibrate-button"
              onClick={() => startCalibration('strong')}
              aria-label="Calibrate strong hold"
              title="Calibrate strong hold"
            >
              <Target size={17} />
              <span>{calibrating && calibrationKind === 'strong' ? 'Calibrating' : hasCalibration ? 'Strong' : 'Strong'}</span>
            </button>
            <InlineExplain label="Explain strong calibration" text={EXPLAIN.strong} compact />
            <button
              className="tool-button calibrate-button weak"
              onClick={() => startCalibration('weak')}
              aria-label="Calibrate weak hold"
              title="Calibrate weak hold"
            >
              <Target size={17} />
              <span>{calibrating && calibrationKind === 'weak' ? 'Calibrating' : 'Weak'}</span>
            </button>
            <InlineExplain label="Explain weak calibration" text={EXPLAIN.weak} compact />
          </div>
          <button className="model-pill model-action" onClick={retryModels} aria-label="Retry model loading">
            <Activity size={16} />
            <span>{modelSummary}</span>
          </button>
        </div>

        {cameraState !== 'live' && (
          <div className="permission-panel">
            <div className="permission-icon">
              <Camera size={30} />
            </div>
            <h1>Live grip analysis</h1>
            <p>
              Start the camera, hold an object naturally, then click the object if the automatic lock needs help.
            </p>
            <button className="start-button" onClick={startCamera}>
              <Camera size={20} />
              <span>Start camera</span>
            </button>
          </div>
        )}
      </section>

      <aside className="analysis-rail" aria-label="Grip analysis">
        <div className={`score-orb ${analysis.guidance.toLowerCase().replaceAll(' ', '-')}`}>
          <span>{analysis.gripPercentage}</span>
          <small>%</small>
        </div>
        <div>
          <p className="eyebrow">Grip quality</p>
          <h2 className="explain-heading">
            {analysis.guidance}
            <InlineExplain label="Explain grip quality" text={EXPLAIN.gripQuality} />
          </h2>
          <p className="guidance-copy">{analysis.message}</p>
          <div className="state-strip">
            <span>
              {analysis.diagnostics.state}
              <InlineExplain label="Explain tracking state" text={EXPLAIN.state} />
            </span>
            <strong>
              {analysis.diagnostics.mode}
              <InlineExplain label="Explain grip mode" text={EXPLAIN.mode} />
            </strong>
          </div>
        </div>

        <div className="metric-grid">
          <div className="lock-quality">
            <div className="metric-label">
              <span className="metric-name">
                Object lock quality
                <InlineExplain label="Explain object lock quality" text={EXPLAIN.objectLockQuality} />
              </span>
              <strong>{Math.round(analysis.objectLockQuality * 100)}%</strong>
            </div>
            <div className="metric-track">
              <span style={{ width: `${Math.round(analysis.objectLockQuality * 100)}%` }} />
            </div>
            <small>
              {analysis.objectLockQuality < 0.38
                ? 'Lock is uncertain. Click the object.'
                : analysis.objectLockQuality < 0.62
                  ? 'Lock is usable but imperfect.'
                  : 'Object lock looks stable.'}
            </small>
          </div>
          <Metric label="Confidence" value={analysis.confidence} info={METRIC_INFO.confidence} />
          <Metric
            label="Contacts"
            value={analysis.contactPoints / 5}
            text={`${analysis.contactPoints}/5`}
            info={METRIC_INFO.contacts}
          />
          <Metric label="Closure" value={analysis.closureScore} info={METRIC_INFO.closure} />
          <Metric label="Thumb" value={analysis.thumbOpposition} info={METRIC_INFO.thumb} />
          <Metric label="Enclosure" value={analysis.enclosureScore} info={METRIC_INFO.enclosure} />
          <Metric label="Coupling" value={analysis.motionCoupling} info={METRIC_INFO.coupling} />
        </div>

        <div className="trainer-panel">
          <div className="motion-header">
            <Images size={18} />
            <span>Object trainer</span>
          </div>
          <label className="object-name-field">
            <span>Object name</span>
            <input
              value={objectName}
              onChange={(event) => setObjectName(event.target.value)}
              placeholder="Phone, mug, remote..."
              maxLength={36}
            />
          </label>
          <div className="trainer-actions">
            <button type="button" onClick={captureObjectTrainingView}>
              <Camera size={16} />
              Capture view
            </button>
            <button type="button" onClick={trainObjectProfile}>
              <Sparkles size={16} />
              Train
            </button>
            <button type="button" onClick={clearTrainingSamples}>
              Clear
            </button>
          </div>
          <p className="diagnostic-copy">{trainingStatus}</p>
          {trainingSamples.length > 0 && (
            <div className="sample-strip" aria-label="Captured object views">
              {trainingSamples.map((sample, index) => (
                <img src={sample.imageDataUrl} alt={`Captured object angle ${index + 1}`} key={sample.id} />
              ))}
            </div>
          )}
          <div className={objectDetection?.matched ? 'detected-object matched' : 'detected-object'}>
            <Box size={17} />
            <span>
              {objectDetection?.matched
                ? `Object detected: ${objectDetection.name}`
                : objectProfiles.length
                  ? 'No trained object detected'
                  : 'No trained object yet'}
            </span>
            <strong>{objectDetection ? `${Math.round(objectDetection.score * 100)}%` : '--'}</strong>
          </div>
          {objectProfiles.length > 0 && (
            <div className="profile-list">
              {objectProfiles.map((profile) => (
                <div className="profile-row" key={profile.id}>
                  <span>{profile.name}</span>
                  <strong>{profile.samples.length} views</strong>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="motion-panel">
          <div className="motion-header">
            <ShieldCheck size={18} />
            <span>
              Motion state
              <InlineExplain label="Explain motion state" text={EXPLAIN.motion} />
            </span>
          </div>
          <strong>{analysis.motionState.replaceAll('-', ' ')}</strong>
          <div className="slip-track">
            <span style={{ width: `${Math.round(analysis.slipRisk * 100)}%` }} />
          </div>
          <small>
            Slip risk {Math.round(analysis.slipRisk * 100)}%
            <InlineExplain label="Explain slip risk" text={EXPLAIN.slip} />
          </small>
        </div>

        {hasCalibration && (
          <div className="calibration-note">
            Strong-hold calibration active{analysis.calibrated ? ' and matched to this pose.' : '.'}
          </div>
        )}

        <div className="diagnostics-panel">
          <div className="motion-header">
            <Activity size={18} />
            <span>
              Grip evidence
              <InlineExplain label="Explain grip evidence" text={EXPLAIN.gripEvidence} />
            </span>
          </div>
          {analysis.diagnostics.scoreBreakdown.map((item) => (
            <div className={`diagnostic-row ${item.impact}`} key={item.label}>
              <span>
                {item.label}
                <InlineExplain label={`Explain ${item.label}`} text={explainBreakdown(item.label)} />
              </span>
              <strong>{item.impact === 'negative' ? '-' : item.impact === 'positive' ? '+' : ''}{Math.round(Math.abs(item.value) * 100)}%</strong>
            </div>
          ))}
          {analysis.evidence.positiveReasons.length > 0 && (
            <p className="diagnostic-copy">Helps: {analysis.evidence.positiveReasons.slice(0, 3).join(', ')}.</p>
          )}
          {analysis.evidence.negativeReasons.length > 0 && (
            <p className="diagnostic-copy warn">Hurts: {analysis.evidence.negativeReasons.slice(0, 3).join(', ')}.</p>
          )}
        </div>

        <div className="diagnostics-panel">
          <div className="motion-header">
            <Crosshair size={18} />
            <span>
              Object evidence
              <InlineExplain label="Explain object evidence" text={EXPLAIN.objectEvidence} />
            </span>
          </div>
          <div className="diagnostic-row neutral">
            <span>
              Shape
              <InlineExplain label="Explain shape" text={EXPLAIN.shape} />
            </span>
            <strong>{previousObjectRef.current?.shape ?? 'unknown'}</strong>
          </div>
          <div className="diagnostic-row neutral">
            <span>
              Lock age
              <InlineExplain label="Explain lock age" text={EXPLAIN.lockAge} />
            </span>
            <strong>{previousObjectRef.current?.lockAgeFrames ?? 0}</strong>
          </div>
          <div className="diagnostic-row neutral">
            <span>
              Manual lock
              <InlineExplain label="Explain manual lock" text={EXPLAIN.manualLock} />
            </span>
            <strong>{previousObjectRef.current?.manuallyAdjusted ? 'yes' : 'no'}</strong>
          </div>
          {analysis.diagnostics.objectIssue && <p className="diagnostic-copy warn">{analysis.diagnostics.objectIssue}</p>}
          {analysis.diagnostics.gripIssue && <p className="diagnostic-copy warn">{analysis.diagnostics.gripIssue}</p>}
        </div>

        <div className="grip-points">
          <div className="motion-header">
            <Target size={18} />
            <span>
              Suggested points
              <InlineExplain label="Explain suggested points" text={EXPLAIN.suggestedPoints} />
            </span>
          </div>
          {analysis.recommendedGripPoints.length ? (
            analysis.recommendedGripPoints.map((point, index) => (
              <div className="point-row" key={`${point.label}-${index}`}>
                <span>
                  {point.label}
                  <InlineExplain label={`Explain ${point.label} point`} text={explainSuggestedPoint(point.label)} />
                </span>
                <strong>{Math.round(point.score * 100)}%</strong>
              </div>
            ))
          ) : (
            <p>Lock an object to reveal grip points.</p>
          )}
        </div>
      </aside>
    </main>
  );
}

function Metric({ label, value, text, info }: { label: string; value: number; text?: string; info: string }) {
  return (
    <div className="metric">
      <div className="metric-label">
        <span className="metric-name">
          {label}
          <InlineExplain label={`Explain ${label}`} text={info} />
        </span>
        <strong>{text ?? `${Math.round(value * 100)}%`}</strong>
      </div>
      <div className="metric-track">
        <span style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

function InlineExplain({ label, text, compact = false }: { label: string; text: string; compact?: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <span className={compact ? 'explain-wrap compact' : 'explain-wrap'}>
      <button
        className="eye-button"
        aria-label={label}
        aria-expanded={open}
        title={label}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((value) => !value);
        }}
        type="button"
      >
        <Eye size={compact ? 14 : 15} />
      </button>
      {open && <span className={compact ? 'explain-popover toolbar' : 'explain-popover'}>{text}</span>}
    </span>
  );
}

function explainBreakdown(label: string) {
  const normalized = label.toLowerCase();
  if (normalized.includes('mode')) return EXPLAIN.modeFit;
  if (normalized.includes('object')) return EXPLAIN.objectLockQuality;
  if (normalized.includes('independent')) return 'Whether the object has evidence separate from the hand: detector, manual click, texture, or clear edges.';
  if (normalized.includes('temporal')) return 'How long the same object lock has stayed stable across recent video frames.';
  if (normalized.includes('contact')) return EXPLAIN.contact;
  if (normalized.includes('finger')) return EXPLAIN.fingerWrap;
  if (normalized.includes('thumb')) return EXPLAIN.thumbSupport;
  if (normalized.includes('motion')) return EXPLAIN.motionStability;
  if (normalized.includes('calibration')) return EXPLAIN.calibration;
  return 'This diagnostic contributes to the current visual grip stability score.';
}

function explainSuggestedPoint(label: string) {
  const normalized = label.toLowerCase();
  if (normalized.includes('thumb')) return 'A suggested place for the thumb to oppose the fingers and stabilize the object.';
  if (normalized.includes('opposition')) return 'A matching point across the object that creates an opposing force pair with the thumb.';
  if (normalized.includes('support')) return 'A lower or side support point that helps keep the object from rotating or sliding.';
  if (normalized.includes('edge')) return 'A reachable object edge that appears useful for side grip contact, especially on phones and remotes.';
  return 'A reachable object boundary point that may improve visual grip stability.';
}

function averageBaseline(samples: GripCalibrationBaseline[]): GripCalibrationBaseline {
  const mode = mostCommonMode(samples);
  return {
    mode,
    gripPercentage: average(samples.map((sample) => sample.gripPercentage)),
    closureScore: average(samples.map((sample) => sample.closureScore)),
    enclosureScore: average(samples.map((sample) => sample.enclosureScore)),
    fingerCurlScore: average(samples.map((sample) => sample.fingerCurlScore)),
    fingerSegmentContactScore: average(samples.map((sample) => sample.fingerSegmentContactScore)),
    phoneSideGripScore: average(samples.map((sample) => sample.phoneSideGripScore)),
    pinchScore: average(samples.map((sample) => sample.pinchScore)),
    powerGripScore: average(samples.map((sample) => sample.powerGripScore)),
    thumbSupportScore: average(samples.map((sample) => sample.thumbSupportScore)),
    objectLockQuality: average(samples.map((sample) => sample.objectLockQuality)),
    createdAt: Date.now()
  };
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function describeObjectPatch(video: HTMLVideoElement, object: ObjectRegion) {
  const canvas = renderObjectPatch(video, object, DESCRIPTOR_SIZE);
  if (!canvas) return null;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return null;
  const { data } = context.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE);
  const bins = new Array(16).fill(0);
  let edgeTotal = 0;
  let samples = 0;

  for (let y = 1; y < DESCRIPTOR_SIZE - 1; y += 1) {
    for (let x = 1; x < DESCRIPTOR_SIZE - 1; x += 1) {
      const index = (y * DESCRIPTOR_SIZE + x) * 4;
      const alpha = data[index + 3];
      if (alpha < 16) continue;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const luminance = r * 0.2126 + g * 0.7152 + b * 0.0722;
      const right = (y * DESCRIPTOR_SIZE + x + 1) * 4;
      const down = ((y + 1) * DESCRIPTOR_SIZE + x) * 4;
      const rightLum = data[right] * 0.2126 + data[right + 1] * 0.7152 + data[right + 2] * 0.0722;
      const downLum = data[down] * 0.2126 + data[down + 1] * 0.7152 + data[down + 2] * 0.0722;
      bins[Math.min(3, Math.floor(r / 64))] += 1;
      bins[4 + Math.min(3, Math.floor(g / 64))] += 1;
      bins[8 + Math.min(3, Math.floor(b / 64))] += 1;
      bins[12 + Math.min(3, Math.floor(luminance / 64))] += 1;
      edgeTotal += Math.abs(luminance - rightLum) + Math.abs(luminance - downLum);
      samples += 1;
    }
  }

  if (samples < 80) return null;
  const normalizedBins = bins.map((value) => value / samples);
  const aspect = Math.max(object.radiusX, object.radiusY) / Math.max(1, Math.min(object.radiusX, object.radiusY));
  return [...normalizedBins, Math.min(1, edgeTotal / samples / 110), Math.min(1, aspect / 3)];
}

function createObjectThumbnail(video: HTMLVideoElement, object: ObjectRegion) {
  const canvas = renderObjectPatch(video, object, 144);
  return canvas?.toDataURL('image/jpeg', 0.82) ?? null;
}

function renderObjectPatch(video: HTMLVideoElement, object: ObjectRegion, size: number) {
  if (!video.videoWidth || !video.videoHeight) return null;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  if (!context) return null;

  const cropRadius = Math.max(object.radiusX, object.radiusY) * 1.35;
  const sourceSize = Math.max(16, Math.min(cropRadius * 2, video.videoWidth, video.videoHeight));
  const sourceX = Math.min(Math.max(0, object.center.x - sourceSize / 2), Math.max(0, video.videoWidth - sourceSize));
  const sourceY = Math.min(Math.max(0, object.center.y - sourceSize / 2), Math.max(0, video.videoHeight - sourceSize));

  context.clearRect(0, 0, size, size);
  context.save();
  context.beginPath();
  context.ellipse(size / 2, size / 2, size * 0.38, size * 0.38, object.angle, 0, Math.PI * 2);
  context.clip();
  context.drawImage(video, sourceX, sourceY, sourceSize, sourceSize, 0, 0, size, size);
  context.restore();
  return canvas;
}

function averageDescriptor(descriptors: number[][]) {
  if (!descriptors.length) return [];
  return descriptors[0].map((_value, index) => average(descriptors.map((descriptor) => descriptor[index] ?? 0)));
}

function matchObjectProfile(video: HTMLVideoElement, object: ObjectRegion | null, profiles: ObjectProfile[]): ObjectDetection {
  if (!object?.locked || !profiles.length) return null;
  const descriptor = describeObjectPatch(video, object);
  if (!descriptor) return null;
  const ranked = profiles
    .map((profile) => {
      const distance = descriptorDistance(descriptor, profile.descriptor);
      const score = Math.max(0, 1 - distance * 1.9);
      return {
        profileId: profile.id,
        name: profile.name,
        score,
        matched: score > 0.62
      };
    })
    .sort((a, b) => b.score - a.score);
  return ranked[0] ?? null;
}

function descriptorDistance(a: number[], b: number[]) {
  const length = Math.max(a.length, b.length, 1);
  let total = 0;
  for (let index = 0; index < length; index += 1) {
    total += Math.abs((a[index] ?? 0) - (b[index] ?? 0));
  }
  return total / length;
}

function mostCommonMode(samples: GripCalibrationBaseline[]): GripMode {
  const counts = samples.reduce<Record<string, number>>((accumulator, sample) => {
    accumulator[sample.mode] = (accumulator[sample.mode] ?? 0) + 1;
    return accumulator;
  }, {});
  return (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as GripMode | undefined) ?? 'uncertain';
}

function selectCalibrationBaseline(
  profiles: GripCalibrationProfiles,
  mode: GripMode,
  kind: 'strong' | 'weak'
) {
  return profiles[mode]?.[kind] ?? null;
}

function readInitialAlgorithmVersion(): AlgorithmVersion {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('version');
  if (fromUrl === 'v1' || fromUrl === 'v2') return fromUrl;
  const fromStorage = window.localStorage.getItem(ALGORITHM_VERSION_STORAGE_KEY);
  return fromStorage === 'v1' || fromStorage === 'v2' ? fromStorage : 'v2';
}

function saveAlgorithmVersion(version: AlgorithmVersion) {
  try {
    window.localStorage.setItem(ALGORITHM_VERSION_STORAGE_KEY, version);
    const url = new URL(window.location.href);
    url.searchParams.set('version', version);
    window.history.replaceState(null, '', url);
  } catch (error) {
    console.warn('Failed to save algorithm version', error);
  }
}

function loadObjectProfiles(): ObjectProfile[] {
  try {
    const raw = window.localStorage.getItem(OBJECT_PROFILES_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ObjectProfile[]) : [];
  } catch (error) {
    console.warn('Failed to load object profiles', error);
    return [];
  }
}

function saveObjectProfiles(profiles: ObjectProfile[]) {
  try {
    window.localStorage.setItem(OBJECT_PROFILES_STORAGE_KEY, JSON.stringify(profiles));
  } catch (error) {
    console.warn('Failed to save object profiles', error);
  }
}

function loadCalibrationProfiles(): GripCalibrationProfiles {
  try {
    const raw = window.localStorage.getItem(CALIBRATION_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as GripCalibrationProfiles) : {};
  } catch (error) {
    console.warn('Failed to load calibration profiles', error);
    return {};
  }
}

function saveCalibrationProfiles(profiles: GripCalibrationProfiles) {
  try {
    window.localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(profiles));
  } catch (error) {
    console.warn('Failed to save calibration profiles', error);
  }
}

function hasAnyCalibration(profiles: GripCalibrationProfiles) {
  return Object.values(profiles).some((profile) => profile?.strong || profile?.weak);
}
