import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Box,
  Camera,
  CheckCircle,
  Crosshair,
  Eye,
  FlipHorizontal2,
  FolderOpen,
  Hand,
  Images,
  Minus,
  Pause,
  Play,
  Power,
  Plus,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Target,
  Upload,
  X
} from 'lucide-react';
import { analyzeGrip, createEmptyAnalysis } from './vision/gripAnalysis';
import { analyzeGripV3 } from './vision/v3GripAnalysis';
import { palmCenter, pointsToPixelSpace, subtract } from './vision/geometry';
import { inferObjectRegion } from './vision/objectTracking';
import {
  browserObjectDescriptorProvider,
  createMaskedCanvasObjectTrainingSample,
  cropBoundsFor,
  matchObjectProfiles,
  trainingReadiness,
  trainObjectProfileV2,
  type ObjectProfileMatch,
  type ObjectProfileV2,
  type ObjectTrainingSampleV2,
  type CanvasObjectMaskOptions
} from './vision/objectProfile';
import { drawTrackingOverlay } from './vision/drawing';
import { createVisionEngine, type VisionEngine, type VisionModelStatus } from './vision/visionEngine';
import { TrackingStabilizer } from './vision/stabilization';
import { createV3AnalyzeFrameRequest, DEFAULT_V3_ENDPOINT, requestV3FrameAnalysis } from './vision/v3Inference';
import type {
  AlgorithmVersion,
  GripAnalysis,
  GripCalibrationBaseline,
  GripCalibrationProfiles,
  GripMode,
  Landmark,
  DetectedObjectBox,
  ObjectIdentitySignal,
  ObjectRegion,
  Point,
  V3PerceptionResponse
} from './vision/types';

const INITIAL_MODEL_STATUS: VisionModelStatus = {
  hands: 'idle',
  detector: 'idle',
  segmenter: 'idle'
};

const CALIBRATION_STORAGE_KEY = 'grip-lab-calibration-profiles-v2';
const ALGORITHM_VERSION_STORAGE_KEY = 'grip-lab-algorithm-version';
const OBJECT_PROFILES_STORAGE_KEY = 'grip-lab-object-profiles-v2';
const VITE_ENV = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env;
const V3_ENDPOINT = VITE_ENV?.VITE_GRIPSENSE_V3_ENDPOINT ?? DEFAULT_V3_ENDPOINT;
const V3_REQUEST_INTERVAL_MS = 420;

type LocalWritableFile = {
  write(data: Blob | string): Promise<void>;
  close(): Promise<void>;
};

type LocalFileHandle = {
  createWritable(): Promise<LocalWritableFile>;
};

type LocalDirectoryHandle = {
  getFileHandle(name: string, options?: { create?: boolean }): Promise<LocalFileHandle>;
};

type WindowWithFolderPicker = Window & {
  showDirectoryPicker?: () => Promise<LocalDirectoryHandle>;
};

type V3Runtime = {
  status: 'idle' | 'pending' | 'ready' | 'fallback';
  message: string;
  endpoint: string;
  result: V3PerceptionResponse | null;
  receivedAt: number | null;
  lastRequestAt: number;
  latencyMs: number | null;
};

type PendingUploadReview = {
  id: string;
  name: string;
  canvas: HTMLCanvasElement;
  imageDataUrl: string;
  cropX: number;
  cropY: number;
  cropSize: number;
  maskScale: number;
  maskShape: CanvasObjectMaskOptions['maskShape'];
  source: ObjectTrainingSampleV2['source'];
};

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
  version: 'Choose V1 for the original permissive heuristic, V2 for stricter object-first scoring, or V3 for local-server perception fusion with V2 fallback.',
  gripQuality: 'Visual grip stability estimated from the camera. It is not real physical force.',
  state: 'The tracking state says what the app believes is happening: no hand, hand only, object uncertain, grip detected, strong hold, or slip risk.',
  mode: 'Grip mode is the type of hold the app thinks it sees, such as phone-side, pinch, power, hook, open hand, or uncertain.',
  objectLockQuality: 'How much the app trusts that the highlighted region is a real object rather than your hand or background.',
  motion: 'Motion state compares hand and object movement. Sustained mismatch raises slip risk.',
  slip: 'Slip risk rises only when the object and hand move differently across several frames.',
  gripEvidence: 'These rows show what raised or lowered the grip score.',
  objectEvidence: 'These rows describe the object tracker: shape, how long it has been locked, and whether you manually adjusted it.',
  detectorLabel: 'The class label from the generic object detector. For phones this should usually say cell phone or phone. If it says unknown, the tracker is using geometry/profile evidence instead.',
  shape: 'The object shape guessed by the tracker: phone-like, ellipse, unknown, or detector/manual fallback.',
  lockAge: 'How many video frames the current object lock has survived. A higher value usually means a more stable lock.',
  manualLock: 'Yes means you clicked or dragged the object lock yourself. Manual locks are trusted more than automatic guesses.',
  suggestedPoints: 'Suggested grip points are possible places for thumb, fingers, or support contact based on the current object outline.',
  modeFit: 'How well the current hand-object pose matches the selected grip mode.',
  contact: 'How much visible finger segment contact appears near the object boundary.',
  fingerWrap: 'How much the fingers appear to curl around or contain the object.',
  thumbSupport: 'How much the thumb appears to support or oppose the fingers.',
  motionStability: 'How stable the object-hand motion is over recent frames.',
  calibration: 'How much saved strong/weak calibration is affecting the current score.',
  objectTrainer: 'Open a separate enrollment portal. Live grip scoring pauses there so you can capture or upload object images without needing the app to believe a grip is already happening.',
  trainerSteps: 'The guided flow is add object images, review quality suggestions, train a local profile, save it, then enable it for live detection.',
  captureView: 'Captures the current webcam frame as an object training image. Center the object in the frame; the app will warn if the image looks weak but will not block you.',
  uploadView: 'Adds object images from your computer. Use multiple angles, backgrounds, and distances to make matching more reliable.',
  trainProfile: 'Asks for an object name, then builds a local visual profile. This is profile matching, not a neural fine-tune.',
  clearViews: 'Removes the temporary captured views before training. Already trained profiles stay saved.',
  folderSave: 'Mirrors trained profiles and thumbnails into a local folder when the browser supports folder access.',
  objectIdentity: 'How closely the current locked object matches the trained profile. Low match blocks strong grip in V2.',
  trainedProfiles: 'Saved local object profiles. Enabled profiles are used for live matching; disabled profiles stay saved but are ignored.'
} as const;

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const trainingVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offlineVideoInputRef = useRef<HTMLInputElement | null>(null);
  const engineRef = useRef<VisionEngine | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const offlineVideoUrlRef = useRef<string | null>(null);
  const previousObjectRef = useRef<ObjectRegion | null>(null);
  const previousPalmRef = useRef<Point | null>(null);
  const manualPointRef = useRef<Point | null>(null);
  const manualScaleRef = useRef(1);
  const draggingObjectRef = useRef(false);
  const lockObjectRef = useRef(false);
  const pausedRef = useRef(false);
  const mirroredRef = useRef(true);
  const lastDetectorRunRef = useRef(0);
  const detectorBoxRef = useRef<DetectedObjectBox | null>(null);
  const autoRetryRef = useRef(false);
  const stabilizerRef = useRef(new TrackingStabilizer());
  const algorithmVersionRef = useRef<AlgorithmVersion>(readInitialAlgorithmVersion());
  const calibrationProfilesRef = useRef<GripCalibrationProfiles>({});
  const objectProfilesRef = useRef<ObjectProfileV2[]>([]);
  const objectDetectionRef = useRef<ObjectProfileMatch>(null);
  const lastObjectMatchRef = useRef(0);
  const v3RuntimeRef = useRef<V3Runtime>({
    status: 'idle',
    message: 'V3 server idle. Select V3 and start tracking to begin fusion.',
    endpoint: V3_ENDPOINT,
    result: null,
    receivedAt: null,
    lastRequestAt: 0,
    latencyMs: null
  });
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const profileDirectoryRef = useRef<LocalDirectoryHandle | null>(null);
  const pausedBeforeTrainerRef = useRef(false);
  const calibrationCaptureRef = useRef<{
    active: boolean;
    kind: 'strong' | 'weak';
    start: number;
    samples: GripCalibrationBaseline[];
  }>({ active: false, kind: 'strong', start: 0, samples: [] });

  const [cameraState, setCameraState] = useState<'idle' | 'requesting' | 'live' | 'blocked'>('idle');
  const [mediaMode, setMediaMode] = useState<'live' | 'offline'>('live');
  const [offlineVideoName, setOfflineVideoName] = useState('');
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
  const [trainingSamples, setTrainingSamples] = useState<ObjectTrainingSampleV2[]>([]);
  const [pendingUploads, setPendingUploads] = useState<PendingUploadReview[]>([]);
  const [objectProfiles, setObjectProfiles] = useState<ObjectProfileV2[]>([]);
  const [objectDetection, setObjectDetection] = useState<ObjectProfileMatch>(null);
  const [v3Runtime, setV3Runtime] = useState<V3Runtime>(() => v3RuntimeRef.current);
  const [trainingStatus, setTrainingStatus] = useState('Open the object portal to capture or upload training images.');
  const [trainerOpen, setTrainerOpen] = useState(false);
  const [namePromptOpen, setNamePromptOpen] = useState(false);
  const [folderStatus, setFolderStatus] = useState('Folder save not connected.');

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
    const video = trainingVideoRef.current;
    const stream = streamRef.current;
    if (!trainerOpen || !video || !stream) return;
    video.srcObject = stream;
    void video.play();
  }, [trainerOpen]);

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      engineRef.current?.dispose();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (offlineVideoUrlRef.current) URL.revokeObjectURL(offlineVideoUrlRef.current);
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

  const trainerReadiness = useMemo(() => trainingReadiness(trainingSamples), [trainingSamples]);
  const pendingUpload = pendingUploads[0] ?? null;
  const pendingUploadPreview = useMemo(() => {
    if (!pendingUpload) return null;
    return createMaskedCanvasObjectTrainingSample(pendingUpload.canvas, {
      cropBounds: {
        x: pendingUpload.cropX,
        y: pendingUpload.cropY,
        size: pendingUpload.cropSize
      },
      maskScale: pendingUpload.maskScale,
      maskShape: pendingUpload.maskShape,
      source: pendingUpload.source,
      sourceName: pendingUpload.name
    });
  }, [pendingUpload]);

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

  const updateV3Runtime = useCallback((next: V3Runtime) => {
    v3RuntimeRef.current = next;
    setV3Runtime(next);
  }, []);

  const resetV3Runtime = useCallback((message = 'V3 server idle. Select V3 and start tracking to begin fusion.') => {
    updateV3Runtime({
      status: 'idle',
      message,
      endpoint: V3_ENDPOINT,
      result: null,
      receivedAt: null,
      lastRequestAt: 0,
      latencyMs: null
    });
  }, [updateV3Runtime]);

  const resetTrackingRefs = useCallback(() => {
    manualPointRef.current = null;
    manualScaleRef.current = 1;
    draggingObjectRef.current = false;
    previousObjectRef.current = null;
    previousPalmRef.current = null;
    detectorBoxRef.current = null;
    objectDetectionRef.current = null;
    resetV3Runtime();
    stabilizerRef.current.reset();
    setLocked(false);
    setObjectDetection(null);
  }, [resetV3Runtime]);

  const startCamera = useCallback(async () => {
    if (cameraState === 'requesting' || (cameraState === 'live' && mediaMode === 'live')) return;
    setCameraState('requesting');
    try {
      if (offlineVideoUrlRef.current) {
        URL.revokeObjectURL(offlineVideoUrlRef.current);
        offlineVideoUrlRef.current = null;
      }
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
      video.removeAttribute('src');
      video.srcObject = stream;
      await video.play();
      setMediaMode('live');
      setOfflineVideoName('');
      resetTrackingRefs();
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
  }, [cameraState, loadVisionEngine, mediaMode, resetTrackingRefs]);

  const startOfflineVideo = useCallback(async (file: File) => {
    if (!file.type.startsWith('video/')) {
      setAnalysis(createEmptyAnalysis('Upload a video file to start offline review.'));
      return;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (offlineVideoUrlRef.current) URL.revokeObjectURL(offlineVideoUrlRef.current);
    const url = URL.createObjectURL(file);
    offlineVideoUrlRef.current = url;

    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.srcObject = null;
    video.src = url;
    video.loop = false;
    video.muted = true;
    setMediaMode('offline');
    setOfflineVideoName(file.name);
    setCameraState('live');
    setPaused(false);
    resetTrackingRefs();
    setAnalysis(createEmptyAnalysis('Offline review loaded. Press play to visualize grip over this video.'));

    await loadVisionEngine();
    await waitForVideoMetadata(video);
    await video.play().catch(() => {
      setAnalysis(createEmptyAnalysis('Offline video loaded. Press play on the video to begin analysis.'));
    });
    runLoop();
  }, [loadVisionEngine, resetTrackingRefs]);

  const handleOfflineVideoUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) void startOfflineVideo(file);
  }, [startOfflineVideo]);

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

  const scheduleV3Inference = useCallback((
    video: HTMLVideoElement,
    hand: Landmark[] | null,
    object: ObjectRegion | null,
    v2Analysis: GripAnalysis,
    objectIdentity: ObjectIdentitySignal,
    timestamp: number
  ) => {
    if (algorithmVersionRef.current !== 'v3') return;

    const current = v3RuntimeRef.current;
    if (current.status === 'pending' || timestamp - current.lastRequestAt < V3_REQUEST_INTERVAL_MS) return;

    const request = createV3AnalyzeFrameRequest({
      video,
      mirrored: mirroredRef.current,
      timestamp,
      hand,
      object,
      v2Analysis,
      objectIdentity
    });

    if (!request) {
      updateV3Runtime({
        ...current,
        status: 'fallback',
        message: 'V3 frame unavailable; V2 fallback active.',
        result: null,
        receivedAt: performance.now(),
        lastRequestAt: timestamp,
        latencyMs: null
      });
      return;
    }

    updateV3Runtime({
      ...current,
      status: 'pending',
      message: 'V3 server analyzing frame.',
      lastRequestAt: timestamp
    });

    void requestV3FrameAnalysis(request, { endpoint: current.endpoint }).then((result) => {
      if (algorithmVersionRef.current !== 'v3') return;
      const latest = v3RuntimeRef.current;
      if (result.ok) {
        updateV3Runtime({
          ...latest,
          status: 'ready',
          message: 'V3 server active; fusing mask, mesh, contact, and temporal evidence.',
          result: result.response,
          receivedAt: result.receivedAt,
          latencyMs: result.response.latencyMs
        });
        return;
      }

      updateV3Runtime({
        ...latest,
        status: 'fallback',
        message: `${formatV3ClientStatus(result.status)}; V2 fallback active.`,
        result: null,
        receivedAt: result.receivedAt,
        latencyMs: null
      });
    });
  }, [updateV3Runtime]);

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
        const activeAlgorithmVersion = algorithmVersionRef.current;
        const fallbackAlgorithmVersion: AlgorithmVersion = activeAlgorithmVersion === 'v3' ? 'v2' : activeAlgorithmVersion;
        const rawObject = inferObjectRegion({
          video,
          hand,
          previous: previousObjectRef.current,
          manualPoint: manualPointRef.current,
          manualScale: manualScaleRef.current,
          locked: lockObjectRef.current,
          detectorBox: detectorBoxRef.current,
          algorithmVersion: fallbackAlgorithmVersion
        });
        object = stabilizerRef.current.stabilizeObject(rawObject, timestamp);
        if (timestamp - lastObjectMatchRef.current > 420) {
          lastObjectMatchRef.current = timestamp;
          const descriptor = object ? browserObjectDescriptorProvider.describe(video, object) : null;
          const enabledProfiles = objectProfilesRef.current.filter((profile) => profile.enabled !== false);
          const match = matchObjectProfiles(descriptor, enabledProfiles);
          objectDetectionRef.current = match;
          setObjectDetection(match);
        }
        const enabledProfileCount = objectProfilesRef.current.filter((profile) => profile.enabled !== false).length;
        const objectIdentity = {
          hasProfiles: enabledProfileCount > 0,
          score: objectDetectionRef.current?.score ?? 0,
          matched: objectDetectionRef.current?.matched ?? false,
          name: objectDetectionRef.current?.name ?? null
        };
        const handVelocityForSlip =
          hand && previousPalmRef.current ? subtract(palmCenter(hand), previousPalmRef.current) : { x: 0, y: 0 };
        const persistentSlipScore = stabilizerRef.current.updatePersistentSlip(handVelocityForSlip, object);
        const rawFrameAnalysis = analyzeGrip(hand, object, previousPalmRef.current, {
          persistentSlipScore,
          algorithmVersion: fallbackAlgorithmVersion,
          objectIdentity
        });
        const baseFrameAnalysis = analyzeGrip(hand, object, previousPalmRef.current, {
          persistentSlipScore,
          calibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'strong'),
          weakCalibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'weak'),
          algorithmVersion: fallbackAlgorithmVersion,
          objectIdentity
        });
        if (activeAlgorithmVersion === 'v3') {
          scheduleV3Inference(video, hand, object, baseFrameAnalysis, objectIdentity, timestamp);
          frameAnalysis = stabilizerRef.current.stabilizeAnalysis(
            analyzeGripV3({
              baseAnalysis: baseFrameAnalysis,
              hand,
              object,
              response: v3RuntimeRef.current.result,
              receivedAt: v3RuntimeRef.current.receivedAt,
              now: timestamp,
              endpoint: v3RuntimeRef.current.endpoint
            }),
            timestamp
          );
        } else {
          frameAnalysis = stabilizerRef.current.stabilizeAnalysis(baseFrameAnalysis, timestamp);
        }
        updateCalibrationCapture(frameAnalysis, timestamp);
        previousObjectRef.current = object;
        previousPalmRef.current = frameAnalysis.palmCenter;
        setAnalysis(frameAnalysis);
      }

      drawTrackingOverlay(context, canvas.width, canvas.height, mirroredRef.current, hand, object, frameAnalysis);
      animationRef.current = requestAnimationFrame(tick);
    };

    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = requestAnimationFrame(tick);
  }, [analysis, scheduleV3Inference, updateCalibrationCapture]);

  const resetObject = useCallback(() => {
    manualPointRef.current = null;
    manualScaleRef.current = 1;
    draggingObjectRef.current = false;
    previousObjectRef.current = null;
    detectorBoxRef.current = null;
    stabilizerRef.current.reset();
    calibrationCaptureRef.current = { active: false, kind: calibrationKind, start: 0, samples: [] };
    resetV3Runtime();
    setLocked(false);
    setCalibrating(false);
    setAnalysis(createEmptyAnalysis('Object reset. Place it between your thumb and fingers to relock.'));
  }, [calibrationKind, resetV3Runtime]);

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
      resetV3Runtime(
        version === 'v3'
          ? 'V3 selected. Start tracking to connect to the local perception server.'
          : 'V3 server idle. Select V3 and start tracking to begin fusion.'
      );
      setLocked(false);
      setCalibrating(false);
      setAlgorithmVersion(version);
      saveAlgorithmVersion(version);
      setAnalysis(
        createEmptyAnalysis(
          version === 'v3'
            ? 'V3 selected. It will fuse local-server perception with V2 fallback when the server is unavailable.'
            : version === 'v2'
            ? 'V2 selected. It will require independent object evidence before scoring grip.'
            : 'V1 selected. It uses the original permissive grip heuristic.'
        )
      );
    },
    [algorithmVersion, resetV3Runtime]
  );

  const openTrainerPortal = useCallback(() => {
    pausedBeforeTrainerRef.current = pausedRef.current;
    setPaused(true);
    setTrainerOpen(true);
    setTrainingStatus('Live grip scoring is paused. Capture webcam frames or upload object images.');
  }, []);

  const closeTrainerPortal = useCallback(() => {
    setTrainerOpen(false);
    setNamePromptOpen(false);
    setPaused(pausedBeforeTrainerRef.current);
  }, []);

  const addTrainingSample = useCallback((sample: ObjectTrainingSampleV2) => {
    const nextSamples = [...trainingSamples, sample].slice(-24);
    setTrainingSamples(nextSamples);
    const nextReadiness = trainingReadiness(nextSamples);
    const qualityNote =
      sample.qualityLabel === 'Good view'
        ? 'Good view added.'
        : `${sample.qualityLabel}: ${sample.descriptor.reasons.join(', ') || 'you can train, but add clearer object-only angles if possible'}.`;
    setTrainingStatus(`${qualityNote} ${nextReadiness.message}`);
  }, [trainingSamples]);

  const captureObjectTrainingView = useCallback(() => {
    const video = trainingVideoRef.current ?? videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      setTrainingStatus('Start the camera first, then capture a training frame.');
      return;
    }

    const frame = videoFrameToCanvas(video);
    if (!frame) {
      setTrainingStatus('Could not capture this frame. Keep the object visible and try again.');
      return;
    }
    setPendingUploads((current) => [createPendingUploadReview(frame, `camera-frame-${current.length + 1}`, 'camera'), ...current]);
    setTrainingStatus('Captured frame. Crop and mask the object before adding it to training.');
  }, []);

  const captureLockedObjectTrainingView = useCallback(() => {
    const video = trainingVideoRef.current ?? videoRef.current;
    const object = previousObjectRef.current;
    if (!video || !video.videoWidth || !video.videoHeight || !object?.locked) {
      setTrainingStatus('No object lock is available. Use Capture frame, upload an image, or click the object first.');
      return;
    }
    const frame = videoFrameToCanvas(video);
    const cropBounds = frame ? cropBoundsFor(video, object) : null;
    if (!frame || !cropBounds) {
      setTrainingStatus('Could not crop the locked object. Capture the full frame or upload an image instead.');
      return;
    }
    setPendingUploads((current) => [
      createPendingUploadReview(frame, `locked-object-${current.length + 1}`, 'locked-crop', cropBounds),
      ...current
    ]);
    setTrainingStatus('Locked object captured. Adjust crop/mask if needed before adding it to training.');
  }, []);

  const updatePendingUpload = useCallback((patch: Partial<PendingUploadReview>) => {
    setPendingUploads((current) => {
      const [first, ...rest] = current;
      if (!first) return current;
      const next = { ...first, ...patch };
      const maxSize = Math.min(next.canvas.width, next.canvas.height);
      next.cropSize = clampNumber(next.cropSize, Math.min(80, maxSize), maxSize);
      next.cropX = clampNumber(next.cropX, 0, Math.max(0, next.canvas.width - next.cropSize));
      next.cropY = clampNumber(next.cropY, 0, Math.max(0, next.canvas.height - next.cropSize));
      next.maskScale = clampNumber(next.maskScale, 0.35, 1);
      return [next, ...rest];
    });
  }, []);

  const acceptPendingUpload = useCallback(() => {
    if (!pendingUploadPreview) {
      setTrainingStatus('Could not read that crop. Tighten the crop around the object and try again.');
      return;
    }
    addTrainingSample(pendingUploadPreview);
    setPendingUploads((current) => current.slice(1));
  }, [addTrainingSample, pendingUploadPreview]);

  const skipPendingUpload = useCallback(() => {
    setPendingUploads((current) => current.slice(1));
    setTrainingStatus('Upload skipped. Review the next image or upload another object view.');
  }, []);

  const uploadTrainingImages = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    const uploads: PendingUploadReview[] = [];
    for (const file of files) {
      const canvas = await imageFileToCanvas(file);
      if (canvas) {
        uploads.push(createPendingUploadReview(canvas, file.name, 'upload'));
      }
    }
    event.target.value = '';
    if (!uploads.length) {
      setTrainingStatus('No uploaded images could be read. Try a JPG, PNG, or WebP.');
      return;
    }
    setPendingUploads((current) => [...current, ...uploads]);
    setTrainingStatus(`Review crop and mask for ${uploads.length} uploaded image${uploads.length === 1 ? '' : 's'} before adding to training.`);
  }, []);

  const chooseProfileFolder = useCallback(async () => {
    const picker = (window as WindowWithFolderPicker).showDirectoryPicker;
    if (!picker) {
      setFolderStatus('This browser does not support folder saving. Profiles still persist in browser storage.');
      return;
    }
    try {
      profileDirectoryRef.current = await picker();
      setFolderStatus('Folder save connected.');
      if (objectProfilesRef.current.length) {
        await mirrorObjectProfilesToFolder(profileDirectoryRef.current, objectProfilesRef.current);
        setFolderStatus('Folder save connected and synced.');
      }
    } catch (error) {
      console.warn('Folder selection failed', error);
      setFolderStatus('Folder save skipped. Profiles still persist in browser storage.');
    }
  }, []);

  const finalizeObjectTraining = useCallback(async () => {
    if (pendingUploads.length) {
      setTrainingStatus('Finish the upload crop and mask review before training this profile.');
      return;
    }
    const name = objectName.trim();
    if (!name) {
      setNamePromptOpen(true);
      setTrainingStatus('Name the object to finish training.');
      return;
    }
    if (!trainingSamples.length) {
      setTrainingStatus('Add at least one webcam capture or uploaded image before training.');
      return;
    }
    const result = trainObjectProfileV2(name, trainingSamples);
    if (!result.ok) {
      setTrainingStatus(result.message);
      return;
    }
    const profile = result.profile;
    const profiles = [profile, ...objectProfiles.filter((current) => current.name.toLowerCase() !== name.toLowerCase())].slice(0, 6);
    objectProfilesRef.current = profiles;
    setObjectProfiles(profiles);
    saveObjectProfiles(profiles);
    setTrainingSamples([]);
    setNamePromptOpen(false);
    let folderMessage = '';
    const picker = (window as WindowWithFolderPicker).showDirectoryPicker;
    if (!profileDirectoryRef.current && picker) {
      try {
        profileDirectoryRef.current = await picker();
        setFolderStatus('Folder save connected.');
      } catch {
        setFolderStatus('Folder save skipped. Profiles still persist in browser storage.');
      }
    }
    if (profileDirectoryRef.current) {
      try {
        await mirrorObjectProfilesToFolder(profileDirectoryRef.current, profiles);
        folderMessage = ' Saved to the selected local folder.';
        setFolderStatus('Latest profile saved to folder.');
      } catch (error) {
        console.warn('Profile folder save failed', error);
        folderMessage = ' Browser storage saved; folder write failed.';
        setFolderStatus('Folder write failed.');
      }
    }
    setTrainingStatus(result.message + folderMessage + ' Enable it below, then resume live tracking to verify detection.');
  }, [objectName, objectProfiles, pendingUploads.length, trainingSamples]);

  const trainObjectProfile = useCallback(() => {
    void finalizeObjectTraining();
  }, [finalizeObjectTraining]);

  const toggleObjectProfile = useCallback((id: string) => {
    const profiles = objectProfiles.map((profile) =>
      profile.id === id ? { ...profile, enabled: profile.enabled === false } : profile
    );
    objectProfilesRef.current = profiles;
    setObjectProfiles(profiles);
    saveObjectProfiles(profiles);
    if (profileDirectoryRef.current) {
      void mirrorObjectProfilesToFolder(profileDirectoryRef.current, profiles);
    }
  }, [objectProfiles]);

  const deleteTrainingSample = useCallback((id: string) => {
    setTrainingSamples((current) => {
      const next = current.filter((sample) => sample.id !== id);
      setTrainingStatus(next.length ? trainingReadiness(next).message : 'Training views cleared. Capture new masked views.');
      return next;
    });
  }, []);

  const clearTrainingSamples = useCallback(() => {
    setTrainingSamples([]);
    setPendingUploads([]);
    setTrainingStatus('Training views cleared. Capture new masked views.');
  }, []);

  return (
    <main className="app-shell">
      <section className="camera-workspace" aria-label="Live grip tracking workspace">
        <video
          ref={videoRef}
          className={mirrored ? 'camera-feed mirrored' : 'camera-feed'}
          playsInline
          muted
          controls={mediaMode === 'offline'}
        />
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
              <button
                type="button"
                className={algorithmVersion === 'v3' ? 'version-button active' : 'version-button'}
                onClick={() => selectAlgorithmVersion('v3')}
                aria-pressed={algorithmVersion === 'v3'}
              >
                V3
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
            <button className="tool-button" onClick={() => offlineVideoInputRef.current?.click()} aria-label="Upload offline review video">
              <Upload size={17} />
              <span>Offline video</span>
            </button>
            <input
              ref={offlineVideoInputRef}
              className="hidden-file-input"
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/*"
              onChange={handleOfflineVideoUpload}
            />
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

        {mediaMode === 'offline' && cameraState === 'live' && !trainerOpen && (
          <div className="offline-review-overlay" aria-label="Offline grip video review">
            <div className="offline-glass-panel offline-left">
              <p className="eyebrow">Offline review</p>
              <h2>{analysis.gripPercentage}%</h2>
              <strong>{analysis.guidance}</strong>
              <span>{offlineVideoName || 'Uploaded video'}</span>
              <div className="offline-mini-row">
                <span>State</span>
                <strong>{analysis.diagnostics.state}</strong>
              </div>
              <div className="offline-mini-row">
                <span>Mode</span>
                <strong>{analysis.diagnostics.mode}</strong>
              </div>
              <div className="offline-mini-row">
                <span>Object</span>
                <strong>{objectDetection?.matched ? objectDetection.name : 'not matched'}</strong>
              </div>
            </div>
            <div className="offline-glass-panel offline-right">
              <p className="eyebrow">Parameters</p>
              <GlassMetric label="Confidence" value={analysis.confidence} />
              <GlassMetric label="Lock" value={analysis.objectLockQuality} />
              <GlassMetric label="Closure" value={analysis.closureScore} />
              <GlassMetric label="Contact" value={analysis.evidence.fingerSegmentContactScore} />
              <GlassMetric label="Thumb" value={analysis.thumbOpposition} />
              <GlassMetric label="Slip" value={analysis.slipRisk} danger />
            </div>
          </div>
        )}
      </section>

      {trainerOpen && (
        <section className="training-portal" aria-label="Object training portal">
          <div className="training-portal-shell">
            <div className="portal-head">
              <div>
                <p className="eyebrow">Object enrollment</p>
                <h2>
                  Training portal
                  <InlineExplain label="Explain training portal" text={EXPLAIN.objectTrainer} />
                </h2>
              </div>
              <button className="icon-button" type="button" onClick={closeTrainerPortal} aria-label="Close training portal">
                <X size={18} />
              </button>
            </div>
            <div className="portal-grid">
              <div className="portal-camera">
                <video className={mirrored ? 'portal-video mirrored' : 'portal-video'} ref={trainingVideoRef} playsInline muted />
                <div className="portal-camera-status">
                  <Pause size={16} />
                  Live grip tracking paused
                </div>
              </div>
              <div className="portal-side">
                <div className="trainer-steps" aria-label="Object profile training steps">
                  {['Add images', 'Crop/mask', 'Name object', 'Train', 'Enable live'].map((step, index) => (
                    <span
                      className={
                        (index === 0 && trainingSamples.length > 0) ||
                        (index === 1 && trainingSamples.length > 0) ||
                        (index === 2 && objectName.trim()) ||
                        (index === 3 && objectProfiles.some((profile) => profile.name.toLowerCase() === objectName.trim().toLowerCase())) ||
                        (index === 4 && objectProfiles.some((profile) => profile.enabled !== false))
                          ? 'complete'
                          : ''
                      }
                      key={step}
                    >
                      {index + 1}. {step}
                    </span>
                  ))}
                  <InlineExplain label="Explain training steps" text={EXPLAIN.trainerSteps} compact />
                </div>
                <div className="portal-actions">
                  <button type="button" onClick={captureObjectTrainingView}>
                    <Camera size={16} />
                    Capture frame
                  </button>
                  <button type="button" onClick={captureLockedObjectTrainingView}>
                    <Crosshair size={16} />
                    Capture lock
                  </button>
                  <button type="button" onClick={() => fileInputRef.current?.click()}>
                    <Upload size={16} />
                    Upload images
                  </button>
                  <button type="button" onClick={chooseProfileFolder}>
                    <FolderOpen size={16} />
                    Save folder
                  </button>
                </div>
                <input
                  ref={fileInputRef}
                  className="hidden-file-input"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  multiple
                  onChange={(event) => void uploadTrainingImages(event)}
                />
                <p className="diagnostic-copy">{trainingStatus}</p>
                <p className="diagnostic-copy">{folderStatus}</p>
                {pendingUpload && (
                  <div className="upload-review" aria-label="Uploaded image crop and mask review">
                    <div className="upload-review-head">
                      <div>
                        <p className="eyebrow">Crop and mask</p>
                        <strong>{pendingUpload.name}</strong>
                      </div>
                      <span>{pendingUploads.length} pending</span>
                    </div>
                    <div className="upload-review-grid">
                      <div className="upload-source-preview">
                        <img src={pendingUpload.imageDataUrl} alt="Uploaded training source" />
                        <span className="crop-box" style={cropOverlayStyle(pendingUpload)} />
                      </div>
                      <div className="upload-mask-preview">
                        {pendingUploadPreview ? (
                          <img src={pendingUploadPreview.imageDataUrl} alt="Masked object preview" />
                        ) : (
                          <span>Adjust crop</span>
                        )}
                      </div>
                    </div>
                    <div className="upload-control-grid">
                      <SliderControl
                        label="Crop size"
                        min={Math.min(80, pendingUpload.canvas.width, pendingUpload.canvas.height)}
                        max={Math.min(pendingUpload.canvas.width, pendingUpload.canvas.height)}
                        value={pendingUpload.cropSize}
                        onChange={(value) => updatePendingUpload({ cropSize: value })}
                      />
                      <SliderControl
                        label="Crop X"
                        min={0}
                        max={Math.max(0, pendingUpload.canvas.width - pendingUpload.cropSize)}
                        value={pendingUpload.cropX}
                        onChange={(value) => updatePendingUpload({ cropX: value })}
                      />
                      <SliderControl
                        label="Crop Y"
                        min={0}
                        max={Math.max(0, pendingUpload.canvas.height - pendingUpload.cropSize)}
                        value={pendingUpload.cropY}
                        onChange={(value) => updatePendingUpload({ cropY: value })}
                      />
                      <SliderControl
                        label="Mask"
                        min={35}
                        max={100}
                        value={Math.round(pendingUpload.maskScale * 100)}
                        onChange={(value) => updatePendingUpload({ maskScale: value / 100 })}
                      />
                    </div>
                    <div className="mask-toggle" aria-label="Mask shape">
                      <button
                        type="button"
                        className={pendingUpload.maskShape === 'ellipse' ? 'active' : ''}
                        onClick={() => updatePendingUpload({ maskShape: 'ellipse' })}
                      >
                        Ellipse
                      </button>
                      <button
                        type="button"
                        className={pendingUpload.maskShape === 'rect' ? 'active' : ''}
                        onClick={() => updatePendingUpload({ maskShape: 'rect' })}
                      >
                        Rectangle
                      </button>
                    </div>
                    {pendingUploadPreview && (
                      <p className={pendingUploadPreview.quality >= 0.56 ? 'diagnostic-copy' : 'diagnostic-copy warn'}>
                        Image quality {Math.round(pendingUploadPreview.quality * 100)}% - {pendingUploadPreview.qualityLabel}: {pendingUploadPreview.descriptor.reasons.join(', ') || 'view is usable'}.
                      </p>
                    )}
                    <div className="portal-train-row">
                      <button type="button" onClick={acceptPendingUpload}>
                        <CheckCircle size={16} />
                        Add masked image
                      </button>
                      <button type="button" onClick={skipPendingUpload}>
                        Skip
                      </button>
                    </div>
                  </div>
                )}
                {trainingSamples.length > 0 && (
                  <div className="sample-strip portal-samples" aria-label="Object training images">
                    {trainingSamples.map((sample, index) => (
                      <div className="sample-card" key={sample.id}>
                        <img src={sample.imageDataUrl} alt={`Training image ${index + 1}`} />
                        <span className={sample.quality >= 0.56 ? 'sample-quality good' : 'sample-quality'}>
                          {sample.qualityLabel} {Math.round(sample.quality * 100)}%
                        </span>
                        <small>{sample.sourceName ?? sample.source ?? 'training image'}</small>
                        <button type="button" onClick={() => deleteTrainingSample(sample.id)} aria-label={`Remove training image ${index + 1}`}>
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="portal-train-row">
                  <button type="button" onClick={trainObjectProfile}>
                    <Sparkles size={16} />
                    Train profile
                  </button>
                  <button type="button" onClick={clearTrainingSamples}>
                    Clear images
                  </button>
                </div>
                {namePromptOpen && (
                  <div className="name-prompt" role="dialog" aria-label="Name object before training">
                    <label className="object-name-field">
                      <span>What should I name this object?</span>
                      <input
                        value={objectName}
                        onChange={(event) => setObjectName(event.target.value)}
                        placeholder="Phone, mug, remote..."
                        maxLength={36}
                        autoFocus
                      />
                    </label>
                    <button type="button" onClick={trainObjectProfile}>
                      <CheckCircle size={16} />
                      Train with this name
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

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

        {algorithmVersion === 'v3' && analysis.v3 && (
          <div className="v3-panel">
            <div className="motion-header">
              <Activity size={18} />
              <span>V3 perception</span>
            </div>
            <div className={analysis.v3.status === 'server' ? 'v3-status ready' : 'v3-status fallback'}>
              <span>{v3Runtime.status === 'pending' ? 'pending' : analysis.v3.status}</span>
              <strong>{analysis.v3.usedServerResult ? `${Math.round(analysis.v3.modelConfidence * 100)}%` : 'V2'}</strong>
            </div>
            <p className={analysis.v3.usedServerResult ? 'diagnostic-copy' : 'diagnostic-copy warn'}>{v3Runtime.message}</p>
            <div className="v3-score-grid">
              <V3Score label="Object" value={analysis.v3.subScores.objectEvidence} />
              <V3Score label="Hand" value={analysis.v3.subScores.handEvidence} />
              <V3Score label="Contact" value={analysis.v3.subScores.contactEvidence} />
              <V3Score label="Temporal" value={analysis.v3.subScores.temporalEvidence} />
            </div>
            <div className="diagnostic-row neutral">
              <span>Latency</span>
              <strong>{analysis.v3.serverLatencyMs === null ? '--' : `${Math.round(analysis.v3.serverLatencyMs)} ms`}</strong>
            </div>
            <div className={analysis.v3.reason === 'strong_hold' ? 'diagnostic-row positive' : 'diagnostic-row neutral'}>
              <span>V3 diagnostic</span>
              <strong>{formatIssueCategory(analysis.v3.reason ?? 'none')}</strong>
            </div>
          </div>
        )}

        <div className="trainer-panel">
          <div className="motion-header">
            <Images size={18} />
            <span>
              Object profiles
              <InlineExplain label="Explain object trainer V2" text={EXPLAIN.objectTrainer} />
            </span>
          </div>
          <button className="portal-button" type="button" onClick={openTrainerPortal}>
            <Images size={17} />
            Open training portal
          </button>
          <p className="diagnostic-copy">{trainingStatus}</p>
          <div className={objectDetection?.matched ? 'detected-object matched' : 'detected-object'}>
            <Box size={17} />
            <span>
              {objectDetection?.matched
                ? `Object detected: ${objectDetection.name}`
                : objectProfiles.some((profile) => profile.enabled !== false)
                  ? 'Enabled object not detected'
                  : objectProfiles.length
                    ? 'All profiles disabled'
                    : 'No trained object yet'}
            </span>
            <strong>{objectDetection ? `${Math.round(objectDetection.score * 100)}%` : '--'}</strong>
          </div>
          <div className="identity-meter">
            <span>
              Object identity match
              <InlineExplain label="Explain object identity match" text={EXPLAIN.objectIdentity} />
            </span>
            <strong>{objectDetection ? `${Math.round(objectDetection.score * 100)}%` : '--'}</strong>
          </div>
          {objectProfiles.length > 0 && (
            <div className="profile-list">
              <div className="motion-header compact-heading">
                <span>
                  Saved profiles
                  <InlineExplain label="Explain saved profiles" text={EXPLAIN.trainedProfiles} compact />
                </span>
              </div>
              {objectProfiles.map((profile) => {
                const status = profileLiveStatus(profile, objectDetection, analysis);
                return (
                  <div className={`profile-row ${status.kind}`} key={profile.id}>
                    <button
                      type="button"
                      className="profile-toggle"
                      onClick={() => toggleObjectProfile(profile.id)}
                      aria-pressed={profile.enabled !== false}
                      aria-label={`${profile.enabled === false ? 'Enable' : 'Disable'} ${profile.name}`}
                    >
                      <Power size={14} />
                    </button>
                    <span>{profile.name}</span>
                    <strong>{status.label}</strong>
                  </div>
                );
              })}
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
              Detector
              <InlineExplain label="Explain detector label" text={EXPLAIN.detectorLabel} />
            </span>
            <strong>
              {previousObjectRef.current?.detectorLabel
                ? `${previousObjectRef.current.detectorLabel} ${Math.round((previousObjectRef.current.detectorScore ?? 0) * 100)}%`
                : 'none'}
            </strong>
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
          <div className={analysis.diagnostics.issueCategory === 'none' ? 'diagnostic-row positive' : 'diagnostic-row negative'}>
            <span>
              Issue type
              <InlineExplain label="Explain issue type" text="Classifies the current blocker as object tracking, hand pose, motion/slip, trained-object identity, or none." />
            </span>
            <strong>{formatIssueCategory(analysis.diagnostics.issueCategory)}</strong>
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

function videoFrameToCanvas(video: HTMLVideoElement) {
  if (!video.videoWidth || !video.videoHeight) return null;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext('2d');
  if (!context) return null;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function imageFileToCanvas(file: File) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    if (!context) return null;
    context.drawImage(image, 0, 0);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function createPendingUploadReview(
  canvas: HTMLCanvasElement,
  name: string,
  source: ObjectTrainingSampleV2['source'],
  initialCrop?: { x: number; y: number; size: number }
): PendingUploadReview {
  const cropSize = initialCrop?.size ?? Math.min(canvas.width, canvas.height);
  return {
    id: crypto.randomUUID(),
    name,
    canvas,
    imageDataUrl: canvas.toDataURL('image/jpeg', 0.82),
    cropX: initialCrop?.x ?? Math.max(0, (canvas.width - cropSize) / 2),
    cropY: initialCrop?.y ?? Math.max(0, (canvas.height - cropSize) / 2),
    cropSize,
    maskScale: 0.86,
    maskShape: inferMaskShape(canvas),
    source
  };
}

function cropOverlayStyle(review: PendingUploadReview): React.CSSProperties {
  return {
    left: `${(review.cropX / Math.max(1, review.canvas.width)) * 100}%`,
    top: `${(review.cropY / Math.max(1, review.canvas.height)) * 100}%`,
    width: `${(review.cropSize / Math.max(1, review.canvas.width)) * 100}%`,
    height: `${(review.cropSize / Math.max(1, review.canvas.height)) * 100}%`
  };
}

function inferMaskShape(canvas: HTMLCanvasElement): CanvasObjectMaskOptions['maskShape'] {
  const aspectRatio = Math.max(canvas.width, canvas.height) / Math.max(1, Math.min(canvas.width, canvas.height));
  return aspectRatio > 1.45 ? 'rect' : 'ellipse';
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function profileLiveStatus(profile: ObjectProfileV2, detection: ObjectProfileMatch, analysis: GripAnalysis) {
  if (profile.enabled === false) return { kind: 'disabled', label: 'disabled' };
  if (detection?.profileId !== profile.id || !detection.matched) return { kind: 'enabled', label: 'enabled' };
  if (analysis.gripPercentage > 0 && ['Grip detected', 'Strong hold', 'Slip risk'].includes(analysis.diagnostics.state)) {
    return { kind: 'gripping', label: 'grip active' };
  }
  return { kind: 'visible', label: 'in frame' };
}

function waitForVideoMetadata(video: HTMLVideoElement) {
  if (video.videoWidth && video.videoHeight) return Promise.resolve();
  return new Promise<void>((resolve) => {
    const handleLoaded = () => {
      video.removeEventListener('loadedmetadata', handleLoaded);
      resolve();
    };
    video.addEventListener('loadedmetadata', handleLoaded, { once: true });
  });
}

function GlassMetric({ label, value, danger = false }: { label: string; value: number; danger?: boolean }) {
  const percentage = Math.round(value * 100);
  return (
    <div className={danger ? 'glass-metric danger' : 'glass-metric'}>
      <div>
        <span>{label}</span>
        <strong>{percentage}%</strong>
      </div>
      <div className="glass-track">
        <span style={{ width: `${percentage}%` }} />
      </div>
    </div>
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

function V3Score({ label, value }: { label: string; value: number }) {
  return (
    <div className="v3-score">
      <span>{label}</span>
      <strong>{Math.round(value * 100)}%</strong>
      <div className="metric-track">
        <span style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

function SliderControl({
  label,
  min,
  max,
  value,
  onChange
}: {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const safeMax = Math.max(min, max);
  const safeValue = clampNumber(value, min, safeMax);
  return (
    <label className="slider-control">
      <span>
        {label}
        <strong>{Math.round(safeValue)}</strong>
      </span>
      <input
        type="range"
        min={Math.round(min)}
        max={Math.round(safeMax)}
        value={Math.round(safeValue)}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
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

function formatIssueCategory(category: GripAnalysis['diagnostics']['issueCategory']) {
  if (category === 'object_problem') return 'object';
  if (category === 'pose_problem') return 'pose';
  if (category === 'motion_problem') return 'motion';
  if (category === 'identity_problem') return 'identity';
  if (category === 'object_uncertain') return 'object uncertain';
  if (category === 'hand_occluded') return 'hand occluded';
  if (category === 'contact_uncertain') return 'contact uncertain';
  if (category === 'slip_risk') return 'slip risk';
  if (category === 'server_unavailable') return 'server unavailable';
  if (category === 'strong_hold') return 'strong hold';
  return 'none';
}

function formatV3ClientStatus(status: 'timeout' | 'network_error' | 'http_error' | 'invalid_response' | 'frame_unavailable') {
  if (status === 'timeout') return 'V3 server timeout';
  if (status === 'network_error') return 'V3 server unavailable';
  if (status === 'http_error') return 'V3 server error';
  if (status === 'invalid_response') return 'V3 server response invalid';
  return 'V3 frame unavailable';
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
  if (fromUrl === 'v1' || fromUrl === 'v2' || fromUrl === 'v3') return fromUrl;
  const fromStorage = window.localStorage.getItem(ALGORITHM_VERSION_STORAGE_KEY);
  return fromStorage === 'v1' || fromStorage === 'v2' || fromStorage === 'v3' ? fromStorage : 'v2';
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

function loadObjectProfiles(): ObjectProfileV2[] {
  try {
    const raw = window.localStorage.getItem(OBJECT_PROFILES_STORAGE_KEY);
    return raw ? normalizeObjectProfiles(JSON.parse(raw) as ObjectProfileV2[]) : [];
  } catch (error) {
    console.warn('Failed to load object profiles', error);
    return [];
  }
}

function saveObjectProfiles(profiles: ObjectProfileV2[]) {
  try {
    window.localStorage.setItem(OBJECT_PROFILES_STORAGE_KEY, JSON.stringify(profiles));
  } catch (error) {
    console.warn('Failed to save object profiles', error);
  }
}

function normalizeObjectProfiles(profiles: ObjectProfileV2[]) {
  return profiles.map((profile) => ({
    ...profile,
    enabled: profile.enabled !== false
  }));
}

async function mirrorObjectProfilesToFolder(handle: LocalDirectoryHandle, profiles: ObjectProfileV2[]) {
  await writeLocalFile(
    handle,
    'gripsense-object-profiles.json',
    JSON.stringify(profiles, null, 2),
    'application/json'
  );
  for (const profile of profiles) {
    for (const [index, sample] of profile.samples.entries()) {
      await writeLocalFile(
        handle,
        `${sanitizeFileName(profile.name)}-${index + 1}.jpg`,
        dataUrlToBlob(sample.imageDataUrl),
        'image/jpeg'
      );
    }
  }
}

async function writeLocalFile(handle: LocalDirectoryHandle, name: string, data: Blob | string, type: string) {
  const file = await handle.getFileHandle(name, { create: true });
  const writable = await file.createWritable();
  await writable.write(typeof data === 'string' ? new Blob([data], { type }) : data);
  await writable.close();
}

function dataUrlToBlob(dataUrl: string) {
  const [header, payload] = dataUrl.split(',');
  const mime = header.match(/data:(.*?);base64/)?.[1] ?? 'image/jpeg';
  const bytes = Uint8Array.from(atob(payload ?? ''), (character) => character.charCodeAt(0));
  return new Blob([bytes], { type: mime });
}

function sanitizeFileName(name: string) {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || 'object';
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
