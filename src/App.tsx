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
  Square,
  Target,
  Upload,
  Video,
  X
} from 'lucide-react';
import { analyzeGrip, createEmptyAnalysis } from './vision/gripAnalysis';
import { analyzeGripV3 } from './vision/v3GripAnalysis';
import { clamp as clampUnit, distance, ellipsePoint, FINGERTIP_INDICES, handSize, palmCenter, pointsToPixelSpace, subtract } from './vision/geometry';
import { inferObjectRegion } from './vision/objectTracking';
import {
  browserObjectDescriptorProvider,
  createMaskedCanvasObjectTrainingSample,
  cropBoundsFor,
  findObjectProfileCandidates,
  matchObjectProfiles,
  objectRegionFromProfileCandidate,
  trainingReadiness,
  trainObjectProfileV2,
  trainingCoverage,
  profileStrength,
  type ObjectProfileMatch,
  type ObjectProfileCandidate,
  type ObjectProfileV2,
  type ObjectTrainingSampleV2,
  type CanvasObjectMaskOptions,
  type TrainingViewRole
} from './vision/objectProfile';
import { drawTrackingOverlay } from './vision/drawing';
import { createVisionEngine, type VisionEngine, type VisionModelStatus } from './vision/visionEngine';
import { TrackingStabilizer } from './vision/stabilization';
import { createV3AnalyzeFrameRequest, DEFAULT_V3_ENDPOINT, requestV3FrameAnalysis } from './vision/v3Inference';
import {
  analyzeGripWithRfdetr,
  compensateRfdetrResponseForHandMotion,
  createInitialRfdetrRuntime,
  DEFAULT_RFDETR_ENDPOINT,
  EMPTY_RFDETR_TRACK,
  isRfdetrResultFresh,
  refineRfdetrOfflineTimeline,
  requestRfdetrFrameAnalysis,
  RFDETR_OFFLINE_INTERVAL_MS,
  RFDETR_REQUEST_INTERVAL_MS,
  type RfdetrRuntime,
  type RfdetrTrackState
} from './vision/rfdetr';
import { EMPTY_TEMPORAL_IDENTITY, temporalIdentityToMatch, updateTemporalIdentity, type TemporalIdentityState } from './vision/temporalIdentity';
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
const RFDETR_ENDPOINT = VITE_ENV?.VITE_GRIPSENSE_RFDETR_ENDPOINT ?? DEFAULT_RFDETR_ENDPOINT;
const V3_REQUEST_INTERVAL_MS = 420;
const V3_PROFILE_SEARCH_INTERVAL_MS = 520;
const OFFLINE_TIMELINE_INTERVAL_MS = 180;
const V5_BASE_TARGET_THRESHOLD = 0.22;
const OFFLINE_BASE_TARGET_THRESHOLD = 0.12;
const V6_LIVE_TARGET_THRESHOLD = 0.16;
const V5_BASE_TRACK_GRACE_MISSES = 6;
const TRAINING_VIEW_ROLES: TrainingViewRole[] = ['front', 'side', 'rotated', 'in-hand', 'alone', 'negative'];

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
  viewRole: TrainingViewRole;
};

type BaseObjectCandidate = DetectedObjectBox & {
  candidateId: string;
  index: number;
  trackId: number;
  missedFrames: number;
  seenFrames: number;
  center: Point;
  radiusX: number;
  radiusY: number;
};

type BaseTrackState = {
  nextId: number;
  tracks: BaseObjectCandidate[];
};

type BaseClassSummary = {
  key: string;
  label: string;
  count: number;
  bestScore: number;
  enabled: boolean;
};

type OfflineTimelinePoint = {
  time: number;
  grip: number;
  confidence: number;
  objectMatch: number;
  lock: number;
  contact: number;
  closure: number;
  thumb: number;
  enclosure: number;
  slip: number;
  weak: boolean;
  guidance: string;
  object: string;
  mode: string;
  state: string;
  objectX: number | null;
  objectY: number | null;
  objectRadiusX?: number | null;
  objectRadiusY?: number | null;
  objectAngle?: number | null;
  palmX: number | null;
  palmY: number | null;
  rfdetrObjectScore?: number;
  rfdetrContact?: number;
  rfdetrLatencyMs?: number | null;
};

type OfflineReviewVersion = 'v1' | 'v2';

type RecordedClip = {
  file: File;
  url: string;
  durationMs: number;
};

type OfflineSegment = {
  start: number;
  end: number;
  reason: string;
};

type OfflineReport = {
  generatedAt: string;
  videoName: string;
  duration: number;
  points: number;
  averageGrip: number;
  peakGrip: number;
  averageObjectMatch: number;
  averageLock: number;
  weakSegments: OfflineSegment[];
  slipEvents: OfflineSegment[];
  summary: string;
} | null;

type OfflineV2TrackState = {
  candidate: BaseObjectCandidate | null;
  confidence: number;
  ageFrames: number;
  missedFrames: number;
  lastSeenAt: number;
};

type LiveIdentityMemory = {
  profileId: string;
  name: string;
  score: number;
  matched: boolean;
  seenFrames: number;
  missedFrames: number;
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
  version: 'Choose V1 for the original permissive heuristic, V2 for stricter object-first scoring, V3 for local-server perception fusion, V4 for trained-object-first matching, V5 for target-object selection, V6 for offline-style sticky live tracking, V7 for the Offline V1 live copy, or V8 for RF-DETR live masks.',
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
  trainedProfiles: 'Saved local object profiles. Enabled profiles are used for live matching; disabled profiles stay saved but are ignored.',
  v4Temporal: 'V4/V5/V6 require the same enabled trained object to match across several frames before it turns green. This reduces flicker and wrong detections.',
  v5Target: 'V5/V6 combine the base object detector with trained profiles. V5 prefers explicit target selection; V6 can also auto-follow the best hand-near object with sticky tracking.',
  contactGate: 'V5/V6 require current visual contact between the selected object and hand. If the object drops away, grip cannot remain high.',
  profileStrength: 'Profile strength estimates training coverage across front, side, rotated, in-hand, alone, and negative examples.'
} as const;

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const trainingVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offlineVideoInputRef = useRef<HTMLInputElement | null>(null);
  const uploadV1InputRef = useRef<HTMLInputElement | null>(null);
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
  const mediaModeRef = useRef<'live' | 'offline'>('live');
  const mirroredRef = useRef(true);
  const lastDetectorRunRef = useRef(0);
  const detectorBoxRef = useRef<DetectedObjectBox | null>(null);
  const baseTrackerRef = useRef<BaseTrackState>({ nextId: 1, tracks: [] });
  const baseObjectCandidatesRef = useRef<BaseObjectCandidate[]>([]);
  const autoRetryRef = useRef(false);
  const stabilizerRef = useRef(new TrackingStabilizer());
  const algorithmVersionRef = useRef<AlgorithmVersion>(readInitialAlgorithmVersion());
  const calibrationProfilesRef = useRef<GripCalibrationProfiles>({});
  const objectProfilesRef = useRef<ObjectProfileV2[]>([]);
  const objectDetectionRef = useRef<ObjectProfileMatch>(null);
  const lastObjectMatchRef = useRef(0);
  const v3ProfileCandidatesRef = useRef<ObjectProfileCandidate[]>([]);
  const lastProfileSearchRef = useRef(0);
  const temporalIdentityRef = useRef<TemporalIdentityState>(EMPTY_TEMPORAL_IDENTITY);
  const targetProfileIdRef = useRef<string | null>(null);
  const targetBaseIdRef = useRef<string | null>(null);
  const analysisRef = useRef<GripAnalysis>(createEmptyAnalysis());
  const showObjectLabelsRef = useRef(false);
  const baseClassEnabledRef = useRef<Record<string, boolean>>({ person: false });
  const offlineTimelineRef = useRef<OfflineTimelinePoint[]>([]);
  const offlineReportRef = useRef<OfflineReport>(null);
  const lastOfflineTimelineRef = useRef(0);
  const offlineReviewVersionRef = useRef<OfflineReviewVersion>('v2');
  const offlineBatchProcessingRef = useRef(false);
  const uploadOnlyExportStartedRef = useRef(false);
  const offlineV2TrackRef = useRef<OfflineV2TrackState>({ candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 });
  const liveV6TrackRef = useRef<OfflineV2TrackState>({ candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 });
  const rfdetrTrackRef = useRef<RfdetrTrackState>(EMPTY_RFDETR_TRACK);
  const liveIdentityMemoryRef = useRef<LiveIdentityMemory>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const recordingStartedAtRef = useRef(0);
  const recordedClipUrlRef = useRef<string | null>(null);
  const v3RuntimeRef = useRef<V3Runtime>({
    status: 'idle',
    message: 'V3 server idle. Select V3 and start tracking to begin fusion.',
    endpoint: V3_ENDPOINT,
    result: null,
    receivedAt: null,
    lastRequestAt: 0,
    latencyMs: null
  });
  const rfdetrRuntimeRef = useRef<RfdetrRuntime>(createInitialRfdetrRuntime(RFDETR_ENDPOINT));
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const profileDirectoryRef = useRef<LocalDirectoryHandle | null>(null);
  const cropDragRef = useRef<{
    mode: 'move' | 'resize';
    startX: number;
    startY: number;
    cropX: number;
    cropY: number;
    cropSize: number;
  } | null>(null);
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
  const [offlineAnalysisPhase, setOfflineAnalysisPhase] = useState<'idle' | 'processing' | 'reviewing' | 'complete'>('idle');
  const [offlineReviewVersion, setOfflineReviewVersion] = useState<OfflineReviewVersion>('v2');
  const [offlineVideoExporting, setOfflineVideoExporting] = useState(false);
  const [offlineVideoExportStatus, setOfflineVideoExportStatus] = useState('');
  const [uploadOnlyMode, setUploadOnlyMode] = useState(false);
  const [uploadOnlyStatus, setUploadOnlyStatus] = useState('');
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
  const [v3ProfileCandidates, setV3ProfileCandidates] = useState<ObjectProfileCandidate[]>([]);
  const [baseObjectCandidates, setBaseObjectCandidates] = useState<BaseObjectCandidate[]>([]);
  const [temporalIdentity, setTemporalIdentity] = useState<TemporalIdentityState>(EMPTY_TEMPORAL_IDENTITY);
  const [targetProfileId, setTargetProfileId] = useState<string | null>(null);
  const [targetBaseId, setTargetBaseId] = useState<string | null>(null);
  const [showObjectLabels, setShowObjectLabels] = useState(false);
  const [baseClassEnabled, setBaseClassEnabled] = useState<Record<string, boolean>>({ person: false });
  const [baseClassSummary, setBaseClassSummary] = useState<BaseClassSummary[]>([]);
  const [offlineTimeline, setOfflineTimeline] = useState<OfflineTimelinePoint[]>([]);
  const [offlineReport, setOfflineReport] = useState<OfflineReport>(null);
  const [recordingState, setRecordingState] = useState<'idle' | 'recording' | 'ready' | 'unsupported'>('idle');
  const [recordingElapsedMs, setRecordingElapsedMs] = useState(0);
  const [recordedClip, setRecordedClip] = useState<RecordedClip | null>(null);
  const [v3Runtime, setV3Runtime] = useState<V3Runtime>(() => v3RuntimeRef.current);
  const [rfdetrRuntime, setRfdetrRuntime] = useState<RfdetrRuntime>(() => rfdetrRuntimeRef.current);
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
    mediaModeRef.current = mediaMode;
  }, [mediaMode]);

  useEffect(() => {
    offlineReviewVersionRef.current = offlineReviewVersion;
  }, [offlineReviewVersion]);

  useEffect(() => {
    analysisRef.current = analysis;
  }, [analysis]);

  useEffect(() => {
    lockObjectRef.current = locked;
  }, [locked]);

  useEffect(() => {
    algorithmVersionRef.current = algorithmVersion;
  }, [algorithmVersion]);

  useEffect(() => {
    targetProfileIdRef.current = targetProfileId;
  }, [targetProfileId]);

  useEffect(() => {
    targetBaseIdRef.current = targetBaseId;
  }, [targetBaseId]);

  useEffect(() => {
    showObjectLabelsRef.current = showObjectLabels;
  }, [showObjectLabels]);

  useEffect(() => {
    baseClassEnabledRef.current = baseClassEnabled;
  }, [baseClassEnabled]);

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
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
      engineRef.current?.dispose();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (offlineVideoUrlRef.current) URL.revokeObjectURL(offlineVideoUrlRef.current);
      if (recordedClipUrlRef.current) URL.revokeObjectURL(recordedClipUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (recordingState !== 'recording') return;
    const interval = window.setInterval(() => {
      setRecordingElapsedMs(Date.now() - recordingStartedAtRef.current);
    }, 250);
    return () => window.clearInterval(interval);
  }, [recordingState]);

  const modelSummary = useMemo(() => {
    const ready = Object.values(modelStatus).filter((state) => state === 'ready').length;
    const failed = Object.values(modelStatus).filter((state) => state === 'failed').length;
    if (modelStatus.hands === 'failed') return 'Retry hand model';
    if (modelStatus.hands === 'ready') return `${ready}/3 models ready${failed ? `, ${failed} fallback` : ''}`;
    if (modelStatus.hands === 'loading') return 'Loading hand model';
    return 'Models idle';
  }, [modelStatus]);

  const trainerReadiness = useMemo(() => trainingReadiness(trainingSamples), [trainingSamples]);
  const trainerCoverage = useMemo(() => trainingCoverage(trainingSamples), [trainingSamples]);
  const trainerStrength = useMemo(() => profileStrength(trainingSamples), [trainingSamples]);
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
      sourceName: pendingUpload.name,
      viewRole: pendingUpload.viewRole
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

  const updateRfdetrRuntime = useCallback((next: RfdetrRuntime) => {
    rfdetrRuntimeRef.current = next;
    setRfdetrRuntime(next);
  }, []);

  const resetRfdetrRuntime = useCallback((message = 'RF-DETR server idle. Select V8 or Offline V2 to begin RF-DETR analysis.') => {
    updateRfdetrRuntime({
      status: 'idle',
      message,
      endpoint: RFDETR_ENDPOINT,
      result: null,
      receivedAt: null,
      lastRequestAt: 0,
      latencyMs: null
    });
    rfdetrTrackRef.current = EMPTY_RFDETR_TRACK;
  }, [updateRfdetrRuntime]);

  const resetTrackingRefs = useCallback(() => {
    manualPointRef.current = null;
    manualScaleRef.current = 1;
    draggingObjectRef.current = false;
    previousObjectRef.current = null;
    previousPalmRef.current = null;
    detectorBoxRef.current = null;
    baseTrackerRef.current = { nextId: 1, tracks: [] };
    baseObjectCandidatesRef.current = [];
    offlineV2TrackRef.current = { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 };
    liveV6TrackRef.current = { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 };
    rfdetrTrackRef.current = EMPTY_RFDETR_TRACK;
    liveIdentityMemoryRef.current = null;
    objectDetectionRef.current = null;
    v3ProfileCandidatesRef.current = [];
    temporalIdentityRef.current = EMPTY_TEMPORAL_IDENTITY;
    targetProfileIdRef.current = null;
    targetBaseIdRef.current = null;
    lastProfileSearchRef.current = 0;
    resetV3Runtime();
    resetRfdetrRuntime();
    stabilizerRef.current.reset();
    setLocked(false);
    setObjectDetection(null);
    setV3ProfileCandidates([]);
    setBaseObjectCandidates([]);
    setTemporalIdentity(EMPTY_TEMPORAL_IDENTITY);
  }, [resetRfdetrRuntime, resetV3Runtime]);

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
      mediaModeRef.current = 'live';
      setMediaMode('live');
      setOfflineVideoName('');
      setOfflineAnalysisPhase('idle');
      offlineTimelineRef.current = [];
      offlineReportRef.current = null;
      setOfflineTimeline([]);
      setOfflineReport(null);
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
    mediaModeRef.current = 'offline';
    setMediaMode('offline');
    setOfflineVideoName(file.name);
    setOfflineAnalysisPhase('processing');
    offlineTimelineRef.current = [];
    offlineReportRef.current = null;
    lastOfflineTimelineRef.current = 0;
    offlineV2TrackRef.current = { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 };
    setOfflineTimeline([]);
    setOfflineReport(null);
    setCameraState('live');
    setPaused(false);
    resetTrackingRefs();
    setAnalysis(createEmptyAnalysis('Offline review processing. The app will auto-detect hand-near objects and build a timeline.'));

    await loadVisionEngine();
    await waitForVideoMetadata(video);
    runLoop();

    if (offlineReviewVersionRef.current === 'v2') {
      offlineBatchProcessingRef.current = true;
      setOfflineAnalysisPhase('processing');
      setAnalysis(createEmptyAnalysis('Offline V2 is scanning the full video before review. It will use past and future frames to smooth the result.'));
      video.currentTime = 0;
      video.playbackRate = 2.5;
      await video.play().catch(() => {
        offlineBatchProcessingRef.current = false;
        video.playbackRate = 1;
        setOfflineAnalysisPhase('reviewing');
        setAnalysis(createEmptyAnalysis('Offline video loaded. Press play once to let V2 process the full video.'));
      });
      return;
    }

    offlineBatchProcessingRef.current = false;
    video.playbackRate = 1;
    setOfflineAnalysisPhase('reviewing');
    await video.play().catch(() => {
      setOfflineAnalysisPhase('reviewing');
      setAnalysis(createEmptyAnalysis('Offline video loaded. Press play on the video to begin analysis.'));
    });
  }, [loadVisionEngine, resetTrackingRefs]);

  const handleOfflineVideoUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    setUploadOnlyMode(false);
    setUploadOnlyStatus('');
    if (file) void startOfflineVideo(file);
  }, [startOfflineVideo]);

  const handleUploadV1Video = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setUploadOnlyMode(false);
      setUploadOnlyStatus('');
      setAnalysis(createEmptyAnalysis('Upload V1 needs a video file.'));
      return;
    }
    offlineReviewVersionRef.current = 'v2';
    setOfflineReviewVersion('v2');
    uploadOnlyExportStartedRef.current = false;
    setOfflineVideoExportStatus('');
    setUploadOnlyMode(true);
    setUploadOnlyStatus('Upload V1 is scanning the full clip before download.');
    void startOfflineVideo(file);
  }, [startOfflineVideo]);

  const clearRecordedClip = useCallback(() => {
    if (recordedClipUrlRef.current) URL.revokeObjectURL(recordedClipUrlRef.current);
    recordedClipUrlRef.current = null;
    setRecordedClip(null);
    setRecordingElapsedMs(0);
    setRecordingState('idle');
  }, []);

  const startLiveRecording = useCallback(async () => {
    if (recordingState === 'recording') return;
    if (typeof MediaRecorder === 'undefined') {
      setRecordingState('unsupported');
      setAnalysis(createEmptyAnalysis('This browser cannot record camera clips for offline review.'));
      return;
    }

    if (!streamRef.current || mediaModeRef.current !== 'live') {
      await startCamera();
    }

    const stream = streamRef.current;
    if (!stream) {
      setAnalysis(createEmptyAnalysis('Start the camera before recording an offline clip.'));
      return;
    }

    const mimeType = mediaRecorderTypeFor('webm');
    if (!mimeType) {
      setRecordingState('unsupported');
      setAnalysis(createEmptyAnalysis('This browser cannot record WebM clips for offline review.'));
      return;
    }

    if (recordedClipUrlRef.current) URL.revokeObjectURL(recordedClipUrlRef.current);
    recordedClipUrlRef.current = null;
    setRecordedClip(null);
    recordedChunksRef.current = [];
    recordingStartedAtRef.current = Date.now();
    setRecordingElapsedMs(0);

    const recorder = new MediaRecorder(stream, { mimeType });
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordedChunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const durationMs = Math.max(0, Date.now() - recordingStartedAtRef.current);
      const blob = new Blob(recordedChunksRef.current, { type: mimeType });
      recordedChunksRef.current = [];
      recorderRef.current = null;
      if (!blob.size) {
        setRecordingState('idle');
        setAnalysis(createEmptyAnalysis('Recording did not capture video. Try recording again.'));
        return;
      }
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const file = new File([blob], `gripsense-recording-${stamp}.webm`, { type: mimeType, lastModified: Date.now() });
      const url = URL.createObjectURL(blob);
      recordedClipUrlRef.current = url;
      setRecordedClip({ file, url, durationMs });
      setRecordingElapsedMs(durationMs);
      setRecordingState('ready');
      setAnalysis(createEmptyAnalysis('Recording captured. Choose Offline V1 or Offline V2 to process it.'));
    };
    recorder.start(250);
    setRecordingState('recording');
    setAnalysis(createEmptyAnalysis('Recording live camera. Stop recording to process this clip offline.'));
  }, [recordingState, startCamera]);

  const stopLiveRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== 'recording') return;
    recorder.stop();
  }, []);

  const processRecordedClip = useCallback((version: OfflineReviewVersion) => {
    if (!recordedClip) return;
    offlineReviewVersionRef.current = version;
    setOfflineReviewVersion(version);
    const file = recordedClip.file;
    if (recordedClipUrlRef.current) URL.revokeObjectURL(recordedClipUrlRef.current);
    recordedClipUrlRef.current = null;
    setRecordedClip(null);
    setRecordingState('idle');
    void startOfflineVideo(file);
  }, [recordedClip, startOfflineVideo]);

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

  const scheduleRfdetrInference = useCallback((video: HTMLVideoElement, hand: Landmark[] | null, timestamp: number, offline = false) => {
    const current = rfdetrRuntimeRef.current;
    const interval = offline ? RFDETR_OFFLINE_INTERVAL_MS : RFDETR_REQUEST_INTERVAL_MS;
    if (current.status === 'pending' || timestamp - current.lastRequestAt < interval) return;
    const requestPalm = hand ? palmCenter(hand) : null;

    updateRfdetrRuntime({
      ...current,
      status: 'pending',
      message: offline ? 'RF-DETR scanning offline frame.' : 'RF-DETR analyzing live frame.',
      lastRequestAt: timestamp
    });

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), offline ? 2400 : 1800);
    void requestRfdetrFrameAnalysis({
      video,
      hand,
      endpoint: current.endpoint,
      mirrored: mirroredRef.current,
      signal: controller.signal
    }).then((result) => {
      window.clearTimeout(timeout);
      const latest = rfdetrRuntimeRef.current;
      const stillRelevant = algorithmVersionRef.current === 'v8' || (mediaModeRef.current === 'offline' && offlineReviewVersionRef.current === 'v2');
      if (!stillRelevant) return;
      if (result.ok) {
        updateRfdetrRuntime({
          ...latest,
          status: 'ready',
          message: `RF-DETR active (${result.response.detections.length} detection${result.response.detections.length === 1 ? '' : 's'}).`,
          result: result.response,
          resultPalm: requestPalm,
          receivedAt: result.receivedAt,
          latencyMs: result.response.latencyMs ?? result.response.detections[0]?.latencyMs ?? null
        });
        return;
      }
      updateRfdetrRuntime({
        ...latest,
        status: 'unavailable',
        message: `RF-DETR unavailable: ${result.status}.`,
        result: null,
        resultPalm: null,
        receivedAt: result.receivedAt,
        latencyMs: null
      });
    });
  }, [updateRfdetrRuntime]);

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
        const offlineReview = mediaModeRef.current === 'offline';
        const offlineV2Review = offlineReview && offlineReviewVersionRef.current === 'v2';
        const activeAlgorithmVersion = algorithmVersionRef.current;
        const liveOfflineV1Review = activeAlgorithmVersion === 'v7' && !offlineReview;
        const liveRfdetrReview = activeAlgorithmVersion === 'v8' && !offlineReview;
        const offlineV1StyleReview = offlineReview || liveOfflineV1Review;
        const targetDetectorAlgorithm = activeAlgorithmVersion === 'v5' || activeAlgorithmVersion === 'v6' || liveOfflineV1Review;
        const liveV6Review = activeAlgorithmVersion === 'v6' && !offlineReview;
        const trackFirstReview = offlineV2Review || liveV6Review;
        const profileFirstAlgorithm =
          activeAlgorithmVersion === 'v3' || activeAlgorithmVersion === 'v4' || targetDetectorAlgorithm;
        const fallbackAlgorithmVersion: AlgorithmVersion =
          activeAlgorithmVersion === 'v3' ? 'v2' : activeAlgorithmVersion;
        const hands = engine.detectHands(video, timestamp);
        const rawHand = hands[0] ? pointsToPixelSpace(hands[0], video.videoWidth, video.videoHeight) : null;
        hand = stabilizerRef.current.stabilizeHand(rawHand, timestamp);
        const usesRfdetr = liveRfdetrReview || offlineV2Review;
        if (usesRfdetr) {
          scheduleRfdetrInference(video, hand, timestamp, offlineV2Review);
        }
        if (timestamp - lastDetectorRunRef.current > 650) {
          const detectorBoxes = engine.detectObjectBoxes(video, timestamp);
          const nextBaseTracker = updateBaseObjectTracks(
            baseTrackerRef.current,
            detectorBoxes,
            timestamp,
            targetBaseIdRef.current,
            hand
          );
          baseTrackerRef.current = nextBaseTracker;
          const allBaseCandidates = nextBaseTracker.tracks;
          const baseCandidates = allBaseCandidates.filter(
            (candidate) =>
              isGripTargetEligible(candidate, video.videoWidth, video.videoHeight, trackFirstReview) &&
              (candidate.candidateId === targetBaseIdRef.current ||
                (candidate.missedFrames === 0 &&
                  (trackFirstReview || isBaseClassEnabled(candidate.label, baseClassEnabledRef.current))))
          );
          setBaseClassSummary(summarizeBaseClasses(allBaseCandidates.filter((candidate) => candidate.missedFrames === 0), baseClassEnabledRef.current));
          baseObjectCandidatesRef.current = baseCandidates;
          setBaseObjectCandidates(baseCandidates);
          detectorBoxRef.current =
            baseCandidates.find((candidate) => candidate.candidateId === targetBaseIdRef.current) ??
            baseCandidates[0] ??
            null;
          lastDetectorRunRef.current = timestamp;
        }
        const enabledProfiles = objectProfilesRef.current.filter((profile) => profile.enabled !== false);
        if (
          profileFirstAlgorithm &&
          enabledProfiles.length &&
          timestamp - lastProfileSearchRef.current > V3_PROFILE_SEARCH_INTERVAL_MS
        ) {
          lastProfileSearchRef.current = timestamp;
          const candidates = findObjectProfileCandidates(
            video,
            enabledProfiles,
            hand,
            targetDetectorAlgorithm
              ? { limit: 10, scanMode: 'all-frame', targetProfileId: null, relaxed: true }
              : 8
          );
          v3ProfileCandidatesRef.current = candidates;
          setV3ProfileCandidates(candidates);
        } else if (!profileFirstAlgorithm || !enabledProfiles.length) {
          v3ProfileCandidatesRef.current = [];
          setV3ProfileCandidates([]);
          temporalIdentityRef.current = EMPTY_TEMPORAL_IDENTITY;
          setTemporalIdentity(EMPTY_TEMPORAL_IDENTITY);
        }

        const selectedProfileTargetId = targetProfileIdRef.current;
        const selectedBaseCandidate =
          targetDetectorAlgorithm && targetBaseIdRef.current
            ? (baseObjectCandidatesRef.current.find((candidate) => candidate.candidateId === targetBaseIdRef.current) ?? null)
            : null;
        const stickyTrack =
          targetDetectorAlgorithm && trackFirstReview && !selectedProfileTargetId
            ? updateOfflineV2Track(
                liveV6Review ? liveV6TrackRef.current : offlineV2TrackRef.current,
                baseObjectCandidatesRef.current,
                hand,
                previousObjectRef.current,
                video,
                timestamp
              )
            : null;
        if (trackFirstReview) {
          const nextTrack = stickyTrack ?? { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: timestamp };
          if (liveV6Review) liveV6TrackRef.current = nextTrack;
          else offlineV2TrackRef.current = nextTrack;
        }
        const offlineBaseCandidate =
          targetDetectorAlgorithm && (offlineV1StyleReview || liveV6Review) && !selectedProfileTargetId
            ? trackFirstReview
              ? stickyTrack?.candidate ?? null
              : selectOfflineHandObjectCandidate(baseObjectCandidatesRef.current, hand) ??
                inferOfflinePixelObjectCandidate(video, previousObjectRef.current)
            : null;
        if (offlineReview && !hand && offlineBaseCandidate) {
          hand = createOfflineSurrogateHand(offlineBaseCandidate);
        }
        const activeBaseCandidate =
          selectedBaseCandidate && isGripTargetEligible(selectedBaseCandidate, video.videoWidth, video.videoHeight, trackFirstReview)
            ? selectedBaseCandidate
            : offlineBaseCandidate && isGripTargetEligible(offlineBaseCandidate, video.videoWidth, video.videoHeight, trackFirstReview)
              ? offlineBaseCandidate
              : null;
        const activeBaseConfidence =
          activeBaseCandidate && trackFirstReview && stickyTrack?.candidate === activeBaseCandidate
            ? stickyTrack.confidence
            : activeBaseCandidate && offlineBaseCandidate === activeBaseCandidate
              ? offlineCandidateConfidence(activeBaseCandidate, hand)
              : activeBaseCandidate?.score ?? 0;
        const bestProfileCandidate =
          profileFirstAlgorithm
            ? targetDetectorAlgorithm
              ? selectedProfileTargetId
                ? (v3ProfileCandidatesRef.current.find((candidate) => candidate.profileId === selectedProfileTargetId && candidate.matched) ?? null)
                : null
              : (v3ProfileCandidatesRef.current.find((candidate) => candidate.matched) ?? null)
            : null;
        const selectedTemporalCandidate =
          bestProfileCandidate ?? (activeBaseCandidate ? baseCandidateToTemporalCandidate(activeBaseCandidate, activeBaseConfidence) : null);
        const v5TargetRequested =
          targetDetectorAlgorithm && (offlineV1StyleReview || liveV6Review || Boolean(targetBaseIdRef.current || targetProfileIdRef.current));
        const v5TargetActive = !targetDetectorAlgorithm || Boolean(bestProfileCandidate || activeBaseCandidate);
        const baseThreshold = offlineV1StyleReview ? OFFLINE_BASE_TARGET_THRESHOLD : liveV6Review ? V6_LIVE_TARGET_THRESHOLD : V5_BASE_TARGET_THRESHOLD;
        const v5TargetLockReady = !targetDetectorAlgorithm || Boolean(bestProfileCandidate || (activeBaseCandidate && activeBaseConfidence >= baseThreshold));
        const requiresEnabledProfileMatch =
          targetDetectorAlgorithm
            ? !offlineV1StyleReview && !manualPointRef.current && !activeBaseCandidate && !bestProfileCandidate
            : profileFirstAlgorithm && enabledProfiles.length > 0 && !manualPointRef.current;
        const heuristicObject =
          activeBaseCandidate && v5TargetLockReady
            ? objectRegionFromBaseCandidate(activeBaseCandidate, previousObjectRef.current, activeBaseConfidence)
            : targetDetectorAlgorithm && (!v5TargetRequested || !v5TargetActive || !v5TargetLockReady)
            ? null
            : inferObjectRegion({
                video,
                hand,
                previous: previousObjectRef.current,
                manualPoint: targetDetectorAlgorithm ? null : manualPointRef.current,
                manualScale: manualScaleRef.current,
                locked: lockObjectRef.current,
                detectorBox: activeBaseCandidate ?? detectorBoxRef.current,
                algorithmVersion: offlineV1StyleReview ? 'v2' : fallbackAlgorithmVersion
              });
        let rawObject: ObjectRegion | null = heuristicObject;
        if (bestProfileCandidate) {
          rawObject = objectRegionFromProfileCandidate(bestProfileCandidate, previousObjectRef.current);
        } else if (requiresEnabledProfileMatch || (targetDetectorAlgorithm && (!v5TargetRequested || !v5TargetActive || !v5TargetLockReady))) {
          rawObject = null;
        }
        if (targetDetectorAlgorithm && !rawObject) {
          previousObjectRef.current = null;
          object = null;
        } else {
          object = stabilizerRef.current.stabilizeObject(rawObject, timestamp);
        }
        if ((requiresEnabledProfileMatch || (targetDetectorAlgorithm && !rawObject)) && objectDetectionRef.current) {
          objectDetectionRef.current = null;
          setObjectDetection(null);
        }
        if (timestamp - lastObjectMatchRef.current > 420) {
          lastObjectMatchRef.current = timestamp;
          const descriptor = object && !bestProfileCandidate ? browserObjectDescriptorProvider.describe(video, object) : null;
          const descriptorMatch =
            descriptor && enabledProfiles.length
              ? matchObjectProfiles(descriptor, enabledProfiles, {
                  threshold: liveV6Review ? 0.5 : 0.56,
                  margin: liveV6Review ? 0.04 : 0.06,
                  useExemplars: true
                })
              : null;
          const baseMatch = activeBaseCandidate
            ? {
                profileId: activeBaseCandidate.candidateId,
                name: offlineV1StyleReview ? baseObjectName(activeBaseCandidate, false) : baseObjectName(activeBaseCandidate, showObjectLabelsRef.current),
                score: activeBaseConfidence,
                matched: activeBaseConfidence >= baseThreshold
              }
            : null;
          const instantMatch = bestProfileCandidate
            ? {
                profileId: bestProfileCandidate.profileId,
                name: bestProfileCandidate.name,
                score: bestProfileCandidate.score,
                matched: bestProfileCandidate.matched
              }
            : descriptorMatch && baseMatch
              ? {
                  ...descriptorMatch,
                  score: clampUnit(Math.max(descriptorMatch.score, baseMatch.score * 0.72)),
                  matched: descriptorMatch.matched || (descriptorMatch.score >= 0.5 && baseMatch.matched)
                }
              : baseMatch ?? descriptorMatch;
          let match = instantMatch;
          if (activeAlgorithmVersion === 'v4' || targetDetectorAlgorithm) {
            const nextTemporal = updateTemporalIdentity(
              temporalIdentityRef.current,
              selectedTemporalCandidate,
              activeBaseCandidate
                ? { stableFrames: offlineV1StyleReview ? 1 : 2, decay: 0.72, threshold: baseThreshold }
                : { stableFrames: 3, decay: 0.72, threshold: 0.62 }
            );
            temporalIdentityRef.current = nextTemporal;
            setTemporalIdentity(nextTemporal);
            match = temporalIdentityToMatch(nextTemporal);
          }
          if (liveV6Review) {
            const memory = updateLiveIdentityMemory(liveIdentityMemoryRef.current, descriptorMatch ?? match, activeBaseConfidence >= baseThreshold);
            liveIdentityMemoryRef.current = memory;
            match = liveIdentityMemoryToMatch(memory);
          }
          objectDetectionRef.current = match;
          setObjectDetection(match);
        }
        const identityScore = activeBaseCandidate
          ? Math.max(activeBaseConfidence, (objectDetectionRef.current?.score ?? 0) * 0.86)
          : objectDetectionRef.current?.score ?? 0;
        const identityMatched = activeBaseCandidate
          ? activeBaseConfidence >= baseThreshold || Boolean(objectDetectionRef.current?.matched)
          : objectDetectionRef.current?.matched ?? false;
        const identityName =
          objectDetectionRef.current?.name ??
          (activeBaseCandidate
            ? offlineV1StyleReview ? baseObjectName(activeBaseCandidate, false) : baseObjectName(activeBaseCandidate, showObjectLabelsRef.current)
            : null);
        let objectIdentity: ObjectIdentitySignal = {
          hasProfiles: enabledProfiles.length > 0 || Boolean(activeBaseCandidate),
          score: identityScore,
          matched: identityMatched,
          name: identityName,
          source: objectDetectionRef.current?.profileId && !objectDetectionRef.current.profileId.startsWith('base-track-') ? 'trained' : activeBaseCandidate ? 'base' : undefined
        };
        const handVelocityForSlip =
          hand && previousPalmRef.current ? subtract(palmCenter(hand), previousPalmRef.current) : { x: 0, y: 0 };
        const persistentSlipScore = stabilizerRef.current.updatePersistentSlip(handVelocityForSlip, object);
        const rawFrameAnalysis = analyzeGrip(hand, object, previousPalmRef.current, {
          persistentSlipScore,
          algorithmVersion: offlineV1StyleReview ? 'v2' : fallbackAlgorithmVersion,
          objectIdentity
        });
        let baseFrameAnalysis = analyzeGrip(hand, object, previousPalmRef.current, {
          persistentSlipScore,
          calibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'strong'),
          weakCalibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'weak'),
          algorithmVersion: offlineV1StyleReview ? 'v2' : fallbackAlgorithmVersion,
          objectIdentity
        });
        let rfdetrSelectionMetrics: { objectScore: number; contact: number; latencyMs: number | null } | null = null;
        if (liveRfdetrReview || offlineV2Review) {
          const latestRfdetr = rfdetrRuntimeRef.current;
          const freshRfdetr = isRfdetrResultFresh(latestRfdetr, timestamp, offlineV2Review ? 2600 : 1500);
          const canHoldOfflineRfdetr =
            offlineV2Review &&
            !freshRfdetr &&
            latestRfdetr.status === 'pending' &&
            Boolean(previousObjectRef.current?.detectorLabel?.startsWith('rfdetr:')) &&
            rfdetrTrackRef.current.missedFrames < 2;
          if (liveRfdetrReview || freshRfdetr || canHoldOfflineRfdetr) {
            const compensatedRfdetr =
              freshRfdetr && latestRfdetr.result
                ? compensateRfdetrResponseForHandMotion(latestRfdetr.result, latestRfdetr.resultPalm, hand)
                : null;
            const rfdetrGrip = analyzeGripWithRfdetr({
              hand,
              detections: compensatedRfdetr?.detections ?? [],
              previousPalm: previousPalmRef.current,
              previousObject: previousObjectRef.current,
              previousTrack: rfdetrTrackRef.current,
              now: timestamp,
              persistentSlipScore,
              calibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'strong'),
              weakCalibrationBaseline: selectCalibrationBaseline(calibrationProfilesRef.current, rawFrameAnalysis.diagnostics.mode, 'weak'),
              serverAvailable: freshRfdetr || canHoldOfflineRfdetr,
              unavailableMessage: latestRfdetr.message || 'RF-DETR unavailable. Start the local RF-DETR server to use V8 live analysis.'
            });
            rfdetrTrackRef.current = rfdetrGrip.track;
            if (liveRfdetrReview || rfdetrGrip.object) {
              object = rfdetrGrip.object;
              objectIdentity = rfdetrGrip.objectIdentity;
              objectDetectionRef.current = rfdetrGrip.objectIdentity.name
                ? {
                    profileId: rfdetrGrip.object?.detectorLabel ?? 'rfdetr-object',
                    name: rfdetrGrip.objectIdentity.name,
                    score: rfdetrGrip.objectIdentity.score,
                    matched: rfdetrGrip.objectIdentity.matched
                  }
                : null;
              setObjectDetection(objectDetectionRef.current);
              baseFrameAnalysis = rfdetrGrip.analysis;
              rfdetrSelectionMetrics = {
                objectScore: rfdetrGrip.selection.objectScore,
                contact: rfdetrGrip.selection.contact,
                latencyMs: latestRfdetr.latencyMs
              };
            }
          }
        }
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
        } else if (liveRfdetrReview) {
          frameAnalysis = baseFrameAnalysis;
        } else {
          frameAnalysis = stabilizerRef.current.stabilizeAnalysis(baseFrameAnalysis, timestamp);
        }
        if (
          liveV6Review &&
          targetBaseIdRef.current &&
          frameAnalysis.guidance === 'Object not locked' &&
          frameAnalysis.diagnostics.objectIssue?.includes('open palm')
        ) {
          targetBaseIdRef.current = null;
          setTargetBaseId(null);
          liveV6TrackRef.current = { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: timestamp };
          previousObjectRef.current = null;
          objectDetectionRef.current = null;
          liveIdentityMemoryRef.current = null;
          setObjectDetection(null);
          object = null;
        }
        if (
          mediaModeRef.current === 'offline' &&
          !(offlineReviewVersionRef.current === 'v2' && offlineReportRef.current) &&
          (timestamp - lastOfflineTimelineRef.current > OFFLINE_TIMELINE_INTERVAL_MS || offlineTimelineRef.current.length === 0)
        ) {
          lastOfflineTimelineRef.current = timestamp;
          const point = {
            time: video.currentTime,
            grip: frameAnalysis.gripPercentage,
            confidence: frameAnalysis.confidence,
            objectMatch: objectIdentity.score,
            lock: frameAnalysis.objectLockQuality,
            contact: frameAnalysis.evidence.fingerSegmentContactScore,
            closure: frameAnalysis.closureScore,
            thumb: frameAnalysis.thumbOpposition,
            enclosure: frameAnalysis.enclosureScore,
            slip: frameAnalysis.slipRisk,
            weak: frameAnalysis.gripPercentage < 44 || frameAnalysis.guidance === 'Reposition' || frameAnalysis.guidance === 'Object uncertain',
            guidance: frameAnalysis.guidance,
            object: objectIdentity.name ?? '',
            mode: frameAnalysis.diagnostics.mode,
            state: frameAnalysis.diagnostics.state,
            objectX: object?.center.x ?? null,
            objectY: object?.center.y ?? null,
            objectRadiusX: object?.radiusX ?? null,
            objectRadiusY: object?.radiusY ?? null,
            objectAngle: object?.angle ?? null,
            palmX: frameAnalysis.palmCenter?.x ?? null,
            palmY: frameAnalysis.palmCenter?.y ?? null,
            rfdetrObjectScore: rfdetrSelectionMetrics?.objectScore,
            rfdetrContact: rfdetrSelectionMetrics?.contact,
            rfdetrLatencyMs: rfdetrSelectionMetrics?.latencyMs
          };
          offlineTimelineRef.current =
            offlineReviewVersionRef.current === 'v2'
              ? [...offlineTimelineRef.current, point]
              : [...offlineTimelineRef.current.slice(-239), point];
          setOfflineTimeline(offlineTimelineRef.current);
        }
        updateCalibrationCapture(frameAnalysis, timestamp);
        previousObjectRef.current = object;
        previousPalmRef.current = frameAnalysis.palmCenter;
        setAnalysis(frameAnalysis);
      }

      const overlayBaseCandidates =
        algorithmVersionRef.current === 'v6'
          ? targetBaseIdRef.current
            ? baseObjectCandidatesRef.current.filter((candidate) => candidate.candidateId === targetBaseIdRef.current)
            : []
          : algorithmVersionRef.current === 'v5'
            ? baseObjectCandidatesRef.current
            : [];
      const overlayProfileCandidates =
        algorithmVersionRef.current === 'v6'
          ? targetProfileIdRef.current
            ? v3ProfileCandidatesRef.current.filter((candidate) => candidate.profileId === targetProfileIdRef.current)
            : []
          : algorithmVersionRef.current === 'v5'
            ? v3ProfileCandidatesRef.current
            : algorithmVersionRef.current === 'v3' || algorithmVersionRef.current === 'v4'
              ? v3ProfileCandidatesRef.current
              : [];
      drawTrackingOverlay(
        context,
        canvas.width,
        canvas.height,
        mirroredRef.current,
        hand,
        object,
        frameAnalysis,
        overlayProfileCandidates,
        targetProfileIdRef.current,
        overlayBaseCandidates,
        targetBaseIdRef.current,
        showObjectLabelsRef.current
      );
      animationRef.current = requestAnimationFrame(tick);
    };

    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = requestAnimationFrame(tick);
  }, [analysis, mediaMode, scheduleRfdetrInference, scheduleV3Inference, updateCalibrationCapture]);

  const resetObject = useCallback(() => {
    manualPointRef.current = null;
    manualScaleRef.current = 1;
    draggingObjectRef.current = false;
    previousObjectRef.current = null;
    detectorBoxRef.current = null;
    baseTrackerRef.current = { nextId: 1, tracks: [] };
    baseObjectCandidatesRef.current = [];
    v3ProfileCandidatesRef.current = [];
    temporalIdentityRef.current = EMPTY_TEMPORAL_IDENTITY;
    targetBaseIdRef.current = null;
    lastProfileSearchRef.current = 0;
    rfdetrTrackRef.current = EMPTY_RFDETR_TRACK;
    stabilizerRef.current.reset();
    calibrationCaptureRef.current = { active: false, kind: calibrationKind, start: 0, samples: [] };
    resetV3Runtime();
    resetRfdetrRuntime();
    setLocked(false);
    setCalibrating(false);
    setV3ProfileCandidates([]);
    setBaseObjectCandidates([]);
    setTemporalIdentity(EMPTY_TEMPORAL_IDENTITY);
    setTargetProfileId(null);
    setTargetBaseId(null);
    setAnalysis(createEmptyAnalysis('Object reset. Place it between your thumb and fingers to relock.'));
  }, [calibrationKind, resetRfdetrRuntime, resetV3Runtime]);

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
      baseTrackerRef.current = { nextId: 1, tracks: [] };
      baseObjectCandidatesRef.current = [];
      liveV6TrackRef.current = { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 };
      rfdetrTrackRef.current = EMPTY_RFDETR_TRACK;
      targetProfileIdRef.current = null;
      targetBaseIdRef.current = null;
      stabilizerRef.current.reset();
      resetV3Runtime(
        version === 'v3'
          ? 'V3 selected. Start tracking to connect to the local perception server.'
          : version === 'v4'
          ? 'V4 selected. Enabled trained objects must match across multiple frames before detection turns green.'
          : version === 'v5'
          ? 'V5 selected. Select a target object ID, then grip scoring only follows that object.'
          : version === 'v6'
          ? 'V6 selected. Live mode auto-follows the hand-near object using Offline V2 sticky tracking.'
          : version === 'v7'
          ? 'V7 selected. It mirrors Offline Review V1 auto-search logic on the live camera.'
          : version === 'v8'
          ? 'V8 selected. Start the local RF-DETR server to use live mask/box object evidence.'
          : 'V3 server idle. Select V3 and start tracking to begin fusion.'
      );
      resetRfdetrRuntime(
        version === 'v8'
          ? 'V8 selected. Start tracking to connect to the local RF-DETR server.'
          : 'RF-DETR server idle. Select V8 or Offline V2 to begin RF-DETR analysis.'
      );
      setLocked(false);
      setCalibrating(false);
      setTargetProfileId(null);
      setTargetBaseId(null);
      setBaseObjectCandidates([]);
      setAlgorithmVersion(version);
      saveAlgorithmVersion(version);
      setAnalysis(
        createEmptyAnalysis(
          version === 'v3'
            ? 'V3 selected. It will fuse local-server perception with V2 fallback when the server is unavailable.'
            : version === 'v4'
            ? 'V4 selected. Train an object, enable it, then V4 will require stable identity before scoring grip.'
            : version === 'v5'
            ? 'V5 selected. It scans enabled object IDs, waits for target selection, and requires contact before grip can score high.'
            : version === 'v6'
            ? 'V6 selected. It applies Offline V2 track-first logic to live video, then uses V5 contact-gated grip scoring.'
            : version === 'v7'
            ? 'V7 selected. It uses the same auto-search implementation as Offline Review V1, without sticky tracking or trained target gating.'
            : version === 'v8'
            ? 'V8 selected. It uses RF-DETR segmentation masks from the local CPU server and will not score grip when RF-DETR is unavailable.'
            : version === 'v2'
            ? 'V2 selected. It will require independent object evidence before scoring grip.'
            : 'V1 selected. It uses the original permissive grip heuristic.'
        )
      );
    },
    [algorithmVersion, resetRfdetrRuntime, resetV3Runtime]
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

  const pointInPendingUpload = useCallback((event: React.PointerEvent<HTMLElement>, review: PendingUploadReview) => {
    const preview = event.currentTarget.closest('.upload-source-preview') as HTMLElement | null;
    const rect = (preview ?? event.currentTarget).getBoundingClientRect();
    const imageAspect = review.canvas.width / Math.max(1, review.canvas.height);
    const boxAspect = rect.width / Math.max(1, rect.height);
    let renderWidth = rect.width;
    let renderHeight = rect.height;
    let offsetX = 0;
    let offsetY = 0;
    if (imageAspect > boxAspect) {
      renderHeight = rect.width / imageAspect;
      offsetY = (rect.height - renderHeight) / 2;
    } else {
      renderWidth = rect.height * imageAspect;
      offsetX = (rect.width - renderWidth) / 2;
    }
    const x = clampNumber(((event.clientX - rect.left - offsetX) / Math.max(1, renderWidth)) * review.canvas.width, 0, review.canvas.width);
    const y = clampNumber(((event.clientY - rect.top - offsetY) / Math.max(1, renderHeight)) * review.canvas.height, 0, review.canvas.height);
    return { x, y };
  }, []);

  const startPendingCropDrag = useCallback(
    (event: React.PointerEvent<HTMLElement>, mode: 'move' | 'resize') => {
      if (!pendingUpload) return;
      event.preventDefault();
      event.stopPropagation();
      const point = pointInPendingUpload(event, pendingUpload);
      cropDragRef.current = {
        mode,
        startX: point.x,
        startY: point.y,
        cropX: pendingUpload.cropX,
        cropY: pendingUpload.cropY,
        cropSize: pendingUpload.cropSize
      };
      event.currentTarget.setPointerCapture?.(event.pointerId);
    },
    [pendingUpload, pointInPendingUpload]
  );

  const movePendingCropDrag = useCallback(
    (event: React.PointerEvent<HTMLElement>) => {
      if (!pendingUpload || !cropDragRef.current) return;
      event.preventDefault();
      const point = pointInPendingUpload(event, pendingUpload);
      const drag = cropDragRef.current;
      const deltaX = point.x - drag.startX;
      const deltaY = point.y - drag.startY;
      if (drag.mode === 'resize') {
        updatePendingUpload({ cropSize: drag.cropSize + Math.max(deltaX, deltaY) });
        return;
      }
      updatePendingUpload({
        cropX: drag.cropX + deltaX,
        cropY: drag.cropY + deltaY
      });
    },
    [pendingUpload, pointInPendingUpload, updatePendingUpload]
  );

  const stopPendingCropDrag = useCallback((event?: React.PointerEvent<HTMLElement>) => {
    if (event) {
      event.preventDefault();
      try {
        event.currentTarget.releasePointerCapture?.(event.pointerId);
      } catch {
        // Pointer capture can already be released by the browser.
      }
    }
    cropDragRef.current = null;
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
    targetProfileIdRef.current = profile.id;
    targetBaseIdRef.current = null;
    setTargetProfileId(profile.id);
    setTargetBaseId(null);
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
    v3ProfileCandidatesRef.current = [];
    lastProfileSearchRef.current = 0;
    setV3ProfileCandidates([]);
    objectDetectionRef.current = null;
    setObjectDetection(null);
    const toggled = profiles.find((profile) => profile.id === id);
    if (toggled?.enabled === false && targetProfileIdRef.current === id) {
      targetProfileIdRef.current = null;
      setTargetProfileId(null);
    }
    if (profileDirectoryRef.current) {
      void mirrorObjectProfilesToFolder(profileDirectoryRef.current, profiles);
    }
  }, [objectProfiles]);

  const selectTargetProfile = useCallback((id: string) => {
    const profile = objectProfilesRef.current.find((item) => item.id === id);
    if (!profile || profile.enabled === false) {
      setTrainingStatus('Enable that profile before selecting it as the V5 target.');
      return;
    }
    targetProfileIdRef.current = id;
    targetBaseIdRef.current = null;
    setTargetProfileId(id);
    setTargetBaseId(null);
    temporalIdentityRef.current = EMPTY_TEMPORAL_IDENTITY;
    setTemporalIdentity(EMPTY_TEMPORAL_IDENTITY);
    objectDetectionRef.current = null;
    setObjectDetection(null);
    setTrainingStatus(`V5 target selected: ${profile.name}. Grip scoring now follows only this object.`);
  }, []);

  const selectBaseTarget = useCallback((id: string) => {
    const candidate = baseObjectCandidatesRef.current.find((item) => item.candidateId === id);
    if (!candidate) {
      setTrainingStatus('That base detector object is no longer visible. Move it into frame and try again.');
      return;
    }
    const video = videoRef.current;
    if (!isGripTargetEligible(candidate, video?.videoWidth ?? 0, video?.videoHeight ?? 0)) {
      setTrainingStatus('That detection looks like a person/background region, not a handheld object. Select a smaller object ID near the hand.');
      return;
    }
    targetBaseIdRef.current = id;
    targetProfileIdRef.current = null;
    setTargetBaseId(id);
    setTargetProfileId(null);
    temporalIdentityRef.current = EMPTY_TEMPORAL_IDENTITY;
    setTemporalIdentity(EMPTY_TEMPORAL_IDENTITY);
    objectDetectionRef.current = null;
    setObjectDetection(null);
    setTrainingStatus(`V5 base target selected: ${baseObjectName(candidate, showObjectLabelsRef.current)}. Train a profile later if you need stronger identity verification.`);
  }, []);

  const toggleBaseClass = useCallback((key: string) => {
    setBaseClassEnabled((current) => {
      const nextEnabled = !(current[key] ?? key !== 'person');
      const next = { ...current, [key]: nextEnabled };
      const selected = baseObjectCandidatesRef.current.find((candidate) => candidate.candidateId === targetBaseIdRef.current);
      if (!nextEnabled && selected && baseClassKey(selected.label) === key) {
        targetBaseIdRef.current = null;
        setTargetBaseId(null);
        temporalIdentityRef.current = EMPTY_TEMPORAL_IDENTITY;
        setTemporalIdentity(EMPTY_TEMPORAL_IDENTITY);
        objectDetectionRef.current = null;
        setObjectDetection(null);
        previousObjectRef.current = null;
        setTrainingStatus(`${formatBaseClassLabel(key)} disabled. Select another target object ID.`);
      }
      const filtered = baseObjectCandidatesRef.current.filter((candidate) => isBaseClassEnabled(candidate.label, next));
      baseObjectCandidatesRef.current = filtered;
      setBaseObjectCandidates(filtered);
      setBaseClassSummary((summary) => summary.map((item) => (item.key === key ? { ...item, enabled: nextEnabled } : item)));
      return next;
    });
  }, []);

  const switchOfflineReviewVersion = useCallback((version: OfflineReviewVersion) => {
    offlineReviewVersionRef.current = version;
    setOfflineReviewVersion(version);

    const video = videoRef.current;
    if (mediaModeRef.current !== 'offline' || !video || !video.src) return;

    if (version === 'v1') {
      offlineBatchProcessingRef.current = false;
      video.playbackRate = 1;
      setOfflineAnalysisPhase(video.ended ? 'complete' : 'reviewing');
      return;
    }

    video.pause();
    video.currentTime = 0;
    video.playbackRate = 2.5;
    offlineBatchProcessingRef.current = true;
    offlineTimelineRef.current = [];
    offlineReportRef.current = null;
    offlineV2TrackRef.current = { candidate: null, confidence: 0, ageFrames: 0, missedFrames: 0, lastSeenAt: 0 };
    rfdetrTrackRef.current = EMPTY_RFDETR_TRACK;
    resetRfdetrRuntime('Offline V2 selected. RF-DETR will run when the local server is available.');
    lastOfflineTimelineRef.current = 0;
    setOfflineTimeline([]);
    setOfflineReport(null);
    setOfflineAnalysisPhase('processing');
    setAnalysis(createEmptyAnalysis('Offline V2 is reprocessing the full video with future and past frame correction.'));
    runLoop();
    void video.play().catch(() => {
      offlineBatchProcessingRef.current = false;
      video.playbackRate = 1;
      setOfflineAnalysisPhase('reviewing');
      setAnalysis(createEmptyAnalysis('Offline video loaded. Press play once to let V2 process the full video.'));
    });
  }, [resetRfdetrRuntime, runLoop]);

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

  const exportOfflineTimeline = useCallback((format: 'csv' | 'json') => {
    if (!offlineTimelineRef.current.length) return;
    const fileBase = (offlineVideoName || 'gripsense-offline-review').replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-');
    const report = offlineReportRef.current ?? buildOfflineReport(offlineTimelineRef.current, offlineVideoName, videoRef.current?.duration ?? 0);
    const payload =
      format === 'json'
        ? JSON.stringify({ report, timeline: offlineTimelineRef.current }, null, 2)
        : [
            `# ${report?.summary ?? 'GripSense offline report'}`,
            `# averageGrip=${report?.averageGrip ?? 0},averageObjectMatch=${report?.averageObjectMatch ?? 0},weakSegments=${report?.weakSegments.length ?? 0},slipEvents=${report?.slipEvents.length ?? 0}`,
            'time,grip,confidence,objectMatch,lock,contact,closure,thumb,enclosure,slip,weak,guidance,object,mode,state,objectX,objectY,objectRadiusX,objectRadiusY,objectAngle,palmX,palmY,rfdetrObjectScore,rfdetrContact,rfdetrLatencyMs',
            ...offlineTimelineRef.current.map((point) =>
              [
                point.time.toFixed(2),
                point.grip,
                Math.round(point.confidence * 100),
                Math.round(point.objectMatch * 100),
                Math.round(point.lock * 100),
                Math.round(point.contact * 100),
                Math.round(point.closure * 100),
                Math.round(point.thumb * 100),
                Math.round(point.enclosure * 100),
                Math.round(point.slip * 100),
                point.weak ? 'yes' : 'no',
                csvEscape(point.guidance),
                csvEscape(point.object),
                csvEscape(point.mode),
                csvEscape(point.state),
                point.objectX?.toFixed(1) ?? '',
                point.objectY?.toFixed(1) ?? '',
                point.objectRadiusX?.toFixed(1) ?? '',
                point.objectRadiusY?.toFixed(1) ?? '',
                point.objectAngle?.toFixed(3) ?? '',
                point.palmX?.toFixed(1) ?? '',
                point.palmY?.toFixed(1) ?? '',
                point.rfdetrObjectScore === undefined ? '' : Math.round(point.rfdetrObjectScore * 100),
                point.rfdetrContact === undefined ? '' : Math.round(point.rfdetrContact * 100),
                point.rfdetrLatencyMs === null || point.rfdetrLatencyMs === undefined ? '' : Math.round(point.rfdetrLatencyMs)
              ].join(',')
            )
          ].join('\n');
    downloadTextFile(`${fileBase}.${format}`, payload, format === 'json' ? 'application/json' : 'text/csv');
  }, [offlineVideoName]);

  const finalizeOfflineReview = useCallback(() => {
    if (mediaModeRef.current !== 'offline') return;
    offlineBatchProcessingRef.current = false;
    if (videoRef.current) videoRef.current.playbackRate = 1;
    if (offlineReviewVersionRef.current === 'v2' && offlineTimelineRef.current.length > 2) {
      const sanitized = sanitizeOfflineV2TimelineGeometry(
        offlineTimelineRef.current,
        videoRef.current?.videoWidth ?? 0,
        videoRef.current?.videoHeight ?? 0
      );
      const smoothed = refineOfflineTimeline(refineRfdetrOfflineTimeline(sanitized));
      offlineTimelineRef.current = smoothed;
      setOfflineTimeline(smoothed);
    }
    const report = buildOfflineReport(offlineTimelineRef.current, offlineVideoName, videoRef.current?.duration ?? 0);
    offlineReportRef.current = report;
    setOfflineReport(report);
    setOfflineAnalysisPhase('complete');
    if (offlineReviewVersionRef.current === 'v2' && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.playbackRate = 1;
      videoRef.current.currentTime = 0;
    }
  }, [offlineVideoName]);

  const playOfflineReview = useCallback(() => {
    const video = videoRef.current;
    if (mediaModeRef.current !== 'offline' || !video) return;
    offlineBatchProcessingRef.current = false;
    video.playbackRate = 1;
    if (video.ended || video.currentTime >= Math.max(0, video.duration - 0.05)) {
      video.currentTime = 0;
    }
    setPaused(false);
    void video.play();
  }, []);

  const exportOfflineAnnotatedVideo = useCallback(async (format: 'mp4' | 'webm', layout: 'full' | 'compact' = 'full') => {
    const video = videoRef.current;
    const overlay = canvasRef.current;
    if (mediaModeRef.current !== 'offline' || !video || !overlay || !video.duration || Number.isNaN(video.duration)) {
      setOfflineVideoExportStatus('Upload an offline video first.');
      return;
    }
    if (offlineReviewVersionRef.current === 'v2' && offlineAnalysisPhase !== 'complete') {
      setOfflineVideoExportStatus('Offline V2 must finish full-video processing before export.');
      return;
    }

    const composite = document.createElement('canvas');
    composite.width = video.videoWidth || overlay.width;
    composite.height = video.videoHeight || overlay.height;
    const ctx = composite.getContext('2d');
    if (!ctx || !('captureStream' in composite) || typeof MediaRecorder === 'undefined') {
      setOfflineVideoExportStatus('This browser cannot export annotated video.');
      return;
    }

    const captureStream = composite.captureStream(30);
    const supportedType = mediaRecorderTypeFor(format);
    if (!supportedType) {
      setOfflineVideoExportStatus(
        format === 'mp4'
          ? 'MP4 export is not supported by this browser. Use WebM here, or convert with ffmpeg.'
          : 'WebM export is not supported by this browser.'
      );
      return;
    }
    const recorder = new MediaRecorder(captureStream, supportedType ? { mimeType: supportedType } : undefined);
    const chunks: BlobPart[] = [];
    const fileBase = (offlineVideoName || 'gripsense-offline-review').replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-');
    const previousTime = video.currentTime;
    const wasPaused = video.paused;
    let raf = 0;
    let stopped = false;

    const finish = () => {
      if (stopped) return;
      stopped = true;
      if (raf) cancelAnimationFrame(raf);
      if (recorder.state !== 'inactive') recorder.stop();
      captureStream.getTracks().forEach((track) => track.stop());
      video.removeEventListener('ended', finish);
    };

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: supportedType });
      const suffix = layout === 'compact' ? 'annotated-compact' : 'annotated';
      downloadBlobFile(`${fileBase}-${suffix}.${format}`, blob);
      setOfflineVideoExporting(false);
      setOfflineVideoExportStatus(`Downloaded ${fileBase}-${suffix}.${format}`);
      video.currentTime = Math.min(previousTime, video.duration || previousTime);
      if (wasPaused) video.pause();
    };

    const drawFrame = () => {
      ctx.clearRect(0, 0, composite.width, composite.height);
      drawExportVideoFrame(ctx, video, composite.width, composite.height, mirroredRef.current);
      const refinedPoint =
        offlineReviewVersionRef.current === 'v2' ? nearestOfflineTimelinePoint(offlineTimelineRef.current, video.currentTime) : null;
      const exportAnalysis = refinedPoint ? analysisFromOfflineTimelinePoint(analysisRef.current, refinedPoint) : analysisRef.current;
      if (offlineReviewVersionRef.current === 'v2' && refinedPoint) {
        const timelineObject = objectRegionFromOfflineTimelinePoint(refinedPoint, composite.width, composite.height);
        drawOfflineV2TimelineOverlay(ctx, composite.width, composite.height, mirroredRef.current, refinedPoint, timelineObject, exportAnalysis);
      } else {
        ctx.drawImage(overlay, 0, 0, composite.width, composite.height);
      }
      if (layout === 'compact') {
        drawCompactOfflineExportOverlay(ctx, composite.width, composite.height, exportAnalysis, offlineTimelineRef.current, offlineVideoName, offlineReportRef.current);
      } else {
        drawOfflineExportOverlay(ctx, composite.width, composite.height, exportAnalysis, offlineTimelineRef.current, offlineVideoName, offlineReportRef.current);
      }
      if (!video.ended && video.currentTime < video.duration - 0.04) {
        raf = requestAnimationFrame(drawFrame);
      } else {
        finish();
      }
    };

    try {
      setOfflineVideoExporting(true);
      setOfflineVideoExportStatus(
        layout === 'compact'
          ? 'Encoding compact annotated video from finalized timeline...'
          : 'Encoding annotated video from finalized timeline...'
      );
      video.currentTime = 0;
      await video.play();
      recorder.start(500);
      video.addEventListener('ended', finish, { once: true });
      drawFrame();
    } catch {
      finish();
      setOfflineVideoExporting(false);
      setOfflineVideoExportStatus('Video export could not start. Try pressing play once, then export again.');
    }
  }, [offlineAnalysisPhase, offlineVideoName]);

  useEffect(() => {
    if (!uploadOnlyMode) return;
    if (offlineAnalysisPhase !== 'complete' || offlineVideoExporting || !offlineTimeline.length) return;
    if (uploadOnlyExportStartedRef.current) return;
    uploadOnlyExportStartedRef.current = true;
    setUploadOnlyStatus('Upload V1 is encoding the MP4 download from the finalized timeline.');
    void exportOfflineAnnotatedVideo('mp4', 'compact');
  }, [exportOfflineAnnotatedVideo, offlineAnalysisPhase, offlineTimeline.length, offlineVideoExporting, uploadOnlyMode]);

  useEffect(() => {
    if (!uploadOnlyMode) return;
    if (!offlineVideoExportStatus.startsWith('Downloaded')) return;
    setUploadOnlyStatus('Upload V1 download is ready.');
    const timer = window.setTimeout(() => setUploadOnlyMode(false), 1200);
    return () => window.clearTimeout(timer);
  }, [offlineVideoExportStatus, uploadOnlyMode]);

  const offlineBatchProcessing = mediaMode === 'offline' && offlineReviewVersion === 'v2' && offlineAnalysisPhase === 'processing';
  const offlineV2ExportLocked = offlineReviewVersion === 'v2' && offlineAnalysisPhase !== 'complete';

  return (
    <main
      className={[
        'app-shell',
        mediaMode === 'offline' ? 'offline-shell' : '',
        offlineBatchProcessing ? 'offline-batch-processing' : '',
        uploadOnlyMode ? 'upload-only-processing' : ''
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <section className="camera-workspace" aria-label="Live grip tracking workspace">
        <video
          ref={videoRef}
          className={mirrored ? 'camera-feed mirrored' : 'camera-feed'}
          playsInline
          muted
          controls={mediaMode === 'offline' && !offlineBatchProcessing}
          onPlay={() => {
            if (mediaMode !== 'offline') return;
            if (offlineReviewVersionRef.current === 'v2' && offlineBatchProcessingRef.current) return;
            if (offlineReviewVersionRef.current === 'v2' && offlineAnalysisPhase === 'complete') return;
            setOfflineAnalysisPhase('reviewing');
          }}
          onEnded={finalizeOfflineReview}
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
            <label className="version-select" aria-label="Algorithm version">
              <span>Version</span>
              <select
                value={algorithmVersion}
                onChange={(event) => selectAlgorithmVersion(event.target.value as AlgorithmVersion)}
              >
                <option value="v1">V1 · original heuristic</option>
                <option value="v2">V2 · object-first</option>
                <option value="v3">V3 · server fusion</option>
                <option value="v4">V4 · trained identity</option>
                <option value="v5">V5 · target detector</option>
                <option value="v6">V6 · sticky live detector</option>
                <option value="v7">V7 · offline V1 live copy</option>
                <option value="v8">V8 · RF-DETR live</option>
              </select>
            </label>
            <InlineExplain label="Explain algorithm version" text={EXPLAIN.version} compact />
            <button
              className={mediaMode === 'offline' ? 'tool-button primary offline-active' : 'tool-button primary'}
              onClick={startCamera}
              disabled={cameraState === 'requesting'}
              aria-label={mediaMode === 'offline' ? 'Switch back to camera' : cameraState === 'live' ? 'Camera live' : 'Start camera'}
            >
              {mediaMode === 'offline' ? <Upload size={18} /> : <Camera size={18} />}
              <span>
                {mediaMode === 'offline'
                  ? 'Offline review'
                  : cameraState === 'live'
                    ? 'Camera live'
                    : cameraState === 'requesting'
                      ? 'Starting'
                      : 'Start'}
              </span>
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
            <button className="tool-button" onClick={() => uploadV1InputRef.current?.click()} aria-label="Upload V1 process and download">
              <Upload size={17} />
              <span>Upload V1</span>
            </button>
            <input
              ref={uploadV1InputRef}
              className="hidden-file-input"
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/*"
              onChange={handleUploadV1Video}
            />
            {mediaMode === 'offline' && offlineReviewVersion === 'v2' && offlineAnalysisPhase === 'complete' && (
              <button className="tool-button primary" onClick={playOfflineReview} aria-label="Play completed offline V2 review">
                <Play size={17} />
                <span>Play review</span>
              </button>
            )}
            {recordingState === 'recording' ? (
              <button className="tool-button recording-active" onClick={stopLiveRecording} aria-label="Stop recording offline review clip">
                <Square size={16} />
                <span>Stop {formatRecordingDuration(recordingElapsedMs)}</span>
              </button>
            ) : (
              <button
                className={recordedClip ? 'tool-button recorded-ready' : 'tool-button'}
                onClick={startLiveRecording}
                disabled={mediaMode === 'offline' || recordingState === 'unsupported'}
                aria-label="Record camera clip for offline review"
              >
                <Video size={17} />
                <span>{recordedClip ? 'Recorded' : 'Record'}</span>
              </button>
            )}
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

        {recordedClip && mediaMode === 'live' && !trainerOpen && (
          <div className="recording-review-panel" role="dialog" aria-label="Process recorded clip">
            <div className="recording-preview">
              <video src={recordedClip.url} muted playsInline controls />
            </div>
            <div className="recording-review-copy">
              <p className="eyebrow">Recorded offline clip</p>
              <h2>Process recording</h2>
              <p>
                Choose the offline engine for this camera recording. V1 starts review quickly; V2 scans the whole clip and then unlocks CSV, JSON, MP4, compact MP4, and WebM export.
              </p>
              <div className="recording-meta">
                <span>Duration</span>
                <strong>{formatRecordingDuration(recordedClip.durationMs)}</strong>
              </div>
              <div className="recording-actions">
                <button type="button" className="tool-button primary" onClick={() => processRecordedClip('v1')}>
                  <Upload size={17} />
                  <span>Process V1</span>
                </button>
                <button type="button" className="tool-button primary" onClick={() => processRecordedClip('v2')}>
                  <Sparkles size={17} />
                  <span>Process V2</span>
                </button>
                <button type="button" className="tool-button" onClick={clearRecordedClip}>
                  <X size={17} />
                  <span>Discard</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {mediaMode === 'offline' && cameraState === 'live' && !trainerOpen && (
          <div className="offline-review-overlay" aria-label="Offline grip video review">
            <div className="offline-glass-panel offline-left">
              <p className="eyebrow">Offline review</p>
              <div className="offline-version-toggle" aria-label="Offline review algorithm version">
                <button
                  type="button"
                  className={offlineReviewVersion === 'v1' ? 'active' : ''}
                  onClick={() => switchOfflineReviewVersion('v1')}
                >
                  V1
                </button>
                <button
                  type="button"
                  className={offlineReviewVersion === 'v2' ? 'active' : ''}
                  onClick={() => switchOfflineReviewVersion('v2')}
                >
                  V2
                </button>
              </div>
              <h2>{analysis.gripPercentage}%</h2>
              <strong>
                {offlineBatchProcessing
                  ? 'Batch processing video'
                  : offlineAnalysisPhase === 'processing'
                    ? 'Processing video'
                    : analysis.guidance}
              </strong>
              <span>{offlineVideoName || 'Uploaded video'}</span>
              <div className="offline-mini-row">
                <span>Phase</span>
                <strong>{formatOfflinePhase(offlineAnalysisPhase)}</strong>
              </div>
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
                <strong>{objectDetection?.matched ? objectDetection.name : mediaMode === 'offline' ? 'auto-search' : 'not matched'}</strong>
              </div>
              <div className="offline-mini-row">
                <span>Timeline</span>
                <strong>{offlineTimeline.length} pts</strong>
              </div>
              {offlineReviewVersion === 'v2' && (
                <div className="offline-mini-row">
                  <span>RF-DETR</span>
                  <strong>{formatRfdetrRuntimeStatus(rfdetrRuntime)}</strong>
                </div>
              )}
              {uploadOnlyMode && uploadOnlyStatus && (
                <div className="offline-report-summary">
                  <span>Upload V1</span>
                  <strong>{uploadOnlyStatus}</strong>
                </div>
              )}
              {offlineReport && (
                <div className="offline-report-summary">
                  <span>Report</span>
                  <strong>{offlineReport.summary}</strong>
                </div>
              )}
            </div>
            <div className="offline-glass-panel offline-right">
              <p className="eyebrow">Parameters</p>
              <GlassMetric label="Confidence" value={analysis.confidence} />
              <GlassMetric label="Lock" value={analysis.objectLockQuality} />
              <GlassMetric label="Closure" value={analysis.closureScore} />
              <GlassMetric label="Contact" value={analysis.evidence.fingerSegmentContactScore} />
              {offlineReviewVersion === 'v2' && (
                <>
                  <GlassMetric label="RF object" value={offlineTimeline[offlineTimeline.length - 1]?.rfdetrObjectScore ?? 0} />
                  <GlassMetric label="RF contact" value={offlineTimeline[offlineTimeline.length - 1]?.rfdetrContact ?? 0} />
                </>
              )}
              <GlassMetric label="Thumb" value={analysis.thumbOpposition} />
              <GlassMetric label="Slip" value={analysis.slipRisk} danger />
              {offlineAnalysisPhase === 'processing' && (
                <div className="offline-processing">
                  <span />
                  <strong>
                    {uploadOnlyMode
                      ? `${uploadOnlyStatus} ${offlineTimeline.length} pts`
                      : offlineReviewVersion === 'v2'
                      ? `Scanning full video for V2 RF-DETR correction before preview/export... ${offlineTimeline.length} pts`
                      : 'Preparing frame analysis...'}
                  </strong>
                </div>
              )}
              <div className="offline-timeline" aria-label="Offline analysis timeline">
                {offlineTimeline.slice(-60).map((point, index) => (
                  <span
                    className={point.slip > 0.45 ? 'slip' : point.weak ? 'weak' : 'ok'}
                    style={{ height: `${Math.max(12, point.grip)}%` }}
                    title={`${point.time.toFixed(1)}s ${point.grip}% ${point.guidance}`}
                    key={`${point.time}-${index}`}
                  />
                ))}
              </div>
              {offlineReport && (
                <div className="offline-report-grid">
                  <span>Avg grip <strong>{offlineReport.averageGrip}%</strong></span>
                  <span>Peak <strong>{offlineReport.peakGrip}%</strong></span>
                  <span>Weak <strong>{offlineReport.weakSegments.length}</strong></span>
                  <span>Slip <strong>{offlineReport.slipEvents.length}</strong></span>
                </div>
              )}
              <div className="offline-export-row">
                <button type="button" onClick={() => exportOfflineTimeline('csv')} disabled={!offlineTimeline.length || offlineBatchProcessing}>
                  CSV
                </button>
                <button type="button" onClick={() => exportOfflineTimeline('json')} disabled={!offlineTimeline.length || offlineBatchProcessing}>
                  JSON
                </button>
                <button type="button" onClick={() => exportOfflineAnnotatedVideo('mp4')} disabled={offlineVideoExporting || offlineBatchProcessing || offlineV2ExportLocked}>
                  {offlineVideoExporting ? 'Rendering' : 'MP4'}
                </button>
                <button type="button" onClick={() => exportOfflineAnnotatedVideo('mp4', 'compact')} disabled={offlineVideoExporting || offlineBatchProcessing || offlineV2ExportLocked}>
                  MP4 Compact
                </button>
                <button type="button" onClick={() => exportOfflineAnnotatedVideo('webm')} disabled={offlineVideoExporting || offlineBatchProcessing || offlineV2ExportLocked}>
                  WebM
                </button>
              </div>
              {offlineVideoExportStatus && <p className="offline-export-status">{offlineVideoExportStatus}</p>}
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
                <div className="training-coverage-panel">
                  <div className="motion-header compact-heading">
                    <span>
                      V4 training coverage
                      <InlineExplain label="Explain profile strength" text={EXPLAIN.profileStrength} compact />
                    </span>
                    <strong>{Math.round(trainerCoverage * 100)}%</strong>
                  </div>
                  <div className="metric-track">
                    <span style={{ width: `${Math.round(trainerCoverage * 100)}%` }} />
                  </div>
                  <div className="role-chip-grid">
                    {TRAINING_VIEW_ROLES.map((role) => (
                      <span className={trainingSamples.some((sample) => sample.viewRole === role) ? 'role-chip complete' : 'role-chip'} key={role}>
                        {formatTrainingRole(role)}
                      </span>
                    ))}
                  </div>
                  <p className="diagnostic-copy">Expected result: {trainerStrength}. Add negative examples to reduce false positives.</p>
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
                      <div
                        className="upload-source-preview"
                        onPointerMove={movePendingCropDrag}
                        onPointerUp={stopPendingCropDrag}
                        onPointerCancel={stopPendingCropDrag}
                      >
                        <img src={pendingUpload.imageDataUrl} alt="Uploaded training source" />
                        <span
                          className="crop-box"
                          style={cropOverlayStyle(pendingUpload)}
                          onPointerDown={(event) => startPendingCropDrag(event, 'move')}
                          title="Drag to move crop"
                        >
                          <span
                            className="crop-resize-handle"
                            onPointerDown={(event) => startPendingCropDrag(event, 'resize')}
                            aria-label="Resize crop"
                            role="button"
                            tabIndex={0}
                          />
                        </span>
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
                    <div className="role-picker" aria-label="Training image role">
                      {TRAINING_VIEW_ROLES.map((role) => (
                        <button
                          type="button"
                          className={pendingUpload.viewRole === role ? 'active' : ''}
                          onClick={() => updatePendingUpload({ viewRole: role })}
                          key={role}
                        >
                          {formatTrainingRole(role)}
                        </button>
                      ))}
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
                        <small>{formatTrainingRole(sample.viewRole ?? 'front')} - {sample.sourceName ?? sample.source ?? 'training image'}</small>
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

      {mediaMode === 'live' && <aside className="analysis-rail" aria-label="Grip analysis">
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

        {algorithmVersion === 'v8' && (
          <div className="v3-panel">
            <div className="motion-header">
              <Activity size={18} />
              <span>V8 RF-DETR</span>
            </div>
            <div className={rfdetrRuntime.status === 'ready' ? 'v3-status ready' : 'v3-status fallback'}>
              <span>{formatRfdetrRuntimeStatus(rfdetrRuntime)}</span>
              <strong>{rfdetrRuntime.status === 'ready' ? `${rfdetrRuntime.result?.detections.length ?? 0} obj` : '--'}</strong>
            </div>
            <p className={rfdetrRuntime.status === 'ready' ? 'diagnostic-copy' : 'diagnostic-copy warn'}>{rfdetrRuntime.message}</p>
            <div className="v3-score-grid">
              <V3Score label="RF object" value={objectDetection?.score ?? 0} />
              <V3Score label="Contact" value={analysis.evidence.visibleContactScore} />
              <V3Score label="Lock" value={analysis.objectLockQuality} />
              <V3Score label="Confidence" value={analysis.confidence} />
            </div>
            <div className="diagnostic-row neutral">
              <span>Latency</span>
              <strong>{rfdetrRuntime.latencyMs === null ? '--' : `${Math.round(rfdetrRuntime.latencyMs)} ms`}</strong>
            </div>
          </div>
        )}

        {(algorithmVersion === 'v4' || algorithmVersion === 'v5' || algorithmVersion === 'v6') && (
          <div className="v3-panel">
            <div className="motion-header">
              <Activity size={18} />
              <span>
                {algorithmVersion === 'v5' ? 'V5 target identity' : algorithmVersion === 'v6' ? 'V6 sticky identity' : 'V4 identity'}
                <InlineExplain label="Explain temporal identity" text={EXPLAIN.v4Temporal} />
              </span>
            </div>
            <div className={temporalIdentity.stable ? 'v3-status ready' : 'v3-status fallback'}>
              <span>{temporalIdentity.stable ? 'stable target' : 'warming up'}</span>
              <strong>{Math.round(temporalIdentity.score * 100)}%</strong>
            </div>
            <div className="diagnostic-row neutral">
              <span>{algorithmVersion === 'v5' || algorithmVersion === 'v6' ? 'Selected target' : 'Target'}</span>
              <strong>{selectedTargetName(objectProfiles, targetProfileId, baseObjectCandidates, targetBaseId, showObjectLabels) ?? temporalIdentity.name ?? 'none'}</strong>
            </div>
            <div className="diagnostic-row neutral">
              <span>Match streak</span>
              <strong>{temporalIdentity.streak}/3</strong>
            </div>
            <p className={temporalIdentity.stable ? 'diagnostic-copy' : 'diagnostic-copy warn'}>
              {(algorithmVersion === 'v5' || algorithmVersion === 'v6') && !targetProfileId && !targetBaseId
                ? algorithmVersion === 'v6'
                  ? 'V6 is auto-following the best hand-near object. Select an ID below if you want to force one target.'
                  : 'Select a base object ID or trained profile ID below. V5 will then score grip only for that target.'
                : temporalIdentity.stable
                  ? 'Selected target is stable across recent frames.'
                  : targetBaseId
                    ? `${algorithmVersion.toUpperCase()} waits for the selected base object to stay detected for a few frames before marking it active.`
                    : `${algorithmVersion.toUpperCase()} waits for repeated trained-object matches before marking the object detected.`}
            </p>
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
              {(algorithmVersion === 'v5' || algorithmVersion === 'v6') && !targetProfileId && !targetBaseId
                ? algorithmVersion === 'v6' ? 'Auto-following hand-near object' : 'Select a target object ID'
                : objectDetection?.matched
                ? `Object detected: ${objectDetection.name}`
                : objectProfiles.some((profile) => profile.enabled !== false)
                  ? 'Enabled object not detected'
                  : objectProfiles.length
                    ? 'All profiles disabled'
                    : 'No trained object yet'}
            </span>
            <strong>{objectDetection ? `${Math.round(objectDetection.score * 100)}%` : '--'}</strong>
          </div>
          {(algorithmVersion === 'v5' || algorithmVersion === 'v6') && (
            <div className="candidate-list">
              <div className="motion-header compact-heading">
                <span>
                  Base detector objects
                  <InlineExplain
                    label="Explain base detector objects"
                    text="These come from the pretrained MediaPipe EfficientDet object detector before any custom object profile is used. Select one for generic target tracking, then train a profile if you need stronger identity matching."
                    compact
                  />
                </span>
                <button
                  type="button"
                  className="label-toggle"
                  onClick={() => setShowObjectLabels((value) => !value)}
                  aria-pressed={showObjectLabels}
                >
                  {showObjectLabels ? 'Labels' : 'IDs'}
                </button>
              </div>
              {baseObjectCandidates.length ? (
                baseObjectCandidates.slice(0, 8).map((candidate) => (
                  <button
                    type="button"
                    className={[
                      'candidate-row',
                      candidate.score > 0.42 ? 'matched' : '',
                      targetBaseId === candidate.candidateId ? 'selected' : ''
                    ].filter(Boolean).join(' ')}
                    onClick={() => selectBaseTarget(candidate.candidateId)}
                    key={candidate.candidateId}
                  >
                    <Box size={14} />
                    <span>{baseObjectName(candidate, showObjectLabels)}</span>
                    <strong>{Math.round(candidate.score * 100)}%</strong>
                  </button>
                ))
              ) : (
                <p className="diagnostic-copy">No base-model objects visible yet. The base detector knows common COCO objects like bottle, cup, phone, laptop, and similar categories.</p>
              )}
            </div>
          )}
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
                const candidate = v3ProfileCandidates.find((item) => item.profileId === profile.id);
                const status = profileLiveStatus(profile, objectDetection, analysis, candidate);
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
                    <span>
                      {profile.name}
                      {(algorithmVersion === 'v3' || algorithmVersion === 'v4' || algorithmVersion === 'v5' || algorithmVersion === 'v6') && candidate && (
                        <small>{Math.round(candidate.score * 100)}%</small>
                      )}
                    </span>
                    {(algorithmVersion === 'v5' || algorithmVersion === 'v6') && (
                      <button
                        type="button"
                        className={targetProfileId === profile.id ? 'target-chip active' : 'target-chip'}
                        onClick={() => selectTargetProfile(profile.id)}
                        disabled={profile.enabled === false}
                      >
                        {targetProfileId === profile.id ? 'Target' : 'Track'}
                      </button>
                    )}
                    <strong>{status.label}</strong>
                  </div>
                );
              })}
              {(algorithmVersion === 'v3' || algorithmVersion === 'v4' || algorithmVersion === 'v5' || algorithmVersion === 'v6') && (
                <div className="candidate-list">
                  <div className="motion-header compact-heading">
                    <span>
                      {algorithmVersion === 'v5' || algorithmVersion === 'v6' ? `${algorithmVersion.toUpperCase()} object IDs in frame` : `${algorithmVersion.toUpperCase()} detectable now`}
                      <InlineExplain
                        label="Explain detectable profiles"
                        text={algorithmVersion === 'v5' || algorithmVersion === 'v6' ? EXPLAIN.v5Target : 'This mode scans only enabled trained profiles. Disable a profile to completely remove it from live object search.'}
                        compact
                      />
                    </span>
                  </div>
                  {v3ProfileCandidates.length ? (
                    v3ProfileCandidates.slice(0, algorithmVersion === 'v5' || algorithmVersion === 'v6' ? 8 : 5).map((candidate, index) => (
                      <button
                        type="button"
                        className={[
                          'candidate-row',
                          candidate.matched ? 'matched' : '',
                          targetProfileId === candidate.profileId ? 'selected' : ''
                        ].filter(Boolean).join(' ')}
                        onClick={() => algorithmVersion === 'v5' || algorithmVersion === 'v6' ? selectTargetProfile(candidate.profileId) : undefined}
                        key={candidate.candidateId}
                      >
                        <Target size={14} />
                        <span>
                          {algorithmVersion === 'v5' || algorithmVersion === 'v6' ? `O${index + 1} · ` : ''}
                          {candidate.name}
                        </span>
                        <strong>{Math.round(candidate.score * 100)}%</strong>
                      </button>
                    ))
                  ) : (
                    <p className="diagnostic-copy">
                      {algorithmVersion === 'v5' || algorithmVersion === 'v6'
                        ? 'No enabled trained object IDs are visible yet. Add stronger views or improve lighting.'
                        : 'No enabled trained object is confidently visible yet.'}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
          {(algorithmVersion === 'v5' || algorithmVersion === 'v6') && (
            <div className="class-filter-panel">
              <div className="motion-header compact-heading">
                <span>
                  Detected classes
                  <InlineExplain
                    label="Explain detected classes"
                    text="Classes currently reported by the base detector. Turn off classes you do not want to use as grip targets. Person is off by default."
                    compact
                  />
                </span>
              </div>
              {baseClassSummary.length ? (
                <div className="class-chip-grid">
                  {baseClassSummary.map((item) => (
                    <button
                      type="button"
                      className={item.enabled ? 'class-chip active' : 'class-chip'}
                      onClick={() => toggleBaseClass(item.key)}
                      aria-pressed={item.enabled}
                      key={item.key}
                    >
                      <span>{item.label}</span>
                      <strong>{item.count} · {Math.round(item.bestScore * 100)}%</strong>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="diagnostic-copy">No base-detector classes visible yet.</p>
              )}
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

        {algorithmVersion === 'v6' && !hasCalibration && (
          <div className="calibration-note warning">
            V6 is using RGB-only defaults. Calibrate strong and weak holds once for more stable live percentages.
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
      </aside>}
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
    source,
    viewRole: source === 'locked-crop' ? 'in-hand' : 'front'
  };
}

function cropOverlayStyle(review: PendingUploadReview): React.CSSProperties {
  const imageAspect = review.canvas.width / Math.max(1, review.canvas.height);
  const renderWidth = imageAspect >= 1 ? 100 : imageAspect * 100;
  const renderHeight = imageAspect >= 1 ? (1 / imageAspect) * 100 : 100;
  const offsetX = (100 - renderWidth) / 2;
  const offsetY = (100 - renderHeight) / 2;
  return {
    left: `${offsetX + (review.cropX / Math.max(1, review.canvas.width)) * renderWidth}%`,
    top: `${offsetY + (review.cropY / Math.max(1, review.canvas.height)) * renderHeight}%`,
    width: `${(review.cropSize / Math.max(1, review.canvas.width)) * renderWidth}%`,
    height: `${(review.cropSize / Math.max(1, review.canvas.height)) * renderHeight}%`
  };
}

function inferMaskShape(canvas: HTMLCanvasElement): CanvasObjectMaskOptions['maskShape'] {
  const aspectRatio = Math.max(canvas.width, canvas.height) / Math.max(1, Math.min(canvas.width, canvas.height));
  return aspectRatio > 1.45 ? 'rect' : 'ellipse';
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function formatTrainingRole(role: TrainingViewRole) {
  return role.replace('-', ' ');
}

function csvEscape(value: string) {
  return `"${value.replaceAll('"', '""')}"`;
}

function formatRecordingDuration(durationMs: number) {
  const totalSeconds = Math.max(0, Math.floor(durationMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function downloadTextFile(filename: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadBlobFile(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function mediaRecorderTypeFor(format: 'mp4' | 'webm') {
  const candidates =
    format === 'mp4'
      ? ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4;codecs=h264', 'video/mp4']
      : ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? null;
}

function drawExportVideoFrame(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  width: number,
  height: number,
  mirrored: boolean
) {
  ctx.save();
  if (mirrored) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();
}

function nearestOfflineTimelinePoint(points: OfflineTimelinePoint[], time: number) {
  if (!points.length) return null;
  return points.reduce((best, point) => (Math.abs(point.time - time) < Math.abs(best.time - time) ? point : best), points[0]);
}

function analysisFromOfflineTimelinePoint(base: GripAnalysis, point: OfflineTimelinePoint): GripAnalysis {
  return {
    ...base,
    gripPercentage: point.grip,
    confidence: point.confidence,
    contactPoints: Math.round(point.contact * 5),
    closureScore: point.closure,
    thumbOpposition: point.thumb,
    enclosureScore: point.enclosure,
    slipRisk: point.slip,
    motionState: point.slip > 0.55 ? 'slipping' : 'moving-with-hand',
    guidance: point.guidance as GripAnalysis['guidance'],
    objectLockQuality: point.lock,
    objectIdentityScore: point.objectMatch,
    objectIdentityName: point.object || base.objectIdentityName,
    objectIdentityMatched: point.objectMatch > 0.25,
    evidence: {
      ...base.evidence,
      fingerSegmentContactScore: point.contact,
      objectLockQuality: point.lock,
      persistentSlipScore: point.slip
    },
    diagnostics: {
      ...base.diagnostics,
      mode: point.mode as GripAnalysis['diagnostics']['mode'],
      state: point.state as GripAnalysis['diagnostics']['state'],
      recommendation: point.guidance
    }
  };
}

function objectRegionFromOfflineTimelinePoint(point: OfflineTimelinePoint, frameWidth: number, frameHeight: number): ObjectRegion | null {
  if (point.objectX === null || point.objectY === null || point.lock < 0.22 || point.objectMatch < 0.2) return null;
  const radiusX = point.objectRadiusX ?? 0;
  const radiusY = point.objectRadiusY ?? 0;
  if (!radiusX || !radiusY) return null;
  const maxRadius = Math.max(radiusX, radiusY);
  const minRadius = Math.min(radiusX, radiusY);
  if (maxRadius < 10 || minRadius < 8) return null;
  if (maxRadius > Math.min(frameWidth, frameHeight) * 0.28 || radiusX > frameWidth * 0.22 || radiusY > frameHeight * 0.34) return null;
  if (point.palmX !== null && point.palmY !== null) {
    const palmDistance = distance({ x: point.objectX, y: point.objectY }, { x: point.palmX, y: point.palmY });
    if (palmDistance > Math.max(260, maxRadius * 2.35)) return null;
  }
  const x = clampNumber(point.objectX, 0, frameWidth);
  const y = clampNumber(point.objectY, 0, frameHeight);
  const contour = [
    { x: x - radiusX, y: y - radiusY },
    { x: x + radiusX, y: y - radiusY },
    { x: x + radiusX, y: y + radiusY },
    { x: x - radiusX, y: y + radiusY }
  ];
  return {
    center: { x, y },
    radiusX,
    radiusY,
    angle: point.objectAngle ?? 0,
    confidence: point.objectMatch,
    locked: point.lock >= 0.34,
    source: 'segmenter',
    velocity: { x: 0, y: 0 },
    contour,
    shape: maxRadius / Math.max(1, minRadius) > 1.35 ? 'phone-like' : 'ellipse',
    aspectRatio: maxRadius / Math.max(1, minRadius),
    tightness: point.contact,
    lockAgeFrames: point.lock >= 0.34 ? 3 : 0,
    manuallyAdjusted: false,
    visualEdgeScore: point.objectMatch,
    visualTextureScore: point.objectMatch,
    independentEvidenceScore: point.objectMatch,
    relativeDriftScore: point.slip
  };
}

function drawOfflineV2TimelineOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  mirrored: boolean,
  point: OfflineTimelinePoint,
  object: ObjectRegion | null,
  analysis: GripAnalysis
) {
  ctx.save();
  if (mirrored) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  if (object) drawOfflineV2Object(ctx, object, analysis);
  if (point.palmX !== null && point.palmY !== null) drawOfflineV2Palm(ctx, point.palmX, point.palmY, analysis);
  ctx.restore();
}

function drawOfflineV2Object(ctx: CanvasRenderingContext2D, object: ObjectRegion, analysis: GripAnalysis) {
  const stateColor =
    analysis.guidance === 'Strong grip'
      ? '74, 222, 128'
      : analysis.guidance === 'Improve grip'
        ? '250, 204, 21'
        : '248, 113, 113';
  ctx.save();
  ctx.translate(object.center.x, object.center.y);
  ctx.rotate(object.angle);
  ctx.strokeStyle = `rgba(${stateColor}, 0.88)`;
  ctx.fillStyle = `rgba(${stateColor}, 0.07)`;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.ellipse(0, 0, object.radiusX, object.radiusY, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  if (object.contour.length >= 3) {
    ctx.save();
    ctx.setLineDash([8, 9]);
    ctx.strokeStyle = `rgba(${stateColor}, 0.34)`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    object.contour.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
}

function drawOfflineV2Palm(ctx: CanvasRenderingContext2D, x: number, y: number, analysis: GripAnalysis) {
  const radius = 11 + analysis.closureScore * 4;
  ctx.save();
  ctx.fillStyle = 'rgba(255, 209, 102, 0.92)';
  ctx.strokeStyle = 'rgba(8, 13, 20, 0.85)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawOfflineExportOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  analysis: GripAnalysis,
  timeline: OfflineTimelinePoint[],
  videoName: string,
  report: OfflineReport
) {
  const margin = Math.max(18, Math.round(width * 0.018));
  const leftWidth = Math.min(380, Math.max(260, width * 0.24));
  const rightWidth = Math.min(420, Math.max(300, width * 0.28));
  const leftHeight = 300;
  const rightHeight = 370;
  const bottom = Math.max(26, Math.round(height * 0.035));
  const leftX = margin;
  const leftY = Math.max(96, height - bottom - leftHeight);
  const rightX = width - margin - rightWidth;
  const rightY = Math.max(96, height - bottom - rightHeight);
  drawExportGlassPanel(ctx, leftX, leftY, leftWidth, leftHeight);
  drawExportGlassPanel(ctx, rightX, rightY, rightWidth, rightHeight);

  drawExportText(ctx, 'OFFLINE REVIEW', leftX + 22, leftY + 34, 15, 800, '#67e8f9');
  drawExportText(ctx, `${analysis.gripPercentage}%`, leftX + 22, leftY + 104, 74, 900, '#f8fafc');
  drawExportText(ctx, analysis.guidance, leftX + 22, leftY + 142, 24, 800, '#f8fafc');
  const source = videoName || 'Uploaded video';
  drawExportText(ctx, `${source.slice(0, 33)}${source.length > 33 ? '...' : ''}`, leftX + 22, leftY + 176, 17, 600, 'rgba(226, 232, 240, 0.86)');
  drawExportMiniRow(ctx, leftX, leftY + 208, leftWidth, 'State', analysis.diagnostics.state);
  drawExportMiniRow(ctx, leftX, leftY + 242, leftWidth, 'Mode', analysis.diagnostics.mode);
  drawExportMiniRow(ctx, leftX, leftY + 276, leftWidth, 'Timeline', `${timeline.length} pts`);

  drawExportText(ctx, 'PARAMETERS', rightX + 22, rightY + 34, 15, 800, '#67e8f9');
  const metrics = [
    ['Confidence', analysis.confidence],
    ['Lock', analysis.objectLockQuality],
    ['Closure', analysis.closureScore],
    ['Contact', analysis.evidence.fingerSegmentContactScore],
    ['Thumb', analysis.thumbOpposition],
    ['Slip', analysis.slipRisk]
  ] as const;
  metrics.forEach(([label, value], index) => {
    drawExportMetric(ctx, rightX + 22, rightY + 66 + index * 42, rightWidth - 44, label, value, label === 'Slip');
  });
  if (report) {
    drawExportText(ctx, `Avg ${report.averageGrip}%  Peak ${report.peakGrip}%`, rightX + 22, rightY + 308, 14, 800, 'rgba(240, 253, 250, 0.92)');
  }
  drawExportTimeline(ctx, rightX + 22, rightY + 324, rightWidth - 44, 30, timeline);
}

function drawCompactOfflineExportOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  analysis: GripAnalysis,
  timeline: OfflineTimelinePoint[],
  videoName: string,
  report: OfflineReport
) {
  const margin = Math.max(14, Math.round(width * 0.014));
  const topWidth = Math.min(560, Math.max(390, width * 0.34));
  const topHeight = 92;
  const metricWidth = Math.min(360, Math.max(260, width * 0.2));
  const metricHeight = 150;
  const metricX = width - margin - metricWidth;
  const metricY = height - margin - metricHeight;

  drawExportGlassPanel(ctx, margin, margin, topWidth, topHeight);
  drawExportText(ctx, 'OFFLINE V2 COMPACT', margin + 18, margin + 28, 13, 800, '#67e8f9');
  drawExportText(ctx, `${analysis.gripPercentage}%`, margin + 18, margin + 76, 48, 900, '#f8fafc');
  drawExportText(ctx, analysis.guidance, margin + 150, margin + 48, 24, 850, '#f8fafc');
  const compactSource = videoName || 'Uploaded video';
  drawExportText(
    ctx,
    `${analysis.diagnostics.state} · ${analysis.diagnostics.mode} · ${compactSource.slice(0, 24)}${compactSource.length > 24 ? '...' : ''}`,
    margin + 150,
    margin + 75,
    13,
    700,
    'rgba(226, 232, 240, 0.82)'
  );

  drawExportGlassPanel(ctx, metricX, metricY, metricWidth, metricHeight);
  drawExportText(ctx, 'GRIP METRICS', metricX + 16, metricY + 28, 13, 800, '#67e8f9');
  drawCompactMetric(ctx, metricX + 16, metricY + 48, metricWidth - 32, 'Lock', analysis.objectLockQuality);
  drawCompactMetric(ctx, metricX + 16, metricY + 72, metricWidth - 32, 'Contact', analysis.evidence.fingerSegmentContactScore);
  drawCompactMetric(ctx, metricX + 16, metricY + 96, metricWidth - 32, 'Thumb', analysis.thumbOpposition);
  drawCompactMetric(ctx, metricX + 16, metricY + 120, metricWidth - 32, 'Slip', analysis.slipRisk, true);
  if (report) {
    drawExportText(ctx, `Avg ${report.averageGrip}% · Peak ${report.peakGrip}%`, metricX + 16, metricY + 140, 12, 800, 'rgba(240, 253, 250, 0.9)');
  }

  const timelineWidth = Math.min(420, Math.max(250, width * 0.24));
  const timelineHeight = 26;
  const timelineX = width - margin - timelineWidth;
  const timelineY = margin + 12;
  drawExportGlassPanel(ctx, timelineX, timelineY, timelineWidth, 56);
  drawExportText(ctx, `${timeline.length} pts`, timelineX + 14, timelineY + 22, 12, 800, '#e0f2fe');
  drawExportTimeline(ctx, timelineX + 14, timelineY + 24, timelineWidth - 28, timelineHeight, timeline);
}

function drawExportGlassPanel(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number) {
  ctx.save();
  const gradient = ctx.createLinearGradient(x, y, x + width, y + height);
  gradient.addColorStop(0, 'rgba(255, 255, 255, 0.22)');
  gradient.addColorStop(1, 'rgba(8, 13, 20, 0.32)');
  ctx.fillStyle = gradient;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.lineWidth = 2;
  roundRect(ctx, x, y, width, height, 14);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawExportText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  size: number,
  weight: number,
  color: string
) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px system-ui, -apple-system, BlinkMacSystemFont, sans-serif`;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawExportMiniRow(ctx: CanvasRenderingContext2D, panelX: number, y: number, panelWidth: number, label: string, value: string) {
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.beginPath();
  ctx.moveTo(panelX + 22, y - 20);
  ctx.lineTo(panelX + panelWidth - 22, y - 20);
  ctx.stroke();
  drawExportText(ctx, label, panelX + 22, y, 17, 650, 'rgba(226, 232, 240, 0.75)');
  ctx.textAlign = 'right';
  ctx.fillStyle = '#f8fafc';
  ctx.font = '800 17px system-ui, -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.fillText(value, panelX + panelWidth - 22, y);
  ctx.restore();
}

function drawExportMetric(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  label: string,
  value: number,
  danger = false
) {
  const percent = Math.round(value * 100);
  drawExportText(ctx, label, x, y, 17, 750, 'rgba(226, 232, 240, 0.78)');
  ctx.save();
  ctx.textAlign = 'right';
  ctx.fillStyle = '#f8fafc';
  ctx.font = '800 17px system-ui, -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.fillText(`${percent}%`, x + width, y);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
  roundRect(ctx, x, y + 13, width, 8, 4);
  ctx.fill();
  const fill = Math.max(4, width * clampNumber(value, 0, 1));
  ctx.fillStyle = danger ? 'rgba(248, 113, 113, 0.92)' : 'rgba(103, 232, 249, 0.92)';
  roundRect(ctx, x, y + 13, fill, 8, 4);
  ctx.fill();
  ctx.restore();
}

function drawCompactMetric(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  label: string,
  value: number,
  danger = false
) {
  const percent = Math.round(value * 100);
  drawExportText(ctx, label, x, y, 12, 750, 'rgba(226, 232, 240, 0.78)');
  ctx.save();
  ctx.textAlign = 'right';
  ctx.fillStyle = '#f8fafc';
  ctx.font = '800 12px system-ui, -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.fillText(`${percent}%`, x + width, y);
  const barX = x + 74;
  const barWidth = Math.max(70, width - 124);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
  roundRect(ctx, barX, y - 9, barWidth, 6, 3);
  ctx.fill();
  const fill = Math.max(3, barWidth * clampNumber(value, 0, 1));
  ctx.fillStyle = danger ? 'rgba(248, 113, 113, 0.9)' : 'rgba(103, 232, 249, 0.9)';
  roundRect(ctx, barX, y - 9, fill, 6, 3);
  ctx.fill();
  ctx.restore();
}

function drawExportTimeline(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, timeline: OfflineTimelinePoint[]) {
  ctx.save();
  ctx.fillStyle = 'rgba(15, 23, 42, 0.54)';
  roundRect(ctx, x, y, width, height, 8);
  ctx.fill();
  const points = timeline.slice(-60);
  if (!points.length) {
    drawExportText(ctx, 'Timeline building...', x + 12, y + 20, 13, 650, 'rgba(226, 232, 240, 0.72)');
    ctx.restore();
    return;
  }
  const gap = 2;
  const barWidth = Math.max(2, (width - gap * (points.length - 1)) / points.length);
  points.forEach((point, index) => {
    const barHeight = Math.max(5, height * (point.grip / 100));
    ctx.fillStyle = point.slip > 0.45 ? '#f87171' : point.weak ? '#facc15' : '#86efac';
    roundRect(ctx, x + index * (barWidth + gap), y + height - barHeight, barWidth, barHeight, 2);
    ctx.fill();
  });
  ctx.restore();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function profileLiveStatus(
  profile: ObjectProfileV2,
  detection: ObjectProfileMatch,
  analysis: GripAnalysis,
  candidate?: ObjectProfileCandidate
) {
  if (profile.enabled === false) return { kind: 'disabled', label: 'disabled' };
  const isDetected = detection?.profileId === profile.id && detection.matched;
  if (isDetected && analysis.gripPercentage > 0 && ['Grip detected', 'Strong hold', 'Slip risk'].includes(analysis.diagnostics.state)) {
    return { kind: 'gripping', label: 'grip active' };
  }
  if (isDetected) return { kind: 'visible', label: 'in frame' };
  if (candidate?.matched) return { kind: 'candidate', label: 'candidate' };
  if (candidate) return { kind: 'enabled', label: `${Math.round(candidate.score * 100)}%` };
  return { kind: 'enabled', label: 'enabled' };
}

function updateBaseObjectTracks(
  previous: BaseTrackState,
  boxes: DetectedObjectBox[],
  timestamp: number,
  selectedId: string | null,
  hand: Landmark[] | null
): BaseTrackState {
  const unmatchedTracks = [...previous.tracks];
  const nextTracks: BaseObjectCandidate[] = [];
  let nextId = previous.nextId;
  const sortedBoxes = [...boxes].sort((a, b) => b.score - a.score).slice(0, 12);

  sortedBoxes.forEach((box) => {
    const detection = createBaseCandidateFromBox(box, nextId, 0, timestamp);
    let bestTrack: BaseObjectCandidate | null = null;
    let bestScore = 0;

    unmatchedTracks.forEach((track) => {
      const score = baseTrackMatchScore(track, detection) + (track.candidateId === selectedId ? 0.06 : 0);
      if (score > bestScore) {
        bestScore = score;
        bestTrack = track;
      }
    });

    if (bestTrack && bestScore >= 0.28) {
      const track = smoothBaseTrack(bestTrack, detection, timestamp);
      nextTracks.push(track);
      unmatchedTracks.splice(unmatchedTracks.indexOf(bestTrack), 1);
      return;
    }

    nextTracks.push(createBaseCandidateFromBox(box, nextId, nextId - 1, timestamp));
    nextId += 1;
  });

  unmatchedTracks.forEach((track) => {
    const missedFrames = track.missedFrames + 1;
    const keepMissed = track.candidateId === selectedId ? V5_BASE_TRACK_GRACE_MISSES + 4 : V5_BASE_TRACK_GRACE_MISSES;
    if (missedFrames > keepMissed) return;
    nextTracks.push({
      ...track,
      score: clampUnit(track.score * 0.72),
      missedFrames,
      box: makeDomRectLike(track.box.x, track.box.y, track.box.width, track.box.height)
    });
  });

  const reacquiredTracks = maybeReacquireSelectedBaseTrack(nextTracks, selectedId, hand);

  return {
    nextId,
    tracks: reacquiredTracks
      .sort((a, b) => {
        if (a.candidateId === selectedId) return -1;
        if (b.candidateId === selectedId) return 1;
        if (a.missedFrames !== b.missedFrames) return a.missedFrames - b.missedFrames;
        return b.score - a.score;
      })
      .slice(0, 12)
  };
}

function maybeReacquireSelectedBaseTrack(
  tracks: BaseObjectCandidate[],
  selectedId: string | null,
  hand: Landmark[] | null
) {
  if (!selectedId || !hand) return tracks;
  const selected = tracks.find((track) => track.candidateId === selectedId);
  if (!selected || selected.missedFrames < 1) return tracks;

  const freshCandidates = tracks.filter((track) => track.candidateId !== selectedId && track.missedFrames === 0 && isGripTargetEligible(track));
  let best: { candidate: BaseObjectCandidate; score: number } | null = null;
  for (const candidate of freshCandidates) {
    const handAffinity = baseCandidateHandAffinity(candidate, hand);
    const sizeSimilarity = baseSizeSimilarity(selected, candidate);
    const score = handAffinity * 0.68 + candidate.score * 0.22 + sizeSimilarity * 0.1;
    if (!best || score > best.score) {
      best = { candidate, score };
    }
  }

  if (!best) return tracks;
  const bestCandidate = best.candidate;
  const handAffinity = baseCandidateHandAffinity(bestCandidate, hand);
  if (handAffinity < 0.34 || bestCandidate.score < 0.14) return tracks;
  if (selected.missedFrames < 2 && selected.score > bestCandidate.score + 0.18) return tracks;

  const transferred: BaseObjectCandidate = {
    ...bestCandidate,
    index: selected.index,
    trackId: selected.trackId,
    candidateId: selected.candidateId,
    missedFrames: 0,
    seenFrames: selected.seenFrames + 1,
    score: clampUnit(Math.max(bestCandidate.score, selected.score * 0.82))
  };

  return tracks
    .filter((track) => track.candidateId !== selectedId && track.candidateId !== bestCandidate.candidateId)
    .concat(transferred);
}

function selectOfflineHandObjectCandidate(candidates: BaseObjectCandidate[], hand: Landmark[] | null) {
  if (!hand) return null;
  let best: { candidate: BaseObjectCandidate; confidence: number } | null = null;
  for (const candidate of candidates) {
    if (candidate.missedFrames > 2 || !isGripTargetEligible(candidate)) continue;
    const confidence = offlineCandidateConfidence(candidate, hand);
    if (!best || confidence > best.confidence) best = { candidate, confidence };
  }
  return best && best.confidence >= OFFLINE_BASE_TARGET_THRESHOLD ? best.candidate : null;
}

function offlineCandidateConfidence(candidate: BaseObjectCandidate, hand: Landmark[] | null) {
  if (!hand) return clampUnit(Math.max(0.42, candidate.score));
  const affinity = baseCandidateHandAffinity(candidate, hand);
  const detectorScore = candidate.score;
  const continuity = candidate.missedFrames === 0 ? 0.08 : -0.08 * candidate.missedFrames;
  return clampUnit(Math.max(0.42, affinity * 0.72 + detectorScore * 0.28 + continuity));
}

function updateOfflineV2Track(
  previousTrack: OfflineV2TrackState,
  candidates: BaseObjectCandidate[],
  hand: Landmark[] | null,
  previousObject: ObjectRegion | null,
  video: HTMLVideoElement,
  timestamp: number
): OfflineV2TrackState {
  const pixelCandidate = inferOfflinePixelObjectCandidate(video, previousObject);
  const pool = [...candidates.filter((candidate) => candidate.missedFrames <= 3), ...(pixelCandidate ? [pixelCandidate] : [])];
  let best: { candidate: BaseObjectCandidate; confidence: number } | null = null;

  for (const candidate of pool) {
    if (!isOfflineV2CandidateSizeValid(candidate, video.videoWidth, video.videoHeight)) continue;
    const handAffinity = hand ? offlineV2HandAffinity(candidate, hand) : 0.35;
    const previousAffinity = previousObject ? offlineV2PreviousAffinity(candidate, previousObject) : 0;
    if (hand && handAffinity < 0.18 && previousAffinity < 0.42) continue;
    const continuity =
      previousTrack.candidate && previousTrack.candidate.candidateId === candidate.candidateId
        ? Math.max(0.12, 0.28 - candidate.missedFrames * 0.04)
        : 0;
    const detectorSignal = Math.min(0.75, Math.max(0.18, candidate.score));
    const confidence = clampUnit(handAffinity * 0.48 + previousAffinity * 0.28 + detectorSignal * 0.16 + continuity);
    if (!best || confidence > best.confidence) best = { candidate, confidence };
  }

  if (best && best.confidence >= 0.16) {
    return {
      candidate: best.candidate,
      confidence: Math.max(0.38, best.confidence),
      ageFrames: previousTrack.candidate?.candidateId === best.candidate.candidateId ? previousTrack.ageFrames + 1 : 1,
      missedFrames: 0,
      lastSeenAt: timestamp
    };
  }

  const canHoldPrevious =
    previousTrack.candidate &&
    previousObject &&
    previousTrack.missedFrames < 14 &&
    timestamp - previousTrack.lastSeenAt < 2600 &&
    (!hand || offlineV2ObjectNearHand(previousObject, hand));
  if (canHoldPrevious && previousTrack.candidate) {
    const decayed = Math.max(0.18, previousTrack.confidence * 0.9);
    return {
      ...previousTrack,
      candidate: previousObjectToBaseCandidate(previousObject, previousTrack.candidate, decayed),
      confidence: decayed,
      missedFrames: previousTrack.missedFrames + 1
    };
  }

  return { candidate: null, confidence: 0, ageFrames: 0, missedFrames: previousTrack.missedFrames + 1, lastSeenAt: previousTrack.lastSeenAt };
}

let offlinePixelCanvas: HTMLCanvasElement | null = null;
let offlinePixelContext: CanvasRenderingContext2D | null = null;

function inferOfflinePixelObjectCandidate(video: HTMLVideoElement, previous: ObjectRegion | null): BaseObjectCandidate | null {
  if (!video.videoWidth || !video.videoHeight) return null;
  if (!offlinePixelCanvas) {
    offlinePixelCanvas = document.createElement('canvas');
    offlinePixelContext = offlinePixelCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!offlinePixelContext || !offlinePixelCanvas) return null;

  const sampleWidth = 180;
  const sampleHeight = Math.max(80, Math.round(sampleWidth * (video.videoHeight / Math.max(1, video.videoWidth))));
  offlinePixelCanvas.width = sampleWidth;
  offlinePixelCanvas.height = sampleHeight;
  offlinePixelContext.drawImage(video, 0, 0, sampleWidth, sampleHeight);
  const { data } = offlinePixelContext.getImageData(0, 0, sampleWidth, sampleHeight);

  let minX = sampleWidth;
  let minY = sampleHeight;
  let maxX = 0;
  let maxY = 0;
  let scoreTotal = 0;
  let count = 0;

  for (let y = 0; y < sampleHeight; y += 2) {
    for (let x = 0; x < sampleWidth; x += 2) {
      const index = (y * sampleWidth + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const saturation = max <= 0 ? 0 : (max - min) / max;
      const greenObject = g > 55 && g > r * 1.04 && g > b * 1.02 && saturation > 0.16;
      const coloredObject = saturation > 0.34 && max > 70 && !(r > g * 1.14 && r > b * 1.18);
      if (!greenObject && !coloredObject) continue;
      const weight = greenObject ? 1 : 0.54;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      scoreTotal += weight * saturation;
      count += 1;
    }
  }

  if (count < 24) {
    if (!previous || !previous.locked || (previous.lockAgeFrames ?? 0) < 2) return null;
    return baseCandidateFromRect(
      makeDomRectLike(
        previous.center.x - previous.radiusX,
        previous.center.y - previous.radiusY,
        previous.radiusX * 2,
        previous.radiusY * 2
      ),
      'offline-object',
      clampUnit((previous.confidence ?? 0.4) * 0.82),
      0,
      9001
    );
  }

  const scaleX = video.videoWidth / sampleWidth;
  const scaleY = video.videoHeight / sampleHeight;
  const paddingX = Math.max(18, (maxX - minX) * scaleX * 0.22);
  const paddingY = Math.max(18, (maxY - minY) * scaleY * 0.22);
  const x = clampNumber(minX * scaleX - paddingX, 0, video.videoWidth - 1);
  const y = clampNumber(minY * scaleY - paddingY, 0, video.videoHeight - 1);
  const width = clampNumber((maxX - minX) * scaleX + paddingX * 2, 36, video.videoWidth - x);
  const height = clampNumber((maxY - minY) * scaleY + paddingY * 2, 36, video.videoHeight - y);
  const confidence = clampUnit(Math.max(0.5, 0.28 + (scoreTotal / Math.max(1, count)) * 1.05 + Math.min(0.22, count / 2200)));
  return baseCandidateFromRect(makeDomRectLike(x, y, width, height), 'offline-object', confidence, 0, 9001);
}

function isOfflineV2CandidateSizeValid(candidate: BaseObjectCandidate, frameWidth: number, frameHeight: number) {
  const frameArea = Math.max(1, frameWidth * frameHeight);
  const areaRatio = (candidate.box.width * candidate.box.height) / frameArea;
  if (areaRatio > 0.42 || areaRatio < 0.002) return false;
  if (candidate.box.width > frameWidth * 0.72 || candidate.box.height > frameHeight * 0.86) return false;
  return true;
}

function offlineV2HandAffinity(candidate: BaseObjectCandidate, hand: Landmark[]) {
  const bounds = landmarksBounds(hand);
  const expandedHand = makeDomRectLike(
    bounds.x - bounds.width * 0.18,
    bounds.y - bounds.height * 0.22,
    bounds.width * 1.36,
    bounds.height * 1.44
  );
  const overlap = rectIoU(candidate.box, expandedHand);
  const palm = palmCenter(hand);
  const distanceToPalm = pointToRectDistance(palm, candidate.box);
  const handSpan = Math.max(bounds.width, bounds.height, 80);
  const palmScore = 1 - clampNumber(distanceToPalm / (handSpan * 0.92), 0, 1);
  const fingertipDistances = FINGERTIP_INDICES.map((index) => pointToRectDistance(hand[index], candidate.box));
  const nearTips = fingertipDistances.filter((value) => value < handSpan * 0.32).length / FINGERTIP_INDICES.length;
  return clampUnit(overlap * 0.42 + palmScore * 0.34 + nearTips * 0.24);
}

function offlineV2PreviousAffinity(candidate: BaseObjectCandidate, previous: ObjectRegion) {
  const previousRect = makeDomRectLike(
    previous.center.x - previous.radiusX,
    previous.center.y - previous.radiusY,
    previous.radiusX * 2,
    previous.radiusY * 2
  );
  const overlap = rectIoU(candidate.box, previousRect);
  const centerDistance = distance(candidate.center, previous.center);
  const size = Math.max(previous.radiusX, previous.radiusY, candidate.radiusX, candidate.radiusY, 1);
  const centerScore = 1 - clampNumber(centerDistance / (size * 2.2), 0, 1);
  return clampUnit(overlap * 0.56 + centerScore * 0.44);
}

function offlineV2ObjectNearHand(object: ObjectRegion, hand: Landmark[]) {
  const bounds = landmarksBounds(hand);
  const objectRect = makeDomRectLike(object.center.x - object.radiusX, object.center.y - object.radiusY, object.radiusX * 2, object.radiusY * 2);
  return rectIoU(objectRect, bounds) > 0.08 || pointToRectDistance(palmCenter(hand), objectRect) < Math.max(bounds.width, bounds.height) * 0.8;
}

function previousObjectToBaseCandidate(previous: ObjectRegion, seed: BaseObjectCandidate, score: number): BaseObjectCandidate {
  const box = makeDomRectLike(previous.center.x - previous.radiusX, previous.center.y - previous.radiusY, previous.radiusX * 2, previous.radiusY * 2);
  return {
    ...seed,
    box,
    score,
    center: previous.center,
    radiusX: previous.radiusX,
    radiusY: previous.radiusY,
    missedFrames: seed.missedFrames + 1
  };
}

function smoothOfflineTimeline(points: OfflineTimelinePoint[]) {
  if (points.length < 3) return points;
  return points.map((point, index) => {
    const neighbors = points.slice(Math.max(0, index - 2), Math.min(points.length, index + 3));
    const grip = Math.round(weightedAverage(neighbors.map((item) => item.grip)));
    const confidence = weightedAverage(neighbors.map((item) => item.confidence));
    const objectMatch = weightedAverage(neighbors.map((item) => item.objectMatch));
    const lock = weightedAverage(neighbors.map((item) => item.lock));
    const contact = weightedAverage(neighbors.map((item) => item.contact));
    const closure = weightedAverage(neighbors.map((item) => item.closure));
    const thumb = weightedAverage(neighbors.map((item) => item.thumb));
    const enclosure = weightedAverage(neighbors.map((item) => item.enclosure));
    const slip = weightedAverage(neighbors.map((item) => item.slip));
    const weakVotes = neighbors.filter((item) => item.weak).length;
    const bestGuidance = mostCommon(neighbors.map((item) => item.guidance));
    const bestObject = mostCommon(neighbors.map((item) => item.object).filter(Boolean));
    const bestMode = mostCommon(neighbors.map((item) => item.mode).filter(Boolean));
    const bestState = mostCommon(neighbors.map((item) => item.state).filter(Boolean));
    return {
      ...point,
      grip,
      confidence,
      objectMatch,
      lock,
      contact,
      closure,
      thumb,
      enclosure,
      slip,
      weak: weakVotes > neighbors.length / 2,
      guidance: bestGuidance || point.guidance,
      object: bestObject || point.object,
      mode: bestMode || point.mode,
      state: bestState || point.state
    };
  });
}

function refineOfflineTimeline(points: OfflineTimelinePoint[]) {
  const smoothed = smoothOfflineTimeline(points);
  if (smoothed.length < 3) return smoothed;
  return smoothed.map((point, index) => {
    const context = offlineTemporalContext(smoothed, point.time, 0.95);
    const futurePastObject = context.filter((item) => item.objectMatch > 0.22 || item.lock > 0.24);
    const continuityScore = futurePastObject.length / Math.max(1, context.length);
    const bridgeScore = offlineObjectBridgeScore(smoothed, index);
    const proximityScore = offlinePointHandObjectProximity(point);
    const contextObjectMatch = average(context.map((item) => item.objectMatch));
    const contextLock = average(context.map((item) => item.lock));
    const contextContact = average(context.map((item) => item.contact));
    const contextThumb = average(context.map((item) => item.thumb));
    const contextClosure = average(context.map((item) => item.closure));
    const contextEnclosure = average(context.map((item) => item.enclosure));
    const handEvidence = clampUnit(
      Math.max(point.contact, contextContact) * 0.42 +
        Math.max(point.closure, contextClosure) * 0.2 +
        Math.max(point.enclosure, contextEnclosure) * 0.16 +
        Math.max(point.thumb, contextThumb) * 0.22
    );
    const repairedObjectMatch = Math.max(
      point.objectMatch,
      contextObjectMatch * 0.9,
      continuityScore * 0.52,
      bridgeScore * 0.58,
      proximityScore * 0.46
    );
    const repairedLock = Math.max(point.lock, contextLock * 0.92, repairedObjectMatch * 0.92, continuityScore * 0.38, bridgeScore * 0.52);
    const occlusionAwareHold =
      repairedLock > 0.42 &&
      repairedObjectMatch > 0.28 &&
      (proximityScore > 0.24 || bridgeScore > 0.55) &&
      (Math.max(point.contact, contextContact) > 0.16 || Math.max(point.thumb, contextThumb) > 0.3 || Math.max(point.enclosure, contextEnclosure) > 0.24);
    const wrapHoldScore = clampUnit(
      repairedLock * 0.28 +
        repairedObjectMatch * 0.2 +
        Math.max(proximityScore, bridgeScore * 0.86) * 0.18 +
        Math.max(point.contact, contextContact) * 0.16 +
        Math.max(point.thumb, contextThumb) * 0.1 +
        continuityScore * 0.08
    );
    const contactDrivenGrip =
      repairedObjectMatch > 0.2 && handEvidence > 0.32
        ? Math.round(clampUnit(handEvidence * 0.62 + repairedObjectMatch * 0.38) * 86)
        : point.grip;
    const occlusionDrivenGrip = occlusionAwareHold
      ? Math.round(Math.max(wrapHoldScore * 94, Math.min(82, repairedLock * 84 + continuityScore * 10 + bridgeScore * 8)))
      : point.grip;
    const slip = Math.max(point.slip, offlineMotionDivergence(smoothed, index));
    const gripCeiling = slip > 0.62 ? 68 : 100;
    const grip = Math.max(point.grip, contactDrivenGrip, occlusionDrivenGrip);
    const correctedGrip = Math.min(grip, gripCeiling);
    const weak = correctedGrip < 42 || repairedLock < 0.2 || slip > 0.62;
    const guidance =
      repairedLock < 0.18
        ? 'Object uncertain'
        : slip > 0.62
          ? 'Improve grip'
          : correctedGrip > 68
            ? 'Strong grip'
            : correctedGrip > 36
              ? 'Improve grip'
              : point.guidance;
    return {
      ...point,
      grip: correctedGrip,
      confidence: Math.max(point.confidence, Math.min(0.92, repairedLock * 0.54 + handEvidence * 0.36 + bridgeScore * 0.1)),
      objectMatch: repairedObjectMatch,
      lock: repairedLock,
      slip,
      weak,
      guidance,
      state: repairedLock > 0.22 ? (correctedGrip > 68 ? 'Strong hold' : 'Grip detected') : point.state,
      mode: occlusionAwareHold && point.mode === 'open hand' ? 'power grip' : point.mode,
      contact: Math.max(point.contact, occlusionAwareHold ? Math.min(0.82, Math.max(contextContact, repairedLock * 0.58)) : point.contact),
      thumb: Math.max(point.thumb, occlusionAwareHold ? Math.min(0.78, contextThumb + repairedLock * 0.18) : point.thumb),
      object: point.object || (repairedObjectMatch > 0.22 ? 'tracked object' : '')
    };
  });
}

function sanitizeOfflineV2TimelineGeometry(points: OfflineTimelinePoint[], frameWidth: number, frameHeight: number) {
  return points.map((point) => {
    if (offlineTimelinePointHasSafeObjectGeometry(point, frameWidth, frameHeight)) return point;
    return {
      ...point,
      grip: Math.min(point.grip, 18),
      confidence: Math.min(point.confidence, 0.24),
      objectMatch: Math.min(point.objectMatch, 0.12),
      lock: Math.min(point.lock, 0.12),
      contact: Math.min(point.contact, 0.12),
      weak: true,
      guidance: 'Object uncertain',
      state: 'Object uncertain',
      object: '',
      objectX: null,
      objectY: null,
      objectRadiusX: null,
      objectRadiusY: null,
      objectAngle: null,
      rfdetrObjectScore: point.rfdetrObjectScore === undefined ? undefined : Math.min(point.rfdetrObjectScore, 0.12),
      rfdetrContact: point.rfdetrContact === undefined ? undefined : Math.min(point.rfdetrContact, 0.12)
    };
  });
}

function offlineTimelinePointHasSafeObjectGeometry(point: OfflineTimelinePoint, frameWidth: number, frameHeight: number) {
  if (point.objectX === null || point.objectY === null) return point.objectMatch < 0.18 && point.lock < 0.18;
  const radiusX = point.objectRadiusX ?? 0;
  const radiusY = point.objectRadiusY ?? 0;
  if (!radiusX || !radiusY) return point.objectMatch < 0.18 && point.lock < 0.18;
  const width = frameWidth || 1920;
  const height = frameHeight || 1080;
  const maxRadius = Math.max(radiusX, radiusY);
  const minRadius = Math.min(radiusX, radiusY);
  if (maxRadius < 10 || minRadius < 8) return false;
  if (maxRadius > Math.min(width, height) * 0.28 || radiusX > width * 0.22 || radiusY > height * 0.34) return false;
  if (point.palmX !== null && point.palmY !== null) {
    const palmDistance = distance({ x: point.objectX, y: point.objectY }, { x: point.palmX, y: point.palmY });
    if (palmDistance > Math.max(260, maxRadius * 2.35)) return false;
  }
  return true;
}

function buildOfflineReport(points: OfflineTimelinePoint[], videoName: string, duration: number): OfflineReport {
  if (!points.length) return null;
  const averageGrip = Math.round(average(points.map((point) => point.grip)));
  const peakGrip = Math.max(...points.map((point) => point.grip));
  const averageObjectMatch = Math.round(average(points.map((point) => point.objectMatch)) * 100);
  const averageLock = Math.round(average(points.map((point) => point.lock)) * 100);
  const weakSegments = collectOfflineSegments(points, (point) => point.weak, 'weak grip');
  const slipEvents = collectOfflineSegments(points, (point) => point.slip > 0.55, 'slip risk');
  const summary =
    averageLock < 25
      ? 'Object evidence is weak; use V2 offline or tighter object framing.'
      : averageGrip >= 68
        ? 'Strong visual grip for most of the video.'
        : averageGrip >= 42
          ? 'Mixed grip quality with improvement windows.'
          : 'Low visual grip stability in this run.';
  return {
    generatedAt: new Date().toISOString(),
    videoName,
    duration,
    points: points.length,
    averageGrip,
    peakGrip,
    averageObjectMatch,
    averageLock,
    weakSegments,
    slipEvents,
    summary
  };
}

function collectOfflineSegments(points: OfflineTimelinePoint[], predicate: (point: OfflineTimelinePoint) => boolean, reason: string) {
  const segments: OfflineSegment[] = [];
  let start: number | null = null;
  let end = 0;
  points.forEach((point) => {
    if (predicate(point)) {
      if (start === null) start = point.time;
      end = point.time;
      return;
    }
    if (start !== null && end - start >= 0.25) {
      segments.push({ start, end, reason });
    }
    start = null;
  });
  if (start !== null && end - start >= 0.25) segments.push({ start, end, reason });
  return segments;
}

function offlineTemporalContext(points: OfflineTimelinePoint[], time: number, radiusSeconds: number) {
  const context = points.filter((point) => Math.abs(point.time - time) <= radiusSeconds);
  return context.length ? context : points;
}

function offlineObjectBridgeScore(points: OfflineTimelinePoint[], index: number) {
  const current = points[index];
  if (!current) return 0;
  const previous = findNearestObjectEvidence(points, index, -1);
  const next = findNearestObjectEvidence(points, index, 1);
  if (!previous || !next) return 0;
  const gap = next.time - previous.time;
  if (gap <= 0 || gap > 2.4) return 0;
  const timeBalance = 1 - clampNumber(Math.abs(current.time - (previous.time + gap / 2)) / (gap / 2 + 0.001), 0, 1);
  const previousStrength = Math.max(previous.objectMatch, previous.lock);
  const nextStrength = Math.max(next.objectMatch, next.lock);
  const proximity = offlinePointHandObjectProximity(current);
  return clampUnit(Math.min(previousStrength, nextStrength) * 0.58 + timeBalance * 0.16 + proximity * 0.26);
}

function findNearestObjectEvidence(points: OfflineTimelinePoint[], startIndex: number, direction: -1 | 1) {
  for (let index = startIndex + direction; index >= 0 && index < points.length; index += direction) {
    const point = points[index];
    if (Math.abs(point.time - points[startIndex].time) > 1.4) return null;
    if (point.objectMatch > 0.24 || point.lock > 0.32) return point;
  }
  return null;
}

function offlinePointHandObjectProximity(point: OfflineTimelinePoint) {
  if (point.objectX === null || point.objectY === null || point.palmX === null || point.palmY === null) return 0;
  const distancePx = distance({ x: point.objectX, y: point.objectY }, { x: point.palmX, y: point.palmY });
  return 1 - clampNumber(distancePx / 240, 0, 1);
}

function offlineMotionDivergence(points: OfflineTimelinePoint[], index: number) {
  const previous = points[index - 1];
  const current = points[index];
  if (!previous || !current) return 0;
  if (
    previous.objectX === null ||
    previous.objectY === null ||
    previous.palmX === null ||
    previous.palmY === null ||
    current.objectX === null ||
    current.objectY === null ||
    current.palmX === null ||
    current.palmY === null
  ) {
    return 0;
  }
  const objectMotion = distance({ x: previous.objectX, y: previous.objectY }, { x: current.objectX, y: current.objectY });
  const palmMotion = distance({ x: previous.palmX, y: previous.palmY }, { x: current.palmX, y: current.palmY });
  const relative = Math.abs(objectMotion - palmMotion);
  if (objectMotion < 8 && palmMotion < 8) return 0;
  return clampUnit((relative - 18) / 90);
}

function updateLiveIdentityMemory(
  previous: LiveIdentityMemory,
  candidate: ObjectProfileMatch,
  hasBaseLock: boolean
): LiveIdentityMemory {
  if (candidate && (candidate.matched || candidate.score >= 0.4)) {
    const same = previous?.profileId === candidate.profileId;
    return {
      profileId: candidate.profileId,
      name: candidate.name,
      score: clampUnit(same ? previous.score * 0.62 + candidate.score * 0.38 : candidate.score),
      matched: candidate.matched || (same && previous.seenFrames >= 2 && candidate.score >= 0.36),
      seenFrames: same ? previous.seenFrames + 1 : 1,
      missedFrames: 0
    };
  }
  if (previous && hasBaseLock && previous.missedFrames < 6) {
    return {
      ...previous,
      score: clampUnit(previous.score * 0.88),
      matched: previous.seenFrames >= 2 && previous.missedFrames < 3,
      missedFrames: previous.missedFrames + 1
    };
  }
  return null;
}

function liveIdentityMemoryToMatch(memory: LiveIdentityMemory): ObjectProfileMatch {
  if (!memory) return null;
  return {
    profileId: memory.profileId,
    name: memory.name,
    score: memory.score,
    matched: memory.matched
  };
}

function weightedAverage(values: number[]) {
  if (!values.length) return 0;
  const weights = values.map((_value, index) => (index === Math.floor(values.length / 2) ? 1.4 : 1));
  const total = values.reduce((sum, value, index) => sum + value * weights[index], 0);
  const weightTotal = weights.reduce((sum, value) => sum + value, 0);
  return total / Math.max(1, weightTotal);
}

function mostCommon(values: string[]) {
  const counts = new Map<string, number>();
  values.forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1));
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '';
}

function baseCandidateFromRect(
  rect: DOMRectReadOnly,
  label: string,
  score: number,
  index: number,
  trackId: number
): BaseObjectCandidate {
  return {
    box: rect,
    label,
    score,
    index,
    trackId,
    candidateId: `base-track-${trackId}`,
    missedFrames: 0,
    seenFrames: 1,
    center: {
      x: rect.x + rect.width / 2,
      y: rect.y + rect.height / 2
    },
    radiusX: Math.max(14, rect.width / 2),
    radiusY: Math.max(14, rect.height / 2)
  };
}

function createOfflineSurrogateHand(candidate: BaseObjectCandidate): Landmark[] {
  const cx = candidate.center.x;
  const cy = candidate.center.y;
  const rx = Math.max(26, candidate.radiusX);
  const ry = Math.max(34, candidate.radiusY);
  const wrist = { x: cx + rx * 0.18, y: cy + ry * 1.35 };
  const hand: Landmark[] = Array.from({ length: 21 }, () => ({ ...wrist }));
  hand[0] = wrist;
  hand[1] = { x: cx - rx * 0.2, y: cy + ry * 1.05 };
  hand[2] = { x: cx - rx * 0.42, y: cy + ry * 0.72 };
  hand[3] = { x: cx - rx * 0.62, y: cy + ry * 0.42 };
  hand[4] = { x: cx - rx * 0.76, y: cy + ry * 0.14 };
  hand[5] = { x: cx - rx * 0.42, y: cy + ry * 0.7 };
  hand[6] = { x: cx - rx * 0.58, y: cy + ry * 0.26 };
  hand[7] = { x: cx - rx * 0.5, y: cy - ry * 0.08 };
  hand[8] = { x: cx - rx * 0.18, y: cy - ry * 0.22 };
  hand[9] = { x: cx - rx * 0.08, y: cy + ry * 0.78 };
  hand[10] = { x: cx - rx * 0.16, y: cy + ry * 0.22 };
  hand[11] = { x: cx - rx * 0.03, y: cy - ry * 0.08 };
  hand[12] = { x: cx + rx * 0.18, y: cy - ry * 0.2 };
  hand[13] = { x: cx + rx * 0.2, y: cy + ry * 0.82 };
  hand[14] = { x: cx + rx * 0.16, y: cy + ry * 0.3 };
  hand[15] = { x: cx + rx * 0.28, y: cy + ry * 0.02 };
  hand[16] = { x: cx + rx * 0.48, y: cy - ry * 0.08 };
  hand[17] = { x: cx + rx * 0.46, y: cy + ry * 0.88 };
  hand[18] = { x: cx + rx * 0.42, y: cy + ry * 0.42 };
  hand[19] = { x: cx + rx * 0.52, y: cy + ry * 0.16 };
  hand[20] = { x: cx + rx * 0.72, y: cy + ry * 0.08 };
  return hand;
}

function isGripTargetEligible(candidate: BaseObjectCandidate, frameWidth = 0, frameHeight = 0, ignoreLabel = false) {
  const label = baseClassKey(candidate.label);
  if (!ignoreLabel && label === 'person') return false;
  const width = candidate.box.width;
  const height = candidate.box.height;
  const area = width * height;
  const frameArea = frameWidth > 0 && frameHeight > 0 ? frameWidth * frameHeight : 0;
  if (frameArea > 0 && area / frameArea > 0.34) return false;
  const longerSide = Math.max(width, height);
  const shorterSide = Math.max(1, Math.min(width, height));
  const aspectRatio = longerSide / shorterSide;
  if (frameWidth > 0 && width / frameWidth > 0.72) return false;
  if (frameHeight > 0 && height / frameHeight > 0.78) return false;
  if (aspectRatio > 5.5) return false;
  return true;
}

function createBaseCandidateFromBox(
  box: DetectedObjectBox,
  trackId: number,
  index: number,
  _timestamp: number
): BaseObjectCandidate {
  return {
    ...box,
    box: makeDomRectLike(box.box.x, box.box.y, box.box.width, box.box.height),
    index,
    trackId,
    candidateId: `base-track-${trackId}`,
    missedFrames: 0,
    seenFrames: 1,
    center: {
      x: box.box.x + box.box.width / 2,
      y: box.box.y + box.box.height / 2
    },
    radiusX: Math.max(14, box.box.width / 2),
    radiusY: Math.max(14, box.box.height / 2)
  };
}

function smoothBaseTrack(
  track: BaseObjectCandidate,
  detection: BaseObjectCandidate,
  _timestamp: number
): BaseObjectCandidate {
  const alpha = track.missedFrames > 0 ? 0.74 : 0.58;
  const x = track.box.x * (1 - alpha) + detection.box.x * alpha;
  const y = track.box.y * (1 - alpha) + detection.box.y * alpha;
  const width = track.box.width * (1 - alpha) + detection.box.width * alpha;
  const height = track.box.height * (1 - alpha) + detection.box.height * alpha;
  const center = {
    x: x + width / 2,
    y: y + height / 2
  };
  return {
    ...detection,
    index: track.index,
    trackId: track.trackId,
    candidateId: track.candidateId,
    score: clampUnit(track.score * 0.46 + detection.score * 0.54),
    box: makeDomRectLike(x, y, width, height),
    center,
    radiusX: Math.max(14, width / 2),
    radiusY: Math.max(14, height / 2),
    missedFrames: 0,
    seenFrames: track.seenFrames + 1
  };
}

function baseCandidateHandAffinity(candidate: BaseObjectCandidate, hand: Landmark[]) {
  if (hand.length < 6) return 0;
  const scale = handSize(hand);
  const fingertips = [4, 8, 12, 16, 20].map((index) => hand[index]).filter(Boolean);
  const contactPoints = [...fingertips, palmCenter(hand)];
  const nearestRectDistance = Math.min(...contactPoints.map((point) => pointToRectDistance(point, candidate.box)));
  const rectContact = clampUnit(1 - nearestRectDistance / Math.max(18, scale * 0.82));
  const centerDistance = distance(candidate.center, palmCenter(hand));
  const centerProximity = clampUnit(1 - centerDistance / Math.max(24, scale * 1.75));
  const handBounds = landmarksBounds(hand);
  const overlap = rectIoU(candidate.box, handBounds);
  return clampUnit(rectContact * 0.62 + centerProximity * 0.24 + Math.min(1, overlap * 4) * 0.14);
}

function baseSizeSimilarity(a: BaseObjectCandidate, b: BaseObjectCandidate) {
  const areaA = Math.max(1, a.box.width * a.box.height);
  const areaB = Math.max(1, b.box.width * b.box.height);
  return Math.min(areaA, areaB) / Math.max(areaA, areaB);
}

function baseTrackMatchScore(track: BaseObjectCandidate, detection: BaseObjectCandidate) {
  const overlap = rectIoU(track.box, detection.box);
  const radius = Math.max(24, Math.max(track.radiusX, track.radiusY, detection.radiusX, detection.radiusY));
  const centerScore = clampUnit(1 - distance(track.center, detection.center) / (radius * 2.35));
  const areaA = Math.max(1, track.box.width * track.box.height);
  const areaB = Math.max(1, detection.box.width * detection.box.height);
  const sizeScore = Math.min(areaA, areaB) / Math.max(areaA, areaB);
  return overlap * 0.5 + centerScore * 0.38 + sizeScore * 0.12;
}

function pointToRectDistance(point: Point, rect: DOMRectReadOnly) {
  const dx = Math.max(rect.x - point.x, 0, point.x - (rect.x + rect.width));
  const dy = Math.max(rect.y - point.y, 0, point.y - (rect.y + rect.height));
  return Math.hypot(dx, dy);
}

function landmarksBounds(points: Landmark[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  const padding = Math.max(18, Math.max(maxX - minX, maxY - minY) * 0.18);
  return makeDomRectLike(minX - padding, minY - padding, maxX - minX + padding * 2, maxY - minY + padding * 2);
}

function rectIoU(a: DOMRectReadOnly, b: DOMRectReadOnly) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = Math.max(1, a.width * a.height + b.width * b.height - intersection);
  return intersection / union;
}

function makeDomRectLike(x: number, y: number, width: number, height: number): DOMRectReadOnly {
  return {
    x,
    y,
    width,
    height,
    top: y,
    left: x,
    right: x + width,
    bottom: y + height,
    toJSON: () => ({ x, y, width, height, top: y, left: x, right: x + width, bottom: y + height })
  } as DOMRectReadOnly;
}

function summarizeBaseClasses(candidates: BaseObjectCandidate[], enabledMap: Record<string, boolean>): BaseClassSummary[] {
  const byClass = new Map<string, BaseClassSummary>();
  candidates.forEach((candidate) => {
    const key = baseClassKey(candidate.label);
    const existing = byClass.get(key);
    const label = formatBaseClassLabel(key);
    if (!existing) {
      byClass.set(key, {
        key,
        label,
        count: 1,
        bestScore: candidate.score,
        enabled: isBaseClassEnabled(candidate.label, enabledMap)
      });
      return;
    }
    existing.count += 1;
    existing.bestScore = Math.max(existing.bestScore, candidate.score);
    existing.enabled = isBaseClassEnabled(candidate.label, enabledMap);
  });
  return Array.from(byClass.values()).sort((a, b) => Number(b.enabled) - Number(a.enabled) || b.bestScore - a.bestScore || a.label.localeCompare(b.label));
}

function isBaseClassEnabled(label: string, enabledMap: Record<string, boolean>) {
  const key = baseClassKey(label);
  return enabledMap[key] ?? key !== 'person';
}

function baseClassKey(label: string) {
  return (label || 'unknown').toLowerCase().trim().replace(/\s+/g, ' ');
}

function formatBaseClassLabel(key: string) {
  return key.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatOfflinePhase(phase: 'idle' | 'processing' | 'reviewing' | 'complete') {
  if (phase === 'processing') return 'processing';
  if (phase === 'reviewing') return 'reviewing';
  if (phase === 'complete') return 'complete';
  return 'ready';
}

function baseCandidateToTemporalCandidate(candidate: BaseObjectCandidate, confidenceOverride?: number): ObjectProfileCandidate {
  const score = confidenceOverride ?? candidate.score;
  return {
    candidateId: candidate.candidateId,
    profileId: candidate.candidateId,
    name: baseObjectName(candidate, false),
    score,
    matched: score >= OFFLINE_BASE_TARGET_THRESHOLD,
    center: candidate.center,
    radiusX: candidate.radiusX,
    radiusY: candidate.radiusY,
    aspectRatio: Math.max(candidate.radiusX, candidate.radiusY) / Math.max(1, Math.min(candidate.radiusX, candidate.radiusY)),
    descriptorQuality: Math.min(0.82, Math.max(0.32, score)),
    scanRank: candidate.index
  };
}

function objectRegionFromBaseCandidate(candidate: BaseObjectCandidate, previous: ObjectRegion | null, confidenceOverride?: number): ObjectRegion {
  const radiusX = Math.max(16, candidate.radiusX);
  const radiusY = Math.max(16, candidate.radiusY);
  const aspectRatio = Math.max(radiusX, radiusY) / Math.max(1, Math.min(radiusX, radiusY));
  const detectorLabel = `base:${candidate.candidateId}`;
  const sameTarget = previous?.detectorLabel === detectorLabel;
  const detectorConfidence = confidenceOverride ?? candidate.score;
  const confidence = clampUnit(0.62 + detectorConfidence * 0.38);
  const region: ObjectRegion = {
    center: candidate.center,
    radiusX,
    radiusY,
    angle: sameTarget ? previous?.angle ?? 0 : 0,
    confidence,
    locked: detectorConfidence >= OFFLINE_BASE_TARGET_THRESHOLD,
    source: 'detector',
    velocity: sameTarget && previous ? { x: candidate.center.x - previous.center.x, y: candidate.center.y - previous.center.y } : { x: 0, y: 0 },
    contour: [],
    shape: aspectRatio > 1.35 ? 'phone-like' : aspectRatio > 1.12 ? 'ellipse' : 'unknown',
    aspectRatio,
    tightness: 0.76,
    lockAgeFrames: detectorConfidence >= OFFLINE_BASE_TARGET_THRESHOLD ? (sameTarget ? previous?.lockAgeFrames ?? 0 : 0) + 1 : 0,
    manuallyAdjusted: false,
    visualEdgeScore: Math.min(0.82, 0.32 + detectorConfidence * 0.58),
    visualTextureScore: Math.min(0.72, 0.28 + detectorConfidence * 0.48),
    independentEvidenceScore: Math.min(0.9, 0.56 + detectorConfidence * 0.34),
    relativeDriftScore: sameTarget && previous ? clampUnit(distance(candidate.center, previous.center) / Math.max(1, Math.max(radiusX, radiusY) * 1.8)) : 0,
    detectorLabel,
    detectorScore: detectorConfidence
  };
  region.contour = Array.from({ length: 28 }, (_item, index) => ellipsePoint(region, (index / 28) * Math.PI * 2));
  return region;
}

function baseObjectName(candidate: Pick<BaseObjectCandidate, 'index' | 'label'>, showLabel = true) {
  const id = `B${candidate.index + 1}`;
  if (!showLabel) return id;
  const label = candidate.label && candidate.label !== 'unknown' ? candidate.label : 'object';
  return `${id} · ${label}`;
}

function selectedTargetName(
  profiles: ObjectProfileV2[],
  profileId: string | null,
  baseCandidates: BaseObjectCandidate[],
  baseId: string | null,
  showBaseLabel = true
) {
  if (profileId) return profiles.find((profile) => profile.id === profileId)?.name ?? null;
  if (baseId) {
    const candidate = baseCandidates.find((item) => item.candidateId === baseId);
    return candidate ? baseObjectName(candidate, showBaseLabel) : null;
  }
  return null;
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

function formatRfdetrRuntimeStatus(runtime: RfdetrRuntime) {
  if (runtime.status === 'ready') return 'ready';
  if (runtime.status === 'pending') return 'pending';
  if (runtime.status === 'unavailable') return 'unavailable';
  return 'idle';
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
  if (fromUrl === 'v1' || fromUrl === 'v2' || fromUrl === 'v3' || fromUrl === 'v4' || fromUrl === 'v5' || fromUrl === 'v6' || fromUrl === 'v7' || fromUrl === 'v8') return fromUrl;
  const fromStorage = window.localStorage.getItem(ALGORITHM_VERSION_STORAGE_KEY);
  return fromStorage === 'v1' || fromStorage === 'v2' || fromStorage === 'v3' || fromStorage === 'v4' || fromStorage === 'v5' || fromStorage === 'v6' || fromStorage === 'v7' || fromStorage === 'v8'
    ? fromStorage
    : 'v5';
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
  return profiles.map((profile) => {
    const samples = profile.samples.map((sample) => ({
      ...sample,
      viewRole: sample.viewRole ?? 'front'
    }));
    return {
      ...profile,
      samples,
      enabled: profile.enabled !== false,
      exemplarDescriptors: profile.exemplarDescriptors ?? samples.flatMap((sample) => sample.descriptorVariants ?? [sample.descriptor.vector]).slice(0, 96),
      strength: profile.strength ?? profileStrength(samples),
      coverageScore: profile.coverageScore ?? trainingCoverage(samples)
    };
  });
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
