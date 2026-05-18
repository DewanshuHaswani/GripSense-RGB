export type Point = {
  x: number;
  y: number;
};

export type Landmark = Point & {
  z?: number;
  visibility?: number;
};

export type MotionState = 'idle' | 'moving-with-hand' | 'slipping' | 'uncertain';

export type GripGuidance = 'Strong grip' | 'Improve grip' | 'Reposition' | 'Object not locked';

export type AlgorithmVersion = 'v1' | 'v2';

export type GripMode = 'phone-side grip' | 'pinch grip' | 'power grip' | 'hook grip' | 'open hand' | 'uncertain';

export type GripState = 'No hand' | 'Hand only' | 'Object uncertain' | 'Grip detected' | 'Strong hold' | 'Slip risk';

export type ObjectRegion = {
  center: Point;
  radiusX: number;
  radiusY: number;
  angle: number;
  confidence: number;
  locked: boolean;
  source: 'automatic' | 'manual' | 'detector' | 'segmenter';
  velocity: Point;
  contour: Point[];
  shape?: 'ellipse' | 'phone-like' | 'unknown';
  aspectRatio?: number;
  tightness?: number;
  lockAgeFrames?: number;
  manuallyAdjusted?: boolean;
  visualEdgeScore?: number;
  visualTextureScore?: number;
  independentEvidenceScore?: number;
};

export type GripPoint = Point & {
  score: number;
  label: 'thumb' | 'finger' | 'opposition' | 'support';
};

export type GripEvidence = {
  fingerCurlScore: number;
  fingerSegmentContactScore: number;
  palmObjectContainmentScore: number;
  thumbSupportScore: number;
  phoneSideGripScore: number;
  persistentSlipScore: number;
  objectLockQuality: number;
  pinchScore: number;
  powerGripScore: number;
  hookGripScore: number;
  visibleContactScore: number;
  occlusionResilienceScore: number;
  motionStabilityScore: number;
  independentObjectScore: number;
  temporalLockScore: number;
  modeScores: Record<GripMode, number>;
  positiveReasons: string[];
  negativeReasons: string[];
};

export type GripCalibrationBaseline = {
  mode: GripMode;
  gripPercentage: number;
  closureScore: number;
  enclosureScore: number;
  fingerCurlScore: number;
  fingerSegmentContactScore: number;
  phoneSideGripScore: number;
  pinchScore: number;
  powerGripScore: number;
  thumbSupportScore: number;
  objectLockQuality: number;
  createdAt: number;
};

export type GripCalibrationProfiles = Partial<Record<GripMode, {
  strong?: GripCalibrationBaseline;
  weak?: GripCalibrationBaseline;
}>>;

export type GripDiagnostics = {
  mode: GripMode;
  state: GripState;
  recommendation: string;
  objectIssue: string | null;
  gripIssue: string | null;
  scoreBreakdown: Array<{
    label: string;
    value: number;
    impact: 'positive' | 'negative' | 'neutral';
  }>;
};

export type GripAnalysis = {
  gripPercentage: number;
  confidence: number;
  contactPoints: number;
  closureScore: number;
  thumbOpposition: number;
  enclosureScore: number;
  motionCoupling: number;
  slipRisk: number;
  motionState: MotionState;
  guidance: GripGuidance;
  message: string;
  palmCenter: Point | null;
  handVelocity: Point;
  recommendedGripPoints: GripPoint[];
  objectLockQuality: number;
  evidence: GripEvidence;
  calibrated: boolean;
  diagnostics: GripDiagnostics;
};

export type TrackingFrame = {
  hand: Landmark[] | null;
  object: ObjectRegion | null;
  analysis: GripAnalysis;
};
