export type Point = {
  x: number;
  y: number;
};

export type Landmark = Point & {
  z?: number;
  visibility?: number;
};

export type MotionState = 'idle' | 'moving-with-hand' | 'slipping' | 'uncertain';

export type GripGuidance = 'Strong grip' | 'Improve grip' | 'Reposition' | 'Object not locked' | 'Object uncertain';

export type AlgorithmVersion = 'v1' | 'v2' | 'v3';

export type GripMode = 'phone-side grip' | 'pinch grip' | 'power grip' | 'hook grip' | 'open hand' | 'uncertain';

export type GripState = 'No hand' | 'Hand only' | 'Object uncertain' | 'Grip detected' | 'Strong hold' | 'Slip risk';

export type V3DiagnosticCode =
  | 'object_uncertain'
  | 'hand_occluded'
  | 'contact_uncertain'
  | 'slip_risk'
  | 'server_unavailable'
  | 'strong_hold';

export type GripIssueCategory =
  | 'none'
  | 'object_problem'
  | 'pose_problem'
  | 'motion_problem'
  | 'identity_problem'
  | V3DiagnosticCode;

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
  relativeDriftScore?: number;
};

export type GripPoint = Point & {
  score: number;
  label: 'thumb' | 'finger' | 'opposition' | 'support';
};

export type GripEvidence = {
  fingerCurlScore: number;
  fingerSegmentContactScore: number;
  contactRoles: {
    thumb: number;
    index: number;
    middle: number;
    ring: number;
    pinky: number;
    palm: number;
  };
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
  issueCategory: GripIssueCategory;
  scoreBreakdown: Array<{
    label: string;
    value: number;
    impact: 'positive' | 'negative' | 'neutral';
  }>;
};

export type V3ContactMap = {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
  palm: number;
};

export type V3SubScores = {
  objectEvidence: number;
  handEvidence: number;
  contactEvidence: number;
  temporalEvidence: number;
};

export type V3PerceptionResponse = {
  version: 'v3';
  frameTimestamp: number;
  modelTimestamp?: number;
  latencyMs: number;
  uncertainty: number;
  hand: {
    meshQuality: number;
    occlusion: number;
    handednessConfidence: number;
    fingerArticulation: number;
    joints?: Landmark[];
  };
  object: {
    present: boolean;
    maskConfidence: number;
    maskStability: number;
    identityConfidence: number;
    poseConfidence: number;
    lockConfidence: number;
  };
  contact: V3ContactMap & {
    coverage: number;
    opposingPairs: number;
  };
  temporal: {
    continuity: number;
    coupling: number;
    slipRisk: number;
    jitter: number;
  };
  diagnostics?: V3DiagnosticCode[];
};

export type V3AnalysisDetails = {
  status: 'server' | 'fallback';
  reason: V3DiagnosticCode | null;
  usedServerResult: boolean;
  endpoint: string;
  serverLatencyMs: number | null;
  serverAgeMs: number | null;
  modelConfidence: number;
  uncertainty: number;
  subScores: V3SubScores;
  diagnostics: V3DiagnosticCode[];
};

export type ObjectIdentitySignal = {
  hasProfiles: boolean;
  score: number;
  matched: boolean;
  name: string | null;
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
  objectIdentityScore: number;
  objectIdentityName: string | null;
  objectIdentityMatched: boolean;
  hasObjectProfiles: boolean;
  evidence: GripEvidence;
  calibrated: boolean;
  diagnostics: GripDiagnostics;
  v3?: V3AnalysisDetails;
};

export type TrackingFrame = {
  hand: Landmark[] | null;
  object: ObjectRegion | null;
  analysis: GripAnalysis;
};
