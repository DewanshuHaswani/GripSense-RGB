import type {
  GripAnalysis,
  AlgorithmVersion,
  GripCalibrationBaseline,
  GripDiagnostics,
  GripEvidence,
  GripMode,
  GripPoint,
  GripState,
  Landmark,
  ObjectIdentitySignal,
  ObjectRegion,
  Point
} from './types';
import {
  averagePoint,
  clamp,
  distance,
  ellipsePoint,
  fingertipPoints,
  handSize,
  palmCenter,
  subtract,
  vectorMagnitude
} from './geometry';
import { computeGripEvidence, countContactPoints } from './gripEvidence';
import { DEFAULT_GRIP_SCORING_CONFIG, type GripScoringConfig } from './gripScoringConfig';

const EMPTY_ANALYSIS: GripAnalysis = {
  gripPercentage: 0,
  confidence: 0,
  contactPoints: 0,
  closureScore: 0,
  thumbOpposition: 0,
  enclosureScore: 0,
  motionCoupling: 0,
  slipRisk: 0,
  motionState: 'idle',
  guidance: 'Object not locked',
  message: 'Start camera tracking, then place an object between the thumb and fingers.',
  palmCenter: null,
  handVelocity: { x: 0, y: 0 },
  recommendedGripPoints: [],
  objectLockQuality: 0,
  objectIdentityScore: 0,
  objectIdentityName: null,
  objectIdentityMatched: false,
  hasObjectProfiles: false,
  calibrated: false,
  evidence: {
    fingerCurlScore: 0,
    fingerSegmentContactScore: 0,
    contactRoles: {
      thumb: 0,
      index: 0,
      middle: 0,
      ring: 0,
      pinky: 0,
      palm: 0
    },
    palmObjectContainmentScore: 0,
    thumbSupportScore: 0,
    phoneSideGripScore: 0,
    persistentSlipScore: 0,
    objectLockQuality: 0,
    pinchScore: 0,
    powerGripScore: 0,
    hookGripScore: 0,
    visibleContactScore: 0,
    occlusionResilienceScore: 0,
    motionStabilityScore: 1,
    independentObjectScore: 0,
    temporalLockScore: 0,
    modeScores: {
      'phone-side grip': 0,
      'pinch grip': 0,
      'power grip': 0,
      'hook grip': 0,
      'open hand': 1,
      uncertain: 0
    },
    positiveReasons: [],
    negativeReasons: []
  },
  diagnostics: {
    mode: 'uncertain',
    state: 'No hand',
    recommendation: 'Start camera tracking, then place an object between the thumb and fingers.',
    objectIssue: null,
    gripIssue: null,
    issueCategory: 'none',
    scoreBreakdown: []
  }
};

export function createEmptyAnalysis(message = EMPTY_ANALYSIS.message): GripAnalysis {
  return { ...EMPTY_ANALYSIS, message };
}

export function analyzeGrip(
  hand: Landmark[] | null,
  object: ObjectRegion | null,
  previousPalm: Point | null,
  options: {
    persistentSlipScore?: number;
    calibrationBaseline?: GripCalibrationBaseline | null;
    weakCalibrationBaseline?: GripCalibrationBaseline | null;
    algorithmVersion?: AlgorithmVersion;
    objectIdentity?: ObjectIdentitySignal;
    scoringConfig?: GripScoringConfig;
  } = {}
): GripAnalysis {
  const algorithmVersion = options.algorithmVersion ?? 'v1';
  const fallbackAlgorithmVersion: AlgorithmVersion = algorithmVersion === 'v3' ? 'v2' : algorithmVersion;
  const scoringConfig = options.scoringConfig ?? DEFAULT_GRIP_SCORING_CONFIG;
  const objectIdentity = options.objectIdentity ?? { hasProfiles: false, score: 0, matched: false, name: null };
  if (!hand || hand.length < 21) {
    return createEmptyAnalysis('No hand detected. Keep your hand inside the camera frame.');
  }

  const currentPalm = palmCenter(hand);
  const handVelocity = previousPalm ? subtract(currentPalm, previousPalm) : { x: 0, y: 0 };
  const tips = fingertipPoints(hand);
  const size = handSize(hand);
  const avgTipDistance = tips.reduce((sum, tip) => sum + distance(tip, currentPalm), 0) / tips.length;
  const closureScore = clamp(1 - (avgTipDistance - size * 0.34) / (size * 0.34));

  if (!object || object.confidence < 0.35 || !object.locked) {
    return {
      ...createEmptyAnalysis('Object not locked. Move an object into the hand area or click it to lock tracking.'),
      closureScore,
      palmCenter: currentPalm,
      handVelocity,
      guidance: 'Object not locked',
      objectIdentityScore: objectIdentity.score,
      objectIdentityName: objectIdentity.name,
      objectIdentityMatched: objectIdentity.matched,
      hasObjectProfiles: objectIdentity.hasProfiles,
      diagnostics: {
        ...EMPTY_ANALYSIS.diagnostics,
        state: 'Hand only',
        recommendation: 'Move an object into the hand area or click it to lock tracking.'
      }
    };
  }

  if (fallbackAlgorithmVersion === 'v2') {
    const gate = evaluateV2ObjectGate(object, closureScore);
    if (!gate.accepted) {
      return {
        ...createEmptyAnalysis(gate.message),
        closureScore,
        palmCenter: currentPalm,
        handVelocity,
        guidance: 'Object not locked',
        objectLockQuality: gate.objectLockQuality,
        diagnostics: {
          ...EMPTY_ANALYSIS.diagnostics,
          state: gate.state,
          recommendation: gate.message,
          objectIssue: gate.objectIssue
        }
      };
    }
  }

  const automaticOpenHandHallucination =
    object.source === 'automatic' &&
    !object.manuallyAdjusted &&
    closureScore < 0.28 &&
    object.confidence < 0.82 &&
    (object.tightness ?? 0) < 0.92;
  if (automaticOpenHandHallucination) {
    return {
      ...createEmptyAnalysis('Hand detected, but no reliable object is locked. Click the object if you are holding one.'),
      closureScore,
      palmCenter: currentPalm,
      handVelocity,
        guidance: 'Object not locked',
        objectLockQuality: 0,
        objectIdentityScore: objectIdentity.score,
        objectIdentityName: objectIdentity.name,
        objectIdentityMatched: objectIdentity.matched,
        hasObjectProfiles: objectIdentity.hasProfiles,
        diagnostics: {
        ...EMPTY_ANALYSIS.diagnostics,
        state: 'Hand only',
        recommendation: 'No object detected in the hand. Click an object to lock it if the camera missed it.',
        objectIssue: 'Automatic lock rejected because the hand is open and no distinct object was detected.'
      }
    };
  }

  const evidence = computeGripEvidence(hand, object, options.persistentSlipScore ?? 0);
  const mode = classifyGripMode(evidence);
  const contactPoints = countContactPoints(evidence);
  const contactScore = scoreByMode(evidence, mode, fallbackAlgorithmVersion, scoringConfig);
  const thumbOpposition = evidence.thumbSupportScore;

  const angles = tips
    .filter((tip) => distance(tip, object.center) < Math.max(object.radiusX, object.radiusY) + size * 0.32)
    .map((tip) => Math.atan2(tip.y - object.center.y, tip.x - object.center.x))
    .sort((a, b) => a - b);
  const angularCoverage = circularCoverage(angles);
  const enclosureScore = clamp(
    angularCoverage * 0.26 +
      closureScore * 0.24 +
      evidence.palmObjectContainmentScore * 0.24 +
      evidence.phoneSideGripScore * 0.18 +
      evidence.fingerCurlScore * 0.08
  );

  const handMotion = vectorMagnitude(handVelocity);
  const objectMotion = vectorMagnitude(object.velocity);
  const moving = Math.max(handMotion, objectMotion) > 2.8;
  const driftSlip = object.relativeDriftScore ?? 0;
  const slipRisk = moving ? clamp(evidence.persistentSlipScore * 0.72 + driftSlip * 0.28) : clamp((1 - evidence.objectLockQuality) * 0.08 + driftSlip * 0.18);
  const motionCoupling = moving ? clamp(1 - slipRisk) : 0.88;
  const calibration = calibrationAdjustment(evidence, mode, closureScore, enclosureScore, options.calibrationBaseline ?? null);
  const weakCalibration = weakCalibrationAdjustment(evidence, mode, closureScore, enclosureScore, options.weakCalibrationBaseline ?? null);
  const identityFactor = objectIdentityReadiness(objectIdentity, fallbackAlgorithmVersion);
  const objectReadiness = fallbackAlgorithmVersion === 'v2' ? v2ObjectReadiness(object, evidence) * identityFactor : 1;

  const gripPercentage = Math.round(
    100 *
      clamp(
        (contactScore +
          enclosureScore * 0.1 +
          evidence.objectLockQuality * 0.08 +
          calibration.boost -
          weakCalibration.penalty -
          slipRisk * 0.12) *
          objectReadiness
      )
  );
  const calibratedGripPercentage =
    calibration.similarToBaseline && options.calibrationBaseline
      ? Math.max(gripPercentage, Math.round(options.calibrationBaseline.gripPercentage * 0.88))
      : gripPercentage;
  const confidence = computeConfidence({
    evidence,
    gripPercentage,
    motionCoupling,
    closureScore,
    calibrationMatched: calibration.similarToBaseline,
    weakMatched: weakCalibration.similarToWeak,
    algorithmVersion: fallbackAlgorithmVersion,
    objectIdentity,
    scoringConfig
  });
  const motionState = !moving ? 'idle' : slipRisk > 0.45 ? 'slipping' : motionCoupling > 0.58 ? 'moving-with-hand' : 'uncertain';
  const state = computeGripState(hand, object, evidence, calibratedGripPercentage, motionState, objectIdentity, fallbackAlgorithmVersion);
  const objectUncertainGuidance =
    evidence.objectLockQuality < 0.38 || identityBlocksStrongGrip(objectIdentity, fallbackAlgorithmVersion);
  const guidance =
    objectUncertainGuidance
      ? 'Object uncertain'
      : calibratedGripPercentage >= 70 && slipRisk < 0.38
      ? 'Strong grip'
      : calibratedGripPercentage >= 44 || evidence.fingerCurlScore > 0.62 || Math.max(evidence.phoneSideGripScore, evidence.powerGripScore, evidence.pinchScore) > 0.58
        ? 'Improve grip'
        : 'Reposition';
  const diagnostics = createDiagnostics(
    mode,
    state,
    evidence,
    calibratedGripPercentage,
    gripPercentage,
    calibration.similarToBaseline,
    weakCalibration.similarToWeak,
    fallbackAlgorithmVersion,
    objectIdentity
  );

  return {
    gripPercentage: calibratedGripPercentage,
    confidence,
    contactPoints,
    closureScore,
    thumbOpposition,
    enclosureScore,
    motionCoupling,
    slipRisk,
    motionState,
    guidance,
    message: diagnostics.recommendation,
    palmCenter: currentPalm,
    handVelocity,
    recommendedGripPoints: createRecommendedGripPoints(hand, object),
    objectLockQuality: evidence.objectLockQuality,
    objectIdentityScore: objectIdentity.score,
    objectIdentityName: objectIdentity.name,
    objectIdentityMatched: objectIdentity.matched,
    hasObjectProfiles: objectIdentity.hasProfiles,
    evidence,
    calibrated: calibration.similarToBaseline,
    diagnostics
  };
}

function circularCoverage(angles: number[]) {
  if (angles.length < 2) return 0;
  const gaps = angles.map((angle, index) => {
    const next = angles[(index + 1) % angles.length] + (index === angles.length - 1 ? Math.PI * 2 : 0);
    return next - angle;
  });
  const largestGap = Math.max(...gaps);
  return clamp((Math.PI * 2 - largestGap) / (Math.PI * 1.65));
}

function createRecommendedGripPoints(hand: Landmark[], object: ObjectRegion): GripPoint[] {
  const thumb = hand[4];
  const fingerCenter = averagePoint([hand[8], hand[12], hand[16]]);
  const supportAngle = Math.atan2(hand[0].y - object.center.y, hand[0].x - object.center.x);
  const thumbAngle = Math.atan2(thumb.y - object.center.y, thumb.x - object.center.x);
  const fingerAngle = Math.atan2(fingerCenter.y - object.center.y, fingerCenter.x - object.center.x);
  const candidates: GripPoint[] = [
    { ...ellipsePoint(object, thumbAngle), score: 0.92, label: 'thumb' },
    { ...ellipsePoint(object, fingerAngle), score: 0.9, label: 'finger' },
    { ...ellipsePoint(object, thumbAngle + Math.PI), score: 0.76, label: 'opposition' },
    { ...ellipsePoint(object, supportAngle), score: 0.58, label: 'support' }
  ];

  return dedupeGripPoints(candidates);
}

function dedupeGripPoints(points: GripPoint[]) {
  return points.filter((point, index) => points.findIndex((other) => distance(point, other) < 16) === index);
}

function calibrationAdjustment(
  evidence: GripEvidence,
  mode: GripMode,
  closureScore: number,
  enclosureScore: number,
  baseline: GripCalibrationBaseline | null
) {
  if (!baseline || baseline.mode !== mode) return { boost: 0, similarToBaseline: false };
  const deltas = [
    Math.abs(closureScore - baseline.closureScore),
    Math.abs(enclosureScore - baseline.enclosureScore),
    Math.abs(evidence.fingerCurlScore - baseline.fingerCurlScore),
    Math.abs(evidence.fingerSegmentContactScore - baseline.fingerSegmentContactScore),
    Math.abs(evidence.phoneSideGripScore - baseline.phoneSideGripScore),
    Math.abs(evidence.pinchScore - baseline.pinchScore),
    Math.abs(evidence.powerGripScore - baseline.powerGripScore),
    Math.abs(evidence.thumbSupportScore - baseline.thumbSupportScore)
  ];
  const similarity = clamp(1 - deltas.reduce((sum, delta) => sum + delta, 0) / deltas.length);
  return {
    boost: similarity > 0.58 ? (similarity - 0.58) * 0.2 : 0,
    similarToBaseline: similarity > 0.74 && evidence.objectLockQuality > 0.38
  };
}

function weakCalibrationAdjustment(
  evidence: GripEvidence,
  mode: GripMode,
  closureScore: number,
  enclosureScore: number,
  baseline: GripCalibrationBaseline | null
) {
  if (!baseline || baseline.mode !== mode) return { penalty: 0, similarToWeak: false };
  const deltas = [
    Math.abs(closureScore - baseline.closureScore),
    Math.abs(enclosureScore - baseline.enclosureScore),
    Math.abs(evidence.fingerCurlScore - baseline.fingerCurlScore),
    Math.abs(evidence.fingerSegmentContactScore - baseline.fingerSegmentContactScore),
    Math.abs(evidence.phoneSideGripScore - baseline.phoneSideGripScore),
    Math.abs(evidence.pinchScore - baseline.pinchScore),
    Math.abs(evidence.powerGripScore - baseline.powerGripScore),
    Math.abs(evidence.thumbSupportScore - baseline.thumbSupportScore)
  ];
  const similarity = clamp(1 - deltas.reduce((sum, delta) => sum + delta, 0) / deltas.length);
  return {
    penalty: similarity > 0.68 ? (similarity - 0.68) * 0.16 : 0,
    similarToWeak: similarity > 0.8 && evidence.objectLockQuality > 0.38
  };
}

function classifyGripMode(evidence: GripEvidence): GripMode {
  const candidates = Object.entries(evidence.modeScores)
    .filter(([mode]) => mode !== 'uncertain')
    .sort((a, b) => b[1] - a[1]) as Array<[GripMode, number]>;
  const [mode, score] = candidates[0] ?? ['uncertain', 0];
  if (score < 0.24) return evidence.modeScores['open hand'] > 0.52 ? 'open hand' : 'uncertain';
  return mode;
}

function weightedScore(evidence: GripEvidence, weights: Record<string, number>) {
  return Object.entries(weights).reduce((sum, [key, weight]) => {
    const value = evidence[key as keyof GripEvidence];
    return sum + (typeof value === 'number' ? value * weight : 0);
  }, 0);
}

function scoreByMode(evidence: GripEvidence, mode: GripMode, algorithmVersion: AlgorithmVersion, scoringConfig: GripScoringConfig) {
  const v2ObjectFactor =
    algorithmVersion === 'v2'
      ? clamp(evidence.independentObjectScore * 0.54 + evidence.objectLockQuality * 0.28 + evidence.temporalLockScore * 0.18, 0.58, 1)
      : 1;
  if (mode === 'phone-side grip' || mode === 'pinch grip' || mode === 'power grip' || mode === 'hook grip') {
    return weightedScore(evidence, scoringConfig.modeWeights[mode]) * v2ObjectFactor;
  }
  if (mode === 'open hand') return Math.max(0, evidence.visibleContactScore * 0.2 - 0.1);
  return (
    evidence.fingerSegmentContactScore * 0.2 +
    evidence.fingerCurlScore * 0.16 +
    evidence.palmObjectContainmentScore * 0.16 +
    evidence.thumbSupportScore * 0.14 +
    evidence.motionStabilityScore * 0.12
  ) * v2ObjectFactor;
}

function computeConfidence({
  evidence,
  gripPercentage,
  motionCoupling,
  closureScore,
  calibrationMatched,
  weakMatched,
  algorithmVersion,
  objectIdentity,
  scoringConfig
}: {
  evidence: GripEvidence;
  gripPercentage: number;
  motionCoupling: number;
  closureScore: number;
  calibrationMatched: boolean;
  weakMatched: boolean;
  algorithmVersion: AlgorithmVersion;
  objectIdentity: ObjectIdentitySignal;
  scoringConfig: GripScoringConfig;
}) {
  const weights = scoringConfig.confidenceWeights[algorithmVersion];
  const bestModeScore = Math.max(evidence.phoneSideGripScore, evidence.powerGripScore, evidence.pinchScore);
  return clamp(
    evidence.objectLockQuality * weights.objectLockQuality +
      (weights.independentObjectScore ?? 0) * evidence.independentObjectScore +
      (weights.temporalLockScore ?? 0) * evidence.temporalLockScore +
      gripPercentage / 100 * weights.gripPercentage +
      motionCoupling * weights.motionCoupling +
      closureScore * weights.closureScore +
      bestModeScore * weights.bestModeScore +
      (objectIdentity.hasProfiles ? (objectIdentity.matched ? objectIdentity.score * (weights.identityMatch ?? 0) : weights.identityMiss ?? 0) : 0) +
      (calibrationMatched ? weights.calibration : 0) +
      (weakMatched ? weights.weakCalibration : 0)
  );
}

function evaluateV2ObjectGate(object: ObjectRegion | null, closureScore: number): {
  accepted: boolean;
  state: GripState;
  message: string;
  objectIssue: string | null;
  objectLockQuality: number;
} {
  if (!object || !object.locked) {
    return {
      accepted: false,
      state: 'Hand only',
      message: 'V2 sees the hand, but no independent object is locked. Click the object if the camera missed it.',
      objectIssue: 'No object passed the V2 object-first gate.',
      objectLockQuality: 0
    };
  }

  const manualTrusted = object.manuallyAdjusted || object.source === 'manual' || object.source === 'segmenter';
  const detectorTrusted = object.source === 'detector' && object.confidence > 0.48 && (object.tightness ?? 0.55) > 0.28;
  const independent = object.independentEvidenceScore ?? inferredIndependentEvidence(object);
  const temporal = clamp((object.lockAgeFrames ?? 0) / 28);
  const tightness = object.tightness ?? 0.45;
  const objectLockQuality = clamp(object.confidence * 0.34 + independent * 0.34 + tightness * 0.18 + temporal * 0.14);

  if (manualTrusted || detectorTrusted) {
    return {
      accepted: object.confidence >= 0.3,
      state: object.confidence >= 0.3 ? 'Grip detected' : 'Object uncertain',
      message:
        object.confidence >= 0.3
          ? 'Object lock accepted by V2 because it has manual, segmenter, or detector evidence.'
          : 'V2 rejected this lock because the object confidence collapsed.',
      objectIssue: object.confidence >= 0.3 ? null : 'Object confidence is too low.',
      objectLockQuality
    };
  }

  if (closureScore < 0.34) {
    return {
      accepted: false,
      state: 'Hand only',
      message: 'V2 rejected the object lock because the hand is open and the object has no detector or manual evidence.',
      objectIssue: 'Open hand with automatic object lock looks like background, not a held object.',
      objectLockQuality: Math.min(objectLockQuality, 0.24)
    };
  }

  if (independent < 0.42) {
    return {
      accepted: false,
      state: 'Object uncertain',
      message: 'Object lock is uncertain. Click the object or tighten the region before trusting the grip score.',
      objectIssue: 'V2 needs stronger independent object evidence before scoring grip.',
      objectLockQuality: Math.min(objectLockQuality, 0.34)
    };
  }

  if (tightness < 0.34 || object.confidence < 0.46) {
    return {
      accepted: false,
      state: 'Object uncertain',
      message: 'Object lock is too loose for V2. Click the object or use grow/shrink to correct it.',
      objectIssue: 'Object region is too loose or low-confidence.',
      objectLockQuality: Math.min(objectLockQuality, 0.38)
    };
  }

  return {
    accepted: true,
    state: 'Grip detected',
    message: 'V2 accepted the object lock.',
    objectIssue: null,
    objectLockQuality
  };
}

function inferredIndependentEvidence(object: ObjectRegion) {
  const sourceEvidence =
    object.source === 'manual' ? 0.96 : object.source === 'segmenter' ? 0.88 : object.source === 'detector' ? 0.82 : 0;
  const visualEvidence = clamp(
    (object.visualEdgeScore ?? 0) * 0.56 +
      (object.visualTextureScore ?? 0) * 0.26 +
      (object.shape === 'phone-like' ? 0.12 : 0) +
      clamp((object.lockAgeFrames ?? 0) / 28) * 0.06
  );
  return clamp(Math.max(sourceEvidence, object.independentEvidenceScore ?? 0, visualEvidence));
}

function v2ObjectReadiness(object: ObjectRegion, evidence: GripEvidence) {
  const manualBonus = object.manuallyAdjusted || object.source === 'manual' || object.source === 'segmenter' ? 0.08 : 0;
  const detectorBonus = object.source === 'detector' ? 0.05 : 0;
  return clamp(
    0.74 +
      evidence.independentObjectScore * 0.12 +
      evidence.objectLockQuality * 0.08 +
      evidence.temporalLockScore * 0.06 +
      manualBonus +
      detectorBonus,
    0.72,
    1
  );
}

function objectIdentityReadiness(identity: ObjectIdentitySignal, algorithmVersion: AlgorithmVersion) {
  if (algorithmVersion !== 'v2' || !identity.hasProfiles) return 1;
  if (identity.matched) return clamp(0.9 + identity.score * 0.12, 0.9, 1);
  return 0;
}

function identityBlocksStrongGrip(identity: ObjectIdentitySignal, algorithmVersion: AlgorithmVersion) {
  return algorithmVersion === 'v2' && identity.hasProfiles && !identity.matched;
}

function computeGripState(
  hand: Landmark[] | null,
  object: ObjectRegion | null,
  evidence: GripEvidence,
  gripPercentage: number,
  motionState: GripAnalysis['motionState'],
  identity: ObjectIdentitySignal,
  algorithmVersion: AlgorithmVersion
): GripState {
  if (!hand) return 'No hand';
  if (!object?.locked) return 'Hand only';
  if (evidence.objectLockQuality < 0.38) return 'Object uncertain';
  if (identityBlocksStrongGrip(identity, algorithmVersion)) return 'Object uncertain';
  if (motionState === 'slipping') return 'Slip risk';
  if (gripPercentage >= 70) return 'Strong hold';
  if (gripPercentage >= 38) return 'Grip detected';
  return 'Object uncertain';
}

function createDiagnostics(
  mode: GripMode,
  state: GripState,
  evidence: GripEvidence,
  calibratedGripPercentage: number,
  rawGripPercentage: number,
  calibrated: boolean,
  weakMatched: boolean,
  algorithmVersion: AlgorithmVersion,
  identity: ObjectIdentitySignal
): GripDiagnostics {
  const objectIssue =
    identityBlocksStrongGrip(identity, algorithmVersion)
      ? 'Trained object not found. Relock the intended object or train another profile.'
      : evidence.objectLockQuality < 0.38
      ? 'Object lock is uncertain. Click or resize the object region.'
      : evidence.objectLockQuality < 0.62
        ? 'Object lock is usable but could be tighter.'
        : null;
  const gripIssue =
    state === 'Slip risk'
      ? 'Object and hand have diverged over multiple frames.'
      : evidence.thumbSupportScore < 0.3
        ? 'Thumb support is unclear.'
        : evidence.fingerCurlScore < 0.34 && mode !== 'pinch grip'
          ? 'Finger wrap is weak.'
          : null;
  const recommendation = recommend(mode, state, objectIssue, gripIssue, evidence, calibrated);
  const issueCategory =
    identityBlocksStrongGrip(identity, algorithmVersion)
      ? 'identity_problem'
      : state === 'Slip risk'
        ? 'motion_problem'
        : objectIssue
          ? 'object_problem'
          : gripIssue
            ? 'pose_problem'
            : 'none';

  return {
    mode,
    state,
    recommendation,
    objectIssue,
    gripIssue,
    issueCategory,
    scoreBreakdown: [
      { label: 'Mode fit', value: mode === 'uncertain' ? 0 : evidence.modeScores[mode], impact: 'positive' },
      { label: 'Object lock', value: evidence.objectLockQuality, impact: objectIssue ? 'negative' : 'positive' },
      { label: 'Contact', value: evidence.fingerSegmentContactScore, impact: 'positive' },
      { label: 'Contact roles', value: roleCoverage(evidence.contactRoles), impact: 'positive' },
      { label: 'Finger wrap', value: evidence.fingerCurlScore, impact: 'positive' },
      { label: 'Thumb support', value: evidence.thumbSupportScore, impact: evidence.thumbSupportScore < 0.3 ? 'negative' : 'positive' },
      { label: 'Motion stability', value: evidence.motionStabilityScore, impact: state === 'Slip risk' ? 'negative' : 'positive' },
      ...(algorithmVersion === 'v2'
        ? [
            {
              label: 'Independent object',
              value: evidence.independentObjectScore,
              impact: evidence.independentObjectScore < 0.42 ? 'negative' : 'positive'
            } as const,
            {
              label: 'Temporal lock',
              value: evidence.temporalLockScore,
              impact: evidence.temporalLockScore < 0.22 ? 'neutral' : 'positive'
            } as const,
            {
              label: 'Object identity match',
              value: identity.hasProfiles ? identity.score : 0,
              impact: identity.hasProfiles ? (identity.matched ? 'positive' : 'negative') : 'neutral'
            } as const,
            {
              label: 'Grip stability',
              value: calibratedGripPercentage / 100,
              impact: calibratedGripPercentage >= 44 ? 'positive' : 'negative'
            } as const
          ]
        : []),
      {
        label: weakMatched ? 'Weak calibration' : 'Strong calibration',
        value: calibrated ? calibratedGripPercentage / 100 - rawGripPercentage / 100 : weakMatched ? 0.2 : 0,
        impact: calibrated ? 'positive' : weakMatched ? 'negative' : 'neutral'
      }
    ]
  };
}

function roleCoverage(roles: GripEvidence['contactRoles']) {
  return clamp(roles.thumb * 0.24 + roles.index * 0.2 + roles.middle * 0.18 + roles.ring * 0.14 + roles.pinky * 0.1 + roles.palm * 0.14);
}

function recommend(
  mode: GripMode,
  state: GripState,
  objectIssue: string | null,
  gripIssue: string | null,
  evidence: GripEvidence,
  calibrated: boolean
) {
  if (state === 'No hand') return 'Keep your hand inside the camera frame.';
  if (state === 'Hand only') return 'Move an object into the hand area or click it to lock tracking.';
  if (objectIssue) return objectIssue;
  if (state === 'Slip risk') return 'Hold still for a moment or relock the object if the boundary is drifting.';
  if (state === 'Strong hold') return calibrated ? 'Strong calibrated hold detected.' : `Strong ${mode} detected.`;
  if (mode === 'phone-side grip' && evidence.thumbSupportScore < 0.46) return 'Phone-side grip detected. Add thumb pressure on the opposite edge.';
  if (mode === 'pinch grip') return 'Pinch grip detected. Keep thumb and index opposed while moving.';
  if (mode === 'power grip') return 'Power grip detected. Wrap fingers deeper around the object for a stronger hold.';
  if (mode === 'hook grip') return 'Hook grip detected. Add thumb or palm support if the object feels unstable.';
  if (gripIssue) return gripIssue;
  return 'Grip detected. Calibrate strong and weak holds to personalize the percentage.';
}
