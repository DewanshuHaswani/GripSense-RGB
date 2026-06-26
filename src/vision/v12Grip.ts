import type { GripAnalysis, GripState } from './types';

export type V12ObjectEvidence = {
  objectScore: number;
  contact: number;
  hasObject: boolean;
  missedFrames: number;
};

export type V12DisplayState = {
  analysis: GripAnalysis | null;
  timestamp: number;
  softLossStartedAt: number | null;
};

type V12Validation = {
  objectInHandScore: number;
  emptyHandScore: number;
  partialGripScore: number;
  contactScore: number;
  hasObjectEvidence: boolean;
  hasUsableContact: boolean;
  hasPartialGrip: boolean;
};

const V12_EMPTY_HAND_CAP = 18;
const V12_WEAK_OBJECT_CAP = 24;
const V12_LOW_CONTACT_CAP = 34;
const V12_BRIDGE_MS = 420;

export function applyV12ProductionGripGate(
  analysis: GripAnalysis,
  objectEvidence: V12ObjectEvidence
): GripAnalysis {
  const validation = validateV12ObjectInHand(analysis, objectEvidence);

  if (analysis.diagnostics.issueCategory === 'server_unavailable') {
    return forceV12NoGrip(
      analysis,
      'Hand only',
      analysis.message || 'YOLO unavailable. V12 cannot score grip without detector object evidence.',
      validation
    );
  }

  if (!validation.hasObjectEvidence) {
    return forceV12NoGrip(
      analysis,
      'Object uncertain',
      'No detector-backed object is visible in the hand. Grip scoring is held at no grip until YOLO sees a separate object.',
      validation
    );
  }

  if (validation.emptyHandScore >= 0.62 && validation.objectInHandScore < 0.38) {
    return forceV12NoGrip(
      analysis,
      'Hand only',
      'The hand pose looks empty or too open, so V12 rejected the grip score even though the hand skeleton is stable.',
      validation
    );
  }

  if (!validation.hasUsableContact) {
    return capV12Grip(
      analysis,
      validation.partialGripScore >= 0.46 ? V12_LOW_CONTACT_CAP : V12_WEAK_OBJECT_CAP,
      'Reposition',
      'Object evidence exists, but contact with the thumb/finger corridor is too weak for a confident grip.',
      validation,
      'Object uncertain'
    );
  }

  if (validation.hasPartialGrip && analysis.gripPercentage < 38) {
    return promoteV12PartialGrip(analysis, validation);
  }

  return normalizeV12ObjectLabels(analysis, validation);
}

export function stabilizeV12DisplayAnalysis(
  rawAnalysis: GripAnalysis,
  previousState: V12DisplayState,
  timestamp: number,
  objectEvidence: V12ObjectEvidence
): { analysis: GripAnalysis; state: V12DisplayState } {
  const gated = applyV12ProductionGripGate(rawAnalysis, objectEvidence);
  const previous = previousState.analysis;
  const validation = validateV12ObjectInHand(gated, objectEvidence);

  if (!previous) {
    return { analysis: gated, state: { analysis: gated, timestamp, softLossStartedAt: null } };
  }

  const dt = Math.max(16, timestamp - previousState.timestamp);
  const collapsed = gated.gripPercentage <= V12_EMPTY_HAND_CAP || gated.guidance === 'Object not locked';
  const canBridgeDetectorBlink =
    collapsed &&
    gated.diagnostics.issueCategory !== 'server_unavailable' &&
    objectEvidence.missedFrames > 0 &&
    objectEvidence.missedFrames <= 2 &&
    validation.objectInHandScore >= 0.34 &&
    validation.emptyHandScore < 0.5 &&
    previous.gripPercentage >= 42 &&
    previous.objectLockQuality >= 0.42 &&
    previous.confidence >= 0.34;

  if (collapsed) {
    if (!canBridgeDetectorBlink) {
      return { analysis: gated, state: { analysis: gated, timestamp, softLossStartedAt: null } };
    }

    const softLossStartedAt = previousState.softLossStartedAt ?? timestamp;
    const lossAge = timestamp - softLossStartedAt;
    if (lossAge > V12_BRIDGE_MS) {
      return { analysis: gated, state: { analysis: gated, timestamp, softLossStartedAt: null } };
    }

    const decay = clampUnit(lossAge / V12_BRIDGE_MS);
    const gripPercentage = Math.round(Math.max(26, previous.gripPercentage * (1 - decay * 0.6)));
    const confidence = clampUnit(Math.max(0.24, previous.confidence * (1 - decay * 0.5)));
    const objectLockQuality = clampUnit(Math.max(0.24, previous.objectLockQuality * (1 - decay * 0.54)));
    const analysis: GripAnalysis = {
      ...gated,
      gripPercentage,
      confidence,
      objectLockQuality,
      guidance: gripPercentage >= 45 ? 'Improve grip' : 'Reposition',
      message: 'YOLO briefly missed the object; V12 is holding a degraded estimate while checking for real separation.',
      evidence: {
        ...gated.evidence,
        objectLockQuality,
        temporalLockScore: Math.max(gated.evidence.temporalLockScore, previous.evidence.temporalLockScore * (1 - decay * 0.55)),
        visibleContactScore: Math.max(gated.evidence.visibleContactScore, previous.evidence.visibleContactScore * (1 - decay * 0.52)),
        fingerSegmentContactScore: Math.max(gated.evidence.fingerSegmentContactScore, previous.evidence.fingerSegmentContactScore * (1 - decay * 0.52))
      },
      diagnostics: {
        ...gated.diagnostics,
        state: gripPercentage >= 45 ? 'Grip detected' : 'Object uncertain',
        recommendation: 'Hold steady. V12 only bridges short detector blinks and will drop to no grip if the object stays missing.',
        issueCategory: 'object_uncertain',
        objectIssue: 'Short YOLO miss bridged with fast decay; persistent object loss is not smoothed.'
      }
    };
    return { analysis, state: { analysis, timestamp, softLossStartedAt } };
  }

  const stableObject =
    validation.objectInHandScore >= 0.58 &&
    gated.objectLockQuality >= 0.56 &&
    gated.confidence >= 0.42 &&
    gated.slipRisk < 0.48;
  const gripAlpha = gated.gripPercentage >= previous.gripPercentage ? (stableObject ? 0.18 : 0.32) : (stableObject ? 0.34 : 0.64);
  const timeAlpha = clampUnit(gripAlpha + clampUnit(dt / 260) * 0.12);
  const gripPercentage =
    stableObject && Math.abs(gated.gripPercentage - previous.gripPercentage) <= 6
      ? previous.gripPercentage
      : Math.round(previous.gripPercentage + (gated.gripPercentage - previous.gripPercentage) * timeAlpha);
  const confidence = smooth(previous.confidence, gated.confidence, stableObject ? 0.22 : 0.4);
  const objectLockQuality = smooth(previous.objectLockQuality, gated.objectLockQuality, stableObject ? 0.2 : 0.42);
  const guidance = guidanceForV12Grip(gripPercentage, objectLockQuality, gated.slipRisk, gated.guidance);
  const analysis: GripAnalysis = {
    ...gated,
    gripPercentage,
    confidence,
    objectLockQuality,
    guidance,
    message: guidance === gated.guidance ? gated.message : messageForV12Guidance(guidance),
    diagnostics: {
      ...gated.diagnostics,
      state: stateForV12Guidance(guidance, gated.diagnostics.state)
    }
  };

  return { analysis, state: { analysis, timestamp, softLossStartedAt: null } };
}

export function validateV12ObjectInHand(analysis: GripAnalysis, objectEvidence: V12ObjectEvidence): V12Validation {
  const roles = analysis.evidence.contactRoles;
  const nonThumbFinger = Math.max(roles.index, roles.middle, roles.ring, roles.pinky);
  const supportingFingerCount = [roles.index, roles.middle, roles.ring, roles.pinky].filter((score) => score >= 0.18).length;
  const contactScore = clampUnit(
    objectEvidence.contact * 0.36 +
      analysis.evidence.visibleContactScore * 0.2 +
      analysis.evidence.fingerSegmentContactScore * 0.2 +
      Math.min(1, roles.thumb * 0.58 + nonThumbFinger * 0.42) * 0.24
  );
  const partialGripScore = clampUnit(
    Math.max(analysis.evidence.pinchScore, analysis.evidence.phoneSideGripScore, analysis.evidence.hookGripScore * 0.92) * 0.48 +
      Math.min(1, roles.thumb * 0.55 + nonThumbFinger * 0.45 + supportingFingerCount * 0.08) * 0.32 +
      analysis.thumbOpposition * 0.2
  );
  const objectScore = Math.max(objectEvidence.objectScore, analysis.evidence.independentObjectScore, analysis.objectLockQuality * 0.72);
  const hasObjectEvidence = objectEvidence.hasObject && objectScore >= 0.16;
  const openHandScore = analysis.evidence.modeScores['open hand'] ?? 0;
  const handOnly = analysis.diagnostics.state === 'Hand only' || analysis.diagnostics.state === 'No hand';
  const emptyHandScore = clampUnit(
    openHandScore * 0.38 +
      (analysis.closureScore < 0.28 ? 0.24 : 0) +
      (contactScore < 0.14 ? 0.22 : 0) +
      (!hasObjectEvidence ? 0.28 : 0) +
      (handOnly ? 0.18 : 0)
  );
  const objectInHandScore = clampUnit(
    objectScore * 0.36 +
      contactScore * 0.34 +
      partialGripScore * 0.2 +
      analysis.evidence.temporalLockScore * 0.1 -
      emptyHandScore * 0.22
  );
  const hasPartialGrip = roles.thumb >= 0.16 && nonThumbFinger >= 0.16 && partialGripScore >= 0.38;
  const hasUsableContact = contactScore >= 0.18 || (hasPartialGrip && objectInHandScore >= 0.34);

  return {
    objectInHandScore,
    emptyHandScore,
    partialGripScore,
    contactScore,
    hasObjectEvidence,
    hasUsableContact,
    hasPartialGrip
  };
}

function forceV12NoGrip(
  analysis: GripAnalysis,
  state: GripState,
  reason: string,
  validation: V12Validation
): GripAnalysis {
  return {
    ...analysis,
    gripPercentage: 0,
    confidence: 0,
    contactPoints: 0,
    objectLockQuality: 0,
    objectIdentityScore: 0,
    objectIdentityName: null,
    objectIdentityMatched: false,
    guidance: 'Object not locked',
    message: reason,
    evidence: {
      ...analysis.evidence,
      objectLockQuality: 0,
      independentObjectScore: 0,
      temporalLockScore: 0,
      visibleContactScore: Math.min(analysis.evidence.visibleContactScore, validation.contactScore),
      fingerSegmentContactScore: Math.min(analysis.evidence.fingerSegmentContactScore, validation.contactScore),
      negativeReasons: [...analysis.evidence.negativeReasons, reason]
    },
    diagnostics: {
      ...analysis.diagnostics,
      state,
      recommendation: reason,
      objectIssue: reason,
      gripIssue: null,
      issueCategory: analysis.diagnostics.issueCategory === 'server_unavailable' ? 'server_unavailable' : 'object_problem'
    }
  };
}

function capV12Grip(
  analysis: GripAnalysis,
  cap: number,
  guidance: GripAnalysis['guidance'],
  reason: string,
  validation: V12Validation,
  state: GripState
): GripAnalysis {
  const gripPercentage = Math.min(analysis.gripPercentage, cap);
  return {
    ...normalizeV12ObjectLabels(analysis, validation),
    gripPercentage,
    confidence: Math.min(analysis.confidence, cap / 100 + 0.08),
    objectLockQuality: Math.min(analysis.objectLockQuality, validation.objectInHandScore),
    guidance,
    message: reason,
    diagnostics: {
      ...analysis.diagnostics,
      state,
      recommendation: reason,
      objectIssue: reason,
      issueCategory: 'object_problem'
    }
  };
}

function promoteV12PartialGrip(analysis: GripAnalysis, validation: V12Validation): GripAnalysis {
  const promotedGrip = Math.round(42 + validation.objectInHandScore * 18 + validation.partialGripScore * 12);
  const gripPercentage = Math.max(analysis.gripPercentage, Math.min(68, promotedGrip));
  return {
    ...normalizeV12ObjectLabels(analysis, validation),
    gripPercentage,
    confidence: Math.max(analysis.confidence, Math.min(0.68, validation.objectInHandScore * 0.72 + validation.partialGripScore * 0.28)),
    objectLockQuality: Math.max(analysis.objectLockQuality, Math.min(0.76, validation.objectInHandScore)),
    guidance: gripPercentage >= 58 ? 'Improve grip' : 'Reposition',
    message: 'Partial grip detected: thumb and finger contact are visible, but wrap/enclosure are not strong enough for a full hold.',
    diagnostics: {
      ...analysis.diagnostics,
      state: 'Grip detected',
      mode: analysis.diagnostics.mode === 'open hand' ? 'pinch grip' : analysis.diagnostics.mode,
      recommendation: 'Partial grip detected. Add more finger wrap for a stronger hold.',
      gripIssue: 'Partial grip has contact but limited enclosure.',
      issueCategory: 'none'
    }
  };
}

function normalizeV12ObjectLabels(analysis: GripAnalysis, validation: V12Validation): GripAnalysis {
  return {
    ...analysis,
    objectIdentityName: null,
    evidence: {
      ...analysis.evidence,
      positiveReasons: [
        ...analysis.evidence.positiveReasons,
        `V12 object-in-hand score ${Math.round(validation.objectInHandScore * 100)}`
      ]
    }
  };
}

function guidanceForV12Grip(
  gripPercentage: number,
  objectLockQuality: number,
  slipRisk: number,
  fallback: GripAnalysis['guidance']
): GripAnalysis['guidance'] {
  if (gripPercentage >= 76 && objectLockQuality >= 0.58 && slipRisk < 0.42) return 'Strong grip';
  if (gripPercentage >= 38) return 'Improve grip';
  if (gripPercentage >= 24) return 'Reposition';
  return fallback;
}

function stateForV12Guidance(guidance: GripAnalysis['guidance'], fallback: GripState): GripState {
  if (guidance === 'Strong grip') return 'Strong hold';
  if (guidance === 'Improve grip') return 'Grip detected';
  if (guidance === 'Object not locked') return 'Hand only';
  return fallback;
}

function messageForV12Guidance(guidance: GripAnalysis['guidance']) {
  if (guidance === 'Strong grip') return 'V12 sees stable object contact in the hand.';
  if (guidance === 'Improve grip') return 'V12 sees object contact, but the hold can be more stable.';
  if (guidance === 'Reposition') return 'Move the object deeper between the thumb and fingers.';
  return 'Object not locked. V12 requires detector-backed object contact before scoring grip.';
}

function smooth(previous: number, next: number, alpha: number) {
  return clampUnit(previous + (next - previous) * alpha);
}

function clampUnit(value: number) {
  return Math.min(1, Math.max(0, value));
}
