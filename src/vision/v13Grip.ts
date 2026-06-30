import { distance } from './geometry';
import { stabilizeV12DisplayAnalysis, validateV12ObjectInHand, type V12DisplayState, type V12ObjectEvidence } from './v12Grip';
import type { GripAnalysis, ObjectRegion } from './types';

export type V13ObjectEvidence = V12ObjectEvidence;

export type V13DisplayState = V12DisplayState & {
  visualObject: ObjectRegion | null;
  visualTimestamp: number;
  visualMissedFrames: number;
};

const V13_OCCLUSION_BRIDGE_MS = 1040;
const V13_VISUAL_HOLD_MS = 820;
const V13_VISUAL_JUMP_WINDOW_MS = 700;

export function createInitialV13DisplayState(): V13DisplayState {
  return {
    analysis: null,
    timestamp: 0,
    softLossStartedAt: null,
    lastStableAt: null,
    visualObject: null,
    visualTimestamp: 0,
    visualMissedFrames: 0
  };
}

export function stabilizeV13DisplayAnalysis(
  rawAnalysis: GripAnalysis,
  previousState: V13DisplayState,
  timestamp: number,
  objectEvidence: V13ObjectEvidence
): { analysis: GripAnalysis; state: V13DisplayState } {
  const v12 = stabilizeV12DisplayAnalysis(rawAnalysis, previousState, timestamp, objectEvidence);
  const previous = previousState.analysis;
  const validation = validateV12ObjectInHand(rawAnalysis, objectEvidence);

  if (!previous) {
    return { analysis: v12.analysis, state: preserveVisualState(v12.state, previousState) };
  }

  const collapsed = v12.analysis.guidance === 'Object not locked' || v12.analysis.gripPercentage <= 18;
  const previousStable =
    previous.gripPercentage >= 42 &&
    previous.objectLockQuality >= 0.38 &&
    previous.confidence >= 0.32 &&
    previous.diagnostics.issueCategory !== 'server_unavailable';
  const stillWrapped = handStillWrapped(rawAnalysis, validation);
  const clearlyGone = handClearlyEmptyOrGone(rawAnalysis, validation);
  const detectorBlink = objectEvidence.missedFrames > 0 || !objectEvidence.hasObject;
  const stableAge = timestamp - (previousState.lastStableAt ?? previousState.timestamp);

  if (collapsed && previousStable && detectorBlink && stillWrapped && !clearlyGone && stableAge <= V13_OCCLUSION_BRIDGE_MS) {
    const softLossStartedAt = previousState.softLossStartedAt ?? timestamp;
    const lossAge = timestamp - softLossStartedAt;
    if (lossAge <= V13_OCCLUSION_BRIDGE_MS) {
      const inferredMissAge = Math.max(80, objectEvidence.missedFrames * 80);
      const decay = clampUnit(Math.max(lossAge, inferredMissAge) / V13_OCCLUSION_BRIDGE_MS);
      const gripPercentage = Math.round(
        Math.max(24, previous.gripPercentage * (1 - decay * 0.6) + rawAnalysis.gripPercentage * 0.12)
      );
      const confidence = clampUnit(Math.max(0.22, previous.confidence * (1 - decay * 0.62)));
      const objectLockQuality = clampUnit(Math.max(0.2, previous.objectLockQuality * (1 - decay * 0.58)));
      const analysis: GripAnalysis = {
        ...v12.analysis,
        gripPercentage,
        confidence,
        objectLockQuality,
        guidance: gripPercentage >= 48 ? 'Improve grip' : 'Reposition',
        message:
          'V13 is treating this as short object occlusion by the fingers. It holds a degraded grip estimate briefly, then drops if YOLO still cannot see a separate object.',
        evidence: {
          ...v12.analysis.evidence,
          objectLockQuality,
          temporalLockScore: Math.max(v12.analysis.evidence.temporalLockScore, previous.evidence.temporalLockScore * (1 - decay * 0.5)),
          visibleContactScore: Math.max(v12.analysis.evidence.visibleContactScore, previous.evidence.visibleContactScore * (1 - decay * 0.52)),
          fingerSegmentContactScore: Math.max(
            v12.analysis.evidence.fingerSegmentContactScore,
            previous.evidence.fingerSegmentContactScore * (1 - decay * 0.52)
          ),
          negativeReasons: [...v12.analysis.evidence.negativeReasons, 'V13 occlusion bridge active']
        },
        diagnostics: {
          ...v12.analysis.diagnostics,
          state: gripPercentage >= 48 ? 'Grip detected' : 'Object uncertain',
          recommendation:
            'Keep the object partly visible. V13 allows short finger occlusion, but persistent missing object evidence becomes no grip.',
          issueCategory: 'object_uncertain',
          objectIssue: 'Short YOLO object occlusion bridged with decay.'
        }
      };
      return {
        analysis,
        state: preserveVisualState(
          {
            analysis,
            timestamp,
            softLossStartedAt,
            lastStableAt: previousState.lastStableAt ?? previousState.timestamp
          },
          previousState
        )
      };
    }
  }

  if (!collapsed && previousStable && v12.analysis.gripPercentage > 0) {
    const stableObject =
      v12.analysis.objectLockQuality >= 0.52 &&
      validation.objectInHandScore >= 0.48 &&
      v12.analysis.diagnostics.issueCategory !== 'server_unavailable';
    const gripAlpha = stableObject ? 0.2 : 0.34;
    const smoothedGrip = Math.round(previous.gripPercentage + (v12.analysis.gripPercentage - previous.gripPercentage) * gripAlpha);
    const smoothedConfidence = smooth(previous.confidence, v12.analysis.confidence, stableObject ? 0.24 : 0.38);
    const smoothedLock = smooth(previous.objectLockQuality, v12.analysis.objectLockQuality, stableObject ? 0.22 : 0.4);
    const analysis: GripAnalysis = {
      ...v12.analysis,
      gripPercentage: smoothedGrip,
      confidence: smoothedConfidence,
      objectLockQuality: smoothedLock,
      evidence: {
        ...v12.analysis.evidence,
        objectLockQuality: smoothedLock
      }
    };
    return {
      analysis,
      state: preserveVisualState(
        {
          ...v12.state,
          analysis
        },
        previousState
      )
    };
  }

  return { analysis: v12.analysis, state: preserveVisualState(v12.state, previousState) };
}

export function stabilizeV13VisualObject(
  currentObject: ObjectRegion | null,
  previousState: V13DisplayState,
  timestamp: number,
  analysis: GripAnalysis
): { object: ObjectRegion | null; state: V13DisplayState } {
  const previousObject = previousState.visualObject;

  if (currentObject && previousObject) {
    const jump = distance(previousObject.center, currentObject.center);
    const previousRadius = Math.max(previousObject.radiusX, previousObject.radiusY, 1);
    const currentRadius = Math.max(currentObject.radiusX, currentObject.radiusY, 1);
    const jumpLimit = Math.max(48, Math.min(150, (previousRadius + currentRadius) * 0.58));
    const recentVisualLock = timestamp - previousState.visualTimestamp <= V13_VISUAL_JUMP_WINDOW_MS;
    const shouldDampenJump = jump > jumpLimit && recentVisualLock && analysis.gripPercentage >= 20;
    const alpha = shouldDampenJump ? 0.24 : 0.46;
    const blended = blendObjectRegion(previousObject, currentObject, alpha, shouldDampenJump ? 0.82 : 1);
    return {
      object: blended,
      state: {
        ...previousState,
        visualObject: blended,
        visualTimestamp: timestamp,
        visualMissedFrames: 0
      }
    };
  }

  if (currentObject) {
    return {
      object: currentObject,
      state: {
        ...previousState,
        visualObject: currentObject,
        visualTimestamp: timestamp,
        visualMissedFrames: 0
      }
    };
  }

  if (previousObject && canHoldVisualObject(previousState, timestamp, analysis)) {
    const age = timestamp - previousState.visualTimestamp;
    const confidenceScale = Math.max(0.18, 1 - age / V13_VISUAL_HOLD_MS);
    const held = {
      ...previousObject,
      confidence: previousObject.confidence * confidenceScale,
      locked: false,
      detectorScore: previousObject.detectorScore == null ? previousObject.detectorScore : previousObject.detectorScore * confidenceScale,
      velocity: { x: 0, y: 0 }
    };
    return {
      object: held,
      state: {
        ...previousState,
        visualObject: held,
        visualMissedFrames: previousState.visualMissedFrames + 1
      }
    };
  }

  return {
    object: null,
    state: {
      ...previousState,
      visualObject: null,
      visualMissedFrames: previousState.visualMissedFrames + 1
    }
  };
}

function preserveVisualState(state: V12DisplayState, previous: V13DisplayState): V13DisplayState {
  return {
    ...state,
    visualObject: previous.visualObject,
    visualTimestamp: previous.visualTimestamp,
    visualMissedFrames: previous.visualMissedFrames
  };
}

function handStillWrapped(rawAnalysis: GripAnalysis, validation: ReturnType<typeof validateV12ObjectInHand>) {
  const roles = rawAnalysis.evidence.contactRoles;
  const nonThumbFinger = Math.max(roles.index, roles.middle, roles.ring, roles.pinky);
  const phoneSide = rawAnalysis.evidence.modeScores['phone-side grip'] ?? 0;
  const power = rawAnalysis.evidence.modeScores['power grip'] ?? 0;
  const pinch = rawAnalysis.evidence.modeScores['pinch grip'] ?? 0;
  return (
    validation.partialGripScore >= 0.34 ||
    validation.contactScore >= 0.14 ||
    (roles.thumb >= 0.18 && nonThumbFinger >= 0.16) ||
    phoneSide >= 0.36 ||
    power >= 0.38 ||
    pinch >= 0.38 ||
    rawAnalysis.closureScore >= 0.4
  );
}

function handClearlyEmptyOrGone(rawAnalysis: GripAnalysis, validation: ReturnType<typeof validateV12ObjectInHand>) {
  const openHand = rawAnalysis.evidence.modeScores['open hand'] ?? 0;
  return (
    rawAnalysis.diagnostics.issueCategory === 'server_unavailable' ||
    rawAnalysis.diagnostics.state === 'No hand' ||
    rawAnalysis.motionState === 'slipping' ||
    (openHand >= 0.78 && rawAnalysis.closureScore < 0.28 && validation.contactScore < 0.16) ||
    (validation.emptyHandScore >= 0.8 && validation.partialGripScore < 0.28)
  );
}

function canHoldVisualObject(previousState: V13DisplayState, timestamp: number, analysis: GripAnalysis) {
  if (analysis.diagnostics.issueCategory === 'server_unavailable') return false;
  if (analysis.guidance === 'Object not locked' && analysis.gripPercentage < 22) return false;
  if (analysis.motionState === 'slipping') return false;
  return timestamp - previousState.visualTimestamp <= V13_VISUAL_HOLD_MS;
}

function blendObjectRegion(previous: ObjectRegion, next: ObjectRegion, alpha: number, confidenceScale: number): ObjectRegion {
  return {
    ...next,
    center: {
      x: previous.center.x + (next.center.x - previous.center.x) * alpha,
      y: previous.center.y + (next.center.y - previous.center.y) * alpha
    },
    radiusX: previous.radiusX + (next.radiusX - previous.radiusX) * alpha,
    radiusY: previous.radiusY + (next.radiusY - previous.radiusY) * alpha,
    angle: previous.angle + shortestAngleDelta(previous.angle, next.angle) * alpha,
    confidence: clampUnit(next.confidence * confidenceScale),
    detectorScore: next.detectorScore == null ? next.detectorScore : clampUnit(next.detectorScore * confidenceScale),
    locked: confidenceScale >= 0.95 ? next.locked : false,
    velocity: {
      x: previous.velocity.x + (next.velocity.x - previous.velocity.x) * alpha,
      y: previous.velocity.y + (next.velocity.y - previous.velocity.y) * alpha
    }
  };
}

function shortestAngleDelta(previous: number, next: number) {
  return Math.atan2(Math.sin(next - previous), Math.cos(next - previous));
}

function smooth(previous: number, next: number, alpha: number) {
  return clampUnit(previous + (next - previous) * alpha);
}

function clampUnit(value: number) {
  return Math.min(1, Math.max(0, value));
}
