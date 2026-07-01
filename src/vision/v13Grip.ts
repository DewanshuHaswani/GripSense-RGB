import { distance } from './geometry';
import { stabilizeV12DisplayAnalysis, validateV12ObjectInHand, type V12DisplayState, type V12ObjectEvidence } from './v12Grip';
import type { GripAnalysis, Landmark, ObjectRegion, Point } from './types';

export type V13ObjectEvidence = V12ObjectEvidence & {
  object?: ObjectRegion | null;
  hand?: Landmark[] | null;
  frameWidth?: number;
  frameHeight?: number;
};

export type V13DisplayState = V12DisplayState & {
  visualObject: ObjectRegion | null;
  visualTimestamp: number;
  visualMissedFrames: number;
  candidateRejected: boolean;
  rejectedCandidateReason: string | null;
};

type V13CandidateGate = {
  rejected: boolean;
  reason: string | null;
  risk: number;
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
    visualMissedFrames: 0,
    candidateRejected: false,
    rejectedCandidateReason: null
  };
}

export function stabilizeV13DisplayAnalysis(
  rawAnalysis: GripAnalysis,
  previousState: V13DisplayState,
  timestamp: number,
  objectEvidence: V13ObjectEvidence
): { analysis: GripAnalysis; state: V13DisplayState } {
  const candidateGate = classifyV13Candidate(rawAnalysis, previousState, timestamp, objectEvidence);
  const effectiveEvidence = candidateGate.rejected ? suppressBackgroundCandidateEvidence(objectEvidence) : objectEvidence;
  const v12 = stabilizeV12DisplayAnalysis(rawAnalysis, previousState, timestamp, effectiveEvidence);
  const v12Analysis = candidateGate.rejected ? annotateV13CandidateRejection(v12.analysis, candidateGate) : v12.analysis;
  const v12State = v12.analysis === v12Analysis ? v12.state : { ...v12.state, analysis: v12Analysis };
  const previous = previousState.analysis;
  const validation = validateV12ObjectInHand(rawAnalysis, effectiveEvidence);

  if (!previous) {
    return { analysis: v12Analysis, state: preserveVisualState(v12State, previousState, candidateGate) };
  }

  const collapsed = v12Analysis.guidance === 'Object not locked' || v12Analysis.gripPercentage <= 18;
  const previousStable =
    previous.gripPercentage >= 42 &&
    previous.objectLockQuality >= 0.38 &&
    previous.confidence >= 0.32 &&
    previous.diagnostics.issueCategory !== 'server_unavailable';
  const stillWrapped = handStillWrapped(rawAnalysis, validation);
  const clearlyGone = handClearlyEmptyOrGone(rawAnalysis, validation);
  const detectorBlink = effectiveEvidence.missedFrames > 0 || !effectiveEvidence.hasObject;
  const stableAge = timestamp - (previousState.lastStableAt ?? previousState.timestamp);

  if (collapsed && previousStable && detectorBlink && stillWrapped && !clearlyGone && stableAge <= V13_OCCLUSION_BRIDGE_MS) {
    const softLossStartedAt = previousState.softLossStartedAt ?? timestamp;
    const lossAge = timestamp - softLossStartedAt;
    if (lossAge <= V13_OCCLUSION_BRIDGE_MS) {
      const inferredMissAge = Math.max(80, effectiveEvidence.missedFrames * 80);
      const decay = clampUnit(Math.max(lossAge, inferredMissAge) / V13_OCCLUSION_BRIDGE_MS);
      const gripPercentage = Math.round(
        Math.max(24, previous.gripPercentage * (1 - decay * 0.6) + rawAnalysis.gripPercentage * 0.12)
      );
      const confidence = clampUnit(Math.max(0.22, previous.confidence * (1 - decay * 0.62)));
      const objectLockQuality = clampUnit(Math.max(0.2, previous.objectLockQuality * (1 - decay * 0.58)));
      const analysis: GripAnalysis = {
        ...v12Analysis,
        gripPercentage,
        confidence,
        objectLockQuality,
        guidance: gripPercentage >= 48 ? 'Improve grip' : 'Reposition',
        message:
          'V13 is treating this as short object occlusion by the fingers. It holds a degraded grip estimate briefly, then drops if YOLO still cannot see a separate object.',
        evidence: {
          ...v12Analysis.evidence,
          objectLockQuality,
          temporalLockScore: Math.max(v12Analysis.evidence.temporalLockScore, previous.evidence.temporalLockScore * (1 - decay * 0.5)),
          visibleContactScore: Math.max(v12Analysis.evidence.visibleContactScore, previous.evidence.visibleContactScore * (1 - decay * 0.52)),
          fingerSegmentContactScore: Math.max(
            v12Analysis.evidence.fingerSegmentContactScore,
            previous.evidence.fingerSegmentContactScore * (1 - decay * 0.52)
          ),
          negativeReasons: [
            ...v12Analysis.evidence.negativeReasons,
            'V13 occlusion bridge active',
            ...(candidateGate.rejected ? ['V13 ignored background-like detector candidate during bridge'] : [])
          ]
        },
        diagnostics: {
          ...v12Analysis.diagnostics,
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
          previousState,
          candidateGate
        )
      };
    }
  }

  if (!collapsed && previousStable && v12Analysis.gripPercentage > 0) {
    const stableObject =
      v12Analysis.objectLockQuality >= 0.52 &&
      validation.objectInHandScore >= 0.48 &&
      v12Analysis.diagnostics.issueCategory !== 'server_unavailable';
    const gripAlpha = stableObject ? 0.2 : 0.34;
    const smoothedGrip = Math.round(previous.gripPercentage + (v12Analysis.gripPercentage - previous.gripPercentage) * gripAlpha);
    const smoothedConfidence = smooth(previous.confidence, v12Analysis.confidence, stableObject ? 0.24 : 0.38);
    const smoothedLock = smooth(previous.objectLockQuality, v12Analysis.objectLockQuality, stableObject ? 0.22 : 0.4);
    const analysis: GripAnalysis = {
      ...v12Analysis,
      gripPercentage: smoothedGrip,
      confidence: smoothedConfidence,
      objectLockQuality: smoothedLock,
      evidence: {
        ...v12Analysis.evidence,
        objectLockQuality: smoothedLock
      }
    };
    return {
      analysis,
      state: preserveVisualState(
        {
          ...v12State,
          analysis
        },
        previousState,
        candidateGate
      )
    };
  }

  return { analysis: v12Analysis, state: preserveVisualState(v12State, previousState, candidateGate) };
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

function preserveVisualState(state: V12DisplayState, previous: V13DisplayState, candidateGate?: V13CandidateGate): V13DisplayState {
  return {
    ...state,
    visualObject: previous.visualObject,
    visualTimestamp: previous.visualTimestamp,
    visualMissedFrames: previous.visualMissedFrames,
    candidateRejected: candidateGate?.rejected ?? false,
    rejectedCandidateReason: candidateGate?.reason ?? null
  };
}

function classifyV13Candidate(
  analysis: GripAnalysis,
  previousState: V13DisplayState,
  timestamp: number,
  evidence: V13ObjectEvidence
): V13CandidateGate {
  const object = evidence.object;
  const hand = evidence.hand;
  const frameWidth = evidence.frameWidth ?? 0;
  const frameHeight = evidence.frameHeight ?? 0;
  if (!evidence.hasObject || !object || !hand || hand.length < 21 || frameWidth <= 0 || frameHeight <= 0) {
    return { rejected: false, reason: null, risk: 0 };
  }

  const objectWidth = object.radiusX * 2;
  const objectHeight = object.radiusY * 2;
  const frameArea = Math.max(1, frameWidth * frameHeight);
  const areaRatio = (objectWidth * objectHeight) / frameArea;
  const widthRatio = objectWidth / frameWidth;
  const heightRatio = objectHeight / frameHeight;
  const aspect = Math.max(objectWidth, objectHeight) / Math.max(1, Math.min(objectWidth, objectHeight));
  const handBox = boundsForPoints(hand);
  const handWidth = Math.max(1, handBox.maxX - handBox.minX);
  const handHeight = Math.max(1, handBox.maxY - handBox.minY);
  const handScale = Math.max(36, Math.hypot(handWidth, handHeight));
  const palm = palmCenterFromLandmarks(hand);
  const nearestHandDistance = Math.min(...hand.map((point) => distance(point, object.center)));
  const centerDistance = distance(palm, object.center);
  const overlap = rectIoU(
    { x: object.center.x - object.radiusX, y: object.center.y - object.radiusY, width: objectWidth, height: objectHeight },
    { x: handBox.minX, y: handBox.minY, width: handWidth, height: handHeight }
  );
  const contact = Math.max(
    evidence.contact,
    analysis.evidence.visibleContactScore,
    analysis.evidence.fingerSegmentContactScore * 0.72
  );
  const continuity = previousState.visualObject && timestamp - previousState.visualTimestamp <= V13_VISUAL_HOLD_MS
    ? clampUnit(1 - distance(previousState.visualObject.center, object.center) / Math.max(60, handScale * 1.7))
    : 0;
  const strongHandObjectEvidence =
    contact >= 0.3 ||
    overlap >= 0.1 ||
    (evidence.objectScore >= 0.58 && contact >= 0.2) ||
    (nearestHandDistance <= handScale * 0.2 && overlap >= 0.045) ||
    continuity >= 0.62;

  const largePlaneRisk =
    areaRatio > 0.16 ||
    widthRatio > 0.46 ||
    heightRatio > 0.58 ||
    (Math.max(objectWidth, objectHeight) > handScale * 3.1 && contact < 0.34);
  const flatTableRisk = aspect > 3.2 && areaRatio > 0.045 && contact < 0.34;
  const cubiclePanelRisk =
    aspect <= 1.45 &&
    areaRatio > 0.06 &&
    overlap < 0.055 &&
    contact < 0.28 &&
    nearestHandDistance > handScale * 0.22;
  const farFromHandRisk = centerDistance > handScale * 1.8 && nearestHandDistance > handScale * 0.48;
  const edgeTouchOnlyRisk =
    (largePlaneRisk || flatTableRisk || cubiclePanelRisk) &&
    overlap < 0.04 &&
    contact < 0.22 &&
    nearestHandDistance <= handScale * 0.5;
  const detectorWeakForLargeSurface = evidence.objectScore < 0.42 && (largePlaneRisk || flatTableRisk || cubiclePanelRisk);
  const risk = clampUnit(
    (largePlaneRisk ? 0.34 : 0) +
      (flatTableRisk ? 0.3 : 0) +
      (cubiclePanelRisk ? 0.26 : 0) +
      (farFromHandRisk ? 0.28 : 0) +
      (edgeTouchOnlyRisk ? 0.2 : 0) +
      (detectorWeakForLargeSurface ? 0.16 : 0) -
      (strongHandObjectEvidence ? 0.46 : 0)
  );

  if (risk >= 0.38 && !strongHandObjectEvidence) {
    const reason = flatTableRisk
      ? 'V13 ignored a wide table-like detector candidate.'
      : cubiclePanelRisk
        ? 'V13 ignored a square cubicle/panel-like detector candidate.'
        : 'V13 ignored a large background-like detector candidate.';
    return { rejected: true, reason, risk };
  }

  return { rejected: false, reason: null, risk };
}

function suppressBackgroundCandidateEvidence(evidence: V13ObjectEvidence): V13ObjectEvidence {
  return {
    ...evidence,
    objectScore: Math.min(evidence.objectScore, 0.06),
    contact: Math.min(evidence.contact, 0.04),
    hasObject: false,
    missedFrames: Math.max(1, evidence.missedFrames)
  };
}

function annotateV13CandidateRejection(analysis: GripAnalysis, gate: V13CandidateGate): GripAnalysis {
  if (!gate.rejected || !gate.reason) return analysis;
  const collapsed = analysis.guidance === 'Object not locked' || analysis.gripPercentage <= 18;
  const message = collapsed
    ? `${gate.reason} Grip scoring stays at no grip until YOLO sees a compact object in the hand corridor.`
    : analysis.message;
  return {
    ...analysis,
    message,
    evidence: {
      ...analysis.evidence,
      negativeReasons: [...analysis.evidence.negativeReasons, gate.reason]
    },
    diagnostics: {
      ...analysis.diagnostics,
      recommendation: collapsed ? message : analysis.diagnostics.recommendation,
      objectIssue: collapsed ? message : analysis.diagnostics.objectIssue,
      issueCategory: collapsed ? 'object_problem' : analysis.diagnostics.issueCategory
    }
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

function boundsForPoints(points: Point[]) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  return { minX, minY, maxX, maxY };
}

function palmCenterFromLandmarks(hand: Landmark[]): Point {
  const indices = [0, 5, 9, 13, 17];
  const total = indices.reduce(
    (acc, index) => {
      const point = hand[index] ?? hand[0];
      return { x: acc.x + point.x, y: acc.y + point.y };
    },
    { x: 0, y: 0 }
  );
  return { x: total.x / indices.length, y: total.y / indices.length };
}

function rectIoU(
  a: { x: number; y: number; width: number; height: number },
  b: { x: number; y: number; width: number; height: number }
) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function clampUnit(value: number) {
  return Math.min(1, Math.max(0, value));
}
