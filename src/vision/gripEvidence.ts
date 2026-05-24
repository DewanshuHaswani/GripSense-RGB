import type { GripEvidence, GripMode, Landmark, ObjectRegion, Point } from './types';
import {
  averagePoint,
  clamp,
  distance,
  distanceToEllipseBoundary,
  dot,
  FINGER_MCP_INDICES,
  FINGERTIP_INDICES,
  handSize,
  normalize,
  palmCenter,
  subtract,
  toHandLocal
} from './geometry';
import { DEFAULT_GRIP_SCORING_CONFIG } from './gripScoringConfig';

const FINGER_SEGMENTS = [
  [5, 6],
  [6, 7],
  [7, 8],
  [9, 10],
  [10, 11],
  [11, 12],
  [13, 14],
  [14, 15],
  [15, 16],
  [17, 18],
  [18, 19],
  [19, 20]
] as const;

export function computeGripEvidence(
  hand: Landmark[],
  object: ObjectRegion,
  persistentSlipScore: number
): GripEvidence {
  const size = handSize(hand);
  const palm = palmCenter(hand);
  const objectLocalToPalm = toHandLocal(object.center, hand, palm);
  const tips = FINGERTIP_INDICES.map((index) => hand[index]);
  const mcpCenter = averagePoint(FINGER_MCP_INDICES.map((index) => hand[index]));
  const curlScores = [8, 12, 16, 20].map((tipIndex, fingerIndex) => {
    const mcp = hand[FINGER_MCP_INDICES[fingerIndex]];
    const tip = hand[tipIndex];
    const openLength = Math.max(size * 0.35, distance(hand[0], mcp) + size * 0.18);
    const curledTowardPalm = clamp(1 - distance(tip, palm) / openLength);
    const tipNearObject = clamp(1 - distance(tip, object.center) / Math.max(size * 0.78, object.radiusX + object.radiusY));
    const boundaryWrap = clamp(1 - distanceToEllipseBoundary(tip, object) / Math.max(26, size * 0.32));
    return clamp(Math.max(curledTowardPalm * 0.72 + tipNearObject * 0.28, tipNearObject * 0.62 + boundaryWrap * 0.38));
  });
  const fingerCurlScore = average(curlScores);

  const segmentTolerance = Math.max(18, size * 0.16);
  const segmentScores = FINGER_SEGMENTS.map(([from, to]) => {
    const samples = [hand[from], midpoint(hand[from], hand[to]), hand[to]];
    return Math.max(
      ...samples.map((point) => {
        const boundaryDistance = distanceToEllipseBoundary(point, object);
        const insideObject = normalizedEllipseRadius(point, object) <= 1.12 ? 0.2 : 0;
        return clamp(1 - boundaryDistance / segmentTolerance + insideObject);
      })
    );
  });
  const fingerSegmentContactScore = averageTop(segmentScores, 5);
  const contactPoints = segmentScores.filter((score) => score > 0.42).length;
  const visibleContactScore = clamp(contactPoints / 5);

  const objectDistanceFromPalm = distance(palm, object.center);
  const centeredInHandCorridor = clamp(1 - Math.abs(objectLocalToPalm.x) / Math.max(size * 0.52, 1));
  const palmObjectContainmentScore = clamp(
    1 -
      Math.abs(distance(palm, object.center) - distance(mcpCenter, object.center)) / Math.max(size * 0.72, 1) +
      (objectDistanceFromPalm < size * 0.9 ? 0.18 : -0.18) +
      centeredInHandCorridor * 0.12
  );

  const thumb = hand[4];
  const fingerCenter = averagePoint([hand[8], hand[12], hand[16], hand[20]]);
  const thumbVector = normalize(subtract(thumb, object.center));
  const fingerVector = normalize(subtract(fingerCenter, object.center));
  const oppositionAngle = clamp((-dot(thumbVector, fingerVector) + 0.1) / 1.1);
  const thumbNearObject = clamp(1 - distanceToEllipseBoundary(thumb, object) / Math.max(20, size * 0.2));
  const thumbHiddenFallback = clamp(fingerCurlScore * 0.52 + palmObjectContainmentScore * 0.38 + object.confidence * 0.1);
  const thumbSupportScore = clamp(Math.max(oppositionAngle * 0.55 + thumbNearObject * 0.45, thumbHiddenFallback * 0.82));

  const phoneSideGripScore = computePhoneSideGripScore(hand, object, size, segmentScores);
  const contactRoles = computeContactRoles(hand, object, size, segmentScores, palmObjectContainmentScore, thumbNearObject);
  const pinchScore = computePinchScore(hand, object, size);
  const roleCoverage = scoreContactRoles(contactRoles);
  const powerGripScore = clamp(fingerCurlScore * 0.28 + fingerSegmentContactScore * 0.22 + palmObjectContainmentScore * 0.22 + thumbSupportScore * 0.16 + roleCoverage * 0.12);
  const hookGripScore = clamp(fingerCurlScore * 0.4 + fingerSegmentContactScore * 0.3 + palmObjectContainmentScore * 0.1 + (1 - thumbSupportScore) * 0.1 + roleCoverage * 0.1);
  const occlusionResilienceScore = clamp(Math.max(fingerSegmentContactScore, phoneSideGripScore, powerGripScore) - visibleContactScore * 0.18);
  const motionStabilityScore = clamp(1 - persistentSlipScore);
  const temporalLockScore = clamp((object.lockAgeFrames ?? 0) / 28);
  const sourceEvidence =
    object.source === 'manual' ? 0.96 : object.source === 'segmenter' ? 0.88 : object.source === 'detector' ? 0.82 : 0;
  const visualEvidence = clamp(
    (object.visualEdgeScore ?? 0) * 0.56 +
      (object.visualTextureScore ?? 0) * 0.26 +
      (object.shape === 'phone-like' ? 0.12 : 0) +
      temporalLockScore * 0.06
  );
  const independentObjectScore = clamp(
    Math.max(sourceEvidence, object.independentEvidenceScore ?? 0, visualEvidence) * 0.78 +
      object.confidence * 0.14 +
      (object.tightness ?? 0.55) * 0.08
  );
  const objectLockQuality = clamp(
    object.confidence * 0.5 +
      (object.tightness ?? 0.55) * 0.18 +
      (object.shape === 'phone-like' ? 0.14 : 0.04) +
      palmObjectContainmentScore * 0.18
  );

  const modeScores: Record<GripMode, number> = {
    'phone-side grip': phoneSideGripScore,
    'pinch grip': pinchScore,
    'power grip': powerGripScore,
    'hook grip': hookGripScore,
    'open hand': clamp(1 - Math.max(fingerCurlScore, fingerSegmentContactScore, thumbSupportScore)),
    uncertain: clamp(1 - objectLockQuality)
  };
  const positiveReasons = createPositiveReasons({
    fingerCurlScore,
    fingerSegmentContactScore,
    palmObjectContainmentScore,
    thumbSupportScore,
    phoneSideGripScore,
    pinchScore,
    powerGripScore,
    motionStabilityScore
  });
  const negativeReasons = createNegativeReasons({ objectLockQuality, thumbSupportScore, visibleContactScore, persistentSlipScore, palmObjectContainmentScore });

  return {
    fingerCurlScore,
    fingerSegmentContactScore,
    contactRoles,
    palmObjectContainmentScore,
    thumbSupportScore,
    phoneSideGripScore,
    persistentSlipScore,
    objectLockQuality: clamp(objectLockQuality + contactPointsBonus(contactPoints)),
    pinchScore,
    powerGripScore,
    hookGripScore,
    visibleContactScore,
    occlusionResilienceScore,
    motionStabilityScore,
    independentObjectScore,
    temporalLockScore,
    modeScores,
    positiveReasons,
    negativeReasons
  };
}

export function countContactPoints(evidence: GripEvidence) {
  const softCount =
    evidence.fingerSegmentContactScore * 3.2 +
    evidence.phoneSideGripScore * 1.2 +
    evidence.thumbSupportScore * 0.8 +
    evidence.palmObjectContainmentScore * 0.8;
  return Math.min(5, Math.max(0, Math.round(softCount)));
}

function computePhoneSideGripScore(hand: Landmark[], object: ObjectRegion, size: number, segmentScores: number[]) {
  const aspectRatio = object.aspectRatio ?? Math.max(object.radiusX, object.radiusY) / Math.max(1, Math.min(object.radiusX, object.radiusY));
  const phoneLike = object.shape === 'phone-like' || aspectRatio > 2.2;
  if (!phoneLike) return 0;

  const localPoints = [hand[4], hand[8], hand[12], hand[16], hand[20], hand[6], hand[10], hand[14], hand[18]].map((point) =>
    toObjectLocal(point, object)
  );
  const nearLongSide = localPoints.filter((point) => {
    const sideDistance = Math.abs(Math.abs(point.x) - object.radiusX);
    const withinLength = Math.abs(point.y) <= object.radiusY * 1.16;
    return sideDistance < Math.max(26, size * 0.18) && withinLength;
  }).length;
  const bothSides = hasOpposingSides(localPoints, object, size);
  const segmentSupport = averageTop(segmentScores, 4);
  return clamp((nearLongSide / 5) * 0.36 + (bothSides ? 0.34 : 0.08) + segmentSupport * 0.3);
}

function computePinchScore(hand: Landmark[], object: ObjectRegion, size: number) {
  const thumb = hand[4];
  const index = hand[8];
  const thumbIndexDistance = distance(thumb, index);
  const objectScale = Math.max(object.radiusX, object.radiusY);
  const smallObjectBias = clamp(1 - (objectScale - size * 0.18) / Math.max(1, size * 0.45));
  const thumbNearObject = clamp(1 - distanceToEllipseBoundary(thumb, object) / Math.max(18, size * 0.14));
  const indexNearObject = clamp(1 - distanceToEllipseBoundary(index, object) / Math.max(18, size * 0.14));
  const opposition = clamp(1 - thumbIndexDistance / Math.max(1, size * 0.72));
  return clamp(opposition * 0.38 + thumbNearObject * 0.24 + indexNearObject * 0.24 + smallObjectBias * 0.14);
}

function computeContactRoles(
  hand: Landmark[],
  object: ObjectRegion,
  size: number,
  segmentScores: number[],
  palmObjectContainmentScore: number,
  thumbNearObject: number
): GripEvidence['contactRoles'] {
  const roleTolerance = Math.max(18, size * 0.17);
  const roleScore = (indices: number[]) =>
    Math.max(
      ...indices.map((index) => {
        const boundaryScore = clamp(1 - distanceToEllipseBoundary(hand[index], object) / roleTolerance);
        const local = toHandLocal(hand[index], hand);
        const sideSupport = clamp(1 - Math.abs(local.x) / Math.max(size * 0.72, 1)) * 0.18;
        return clamp(boundaryScore + sideSupport);
      })
    );

  return {
    thumb: thumbNearObject,
    index: Math.max(roleScore([5, 6, 7, 8]), averageTop(segmentScores.slice(0, 3), 2)),
    middle: Math.max(roleScore([9, 10, 11, 12]), averageTop(segmentScores.slice(3, 6), 2)),
    ring: Math.max(roleScore([13, 14, 15, 16]), averageTop(segmentScores.slice(6, 9), 2)),
    pinky: Math.max(roleScore([17, 18, 19, 20]), averageTop(segmentScores.slice(9, 12), 2)),
    palm: palmObjectContainmentScore
  };
}

function scoreContactRoles(roles: GripEvidence['contactRoles']) {
  const weights = DEFAULT_GRIP_SCORING_CONFIG.contactRoleWeights;
  return clamp(
    roles.thumb * weights.thumb +
      roles.index * weights.index +
      roles.middle * weights.middle +
      roles.ring * weights.ring +
      roles.pinky * weights.pinky +
      roles.palm * weights.palm
  );
}

function hasOpposingSides(points: Point[], object: ObjectRegion, size: number) {
  const tolerance = Math.max(28, size * 0.2);
  const left = points.some((point) => Math.abs(point.x + object.radiusX) < tolerance);
  const right = points.some((point) => Math.abs(point.x - object.radiusX) < tolerance);
  return left && right;
}

function toObjectLocal(point: Point, object: ObjectRegion): Point {
  const angleCos = Math.cos(-object.angle);
  const angleSin = Math.sin(-object.angle);
  const dx = point.x - object.center.x;
  const dy = point.y - object.center.y;
  return {
    x: dx * angleCos - dy * angleSin,
    y: dx * angleSin + dy * angleCos
  };
}

function normalizedEllipseRadius(point: Point, object: ObjectRegion) {
  const local = toObjectLocal(point, object);
  return Math.hypot(local.x / Math.max(1, object.radiusX), local.y / Math.max(1, object.radiusY));
}

function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function averageTop(values: number[], count: number) {
  return average([...values].sort((a, b) => b - a).slice(0, count));
}

function contactPointsBonus(contactPoints: number) {
  return contactPoints >= 4 ? 0.08 : contactPoints >= 2 ? 0.04 : 0;
}

function createPositiveReasons(scores: {
  fingerCurlScore: number;
  fingerSegmentContactScore: number;
  palmObjectContainmentScore: number;
  thumbSupportScore: number;
  phoneSideGripScore: number;
  pinchScore: number;
  powerGripScore: number;
  motionStabilityScore: number;
}) {
  const reasons: string[] = [];
  if (scores.phoneSideGripScore > 0.58) reasons.push('phone-side support detected');
  if (scores.powerGripScore > 0.62) reasons.push('whole-hand wrap is strong');
  if (scores.pinchScore > 0.62) reasons.push('thumb-index pinch is stable');
  if (scores.fingerSegmentContactScore > 0.55) reasons.push('finger segments are supporting the object');
  if (scores.palmObjectContainmentScore > 0.62) reasons.push('object sits inside the hand corridor');
  if (scores.thumbSupportScore > 0.58) reasons.push('thumb support looks useful');
  if (scores.motionStabilityScore > 0.78) reasons.push('motion is stable');
  return reasons;
}

function createNegativeReasons(scores: {
  objectLockQuality: number;
  thumbSupportScore: number;
  visibleContactScore: number;
  persistentSlipScore: number;
  palmObjectContainmentScore: number;
}) {
  const reasons: string[] = [];
  if (scores.objectLockQuality < 0.42) reasons.push('object lock is uncertain');
  if (scores.visibleContactScore < 0.24) reasons.push('few visible contact points');
  if (scores.thumbSupportScore < 0.32) reasons.push('thumb support is unclear');
  if (scores.persistentSlipScore > 0.45) reasons.push('object and hand motion diverge');
  if (scores.palmObjectContainmentScore < 0.35) reasons.push('object is not clearly inside the grip corridor');
  return reasons;
}
