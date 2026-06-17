import { analyzeGrip, createEmptyAnalysis } from './gripAnalysis';
import { averagePoint, clamp, distance, FINGERTIP_INDICES, handSize, palmCenter, subtract } from './geometry';
import type { GripAnalysis, GripCalibrationBaseline, Landmark, ObjectIdentitySignal, ObjectRegion, Point } from './types';

export const DEFAULT_RFDETR_ENDPOINT = 'http://127.0.0.1:7867/api/rfdetr/analyze';
export const RFDETR_REQUEST_INTERVAL_MS = 520;
export const RFDETR_OFFLINE_INTERVAL_MS = 220;

export type RfdetrDetection = {
  id: string;
  label: string;
  score: number;
  bbox: { x: number; y: number; width: number; height: number };
  maskPolygon: Point[];
  maskArea: number;
  center: Point;
  latencyMs: number;
};

export type RfdetrAnalyzeResponse = {
  detections: RfdetrDetection[];
  latencyMs?: number;
  model?: string;
  device?: string;
};

export type RfdetrRuntime = {
  status: 'idle' | 'pending' | 'ready' | 'unavailable';
  message: string;
  endpoint: string;
  result: RfdetrAnalyzeResponse | null;
  resultPalm?: Point | null;
  receivedAt: number | null;
  lastRequestAt: number;
  latencyMs: number | null;
};

export type RfdetrTrackState = {
  detectionKey: string | null;
  center: Point | null;
  confidence: number;
  continuity: number;
  missedFrames: number;
  lastSeenAt: number;
};

export type RfdetrSelection = {
  detection: RfdetrDetection | null;
  objectScore: number;
  contact: number;
  proximity: number;
  continuity: number;
  rejectedPerson: boolean;
};

export type RfdetrGripResult = {
  analysis: GripAnalysis;
  object: ObjectRegion | null;
  objectIdentity: ObjectIdentitySignal;
  track: RfdetrTrackState;
  selection: RfdetrSelection;
};

export type RfdetrTimelinePoint = {
  grip: number;
  confidence: number;
  objectMatch: number;
  lock: number;
  contact: number;
  slip: number;
  weak: boolean;
  guidance: string;
  state: string;
  object?: string;
  rfdetrObjectScore?: number;
  rfdetrContact?: number;
};

export const EMPTY_RFDETR_TRACK: RfdetrTrackState = {
  detectionKey: null,
  center: null,
  confidence: 0,
  continuity: 0,
  missedFrames: 0,
  lastSeenAt: 0
};

export function createInitialRfdetrRuntime(endpoint = DEFAULT_RFDETR_ENDPOINT): RfdetrRuntime {
  return {
    status: 'idle',
    message: 'RF-DETR server idle. Select V8 or Offline V2 to begin RF-DETR analysis.',
    endpoint,
    result: null,
    resultPalm: null,
    receivedAt: null,
    lastRequestAt: 0,
    latencyMs: null
  };
}

export function analyzeGripWithRfdetr(options: {
  hand: Landmark[] | null;
  detections: RfdetrDetection[];
  previousPalm: Point | null;
  previousObject: ObjectRegion | null;
  previousTrack: RfdetrTrackState;
  now: number;
  persistentSlipScore?: number;
  calibrationBaseline?: GripCalibrationBaseline | null;
  weakCalibrationBaseline?: GripCalibrationBaseline | null;
  serverAvailable: boolean;
  unavailableMessage?: string;
}): RfdetrGripResult {
  const emptyTrack = decayRfdetrTrack(options.previousTrack, options.now);
  const unavailable = !options.serverAvailable;
  if (unavailable) {
    const analysis = createRfdetrUnavailableAnalysis(options.unavailableMessage);
    return {
      analysis,
      object: null,
      objectIdentity: rfdetrIdentity(null, 0, 0),
      track: emptyTrack,
      selection: { detection: null, objectScore: 0, contact: 0, proximity: 0, continuity: emptyTrack.continuity, rejectedPerson: false }
    };
  }

  if (!options.hand || options.hand.length < 21) {
    return {
      analysis: createEmptyAnalysis('No hand detected. Keep your hand inside the camera frame.'),
      object: null,
      objectIdentity: rfdetrIdentity(null, 0, 0),
      track: emptyTrack,
      selection: { detection: null, objectScore: 0, contact: 0, proximity: 0, continuity: emptyTrack.continuity, rejectedPerson: false }
    };
  }

  const selection = selectRfdetrGripObject(options.detections, options.hand, options.previousTrack);
  const track = updateRfdetrTrack(options.previousTrack, selection, options.now);
  if (!selection.detection || !rfdetrObjectLockReady(selection)) {
    const held = createHeldRfdetrObject(options.previousObject, options.hand, options.previousPalm, track);
    if (held) {
      const holdSelection = {
        ...selection,
        objectScore: held.objectScore,
        contact: held.contact,
        proximity: held.proximity,
        continuity: track.continuity
      };
      const objectIdentity = heldRfdetrIdentity(holdSelection.objectScore);
      const baseAnalysis = analyzeGrip(options.hand, held.object, options.previousPalm, {
        persistentSlipScore: options.persistentSlipScore ?? 0,
        calibrationBaseline: options.calibrationBaseline,
        weakCalibrationBaseline: options.weakCalibrationBaseline,
        algorithmVersion: 'v6',
        objectIdentity
      });
      return {
        analysis: applyRfdetrHoldGate(baseAnalysis, holdSelection, track),
        object: held.object,
        objectIdentity,
        track,
        selection: holdSelection
      };
    }

    const analysis = createEmptyAnalysis(
      selection.rejectedPerson
        ? 'RF-DETR rejected person detection as a grip object.'
        : 'RF-DETR did not find a non-person object close enough to the hand.'
    );
    return {
      analysis: {
        ...analysis,
        closureScore: closedHandScore(options.hand),
        palmCenter: palmCenter(options.hand),
        diagnostics: {
          ...analysis.diagnostics,
          state: selection.detection ? 'Object uncertain' : 'Hand only',
          recommendation: analysis.message,
          objectIssue: analysis.message,
          issueCategory: 'object_problem'
        }
      },
      object: null,
      objectIdentity: rfdetrIdentity(null, 0, selection.contact),
      track,
      selection
    };
  }

  const object = rfdetrDetectionToObjectRegion(selection.detection, options.previousObject, selection, track);
  const baseAnalysis = analyzeGrip(options.hand, object, options.previousPalm, {
    persistentSlipScore: options.persistentSlipScore ?? 0,
    calibrationBaseline: options.calibrationBaseline,
    weakCalibrationBaseline: options.weakCalibrationBaseline,
    algorithmVersion: 'v6',
    objectIdentity: rfdetrIdentity(selection.detection, selection.objectScore, selection.contact)
  });
  const analysis = applyRfdetrContactGate(baseAnalysis, selection, track);

  return {
    analysis,
    object,
    objectIdentity: rfdetrIdentity(selection.detection, selection.objectScore, selection.contact),
    track,
    selection
  };
}

export function selectRfdetrGripObject(
  detections: RfdetrDetection[],
  hand: Landmark[] | null,
  previousTrack: RfdetrTrackState = EMPTY_RFDETR_TRACK
): RfdetrSelection {
  const valid = detections.filter((detection) => detection.score > 0.05);
  const nonPerson = valid.filter((detection) => !isPersonLabel(detection.label));
  let best: RfdetrSelection | null = null;

  for (const detection of nonPerson) {
    const contact = hand ? rfdetrMaskContactScore(detection, hand) : 0;
    const proximity = hand ? rfdetrHandProximityScore(detection, hand) : 0.2;
    const spatial = hand ? rfdetrSpatialPlausibilityScore(detection, hand) : 1;
    const continuity = rfdetrTemporalContinuity(detection, previousTrack);
    const rawObjectScore = clamp((detection.score * 0.22 + contact * 0.5 + proximity * 0.18 + continuity * 0.1) * spatial);
    const objectScore = contact < 0.12 && continuity < 0.42 ? Math.min(rawObjectScore, 0.16) : rawObjectScore;
    if (!best || objectScore > best.objectScore) {
      best = {
        detection,
        objectScore,
        contact,
        proximity,
        continuity,
        rejectedPerson: false
      };
    }
  }

  return best ?? {
    detection: null,
    objectScore: 0,
    contact: 0,
    proximity: 0,
    continuity: 0,
    rejectedPerson: valid.length > 0 && nonPerson.length === 0
  };
}

export function rfdetrMaskContactScore(detection: RfdetrDetection, hand: Landmark[]) {
  if (!hand.length) return 0;
  const size = handSize(hand);
  const samples = handContactSamples(hand);
  const polygon = detection.maskPolygon.length >= 3 ? detection.maskPolygon : bboxPolygon(detection.bbox);
  const inside = samples.filter((point) => pointInPolygon(point, polygon)).length / samples.length;
  const nearBoundary = samples.filter((point) => pointToPolygonDistance(point, polygon) < Math.max(16, size * 0.16)).length / samples.length;
  const bboxOverlap = rectIoU(rectFromPoints(handCorridorPoints(hand)), detection.bbox);
  const palmDistance = pointToRectDistance(palmCenter(hand), detection.bbox);
  const palmNear = clamp(1 - palmDistance / Math.max(28, size * 0.72));
  return clamp(inside * 0.38 + nearBoundary * 0.28 + Math.min(1, bboxOverlap * 5.2) * 0.22 + palmNear * 0.12);
}

export function rfdetrDetectionToObjectRegion(
  detection: RfdetrDetection,
  previous: ObjectRegion | null,
  selection: Pick<RfdetrSelection, 'objectScore' | 'contact' | 'proximity' | 'continuity'>,
  track: RfdetrTrackState
): ObjectRegion {
  const same = previous?.detectorLabel === `rfdetr:${detection.id}`;
  const aspectRatio = Math.max(detection.bbox.width, detection.bbox.height) / Math.max(1, Math.min(detection.bbox.width, detection.bbox.height));
  const contour = detection.maskPolygon.length >= 3 ? detection.maskPolygon : bboxPolygon(detection.bbox);
  const confidence = clamp(0.36 + detection.score * 0.26 + selection.contact * 0.26 + selection.continuity * 0.12);
  const center = detection.center;
  return {
    center,
    radiusX: Math.max(12, detection.bbox.width / 2),
    radiusY: Math.max(12, detection.bbox.height / 2),
    angle: 0,
    confidence,
    locked: rfdetrObjectLockReady(selection),
    source: 'segmenter',
    velocity: same && previous ? subtract(center, previous.center) : { x: 0, y: 0 },
    contour,
    shape: aspectRatio > 1.35 ? 'phone-like' : aspectRatio > 1.12 ? 'ellipse' : 'unknown',
    aspectRatio,
    tightness: clamp(0.52 + selection.contact * 0.34 + detection.score * 0.14),
    lockAgeFrames: rfdetrObjectLockReady(selection) ? Math.max(track.continuity * 18, same ? (previous?.lockAgeFrames ?? 0) + 1 : 1) : 0,
    manuallyAdjusted: false,
    visualEdgeScore: clamp(0.34 + detection.score * 0.42),
    visualTextureScore: clamp(0.24 + detection.score * 0.32),
    independentEvidenceScore: clamp(0.5 + detection.score * 0.26 + selection.contact * 0.18),
    relativeDriftScore: same && previous ? clamp(distance(center, previous.center) / Math.max(1, Math.max(detection.bbox.width, detection.bbox.height))) : 0,
    detectorLabel: `rfdetr:${detection.id}`,
    detectorScore: detection.score
  };
}

function rfdetrObjectLockReady(selection: Pick<RfdetrSelection, 'objectScore' | 'contact' | 'proximity' | 'continuity'>) {
  if (selection.objectScore < 0.22 || selection.proximity < 0.2) return false;
  if (selection.contact >= 0.16) return true;
  return selection.contact >= 0.1 && selection.proximity >= 0.42 && selection.continuity >= 0.7;
}

function createHeldRfdetrObject(previous: ObjectRegion | null, hand: Landmark[], previousPalm: Point | null, track: RfdetrTrackState) {
  if (!previous || !previous.detectorLabel?.startsWith('rfdetr:')) return null;
  if (track.missedFrames < 1 || track.missedFrames > 4 || track.confidence < 0.08 || track.continuity < 0.04) return null;
  const currentPalm = palmCenter(hand);
  const palmShift = previousPalm ? subtract(currentPalm, previousPalm) : { x: 0, y: 0 };
  const maxShift = Math.max(30, handSize(hand) * 1.18);
  const compensated = distance({ x: 0, y: 0 }, palmShift) <= maxShift ? translateObjectRegion(previous, palmShift) : previous;
  const proximity = rfdetrObjectRegionProximityScore(compensated, hand);
  if (proximity < 0.28) return null;
  const contact = rfdetrObjectRegionContactScore(compensated, hand);
  const ageDecay = Math.max(0.32, 1 - track.missedFrames * 0.16);
  const objectScore = clamp((track.confidence * 0.5 + proximity * 0.26 + contact * 0.24) * ageDecay);
  if (objectScore < 0.12) return null;
  const object: ObjectRegion = {
    ...compensated,
    confidence: Math.min(compensated.confidence, objectScore),
    locked: false,
    lockAgeFrames: 0,
    velocity: palmShift,
    tightness: Math.min(compensated.tightness ?? 0.45, Math.max(0.16, contact)),
    visualEdgeScore: Math.min(compensated.visualEdgeScore ?? compensated.confidence, objectScore),
    visualTextureScore: Math.min(compensated.visualTextureScore ?? compensated.confidence, objectScore),
    independentEvidenceScore: Math.min(compensated.independentEvidenceScore ?? compensated.confidence, objectScore),
    relativeDriftScore: Math.max(compensated.relativeDriftScore ?? 0, 0.42 + track.missedFrames * 0.08)
  };
  return { object, objectScore, contact, proximity };
}

function applyRfdetrHoldGate(analysis: GripAnalysis, selection: RfdetrSelection, track: RfdetrTrackState): GripAnalysis {
  const holdCap = Math.max(20, Math.round((selection.objectScore * 0.56 + selection.contact * 0.3 + track.continuity * 0.14) * 72));
  const gripPercentage = Math.min(analysis.gripPercentage, holdCap);
  const confidence = Math.min(analysis.confidence, clamp(selection.objectScore * 0.52 + selection.contact * 0.2 + track.continuity * 0.1));
  return {
    ...analysis,
    gripPercentage,
    confidence,
    guidance: gripPercentage >= 34 ? 'Improve grip' : 'Object uncertain',
    objectLockQuality: Math.min(analysis.objectLockQuality, selection.objectScore),
    objectIdentityScore: selection.objectScore,
    objectIdentityMatched: false,
    slipRisk: Math.max(analysis.slipRisk, 0.42),
    evidence: {
      ...analysis.evidence,
      visibleContactScore: Math.min(analysis.evidence.visibleContactScore, selection.contact),
      fingerSegmentContactScore: Math.min(analysis.evidence.fingerSegmentContactScore, Math.max(selection.contact, 0.08)),
      objectLockQuality: Math.min(analysis.evidence.objectLockQuality, selection.objectScore)
    },
    diagnostics: {
      ...analysis.diagnostics,
      state: gripPercentage >= 34 ? 'Grip detected' : 'Object uncertain',
      recommendation: 'RF-DETR missed this frame; holding the previous object boundary briefly.',
      objectIssue: 'RF-DETR did not refresh the object mask on this frame.',
      gripIssue: 'Object boundary is a short temporal hold, so grip confidence is capped.',
      issueCategory: 'object_problem'
    }
  };
}

export function applyRfdetrContactGate(analysis: GripAnalysis, selection: RfdetrSelection, track: RfdetrTrackState): GripAnalysis {
  const quickDecay = selection.contact < 0.14 || selection.proximity < 0.18 || track.missedFrames > 0;
  const contactCap = selection.contact < 0.12 ? 18 : selection.contact < 0.22 ? 32 : selection.contact < 0.34 ? 52 : 100;
  const continuityCap = track.missedFrames > 0 ? Math.max(8, Math.round(track.confidence * 46)) : 100;
  const scoreCap = selection.objectScore < 0.28 ? 34 : 100;
  const cap = Math.min(contactCap, continuityCap, scoreCap);
  if (cap >= 100) {
    return {
      ...analysis,
      objectIdentityScore: Math.max(analysis.objectIdentityScore, selection.objectScore),
      evidence: {
        ...analysis.evidence,
        visibleContactScore: Math.max(analysis.evidence.visibleContactScore, selection.contact),
        objectLockQuality: Math.max(analysis.evidence.objectLockQuality, selection.objectScore)
      }
    };
  }

  const gripPercentage = Math.min(analysis.gripPercentage, cap);
  const confidence = Math.min(analysis.confidence, clamp(selection.objectScore * 0.5 + selection.contact * 0.38 + track.continuity * 0.12));
  return {
    ...analysis,
    gripPercentage,
    confidence,
    guidance: gripPercentage > 36 && !quickDecay ? 'Improve grip' : 'Reposition',
    objectLockQuality: Math.min(analysis.objectLockQuality, selection.objectScore),
    objectIdentityScore: selection.objectScore,
    slipRisk: Math.max(analysis.slipRisk, quickDecay && analysis.gripPercentage > 35 ? 0.48 : analysis.slipRisk),
    evidence: {
      ...analysis.evidence,
      visibleContactScore: Math.min(analysis.evidence.visibleContactScore, selection.contact),
      fingerSegmentContactScore: Math.min(analysis.evidence.fingerSegmentContactScore, Math.max(selection.contact, 0.08)),
      objectLockQuality: Math.min(analysis.evidence.objectLockQuality, selection.objectScore)
    },
    diagnostics: {
      ...analysis.diagnostics,
      state: gripPercentage <= 18 || quickDecay ? 'Object uncertain' : analysis.diagnostics.state,
      recommendation: quickDecay
        ? 'RF-DETR object mask is no longer overlapping the hand corridor.'
        : 'RF-DETR mask contact is weak; reposition object against the hand.',
      gripIssue: 'RF-DETR mask boundary is not visibly in contact with the hand corridor.',
      issueCategory: 'object_problem'
    }
  };
}

export function refineRfdetrOfflineTimeline<T extends RfdetrTimelinePoint>(points: T[]): T[] {
  if (points.length < 3) return points;
  return points.map((point, index) => {
    const neighbors = points.slice(Math.max(0, index - 2), Math.min(points.length, index + 3));
    const strongObjectNeighbors = neighbors.filter((item) => (item.rfdetrObjectScore ?? item.objectMatch) > 0.34);
    const bridgeableMiss =
      (point.rfdetrObjectScore ?? point.objectMatch) < 0.16 &&
      strongObjectNeighbors.length >= Math.min(2, neighbors.length - 1);
    if (!bridgeableMiss) return point;

    const objectScore = average(strongObjectNeighbors.map((item) => item.rfdetrObjectScore ?? item.objectMatch)) * 0.82;
    const contact = average(strongObjectNeighbors.map((item) => item.rfdetrContact ?? item.contact)) * 0.78;
    const lock = Math.max(point.lock, objectScore * 0.92);
    const grip = Math.max(point.grip, Math.round(Math.min(72, objectScore * 44 + contact * 38)));
    return {
      ...point,
      grip,
      confidence: Math.max(point.confidence, Math.min(0.82, objectScore * 0.54 + contact * 0.34)),
      objectMatch: Math.max(point.objectMatch, objectScore),
      lock,
      contact: Math.max(point.contact, contact),
      rfdetrObjectScore: Math.max(point.rfdetrObjectScore ?? 0, objectScore),
      rfdetrContact: Math.max(point.rfdetrContact ?? 0, contact),
      weak: grip < 42 || lock < 0.22,
      guidance: grip > 48 ? 'Improve grip' : point.guidance,
      state: lock > 0.24 ? 'Grip detected' : point.state,
      object: point.object || 'RF-DETR object'
    };
  });
}

export async function requestRfdetrFrameAnalysis(options: {
  video: HTMLVideoElement;
  hand: Landmark[] | null;
  endpoint: string;
  mirrored: boolean;
  signal?: AbortSignal;
}): Promise<{ ok: true; response: RfdetrAnalyzeResponse; receivedAt: number } | { ok: false; status: string; receivedAt: number }> {
  const frame = await captureVideoJpeg(options.video);
  if (!frame) return { ok: false, status: 'frame_unavailable', receivedAt: performance.now() };

  const form = new FormData();
  form.set('frame', frame.blob, 'frame.jpg');
  form.set('width', String(frame.width));
  form.set('height', String(frame.height));
  form.set('mirrored', String(options.mirrored));
  if (options.hand) form.set('handLandmarks', JSON.stringify(options.hand));

  try {
    const response = await fetch(options.endpoint, {
      method: 'POST',
      body: form,
      signal: options.signal
    });
    const receivedAt = performance.now();
    if (!response.ok) return { ok: false, status: `HTTP ${response.status}`, receivedAt };
    const parsed = parseRfdetrAnalyzeResponse(await response.json());
    if (!parsed.ok) return { ok: false, status: parsed.error, receivedAt };
    return { ok: true, response: scaleRfdetrResponseToVideo(parsed.response, frame), receivedAt };
  } catch (error) {
    return { ok: false, status: error instanceof Error ? error.message : 'request_failed', receivedAt: performance.now() };
  }
}

export function parseRfdetrAnalyzeResponse(value: unknown): { ok: true; response: RfdetrAnalyzeResponse } | { ok: false; error: string } {
  if (!value || typeof value !== 'object') return { ok: false, error: 'RF-DETR response was not an object' };
  const raw = value as { detections?: unknown; latencyMs?: unknown; model?: unknown; device?: unknown };
  if (!Array.isArray(raw.detections)) return { ok: false, error: 'RF-DETR response missing detections' };
  const detections: RfdetrDetection[] = [];
  for (const item of raw.detections) {
    const detection = normalizeRfdetrDetection(item);
    if (detection) detections.push(detection);
  }
  return {
    ok: true,
    response: {
      detections,
      latencyMs: typeof raw.latencyMs === 'number' ? raw.latencyMs : undefined,
      model: typeof raw.model === 'string' ? raw.model : undefined,
      device: typeof raw.device === 'string' ? raw.device : undefined
    }
  };
}

export function isRfdetrResultFresh(runtime: RfdetrRuntime, now: number, maxAgeMs: number) {
  return Boolean(runtime.result && runtime.receivedAt !== null && now - runtime.receivedAt <= maxAgeMs);
}

export function compensateRfdetrResponseForHandMotion(
  response: RfdetrAnalyzeResponse,
  sourcePalm: Point | null | undefined,
  currentHand: Landmark[] | null
): RfdetrAnalyzeResponse {
  if (!sourcePalm || !currentHand?.length) return response;
  const currentPalm = palmCenter(currentHand);
  const shift = subtract(currentPalm, sourcePalm);
  const maxShift = Math.max(26, handSize(currentHand) * 1.35);
  const shiftDistance = distance({ x: 0, y: 0 }, shift);
  if (shiftDistance < 1) return response;
  if (shiftDistance > maxShift) return response;
  return {
    ...response,
    detections: response.detections.map((detection) => translateRfdetrDetection(detection, shift))
  };
}

function normalizeRfdetrDetection(value: unknown): RfdetrDetection | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const bbox = normalizeBbox(raw.bbox);
  const score = numberOr(raw.score, 0);
  const center = normalizePoint(raw.center) ?? { x: bbox.x + bbox.width / 2, y: bbox.y + bbox.height / 2 };
  if (!bbox.width || !bbox.height || !Number.isFinite(score)) return null;
  return {
    id: String(raw.id ?? `${raw.label ?? 'object'}-${Math.round(center.x)}-${Math.round(center.y)}`),
    label: String(raw.label ?? 'object'),
    score: clamp(score),
    bbox,
    maskPolygon: Array.isArray(raw.maskPolygon) ? raw.maskPolygon.map(normalizePoint).filter((point): point is Point => Boolean(point)) : [],
    maskArea: numberOr(raw.maskArea, bbox.width * bbox.height),
    center,
    latencyMs: numberOr(raw.latencyMs, 0)
  };
}

function normalizeBbox(value: unknown) {
  if (Array.isArray(value)) {
    const [x, y, width, height] = value.map((item) => numberOr(item, 0));
    return { x, y, width, height };
  }
  const raw = (value ?? {}) as Record<string, unknown>;
  const x = numberOr(raw.x ?? raw.left, 0);
  const y = numberOr(raw.y ?? raw.top, 0);
  const width = numberOr(raw.width, numberOr(raw.right, x) - x);
  const height = numberOr(raw.height, numberOr(raw.bottom, y) - y);
  return { x, y, width, height };
}

function normalizePoint(value: unknown): Point | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const x = numberOr(raw.x, Number.NaN);
  const y = numberOr(raw.y, Number.NaN);
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

async function captureVideoJpeg(video: HTMLVideoElement) {
  if (!video.videoWidth || !video.videoHeight) return null;
  const maxWidth = 720;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return null;
  context.drawImage(video, 0, 0, width, height);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.72));
  return blob ? { blob, width, height, sourceWidth: video.videoWidth, sourceHeight: video.videoHeight } : null;
}

export function scaleRfdetrResponseToVideo(
  response: RfdetrAnalyzeResponse,
  frame: { width: number; height: number; sourceWidth: number; sourceHeight: number }
): RfdetrAnalyzeResponse {
  const scaleX = frame.sourceWidth / Math.max(1, frame.width);
  const scaleY = frame.sourceHeight / Math.max(1, frame.height);
  if (Math.abs(scaleX - 1) < 0.001 && Math.abs(scaleY - 1) < 0.001) return response;
  return {
    ...response,
    detections: response.detections.map((detection) => ({
      ...detection,
      bbox: {
        x: detection.bbox.x * scaleX,
        y: detection.bbox.y * scaleY,
        width: detection.bbox.width * scaleX,
        height: detection.bbox.height * scaleY
      },
      maskPolygon: detection.maskPolygon.map((point) => ({ x: point.x * scaleX, y: point.y * scaleY })),
      maskArea: detection.maskArea * scaleX * scaleY,
      center: {
        x: detection.center.x * scaleX,
        y: detection.center.y * scaleY
      }
    }))
  };
}

function translateRfdetrDetection(detection: RfdetrDetection, offset: Point): RfdetrDetection {
  return {
    ...detection,
    bbox: {
      ...detection.bbox,
      x: detection.bbox.x + offset.x,
      y: detection.bbox.y + offset.y
    },
    maskPolygon: detection.maskPolygon.map((point) => ({ x: point.x + offset.x, y: point.y + offset.y })),
    center: {
      x: detection.center.x + offset.x,
      y: detection.center.y + offset.y
    }
  };
}

function updateRfdetrTrack(previous: RfdetrTrackState, selection: RfdetrSelection, now: number): RfdetrTrackState {
  if (!selection.detection) return decayRfdetrTrack(previous, now);
  const same =
    previous.detectionKey === selection.detection.id ||
    (previous.center ? distance(previous.center, selection.detection.center) < Math.max(selection.detection.bbox.width, selection.detection.bbox.height, 30) : false);
  return {
    detectionKey: selection.detection.id,
    center: selection.detection.center,
    confidence: clamp(same ? previous.confidence * 0.48 + selection.objectScore * 0.52 : selection.objectScore),
    continuity: clamp(same ? previous.continuity * 0.72 + 0.28 : 0.24),
    missedFrames: 0,
    lastSeenAt: now
  };
}

function decayRfdetrTrack(previous: RfdetrTrackState, now: number): RfdetrTrackState {
  if (!previous.detectionKey) return { ...EMPTY_RFDETR_TRACK, lastSeenAt: previous.lastSeenAt };
  return {
    ...previous,
    confidence: clamp(previous.confidence * 0.64),
    continuity: clamp(previous.continuity * 0.68),
    missedFrames: previous.missedFrames + 1,
    lastSeenAt: previous.lastSeenAt || now
  };
}

function translateObjectRegion(object: ObjectRegion, offset: Point): ObjectRegion {
  if (Math.abs(offset.x) < 0.001 && Math.abs(offset.y) < 0.001) return object;
  return {
    ...object,
    center: { x: object.center.x + offset.x, y: object.center.y + offset.y },
    contour: object.contour.map((point) => ({ x: point.x + offset.x, y: point.y + offset.y }))
  };
}

function rfdetrIdentity(detection: RfdetrDetection | null, objectScore: number, contact: number): ObjectIdentitySignal {
  return {
    hasProfiles: Boolean(detection),
    score: objectScore,
    matched: Boolean(detection && objectScore >= 0.28 && contact >= 0.16),
    name: detection ? `RF-DETR ${safeObjectLabel(detection.label)}` : null,
    source: 'base'
  };
}

function heldRfdetrIdentity(objectScore: number): ObjectIdentitySignal {
  return {
    hasProfiles: true,
    score: objectScore,
    matched: false,
    name: 'RF-DETR held object',
    source: 'base'
  };
}

function createRfdetrUnavailableAnalysis(message = 'RF-DETR unavailable. Start the local RF-DETR server to use V8 live analysis.'): GripAnalysis {
  const analysis = createEmptyAnalysis(message);
  return {
    ...analysis,
    confidence: 0,
    diagnostics: {
      ...analysis.diagnostics,
      state: 'Object uncertain',
      recommendation: message,
      objectIssue: message,
      issueCategory: 'server_unavailable'
    }
  };
}

function rfdetrHandProximityScore(detection: RfdetrDetection, hand: Landmark[]) {
  const size = handSize(hand);
  const points = [palmCenter(hand), ...FINGERTIP_INDICES.map((index) => hand[index]).filter(Boolean)];
  const nearest = Math.min(...points.map((point) => pointToRectDistance(point, detection.bbox)));
  const centerDistance = distance(detection.center, palmCenter(hand));
  return clamp((1 - nearest / Math.max(22, size * 0.74)) * 0.68 + (1 - centerDistance / Math.max(30, size * 1.8)) * 0.32);
}

function rfdetrObjectRegionProximityScore(object: ObjectRegion, hand: Landmark[]) {
  const size = handSize(hand);
  const rect = objectRegionRect(object);
  const points = [palmCenter(hand), ...FINGERTIP_INDICES.map((index) => hand[index]).filter(Boolean)];
  const nearest = Math.min(...points.map((point) => pointToRectDistance(point, rect)));
  const centerDistance = distance(object.center, palmCenter(hand));
  return clamp((1 - nearest / Math.max(22, size * 0.82)) * 0.68 + (1 - centerDistance / Math.max(30, size * 2.05)) * 0.32);
}

function rfdetrObjectRegionContactScore(object: ObjectRegion, hand: Landmark[]) {
  const size = handSize(hand);
  const samples = handContactSamples(hand);
  const polygon = object.contour.length >= 3 ? object.contour : bboxPolygon(objectRegionRect(object));
  const inside = samples.filter((point) => pointInPolygon(point, polygon)).length / samples.length;
  const nearBoundary = samples.filter((point) => pointToPolygonDistance(point, polygon) < Math.max(18, size * 0.18)).length / samples.length;
  const bboxOverlap = rectIoU(rectFromPoints(handCorridorPoints(hand)), objectRegionRect(object));
  const palmDistance = pointToRectDistance(palmCenter(hand), objectRegionRect(object));
  const palmNear = clamp(1 - palmDistance / Math.max(32, size * 0.78));
  return clamp(inside * 0.34 + nearBoundary * 0.26 + Math.min(1, bboxOverlap * 4.8) * 0.24 + palmNear * 0.16);
}

function objectRegionRect(object: ObjectRegion) {
  return {
    x: object.center.x - object.radiusX,
    y: object.center.y - object.radiusY,
    width: object.radiusX * 2,
    height: object.radiusY * 2
  };
}

function rfdetrSpatialPlausibilityScore(detection: RfdetrDetection, hand: Landmark[]) {
  const size = handSize(hand);
  const maxSide = Math.max(detection.bbox.width, detection.bbox.height);
  const minSide = Math.max(1, Math.min(detection.bbox.width, detection.bbox.height));
  const centerDistance = distance(detection.center, palmCenter(hand));
  const areaRatio = (detection.bbox.width * detection.bbox.height) / Math.max(1, size * size);
  const sidePenalty = maxSide > size * 4.4 ? clamp(1 - (maxSide - size * 4.4) / Math.max(size * 2.8, 1)) : 1;
  const areaPenalty = areaRatio > 14 ? clamp(1 - (areaRatio - 14) / 18) : 1;
  const centerPenalty = centerDistance > size * 2.9 ? clamp(1 - (centerDistance - size * 2.9) / Math.max(size * 2.2, 1)) : 1;
  const thinObjectBonus = maxSide / minSide > 1.55 && maxSide < size * 5.2 ? 0.12 : 0;
  return clamp(Math.min(sidePenalty, areaPenalty, centerPenalty) + thinObjectBonus);
}

function rfdetrTemporalContinuity(detection: RfdetrDetection, previousTrack: RfdetrTrackState) {
  if (!previousTrack.center) return 0;
  const radius = Math.max(24, detection.bbox.width, detection.bbox.height);
  return clamp(1 - distance(detection.center, previousTrack.center) / (radius * 1.8)) * previousTrack.continuity;
}

function closedHandScore(hand: Landmark[]) {
  const palm = palmCenter(hand);
  const size = handSize(hand);
  const tips = FINGERTIP_INDICES.map((index) => hand[index]).filter(Boolean);
  const avgTipDistance = average(tips.map((tip) => distance(tip, palm)));
  return clamp(1 - (avgTipDistance - size * 0.34) / (size * 0.34));
}

function handContactSamples(hand: Landmark[]) {
  const samples: Point[] = [palmCenter(hand)];
  const segments = [
    [4, 8],
    [5, 8],
    [9, 12],
    [13, 16],
    [17, 20],
    [4, 12]
  ] as const;
  FINGERTIP_INDICES.forEach((index) => samples.push(hand[index]));
  segments.forEach(([from, to]) => {
    samples.push(hand[from], midpoint(hand[from], hand[to]), hand[to]);
  });
  return samples.filter(Boolean);
}

function handCorridorPoints(hand: Landmark[]) {
  return [palmCenter(hand), averagePoint([hand[5], hand[9], hand[13], hand[17]]), ...FINGERTIP_INDICES.map((index) => hand[index]).filter(Boolean)];
}

function rectFromPoints(points: Point[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  const padding = Math.max(18, Math.max(maxX - minX, maxY - minY) * 0.24);
  return { x: minX - padding, y: minY - padding, width: maxX - minX + padding * 2, height: maxY - minY + padding * 2 };
}

function pointInPolygon(point: Point, polygon: Point[]) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const a = polygon[index];
    const b = polygon[previous];
    const intersects = a.y > point.y !== b.y > point.y && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y || 0.0001) + a.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function pointToPolygonDistance(point: Point, polygon: Point[]) {
  if (!polygon.length) return Number.POSITIVE_INFINITY;
  if (pointInPolygon(point, polygon)) return 0;
  let best = Number.POSITIVE_INFINITY;
  for (let index = 0; index < polygon.length; index += 1) {
    best = Math.min(best, pointToSegmentDistance(point, polygon[index], polygon[(index + 1) % polygon.length]));
  }
  return best;
}

function pointToSegmentDistance(point: Point, a: Point, b: Point) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy || 1;
  const t = clamp(((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared);
  return distance(point, { x: a.x + dx * t, y: a.y + dy * t });
}

function pointToRectDistance(point: Point, rect: { x: number; y: number; width: number; height: number }) {
  const dx = Math.max(rect.x - point.x, 0, point.x - (rect.x + rect.width));
  const dy = Math.max(rect.y - point.y, 0, point.y - (rect.y + rect.height));
  return Math.hypot(dx, dy);
}

function rectIoU(a: { x: number; y: number; width: number; height: number }, b: { x: number; y: number; width: number; height: number }) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = Math.max(1, a.width * a.height + b.width * b.height - intersection);
  return intersection / union;
}

function bboxPolygon(bbox: { x: number; y: number; width: number; height: number }): Point[] {
  return [
    { x: bbox.x, y: bbox.y },
    { x: bbox.x + bbox.width, y: bbox.y },
    { x: bbox.x + bbox.width, y: bbox.y + bbox.height },
    { x: bbox.x, y: bbox.y + bbox.height }
  ];
}

function isPersonLabel(label: string) {
  const normalized = label.toLowerCase().trim();
  return normalized === 'person' || normalized === 'human' || normalized.includes('person');
}

function safeObjectLabel(label: string) {
  return isPersonLabel(label) ? 'object' : label || 'object';
}

function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function numberOr(value: unknown, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}
