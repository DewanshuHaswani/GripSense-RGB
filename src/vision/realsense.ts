import { analyzeGripWithRfdetr, type RfdetrGripResult, type RfdetrTimelinePoint } from './rfdetr';
import { clamp } from './geometry';
import type { GripAnalysis, GripCalibrationBaseline, Landmark, ObjectRegion, Point } from './types';
import type { RfdetrDetection, RfdetrTrackState } from './rfdetr';

export const DEFAULT_REALSENSE_ENDPOINT = 'http://127.0.0.1:7867/api/realsense/depth-signal';
export const REALSENSE_REQUEST_INTERVAL_MS = 260;
export const REALSENSE_OFFLINE_INTERVAL_MS = 180;

export type RealSenseDepthSignal = {
  available: boolean;
  frameTimestamp?: number;
  latencyMs?: number;
  handDepthM?: number | null;
  objectDepthM?: number | null;
  depthDeltaM?: number | null;
  contactDepthScore: number;
  depthSeparationScore: number;
  stereoConfidence: number;
  occlusionScore: number;
  surfaceContinuity: number;
  source?: string;
};

export type RealSenseRuntime = {
  status: 'idle' | 'pending' | 'ready' | 'unavailable';
  message: string;
  endpoint: string;
  result: RealSenseDepthSignal | null;
  receivedAt: number | null;
  lastRequestAt: number;
  latencyMs: number | null;
};

export type RealSenseTimelinePoint = RfdetrTimelinePoint & {
  realsenseDepthContact?: number;
  realsenseDepthSeparation?: number;
  realsenseStereoConfidence?: number;
};

export function createInitialRealSenseRuntime(endpoint = DEFAULT_REALSENSE_ENDPOINT): RealSenseRuntime {
  return {
    status: 'idle',
    message: 'RealSense depth idle. Select V9 live, Offline V3, or Offline Max to use aligned RGB-D evidence.',
    endpoint,
    result: null,
    receivedAt: null,
    lastRequestAt: 0,
    latencyMs: null
  };
}

export function analyzeGripWithRealSense(options: {
  hand: Landmark[] | null;
  detections: RfdetrDetection[];
  previousPalm: Point | null;
  previousObject: ObjectRegion | null;
  previousTrack: RfdetrTrackState;
  now: number;
  depthSignal: RealSenseDepthSignal | null;
  persistentSlipScore?: number;
  calibrationBaseline?: GripCalibrationBaseline | null;
  weakCalibrationBaseline?: GripCalibrationBaseline | null;
  serverAvailable: boolean;
  unavailableMessage?: string;
}): RfdetrGripResult {
  const base = analyzeGripWithRfdetr({
    hand: options.hand,
    detections: options.detections,
    previousPalm: options.previousPalm,
    previousObject: options.previousObject,
    previousTrack: options.previousTrack,
    now: options.now,
    persistentSlipScore: options.persistentSlipScore,
    calibrationBaseline: options.calibrationBaseline,
    weakCalibrationBaseline: options.weakCalibrationBaseline,
    serverAvailable: options.serverAvailable,
    unavailableMessage: options.unavailableMessage
  });

  if (!options.depthSignal?.available) {
    return {
      ...base,
      objectIdentity: base.object ? { ...base.objectIdentity, name: 'RealSense object' } : base.objectIdentity,
      analysis: {
        ...base.analysis,
        diagnostics: {
          ...base.analysis.diagnostics,
          recommendation:
            base.object && options.serverAvailable
              ? 'RealSense depth unavailable; using RF-DETR mask evidence only.'
              : base.analysis.diagnostics.recommendation
        }
      }
    };
  }

  return {
    ...base,
    objectIdentity: base.object ? { ...base.objectIdentity, name: 'RealSense depth object' } : base.objectIdentity,
    analysis: applyRealSenseDepthGate(base.analysis, options.depthSignal, Boolean(base.object)),
    selection: {
      ...base.selection,
      objectScore: clamp(base.selection.objectScore * 0.78 + options.depthSignal.stereoConfidence * 0.1 + options.depthSignal.surfaceContinuity * 0.12),
      contact: clamp(base.selection.contact * 0.62 + options.depthSignal.contactDepthScore * 0.38)
    }
  };
}

export function applyRealSenseDepthGate(analysis: GripAnalysis, depth: RealSenseDepthSignal, hasObject: boolean): GripAnalysis {
  if (!hasObject) return analysis;
  const depthContact = clamp(depth.contactDepthScore);
  const separation = clamp(depth.depthSeparationScore);
  const stereo = clamp(depth.stereoConfidence);
  const continuity = clamp(depth.surfaceContinuity);
  const depthQuality = clamp(stereo * 0.42 + depthContact * 0.32 + continuity * 0.18 + (1 - depth.occlusionScore) * 0.08);
  const separated = separation > 0.58 && depthContact < 0.34;
  const contactCap = depthContact < 0.18 ? 24 : depthContact < 0.34 ? 44 : depthContact < 0.48 ? 68 : 100;
  const separationCap = separated ? Math.max(12, Math.round((1 - separation) * 64)) : 100;
  const cap = Math.min(contactCap, separationCap);
  const boostedGrip = cap >= 100 ? Math.round(analysis.gripPercentage * 0.88 + Math.min(100, analysis.gripPercentage + depthQuality * 14) * 0.12) : analysis.gripPercentage;
  const gripPercentage = Math.min(boostedGrip, cap);
  const confidence = Math.min(
    separated ? Math.min(analysis.confidence, 0.42) : Math.max(analysis.confidence, depthQuality * 0.92),
    depthContact < 0.18 ? 0.48 : 1
  );
  return {
    ...analysis,
    gripPercentage,
    confidence,
    objectLockQuality: separated ? Math.min(analysis.objectLockQuality, depthQuality * 0.72) : Math.max(analysis.objectLockQuality, depthQuality),
    slipRisk: separated ? Math.max(analysis.slipRisk, 0.62) : Math.max(0, analysis.slipRisk - depthContact * 0.08),
    evidence: {
      ...analysis.evidence,
      visibleContactScore: clamp(analysis.evidence.visibleContactScore * 0.7 + depthContact * 0.3),
      fingerSegmentContactScore: separated
        ? Math.min(analysis.evidence.fingerSegmentContactScore, depthContact)
        : clamp(analysis.evidence.fingerSegmentContactScore * 0.72 + depthContact * 0.28),
      objectLockQuality: separated ? Math.min(analysis.evidence.objectLockQuality, depthQuality * 0.72) : Math.max(analysis.evidence.objectLockQuality, depthQuality),
      independentObjectScore: Math.max(analysis.evidence.independentObjectScore, depthQuality),
      motionStabilityScore: clamp(analysis.evidence.motionStabilityScore * 0.72 + continuity * 0.28)
    },
    diagnostics: {
      ...analysis.diagnostics,
      state: separated ? 'Slip risk' : analysis.diagnostics.state,
      recommendation: separated ? 'RealSense depth shows hand-object separation; re-seat the object in the grip.' : analysis.diagnostics.recommendation,
      objectIssue: separated ? 'Depth edge moved away from the hand corridor.' : analysis.diagnostics.objectIssue,
      gripIssue: separated ? 'Stereo depth contact is weak even though RGB grip geometry looks plausible.' : analysis.diagnostics.gripIssue,
      issueCategory: separated ? 'motion_problem' : analysis.diagnostics.issueCategory
    }
  };
}

export function refineRealSenseOfflineTimeline<T extends RealSenseTimelinePoint>(points: T[]): T[] {
  if (points.length < 3) return points;
  return points.map((point, index) => {
    const neighbors = points.slice(Math.max(0, index - 3), Math.min(points.length, index + 4));
    const depthNeighbors = neighbors.filter((item) => (item.realsenseDepthContact ?? 0) > 0.38 && (item.realsenseStereoConfidence ?? 0) > 0.4);
    if (!depthNeighbors.length) return point;
    const depthContact = average(depthNeighbors.map((item) => item.realsenseDepthContact ?? 0));
    const stereo = average(depthNeighbors.map((item) => item.realsenseStereoConfidence ?? 0));
    const separated = (point.realsenseDepthSeparation ?? 0) > 0.62 && depthContact < 0.34;
    if (separated) {
      return {
        ...point,
        grip: Math.min(point.grip, 34),
        confidence: Math.min(point.confidence, 0.42),
        contact: Math.min(point.contact, depthContact),
        slip: Math.max(point.slip, 0.62),
        weak: true,
        guidance: 'Reposition',
        state: 'Slip risk'
      };
    }
    const grip = Math.max(point.grip, Math.round(Math.min(86, point.grip * 0.82 + depthContact * 36 + stereo * 10)));
    return {
      ...point,
      grip,
      confidence: Math.max(point.confidence, Math.min(0.92, point.confidence * 0.74 + depthContact * 0.18 + stereo * 0.08)),
      contact: Math.max(point.contact, depthContact),
      lock: Math.max(point.lock, Math.min(0.94, point.lock * 0.76 + stereo * 0.24)),
      weak: grip < 42,
      state: grip > 50 ? 'Grip detected' : point.state
    };
  });
}

export async function requestRealSenseDepthSignal(options: {
  hand: Landmark[] | null;
  object: ObjectRegion | null;
  endpoint: string;
  timestamp: number;
  signal?: AbortSignal;
}): Promise<{ ok: true; response: RealSenseDepthSignal; receivedAt: number } | { ok: false; status: string; receivedAt: number }> {
  try {
    const response = await fetch(options.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        timestamp: options.timestamp,
        hand: options.hand,
        object: options.object ? objectPayload(options.object) : null
      }),
      signal: options.signal
    });
    const receivedAt = performance.now();
    if (!response.ok) return { ok: false, status: `HTTP ${response.status}`, receivedAt };
    return { ok: true, response: normalizeRealSenseDepthSignal(await response.json()), receivedAt };
  } catch (error) {
    return { ok: false, status: error instanceof Error ? error.message : 'request_failed', receivedAt: performance.now() };
  }
}

export function isRealSenseResultFresh(runtime: RealSenseRuntime, now: number, maxAgeMs: number) {
  return Boolean(runtime.result && runtime.receivedAt !== null && now - runtime.receivedAt <= maxAgeMs);
}

export function normalizeRealSenseDepthSignal(value: unknown): RealSenseDepthSignal {
  const raw = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>;
  const available = Boolean(raw.available);
  return {
    available,
    frameTimestamp: numberOr(raw.frameTimestamp, undefined),
    latencyMs: numberOr(raw.latencyMs, undefined),
    handDepthM: nullableNumber(raw.handDepthM),
    objectDepthM: nullableNumber(raw.objectDepthM),
    depthDeltaM: nullableNumber(raw.depthDeltaM),
    contactDepthScore: available ? clamp(numberOr(raw.contactDepthScore, 0)) : 0,
    depthSeparationScore: available ? clamp(numberOr(raw.depthSeparationScore, 1)) : 1,
    stereoConfidence: available ? clamp(numberOr(raw.stereoConfidence, 0)) : 0,
    occlusionScore: available ? clamp(numberOr(raw.occlusionScore, 0)) : 0,
    surfaceContinuity: available ? clamp(numberOr(raw.surfaceContinuity, 0)) : 0,
    source: typeof raw.source === 'string' ? raw.source : undefined
  };
}

function objectPayload(object: ObjectRegion) {
  return {
    center: object.center,
    radiusX: object.radiusX,
    radiusY: object.radiusY,
    angle: object.angle,
    contour: object.contour
  };
}

function nullableNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function numberOr<T extends number | undefined>(value: unknown, fallback: T): number | T {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function average(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}
