import type { GripAnalysis, Landmark, ObjectIdentitySignal, ObjectRegion, V3DiagnosticCode, V3PerceptionResponse } from './types';

export const DEFAULT_V3_ENDPOINT = 'http://127.0.0.1:7867/v3/analyze-frame';
const DEFAULT_TIMEOUT_MS = 700;
const FRAME_MAX_WIDTH = 512;

export type V3FramePayload = {
  dataUrl: string;
  width: number;
  height: number;
  mirrored: boolean;
  coordinateSpace: 'video';
};

export type V3AnalyzeFrameRequest = {
  version: 'v3';
  frame: V3FramePayload;
  timestamp: number;
  hand: Landmark[] | null;
  object: ObjectRegion | null;
  v2Analysis: Pick<GripAnalysis, 'gripPercentage' | 'confidence' | 'diagnostics' | 'evidence' | 'objectLockQuality' | 'slipRisk'>;
  objectIdentity?: ObjectIdentitySignal;
};

export type V3ClientResult =
  | {
      ok: true;
      response: V3PerceptionResponse;
      receivedAt: number;
    }
  | {
      ok: false;
      status: 'timeout' | 'network_error' | 'http_error' | 'invalid_response' | 'frame_unavailable';
      message: string;
      receivedAt: number;
    };

type NumberValidation = { ok: true; value: number } | { ok: false; reason: string };

export async function requestV3FrameAnalysis(
  request: V3AnalyzeFrameRequest,
  options: {
    endpoint?: string;
    timeoutMs?: number;
  } = {}
): Promise<V3ClientResult> {
  const endpoint = options.endpoint ?? DEFAULT_V3_ENDPOINT;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request),
      signal: controller.signal
    });

    if (!response.ok) {
      return {
        ok: false,
        status: 'http_error',
        message: `V3 server returned HTTP ${response.status}.`,
        receivedAt: performance.now()
      };
    }

    const parsed = parseV3PerceptionResponse(await response.json());
    if (!parsed.ok) {
      return {
        ok: false,
        status: 'invalid_response',
        message: parsed.reason,
        receivedAt: performance.now()
      };
    }

    return {
      ok: true,
      response: parsed.response,
      receivedAt: performance.now()
    };
  } catch (error) {
    return {
      ok: false,
      status: error instanceof DOMException && error.name === 'AbortError' ? 'timeout' : 'network_error',
      message: error instanceof Error ? error.message : 'V3 server request failed.',
      receivedAt: performance.now()
    };
  } finally {
    window.clearTimeout(timeout);
  }
}

export function createV3AnalyzeFrameRequest(options: {
  video: HTMLVideoElement;
  mirrored: boolean;
  timestamp: number;
  hand: Landmark[] | null;
  object: ObjectRegion | null;
  v2Analysis: GripAnalysis;
  objectIdentity?: ObjectIdentitySignal;
}): V3AnalyzeFrameRequest | null {
  const frame = captureVideoFrame(options.video, options.mirrored);
  if (!frame) return null;
  return {
    version: 'v3',
    frame,
    timestamp: options.timestamp,
    hand: options.hand,
    object: options.object,
    v2Analysis: {
      gripPercentage: options.v2Analysis.gripPercentage,
      confidence: options.v2Analysis.confidence,
      diagnostics: options.v2Analysis.diagnostics,
      evidence: options.v2Analysis.evidence,
      objectLockQuality: options.v2Analysis.objectLockQuality,
      slipRisk: options.v2Analysis.slipRisk
    },
    objectIdentity: options.objectIdentity
  };
}

export function parseV3PerceptionResponse(value: unknown): { ok: true; response: V3PerceptionResponse } | { ok: false; reason: string } {
  if (!isRecord(value)) return { ok: false, reason: 'V3 response must be an object.' };
  if (value.version !== 'v3') return { ok: false, reason: 'V3 response version must be "v3".' };

  const frameTimestamp = requiredNumber(value.frameTimestamp, 'frameTimestamp');
  const latencyMs = requiredNumber(value.latencyMs, 'latencyMs');
  const uncertainty = requiredUnitNumber(value.uncertainty, 'uncertainty');
  if (!frameTimestamp.ok) return frameTimestamp;
  if (!latencyMs.ok) return latencyMs;
  if (!uncertainty.ok) return uncertainty;

  if (!isRecord(value.hand)) return { ok: false, reason: 'V3 response hand must be an object.' };
  if (!isRecord(value.object)) return { ok: false, reason: 'V3 response object must be an object.' };
  if (!isRecord(value.contact)) return { ok: false, reason: 'V3 response contact must be an object.' };
  if (!isRecord(value.temporal)) return { ok: false, reason: 'V3 response temporal must be an object.' };

  const hand = {
    meshQuality: requiredUnitNumber(value.hand.meshQuality, 'hand.meshQuality'),
    occlusion: requiredUnitNumber(value.hand.occlusion, 'hand.occlusion'),
    handednessConfidence: requiredUnitNumber(value.hand.handednessConfidence, 'hand.handednessConfidence'),
    fingerArticulation: requiredUnitNumber(value.hand.fingerArticulation, 'hand.fingerArticulation')
  };
  const object = {
    present: typeof value.object.present === 'boolean' ? value.object.present : null,
    maskConfidence: requiredUnitNumber(value.object.maskConfidence, 'object.maskConfidence'),
    maskStability: requiredUnitNumber(value.object.maskStability, 'object.maskStability'),
    identityConfidence: requiredUnitNumber(value.object.identityConfidence, 'object.identityConfidence'),
    poseConfidence: requiredUnitNumber(value.object.poseConfidence, 'object.poseConfidence'),
    lockConfidence: requiredUnitNumber(value.object.lockConfidence, 'object.lockConfidence')
  };
  const contact = {
    thumb: requiredUnitNumber(value.contact.thumb, 'contact.thumb'),
    index: requiredUnitNumber(value.contact.index, 'contact.index'),
    middle: requiredUnitNumber(value.contact.middle, 'contact.middle'),
    ring: requiredUnitNumber(value.contact.ring, 'contact.ring'),
    pinky: requiredUnitNumber(value.contact.pinky, 'contact.pinky'),
    palm: requiredUnitNumber(value.contact.palm, 'contact.palm'),
    coverage: requiredUnitNumber(value.contact.coverage, 'contact.coverage'),
    opposingPairs: requiredUnitNumber(value.contact.opposingPairs, 'contact.opposingPairs')
  };
  const temporal = {
    continuity: requiredUnitNumber(value.temporal.continuity, 'temporal.continuity'),
    coupling: requiredUnitNumber(value.temporal.coupling, 'temporal.coupling'),
    slipRisk: requiredUnitNumber(value.temporal.slipRisk, 'temporal.slipRisk'),
    jitter: requiredUnitNumber(value.temporal.jitter, 'temporal.jitter')
  };

  const checks = [
    hand.meshQuality,
    hand.occlusion,
    hand.handednessConfidence,
    hand.fingerArticulation,
    object.maskConfidence,
    object.maskStability,
    object.identityConfidence,
    object.poseConfidence,
    object.lockConfidence,
    contact.thumb,
    contact.index,
    contact.middle,
    contact.ring,
    contact.pinky,
    contact.palm,
    contact.coverage,
    contact.opposingPairs,
    temporal.continuity,
    temporal.coupling,
    temporal.slipRisk,
    temporal.jitter
  ];
  const failed = checks.find((check) => !check.ok);
  if (failed && !failed.ok) return failed;
  if (object.present === null) return { ok: false, reason: 'object.present must be a boolean.' };

  const diagnostics = validateDiagnostics(value.diagnostics);
  if (!diagnostics.ok) return diagnostics;

  return {
    ok: true,
    response: {
      version: 'v3',
      frameTimestamp: numberValue(frameTimestamp),
      modelTimestamp: typeof value.modelTimestamp === 'number' && Number.isFinite(value.modelTimestamp) ? value.modelTimestamp : undefined,
      latencyMs: numberValue(latencyMs),
      uncertainty: numberValue(uncertainty),
      hand: {
        meshQuality: numberValue(hand.meshQuality),
        occlusion: numberValue(hand.occlusion),
        handednessConfidence: numberValue(hand.handednessConfidence),
        fingerArticulation: numberValue(hand.fingerArticulation),
        joints: Array.isArray(value.hand.joints) ? sanitizeJoints(value.hand.joints) : undefined
      },
      object: {
        present: object.present,
        maskConfidence: numberValue(object.maskConfidence),
        maskStability: numberValue(object.maskStability),
        identityConfidence: numberValue(object.identityConfidence),
        poseConfidence: numberValue(object.poseConfidence),
        lockConfidence: numberValue(object.lockConfidence)
      },
      contact: {
        thumb: numberValue(contact.thumb),
        index: numberValue(contact.index),
        middle: numberValue(contact.middle),
        ring: numberValue(contact.ring),
        pinky: numberValue(contact.pinky),
        palm: numberValue(contact.palm),
        coverage: numberValue(contact.coverage),
        opposingPairs: numberValue(contact.opposingPairs)
      },
      temporal: {
        continuity: numberValue(temporal.continuity),
        coupling: numberValue(temporal.coupling),
        slipRisk: numberValue(temporal.slipRisk),
        jitter: numberValue(temporal.jitter)
      },
      diagnostics: diagnostics.value
    }
  };
}

function captureVideoFrame(video: HTMLVideoElement, mirrored: boolean): V3FramePayload | null {
  if (!video.videoWidth || !video.videoHeight) return null;
  const scale = Math.min(1, FRAME_MAX_WIDTH / video.videoWidth);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return null;
  context.drawImage(video, 0, 0, width, height);
  return {
    dataUrl: canvas.toDataURL('image/jpeg', 0.72),
    width,
    height,
    mirrored,
    coordinateSpace: 'video'
  };
}

function requiredNumber(value: unknown, label: string): NumberValidation {
  if (typeof value !== 'number' || !Number.isFinite(value)) return { ok: false, reason: `${label} must be a finite number.` };
  return { ok: true, value };
}

function requiredUnitNumber(value: unknown, label: string): NumberValidation {
  const number = requiredNumber(value, label);
  if (!number.ok) return number;
  if (number.value < 0 || number.value > 1) return { ok: false, reason: `${label} must be between 0 and 1.` };
  return number;
}

function numberValue(result: NumberValidation) {
  return result.ok ? result.value : 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isV3DiagnosticCode(value: unknown): value is V3DiagnosticCode {
  return (
    value === 'object_uncertain' ||
    value === 'hand_occluded' ||
    value === 'contact_uncertain' ||
    value === 'slip_risk' ||
    value === 'server_unavailable' ||
    value === 'strong_hold'
  );
}

function validateDiagnostics(value: unknown): { ok: true; value: V3DiagnosticCode[] | undefined } | { ok: false; reason: string } {
  if (value === undefined) return { ok: true, value: undefined };
  if (!Array.isArray(value)) return { ok: false, reason: 'diagnostics must be an array when provided.' };
  const unknown = value.find((item) => !isV3DiagnosticCode(item));
  if (unknown !== undefined) return { ok: false, reason: `Unknown V3 diagnostic code: ${String(unknown)}.` };
  return { ok: true, value };
}

function sanitizeJoints(value: unknown[]): Landmark[] {
  return value.flatMap((item) => {
    if (!isRecord(item) || typeof item.x !== 'number' || typeof item.y !== 'number') return [];
    return {
      x: item.x,
      y: item.y,
      z: typeof item.z === 'number' ? item.z : undefined,
      visibility: typeof item.visibility === 'number' ? item.visibility : undefined
    };
  });
}
