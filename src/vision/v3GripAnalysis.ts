import type {
  GripAnalysis,
  GripDiagnostics,
  Landmark,
  ObjectRegion,
  V3AnalysisDetails,
  V3DiagnosticCode,
  V3PerceptionResponse,
  V3SubScores
} from './types';
import { clamp } from './geometry';

const DEFAULT_STALE_AFTER_MS = 900;
const EMPTY_V3_SUB_SCORES: V3SubScores = {
  objectEvidence: 0,
  handEvidence: 0,
  contactEvidence: 0,
  temporalEvidence: 0
};

type AnalyzeGripV3Options = {
  baseAnalysis: GripAnalysis;
  hand: Landmark[] | null;
  object: ObjectRegion | null;
  response: V3PerceptionResponse | null;
  receivedAt: number | null;
  now: number;
  endpoint: string;
  staleAfterMs?: number;
};

export function analyzeGripV3({
  baseAnalysis,
  hand,
  object,
  response,
  receivedAt,
  now,
  endpoint,
  staleAfterMs = DEFAULT_STALE_AFTER_MS
}: AnalyzeGripV3Options): GripAnalysis {
  if (!hand || !object?.locked) {
    return withV3Fallback(baseAnalysis, endpoint, 'object_uncertain', null, null);
  }

  const serverAgeMs = receivedAt === null ? null : Math.max(0, now - receivedAt);
  const responseAgeMs = response ? Math.max(0, now - response.frameTimestamp) : null;
  if (!response || receivedAt === null) {
    return withV3Fallback(baseAnalysis, endpoint, 'server_unavailable', null, null);
  }
  if (
    serverAgeMs === null ||
    serverAgeMs > staleAfterMs ||
    responseAgeMs === null ||
    responseAgeMs > staleAfterMs * 1.6
  ) {
    return withV3Fallback(baseAnalysis, endpoint, 'server_unavailable', response, serverAgeMs);
  }

  const subScores = computeV3SubScores(response);
  const modelConfidence = clamp(1 - response.uncertainty);
  if (modelConfidence < 0.42 || response.hand.meshQuality < 0.18 || response.object.maskConfidence < 0.12) {
    const fallbackReason = response.diagnostics?.[0] ?? (response.hand.meshQuality < 0.18 ? 'hand_occluded' : 'server_unavailable');
    return withV3Fallback(baseAnalysis, endpoint, fallbackReason, response, serverAgeMs);
  }

  const issue = chooseV3Issue(response, subScores);
  const serverScore = computeServerGripScore(response, subScores);
  const trust = clamp(modelConfidence * 0.5 + subScores.objectEvidence * 0.16 + subScores.handEvidence * 0.14 + subScores.temporalEvidence * 0.2);
  const fusedScore = clamp(serverScore * Math.max(0.56, trust) + (baseAnalysis.gripPercentage / 100) * (1 - Math.max(0.56, trust)));
  const gripPercentage = capGripPercentage(issue, response, Math.round(fusedScore * 100));
  const guidance = issue === 'strong_hold'
    ? 'Strong grip'
    : issue === 'object_uncertain'
      ? 'Object uncertain'
      : issue === 'contact_uncertain' || issue === 'hand_occluded'
        ? 'Reposition'
        : issue === 'slip_risk'
          ? 'Improve grip'
          : gripPercentage >= 70
            ? 'Strong grip'
            : gripPercentage >= 44
              ? 'Improve grip'
              : 'Reposition';
  const state = issue === 'strong_hold'
    ? 'Strong hold'
    : issue === 'slip_risk'
      ? 'Slip risk'
      : issue === 'object_uncertain'
        ? 'Object uncertain'
        : gripPercentage >= 44
          ? 'Grip detected'
          : 'Object uncertain';
  const contactPoints = countV3Contacts(response);
  const v3: V3AnalysisDetails = {
    status: 'server',
    reason: issue,
    usedServerResult: true,
    endpoint,
    serverLatencyMs: response.latencyMs,
    serverAgeMs,
    modelConfidence,
    uncertainty: response.uncertainty,
    subScores,
    diagnostics: uniqueDiagnostics([issue, ...(response.diagnostics ?? [])])
  };

  const diagnostics: GripDiagnostics = {
    ...baseAnalysis.diagnostics,
    state,
    recommendation: recommendV3(issue, gripPercentage),
    objectIssue: issue === 'object_uncertain' ? 'V3 object mask or pose is uncertain. Relock the object or improve object visibility.' : null,
    gripIssue:
      issue === 'hand_occluded'
        ? 'V3 hand mesh is partially occluded. Keep fingers and thumb visible.'
        : issue === 'contact_uncertain'
          ? 'V3 contact map does not show enough opposing finger or palm support.'
          : issue === 'slip_risk'
            ? 'V3 temporal tracking sees hand-object drift that may indicate slipping.'
            : null,
    issueCategory: issue,
    scoreBreakdown: createV3Breakdown(baseAnalysis, response, subScores, modelConfidence)
  };

  const evidence = {
    ...baseAnalysis.evidence,
    contactRoles: {
      thumb: response.contact.thumb,
      index: response.contact.index,
      middle: response.contact.middle,
      ring: response.contact.ring,
      pinky: response.contact.pinky,
      palm: response.contact.palm
    },
    fingerSegmentContactScore: Math.max(baseAnalysis.evidence.fingerSegmentContactScore, subScores.contactEvidence),
    objectLockQuality: subScores.objectEvidence,
    visibleContactScore: Math.max(baseAnalysis.evidence.visibleContactScore, response.contact.coverage),
    thumbSupportScore: Math.max(baseAnalysis.evidence.thumbSupportScore, response.contact.thumb),
    motionStabilityScore: subScores.temporalEvidence,
    persistentSlipScore: Math.max(baseAnalysis.evidence.persistentSlipScore, response.temporal.slipRisk),
    positiveReasons: createV3PositiveReasons(issue, subScores, baseAnalysis.evidence.positiveReasons),
    negativeReasons: createV3NegativeReasons(issue, baseAnalysis.evidence.negativeReasons)
  };

  return {
    ...baseAnalysis,
    gripPercentage,
    confidence: clamp(baseAnalysis.confidence * 0.36 + modelConfidence * 0.3 + trust * 0.34),
    contactPoints,
    thumbOpposition: Math.max(baseAnalysis.thumbOpposition, response.contact.thumb),
    motionCoupling: response.temporal.coupling,
    slipRisk: Math.max(baseAnalysis.slipRisk, response.temporal.slipRisk),
    motionState: issue === 'slip_risk' ? 'slipping' : baseAnalysis.motionState,
    guidance,
    message: diagnostics.recommendation,
    objectLockQuality: subScores.objectEvidence,
    evidence,
    diagnostics,
    v3
  };
}

export function computeV3SubScores(response: V3PerceptionResponse): V3SubScores {
  const contactAverage =
    response.contact.thumb * 0.22 +
    response.contact.index * 0.18 +
    response.contact.middle * 0.16 +
    response.contact.ring * 0.12 +
    response.contact.pinky * 0.08 +
    response.contact.palm * 0.14 +
    response.contact.coverage * 0.1;

  return {
    objectEvidence: clamp(
      (response.object.present ? 0.08 : 0) +
        response.object.maskConfidence * 0.3 +
        response.object.maskStability * 0.24 +
        response.object.poseConfidence * 0.18 +
        response.object.identityConfidence * 0.1 +
        response.object.lockConfidence * 0.1
    ),
    handEvidence: clamp(
      response.hand.meshQuality * 0.38 +
        (1 - response.hand.occlusion) * 0.24 +
        response.hand.handednessConfidence * 0.16 +
        response.hand.fingerArticulation * 0.22
    ),
    contactEvidence: clamp(contactAverage * 0.58 + response.contact.coverage * 0.24 + response.contact.opposingPairs * 0.18),
    temporalEvidence: clamp(
      response.temporal.continuity * 0.32 +
        response.temporal.coupling * 0.26 +
        (1 - response.temporal.slipRisk) * 0.24 +
        (1 - response.temporal.jitter) * 0.18
    )
  };
}

function withV3Fallback(
  baseAnalysis: GripAnalysis,
  endpoint: string,
  reason: V3DiagnosticCode,
  response: V3PerceptionResponse | null,
  serverAgeMs: number | null
): GripAnalysis {
  const subScores = response ? computeV3SubScores(response) : EMPTY_V3_SUB_SCORES;
  const modelConfidence = response ? clamp(1 - response.uncertainty) : 0;
  return {
    ...baseAnalysis,
    confidence: Math.min(baseAnalysis.confidence, reason === 'object_uncertain' ? baseAnalysis.confidence : 0.62),
    message:
      reason === 'server_unavailable'
        ? `V3 server unavailable or stale; showing V2 fallback. ${baseAnalysis.message}`
        : recommendV3(reason, baseAnalysis.gripPercentage),
    diagnostics: {
      ...baseAnalysis.diagnostics,
      recommendation:
        reason === 'server_unavailable'
          ? `V3 server unavailable or stale; showing V2 fallback. ${baseAnalysis.message}`
          : recommendV3(reason, baseAnalysis.gripPercentage),
      objectIssue:
        reason === 'object_uncertain'
          ? 'V3 object mask or pose is uncertain. Relock the object or improve object visibility.'
          : baseAnalysis.diagnostics.objectIssue,
      gripIssue: fallbackGripIssue(reason, baseAnalysis.diagnostics.gripIssue),
      issueCategory: reason,
      scoreBreakdown: [
        ...baseAnalysis.diagnostics.scoreBreakdown,
        {
          label: reason === 'server_unavailable' ? 'V3 server' : 'V3 fallback',
          value: reason === 'server_unavailable' ? 0 : modelConfidence,
          impact: reason === 'server_unavailable' ? 'negative' : 'neutral'
        }
      ]
    },
    v3: {
      status: 'fallback',
      reason,
      usedServerResult: false,
      endpoint,
      serverLatencyMs: response?.latencyMs ?? null,
      serverAgeMs,
      modelConfidence,
      uncertainty: response?.uncertainty ?? 1,
      subScores,
      diagnostics: uniqueDiagnostics([reason, ...(response?.diagnostics ?? [])])
    }
  };
}

function fallbackGripIssue(reason: V3DiagnosticCode, previous: string | null) {
  if (reason === 'hand_occluded') return 'V3 hand mesh is partially occluded. Keep fingers and thumb visible.';
  if (reason === 'contact_uncertain') return 'V3 contact map does not show enough opposing finger or palm support.';
  if (reason === 'slip_risk') return 'V3 temporal tracking sees hand-object drift that may indicate slipping.';
  return previous;
}

function chooseV3Issue(response: V3PerceptionResponse, subScores: V3SubScores): V3DiagnosticCode {
  if (!response.object.present || subScores.objectEvidence < 0.4) return 'object_uncertain';
  if (response.hand.occlusion > 0.68 || subScores.handEvidence < 0.42) return 'hand_occluded';
  if (response.temporal.slipRisk > 0.56 || subScores.temporalEvidence < 0.38) return 'slip_risk';
  if (subScores.contactEvidence < 0.42 || response.contact.opposingPairs < 0.3) return 'contact_uncertain';
  if (computeServerGripScore(response, subScores) >= 0.72) return 'strong_hold';
  return 'contact_uncertain';
}

function computeServerGripScore(response: V3PerceptionResponse, subScores: V3SubScores) {
  return clamp(
    subScores.objectEvidence * 0.25 +
      subScores.handEvidence * 0.2 +
      subScores.contactEvidence * 0.35 +
      subScores.temporalEvidence * 0.2 -
      response.temporal.slipRisk * 0.08
  );
}

function capGripPercentage(issue: V3DiagnosticCode, response: V3PerceptionResponse, gripPercentage: number) {
  if (issue === 'object_uncertain') return response.object.present ? Math.min(gripPercentage, 28) : 0;
  if (issue === 'hand_occluded') return Math.min(gripPercentage, 36);
  if (issue === 'contact_uncertain') return Math.min(gripPercentage, 46);
  if (issue === 'slip_risk') return Math.min(Math.max(gripPercentage, 38), 68);
  return gripPercentage;
}

function countV3Contacts(response: V3PerceptionResponse) {
  const roles = [
    response.contact.thumb,
    response.contact.index,
    response.contact.middle,
    response.contact.ring,
    response.contact.pinky,
    response.contact.palm
  ];
  return Math.min(5, roles.filter((score) => score > 0.46).length + (response.contact.opposingPairs > 0.54 ? 1 : 0));
}

function createV3Breakdown(
  baseAnalysis: GripAnalysis,
  response: V3PerceptionResponse,
  subScores: V3SubScores,
  modelConfidence: number
) {
  return [
    { label: 'Object evidence', value: subScores.objectEvidence, impact: subScores.objectEvidence < 0.4 ? 'negative' : 'positive' },
    { label: '3D hand evidence', value: subScores.handEvidence, impact: subScores.handEvidence < 0.42 ? 'negative' : 'positive' },
    { label: 'Contact map', value: subScores.contactEvidence, impact: subScores.contactEvidence < 0.42 ? 'negative' : 'positive' },
    { label: 'Temporal consistency', value: subScores.temporalEvidence, impact: response.temporal.slipRisk > 0.56 ? 'negative' : 'positive' },
    { label: 'V2 fallback signal', value: baseAnalysis.gripPercentage / 100, impact: 'neutral' },
    { label: 'Model confidence', value: modelConfidence, impact: modelConfidence < 0.42 ? 'negative' : 'positive' }
  ] as GripDiagnostics['scoreBreakdown'];
}

function recommendV3(issue: V3DiagnosticCode, gripPercentage: number) {
  if (issue === 'strong_hold') return 'V3 sees stable object mask, 3D hand evidence, contact, and temporal coupling.';
  if (issue === 'object_uncertain') return 'V3 cannot verify the object mask or pose. Relock the object or improve visibility.';
  if (issue === 'hand_occluded') return 'V3 hand mesh is occluded. Rotate slightly so thumb and fingers are visible.';
  if (issue === 'slip_risk') return 'V3 sees hand-object drift. Hold steady or relock if the object outline is drifting.';
  if (issue === 'server_unavailable') return 'V3 server is unavailable; V2 fallback is active.';
  if (gripPercentage >= 44) return 'V3 contact is usable, but opposing contact can improve.';
  return 'V3 contact map is weak. Reposition the object between thumb and fingers.';
}

function createV3PositiveReasons(issue: V3DiagnosticCode, subScores: V3SubScores, previous: string[]) {
  const reasons = [...previous];
  if (issue === 'strong_hold') reasons.unshift('V3 fused mask, mesh, contact, and temporal evidence');
  if (subScores.objectEvidence > 0.64) reasons.push('V3 object mask is stable');
  if (subScores.contactEvidence > 0.58) reasons.push('V3 contact map has opposing support');
  return Array.from(new Set(reasons)).slice(0, 5);
}

function createV3NegativeReasons(issue: V3DiagnosticCode, previous: string[]) {
  const reasons = [...previous];
  if (issue === 'object_uncertain') reasons.unshift('V3 object mask or pose is uncertain');
  if (issue === 'hand_occluded') reasons.unshift('V3 hand mesh is occluded');
  if (issue === 'contact_uncertain') reasons.unshift('V3 contact map is weak');
  if (issue === 'slip_risk') reasons.unshift('V3 temporal drift suggests slip');
  return Array.from(new Set(reasons)).slice(0, 5);
}

function uniqueDiagnostics(values: Array<V3DiagnosticCode | null | undefined>) {
  return Array.from(new Set(values.filter(Boolean))) as V3DiagnosticCode[];
}
