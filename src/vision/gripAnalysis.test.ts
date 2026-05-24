import { describe, expect, it } from 'vitest';
import { analyzeGrip } from './gripAnalysis';
import { analyzeGripV3 } from './v3GripAnalysis';
import { parseV3PerceptionResponse } from './v3Inference';
import type { Landmark, ObjectRegion, V3PerceptionResponse } from './types';

describe('analyzeGrip', () => {
  it('returns no score when a hand is open with no object', () => {
    const analysis = analyzeGrip(openHand(), null, null);
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.guidance).toBe('Object not locked');
  });

  it('does not report a strong grip for a closed hand without an object', () => {
    const analysis = analyzeGrip(closedGripHand(), null, null);
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.guidance).toBe('Object not locked');
  });

  it('reports weak or reposition guidance when contact points are poor', () => {
    const analysis = analyzeGrip(openHand(), objectAt(520, 300, 0.82), null);
    expect(analysis.gripPercentage).toBeLessThan(46);
    expect(analysis.guidance).toBe('Reposition');
  });

  it('reports stronger grip when the object has multiple contacts and coupled motion', () => {
    const hand = closedGripHand();
    const previousPalm = { x: 320.6, y: 341.6 };
    const object = objectAt(356, 322, 0.9, { x: 5, y: -2 });
    const analysis = analyzeGrip(hand, object, previousPalm);
    expect(analysis.contactPoints).toBeGreaterThanOrEqual(2);
    expect(analysis.gripPercentage).toBeGreaterThanOrEqual(55);
    expect(analysis.guidance).not.toBe('Object not locked');
  });

  it('falls back when object confidence is low', () => {
    const analysis = analyzeGrip(closedGripHand(), objectAt(356, 322, 0.2), null);
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.guidance).toBe('Object not locked');
  });

  it('scores a tight phone-style grip as strong even with limited visible fingertip contacts', () => {
    const hand = phoneGripHand();
    const analysis = analyzeGrip(hand, phoneObjectAt(350, 316, 0.88), { x: 326, y: 346 }, { persistentSlipScore: 0.04 });
    expect(analysis.evidence.phoneSideGripScore).toBeGreaterThan(0.45);
    expect(analysis.evidence.fingerCurlScore).toBeGreaterThan(0.5);
    expect(analysis.gripPercentage).toBeGreaterThanOrEqual(62);
    expect(analysis.guidance).not.toBe('Reposition');
  });

  it('keeps an idle held object out of slipping state', () => {
    const analysis = analyzeGrip(phoneGripHand(), phoneObjectAt(350, 316, 0.88), null, { persistentSlipScore: 0.9 });
    expect(analysis.motionState).toBe('idle');
    expect(analysis.slipRisk).toBeLessThan(0.12);
  });

  it('reports slipping only when persistent relative motion is high', () => {
    const movingObject = phoneObjectAt(350, 316, 0.88, { x: 18, y: -10 });
    const analysis = analyzeGrip(phoneGripHand(), movingObject, { x: 326, y: 346 }, { persistentSlipScore: 0.72 });
    expect(analysis.motionState).toBe('slipping');
    expect(analysis.slipRisk).toBeGreaterThan(0.45);
  });

  it('uses calibration to lift a matching strong-hold pose', () => {
    const hand = phoneGripHand();
    const uncalibrated = analyzeGrip(hand, phoneObjectAt(350, 316, 0.7), { x: 326, y: 346 });
    const calibrated = analyzeGrip(hand, phoneObjectAt(350, 316, 0.7), { x: 326, y: 346 }, {
      calibrationBaseline: {
        mode: uncalibrated.diagnostics.mode,
        gripPercentage: 84,
        closureScore: uncalibrated.closureScore,
        enclosureScore: uncalibrated.enclosureScore,
        fingerCurlScore: uncalibrated.evidence.fingerCurlScore,
        fingerSegmentContactScore: uncalibrated.evidence.fingerSegmentContactScore,
        phoneSideGripScore: uncalibrated.evidence.phoneSideGripScore,
        pinchScore: uncalibrated.evidence.pinchScore,
        powerGripScore: uncalibrated.evidence.powerGripScore,
        thumbSupportScore: uncalibrated.evidence.thumbSupportScore,
        objectLockQuality: uncalibrated.objectLockQuality,
        createdAt: Date.now()
      }
    });
    expect(calibrated.calibrated).toBe(true);
    expect(calibrated.gripPercentage).toBeGreaterThan(uncalibrated.gripPercentage);
  });

  it('classifies a bottle-like wrap as a power grip', () => {
    const analysis = analyzeGrip(closedGripHand(), bottleObjectAt(354, 334, 0.86), null, { persistentSlipScore: 0.02 });
    expect(analysis.diagnostics.mode).toBe('power grip');
    expect(analysis.gripPercentage).toBeGreaterThan(45);
  });

  it('classifies a small thumb-index hold as a pinch grip', () => {
    const analysis = analyzeGrip(pinchGripHand(), smallObjectAt(306, 310, 0.88), null, { persistentSlipScore: 0.01 });
    expect(analysis.diagnostics.mode).toBe('pinch grip');
    expect(analysis.guidance).not.toBe('Object not locked');
  });

  it('keeps stable moving objects out of slip risk', () => {
    const movingObject = phoneObjectAt(350, 316, 0.88, { x: 3, y: -1 });
    const analysis = analyzeGrip(phoneGripHand(), movingObject, { x: 323, y: 347 }, { persistentSlipScore: 0.04 });
    expect(analysis.motionState).not.toBe('slipping');
    expect(analysis.diagnostics.state).not.toBe('Slip risk');
  });

  it('marks a bad object lock as object uncertain instead of weak grip', () => {
    const object = phoneObjectAt(620, 120, 0.38);
    object.tightness = 0.05;
    const analysis = analyzeGrip(phoneGripHand(), object, null);
    expect(analysis.diagnostics.state).toBe('Object uncertain');
    expect(analysis.diagnostics.objectIssue).toContain('Object lock');
  });

  it('rejects a hallucinated automatic object inside an open empty hand', () => {
    const object = objectAt(304, 304, 0.72);
    object.source = 'automatic';
    object.tightness = 0.8;
    object.lockAgeFrames = 100;
    const analysis = analyzeGrip(openHand(), object, null);
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.diagnostics.state).toBe('Hand only');
    expect(analysis.diagnostics.objectIssue).toContain('Automatic lock rejected');
  });

  it('v2 rejects a confident automatic lock when an open hand has no independent object evidence', () => {
    const object = objectAt(304, 304, 0.9);
    object.source = 'automatic';
    object.tightness = 0.86;
    object.lockAgeFrames = 420;
    object.independentEvidenceScore = 0.18;
    object.visualEdgeScore = 0.12;
    object.visualTextureScore = 0.08;
    const analysis = analyzeGrip(openHand(), object, null, { algorithmVersion: 'v2' });
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.diagnostics.state).toBe('Hand only');
    expect(analysis.diagnostics.objectIssue).toContain('Open hand');
  });

  it('v2 keeps detector-backed phone grips strong when object evidence is real', () => {
    const object = phoneObjectAt(350, 316, 0.88);
    object.independentEvidenceScore = 0.82;
    object.visualEdgeScore = 0.68;
    object.visualTextureScore = 0.44;
    const analysis = analyzeGrip(phoneGripHand(), object, { x: 326, y: 346 }, {
      persistentSlipScore: 0.03,
      algorithmVersion: 'v2'
    });
    expect(analysis.diagnostics.state).not.toBe('Hand only');
    expect(analysis.evidence.independentObjectScore).toBeGreaterThan(0.7);
    expect(analysis.gripPercentage).toBeGreaterThanOrEqual(55);
  });

  it('v2 blocks a strong grip when the trained object identity does not match', () => {
    const object = phoneObjectAt(350, 316, 0.88);
    const analysis = analyzeGrip(phoneGripHand(), object, { x: 326, y: 346 }, {
      persistentSlipScore: 0.03,
      algorithmVersion: 'v2',
      objectIdentity: {
        hasProfiles: true,
        score: 0.24,
        matched: false,
        name: null
      }
    });
    expect(analysis.diagnostics.state).toBe('Object uncertain');
    expect(analysis.guidance).toBe('Object uncertain');
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.diagnostics.objectIssue).toContain('Trained object not found');
  });

  it('v2 still uses generic object-first scoring when no profile is trained', () => {
    const object = phoneObjectAt(350, 316, 0.88);
    const analysis = analyzeGrip(phoneGripHand(), object, { x: 326, y: 346 }, {
      persistentSlipScore: 0.03,
      algorithmVersion: 'v2',
      objectIdentity: {
        hasProfiles: false,
        score: 0,
        matched: false,
        name: null
      }
    });
    expect(analysis.hasObjectProfiles).toBe(false);
    expect(analysis.diagnostics.state).not.toBe('Object uncertain');
    expect(analysis.gripPercentage).toBeGreaterThanOrEqual(55);
  });

  it('v2 reports object uncertain rather than weak grip for loose automatic locks', () => {
    const object = bottleObjectAt(354, 334, 0.52);
    object.source = 'automatic';
    object.tightness = 0.22;
    object.independentEvidenceScore = 0.5;
    const analysis = analyzeGrip(closedGripHand(), object, null, { algorithmVersion: 'v2' });
    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.diagnostics.state).toBe('Object uncertain');
    expect(analysis.diagnostics.objectIssue).toContain('loose');
  });

  it('reports contact-role evidence for a power grip', () => {
    const analysis = analyzeGrip(closedGripHand(), bottleObjectAt(354, 334, 0.86), null, { persistentSlipScore: 0.02 });
    expect(analysis.evidence.contactRoles.thumb).toBeGreaterThan(0.2);
    expect(analysis.evidence.contactRoles.index).toBeGreaterThan(0.2);
    expect(analysis.diagnostics.scoreBreakdown.some((item) => item.label === 'Contact roles')).toBe(true);
  });

  it('keeps mirrored phone grip scoring in the same range', () => {
    const normal = analyzeGrip(phoneGripHand(), phoneObjectAt(350, 316, 0.88), { x: 326, y: 346 }, { persistentSlipScore: 0.03 });
    const mirrored = analyzeGrip(mirrorHand(phoneGripHand(), 700), mirrorObject(phoneObjectAt(350, 316, 0.88), 700), { x: 374, y: 346 }, { persistentSlipScore: 0.03 });
    expect(Math.abs(normal.gripPercentage - mirrored.gripPercentage)).toBeLessThanOrEqual(12);
    expect(mirrored.guidance).not.toBe('Object not locked');
  });

  it('uses relative object drift as slip evidence even when frame motion is modest', () => {
    const object = phoneObjectAt(350, 316, 0.88, { x: 1, y: 0 });
    object.relativeDriftScore = 0.8;
    const analysis = analyzeGrip(phoneGripHand(), object, { x: 317, y: 382 }, {
      persistentSlipScore: 0.1,
      algorithmVersion: 'v2'
    });
    expect(analysis.slipRisk).toBeGreaterThan(0.14);
    expect(analysis.diagnostics.scoreBreakdown.some((item) => item.label === 'Motion stability')).toBe(true);
  });

  it('categorizes identity and pose problems separately', () => {
    const identityBlocked = analyzeGrip(phoneGripHand(), phoneObjectAt(350, 316, 0.88), { x: 326, y: 346 }, {
      algorithmVersion: 'v2',
      objectIdentity: {
        hasProfiles: true,
        score: 0.1,
        matched: false,
        name: null
      }
    });
    const poseBlocked = analyzeGrip(openHand(), objectAt(520, 300, 0.82), null);
    expect(identityBlocked.diagnostics.issueCategory).toBe('identity_problem');
    expect(poseBlocked.diagnostics.issueCategory).toBe('pose_problem');
  });

  it('v3 falls back to V2 when the local server is unavailable', () => {
    const hand = phoneGripHand();
    const object = phoneObjectAt(350, 316, 0.88);
    const base = analyzeGrip(hand, object, { x: 326, y: 346 }, { algorithmVersion: 'v2' });
    const analysis = analyzeGripV3({
      baseAnalysis: base,
      hand,
      object,
      response: null,
      receivedAt: null,
      now: 1200,
      endpoint: 'http://127.0.0.1:7867/v3/analyze-frame'
    });

    expect(analysis.gripPercentage).toBe(base.gripPercentage);
    expect(analysis.v3?.status).toBe('fallback');
    expect(analysis.v3?.reason).toBe('server_unavailable');
    expect(analysis.diagnostics.issueCategory).toBe('server_unavailable');
  });

  it('v3 refuses to report a strong hold when the server sees no object', () => {
    const hand = phoneGripHand();
    const object = phoneObjectAt(350, 316, 0.88);
    const base = analyzeGrip(hand, object, { x: 326, y: 346 }, { algorithmVersion: 'v2' });
    const analysis = analyzeGripV3({
      baseAnalysis: base,
      hand,
      object,
      response: v3Response({ object: { present: false } }),
      receivedAt: 1180,
      now: 1200,
      endpoint: 'local'
    });

    expect(analysis.gripPercentage).toBe(0);
    expect(analysis.guidance).toBe('Object uncertain');
    expect(analysis.diagnostics.issueCategory).toBe('object_uncertain');
  });

  it('v3 reports strong hold only when object, hand, contact, and temporal evidence agree', () => {
    const hand = phoneGripHand();
    const object = phoneObjectAt(350, 316, 0.88);
    const base = analyzeGrip(hand, object, { x: 326, y: 346 }, { algorithmVersion: 'v2' });
    const analysis = analyzeGripV3({
      baseAnalysis: base,
      hand,
      object,
      response: v3Response(),
      receivedAt: 1180,
      now: 1200,
      endpoint: 'local'
    });

    expect(analysis.v3?.usedServerResult).toBe(true);
    expect(analysis.diagnostics.issueCategory).toBe('strong_hold');
    expect(analysis.guidance).toBe('Strong grip');
    expect(analysis.gripPercentage).toBeGreaterThanOrEqual(70);
  });

  it('v3 falls back when server evidence is stale or hand mesh confidence is too low', () => {
    const hand = phoneGripHand();
    const object = phoneObjectAt(350, 316, 0.88);
    const base = analyzeGrip(hand, object, { x: 326, y: 346 }, { algorithmVersion: 'v2' });
    const stale = analyzeGripV3({
      baseAnalysis: base,
      hand,
      object,
      response: v3Response(),
      receivedAt: 0,
      now: 2400,
      endpoint: 'local'
    });
    const lowMesh = analyzeGripV3({
      baseAnalysis: base,
      hand,
      object,
      response: v3Response({ hand: { meshQuality: 0.12 }, diagnostics: ['hand_occluded'] }),
      receivedAt: 1180,
      now: 1200,
      endpoint: 'local'
    });

    expect(stale.v3?.status).toBe('fallback');
    expect(stale.v3?.reason).toBe('server_unavailable');
    expect(lowMesh.v3?.status).toBe('fallback');
    expect(lowMesh.v3?.reason).toBe('hand_occluded');
    expect(lowMesh.diagnostics.issueCategory).toBe('hand_occluded');
    expect(lowMesh.diagnostics.gripIssue).toContain('hand mesh');
  });

  it('validates the V3 server response contract', () => {
    expect(parseV3PerceptionResponse(v3Response()).ok).toBe(true);
    expect(parseV3PerceptionResponse({ ...v3Response(), object: { present: true } }).ok).toBe(false);
    expect(parseV3PerceptionResponse({ ...v3Response(), uncertainty: 1.4 }).ok).toBe(false);
    expect(parseV3PerceptionResponse({ ...v3Response(), diagnostics: ['new_server_code'] }).ok).toBe(false);
  });
});

function baseHand(): Landmark[] {
  return Array.from({ length: 21 }, () => ({ x: 300, y: 340 }));
}

function openHand(): Landmark[] {
  const hand = baseHand();
  hand[0] = { x: 300, y: 380 };
  hand[4] = { x: 190, y: 270 };
  hand[5] = { x: 260, y: 330 };
  hand[8] = { x: 245, y: 185 };
  hand[9] = { x: 300, y: 320 };
  hand[12] = { x: 300, y: 165 };
  hand[13] = { x: 340, y: 326 };
  hand[16] = { x: 358, y: 190 };
  hand[17] = { x: 378, y: 342 };
  hand[20] = { x: 416, y: 238 };
  return hand;
}

function closedGripHand(): Landmark[] {
  const hand = baseHand();
  hand[0] = { x: 314, y: 362 };
  hand[4] = { x: 315, y: 318 };
  hand[5] = { x: 288, y: 335 };
  hand[8] = { x: 370, y: 302 };
  hand[9] = { x: 318, y: 323 };
  hand[12] = { x: 382, y: 330 };
  hand[13] = { x: 342, y: 330 };
  hand[16] = { x: 370, y: 360 };
  hand[17] = { x: 366, y: 348 };
  hand[20] = { x: 342, y: 372 };
  return hand;
}

function phoneGripHand(): Landmark[] {
  const hand = baseHand();
  hand[0] = { x: 318, y: 382 };
  hand[4] = { x: 306, y: 322 };
  hand[5] = { x: 292, y: 346 };
  hand[6] = { x: 326, y: 326 };
  hand[7] = { x: 358, y: 312 };
  hand[8] = { x: 384, y: 306 };
  hand[9] = { x: 320, y: 342 };
  hand[10] = { x: 350, y: 326 };
  hand[11] = { x: 378, y: 324 };
  hand[12] = { x: 398, y: 332 };
  hand[13] = { x: 344, y: 352 };
  hand[14] = { x: 370, y: 344 };
  hand[15] = { x: 388, y: 354 };
  hand[16] = { x: 394, y: 368 };
  hand[17] = { x: 368, y: 366 };
  hand[18] = { x: 388, y: 366 };
  hand[19] = { x: 394, y: 378 };
  hand[20] = { x: 386, y: 390 };
  return hand;
}

function pinchGripHand(): Landmark[] {
  const hand = openHand();
  hand[0] = { x: 300, y: 370 };
  hand[4] = { x: 294, y: 306 };
  hand[8] = { x: 318, y: 312 };
  hand[12] = { x: 360, y: 250 };
  hand[16] = { x: 392, y: 268 };
  hand[20] = { x: 420, y: 292 };
  return hand;
}

function objectAt(x: number, y: number, confidence: number, velocity = { x: 0, y: 0 }): ObjectRegion {
  return {
    center: { x, y },
    radiusX: 48,
    radiusY: 34,
    angle: -0.2,
    confidence,
    locked: true,
    source: 'automatic',
    velocity,
    contour: []
  };
}

function phoneObjectAt(x: number, y: number, confidence: number, velocity = { x: 0, y: 0 }): ObjectRegion {
  return {
    center: { x, y },
    radiusX: 44,
    radiusY: 112,
    angle: -0.35,
    confidence,
    locked: true,
    source: 'detector',
    velocity,
    contour: [],
    shape: 'phone-like',
    aspectRatio: 2.55,
    tightness: 0.82,
    lockAgeFrames: 20,
    visualEdgeScore: 0.68,
    visualTextureScore: 0.42,
    independentEvidenceScore: 0.82
  };
}

function bottleObjectAt(x: number, y: number, confidence: number, velocity = { x: 0, y: 0 }): ObjectRegion {
  return {
    center: { x, y },
    radiusX: 38,
    radiusY: 68,
    angle: 0.1,
    confidence,
    locked: true,
    source: 'automatic',
    velocity,
    contour: [],
    shape: 'ellipse',
    aspectRatio: 1.78,
    tightness: 0.8,
    lockAgeFrames: 16,
    visualEdgeScore: 0.52,
    visualTextureScore: 0.35,
    independentEvidenceScore: 0.56
  };
}

function smallObjectAt(x: number, y: number, confidence: number, velocity = { x: 0, y: 0 }): ObjectRegion {
  return {
    center: { x, y },
    radiusX: 16,
    radiusY: 14,
    angle: 0,
    confidence,
    locked: true,
    source: 'manual',
    velocity,
    contour: [],
    shape: 'unknown',
    aspectRatio: 1.14,
    tightness: 0.78,
    lockAgeFrames: 12,
    manuallyAdjusted: true
  };
}

function mirrorHand(hand: Landmark[], width: number): Landmark[] {
  return hand.map((point) => ({ ...point, x: width - point.x }));
}

function mirrorObject(object: ObjectRegion, width: number): ObjectRegion {
  return {
    ...object,
    center: { x: width - object.center.x, y: object.center.y },
    angle: -object.angle,
    velocity: { x: -object.velocity.x, y: object.velocity.y },
    contour: object.contour.map((point) => ({ x: width - point.x, y: point.y }))
  };
}

function v3Response(overrides: {
  frameTimestamp?: number;
  latencyMs?: number;
  uncertainty?: number;
  hand?: Partial<V3PerceptionResponse['hand']>;
  object?: Partial<V3PerceptionResponse['object']>;
  contact?: Partial<V3PerceptionResponse['contact']>;
  temporal?: Partial<V3PerceptionResponse['temporal']>;
  diagnostics?: V3PerceptionResponse['diagnostics'];
} = {}): V3PerceptionResponse {
  return {
    version: 'v3',
    frameTimestamp: overrides.frameTimestamp ?? 1200,
    latencyMs: overrides.latencyMs ?? 42,
    uncertainty: overrides.uncertainty ?? 0.12,
    hand: {
      meshQuality: 0.86,
      occlusion: 0.12,
      handednessConfidence: 0.92,
      fingerArticulation: 0.82,
      ...overrides.hand
    },
    object: {
      present: true,
      maskConfidence: 0.88,
      maskStability: 0.84,
      identityConfidence: 0.8,
      poseConfidence: 0.82,
      lockConfidence: 0.86,
      ...overrides.object
    },
    contact: {
      thumb: 0.82,
      index: 0.78,
      middle: 0.74,
      ring: 0.62,
      pinky: 0.44,
      palm: 0.68,
      coverage: 0.78,
      opposingPairs: 0.84,
      ...overrides.contact
    },
    temporal: {
      continuity: 0.86,
      coupling: 0.84,
      slipRisk: 0.08,
      jitter: 0.12,
      ...overrides.temporal
    },
    diagnostics: overrides.diagnostics
  };
}
