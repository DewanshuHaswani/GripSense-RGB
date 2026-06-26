import { describe, expect, it } from 'vitest';
import { createEmptyAnalysis } from './gripAnalysis';
import { applyV12ProductionGripGate, stabilizeV12DisplayAnalysis, validateV12ObjectInHand, type V12DisplayState } from './v12Grip';
import type { GripAnalysis } from './types';

describe('V12 YOLO production grip gate', () => {
  it('caps an empty hand even when the hand pose looks like a grip', () => {
    const gated = applyV12ProductionGripGate(
      analysisFixture({
        gripPercentage: 67,
        confidence: 0.62,
        objectLockQuality: 0.58,
        closureScore: 0.74,
        mode: 'power grip',
        state: 'Grip detected',
        contact: 0.04,
        openHand: 0.72
      }),
      { objectScore: 0, contact: 0, hasObject: false, missedFrames: 0 }
    );

    expect(gated.gripPercentage).toBe(0);
    expect(gated.guidance).toBe('Object not locked');
    expect(gated.diagnostics.issueCategory).toBe('object_problem');
  });

  it('allows a thumb plus two-finger partial grip when object evidence is real', () => {
    const analysis = analysisFixture({
      gripPercentage: 26,
      confidence: 0.28,
      objectLockQuality: 0.44,
      closureScore: 0.3,
      mode: 'pinch grip',
      state: 'Object uncertain',
      contact: 0.28,
      thumb: 0.72,
      index: 0.62,
      middle: 0.42,
      pinch: 0.64,
      phoneSide: 0.5,
      openHand: 0.18
    });
    const gated = applyV12ProductionGripGate(analysis, { objectScore: 0.68, contact: 0.32, hasObject: true, missedFrames: 0 });

    expect(gated.gripPercentage).toBeGreaterThanOrEqual(42);
    expect(gated.diagnostics.state).toBe('Grip detected');
    expect(gated.guidance).not.toBe('Object not locked');
  });

  it('rejects object-looking evidence when there is no hand corridor contact', () => {
    const gated = applyV12ProductionGripGate(
      analysisFixture({
        gripPercentage: 58,
        confidence: 0.52,
        objectLockQuality: 0.58,
        closureScore: 0.36,
        mode: 'uncertain',
        state: 'Grip detected',
        contact: 0.02,
        thumb: 0.04,
        index: 0.03,
        openHand: 0.28
      }),
      { objectScore: 0.74, contact: 0.03, hasObject: true, missedFrames: 0 }
    );

    expect(gated.gripPercentage).toBeLessThanOrEqual(24);
    expect(gated.diagnostics.state).toBe('Object uncertain');
  });

  it('bridges one short YOLO miss but decays instead of freezing strong grip', () => {
    const previous = analysisFixture({
      gripPercentage: 78,
      confidence: 0.75,
      objectLockQuality: 0.72,
      closureScore: 0.62,
      mode: 'power grip',
      state: 'Strong hold',
      contact: 0.52,
      thumb: 0.68,
      index: 0.62,
      middle: 0.5,
      phoneSide: 0.72,
      openHand: 0.08
    });
    const currentMiss = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      closureScore: 0.6,
      mode: 'power grip',
      state: 'Hand only',
      contact: 0.2,
      thumb: 0.5,
      index: 0.42,
      phoneSide: 0.56,
      openHand: 0.2
    });
    const state: V12DisplayState = { analysis: previous, timestamp: 1000, softLossStartedAt: null };
    const bridged = stabilizeV12DisplayAnalysis(currentMiss, state, 1120, {
      objectScore: 0.3,
      contact: 0.22,
      hasObject: true,
      missedFrames: 1
    });

    expect(bridged.analysis.gripPercentage).toBeGreaterThan(26);
    expect(bridged.analysis.gripPercentage).toBeLessThan(previous.gripPercentage);
    expect(bridged.analysis.guidance).not.toBe('Object not locked');
  });

  it('bridges an edge-on phone angle briefly when the hand still wraps the last stable object', () => {
    const previous = analysisFixture({
      gripPercentage: 84,
      confidence: 0.78,
      objectLockQuality: 0.74,
      closureScore: 0.66,
      mode: 'phone-side grip',
      state: 'Strong hold',
      contact: 0.48,
      thumb: 0.7,
      index: 0.56,
      middle: 0.44,
      phoneSide: 0.78,
      openHand: 0.05
    });
    const edgeOnMiss = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      closureScore: 0.5,
      mode: 'phone-side grip',
      state: 'Hand only',
      contact: 0.12,
      thumb: 0.46,
      index: 0.38,
      middle: 0.26,
      phoneSide: 0.52,
      openHand: 0.18
    });
    const state: V12DisplayState = { analysis: previous, timestamp: 1000, softLossStartedAt: null, lastStableAt: 1000 };
    const bridged = stabilizeV12DisplayAnalysis(edgeOnMiss, state, 1380, {
      objectScore: 0,
      contact: 0,
      hasObject: false,
      missedFrames: 3
    });

    expect(bridged.analysis.gripPercentage).toBeGreaterThan(20);
    expect(bridged.analysis.guidance).not.toBe('Object not locked');
    expect(bridged.analysis.message).toContain('edge-on');
  });

  it('expires the edge-on bridge instead of holding a stale object indefinitely', () => {
    const previous = analysisFixture({
      gripPercentage: 84,
      confidence: 0.78,
      objectLockQuality: 0.74,
      closureScore: 0.66,
      mode: 'phone-side grip',
      state: 'Strong hold',
      contact: 0.48,
      thumb: 0.7,
      index: 0.56,
      phoneSide: 0.78,
      openHand: 0.05
    });
    const current = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      closureScore: 0.5,
      mode: 'phone-side grip',
      state: 'Hand only',
      contact: 0.12,
      thumb: 0.46,
      index: 0.38,
      phoneSide: 0.52,
      openHand: 0.18
    });
    const expired = stabilizeV12DisplayAnalysis(
      current,
      { analysis: previous, timestamp: 1000, softLossStartedAt: 1000, lastStableAt: 1000 },
      2300,
      { objectScore: 0, contact: 0, hasObject: false, missedFrames: 5 }
    );

    expect(expired.analysis.gripPercentage).toBe(0);
    expect(expired.analysis.guidance).toBe('Object not locked');
  });

  it('drops quickly when the object is really gone from the hand', () => {
    const previous = analysisFixture({
      gripPercentage: 78,
      confidence: 0.75,
      objectLockQuality: 0.72,
      closureScore: 0.62,
      mode: 'power grip',
      state: 'Strong hold',
      contact: 0.52,
      thumb: 0.68,
      index: 0.62,
      middle: 0.5,
      phoneSide: 0.72,
      openHand: 0.08
    });
    const current = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      closureScore: 0.18,
      mode: 'open hand',
      state: 'Hand only',
      contact: 0,
      openHand: 0.94
    });
    const dropped = stabilizeV12DisplayAnalysis(
      current,
      { analysis: previous, timestamp: 1000, softLossStartedAt: null },
      1120,
      { objectScore: 0, contact: 0, hasObject: false, missedFrames: 3 }
    );

    expect(dropped.analysis.gripPercentage).toBe(0);
    expect(dropped.analysis.guidance).toBe('Object not locked');
  });

  it('reports server unavailable without synthetic grip confidence', () => {
    const unavailable = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      issueCategory: 'server_unavailable',
      state: 'Hand only'
    });
    const gated = applyV12ProductionGripGate(unavailable, { objectScore: 0, contact: 0, hasObject: false, missedFrames: 0 });

    expect(gated.gripPercentage).toBe(0);
    expect(gated.confidence).toBe(0);
    expect(gated.diagnostics.issueCategory).toBe('server_unavailable');
  });

  it('exposes object-in-hand validation signals', () => {
    const validation = validateV12ObjectInHand(
      analysisFixture({
        gripPercentage: 55,
        contact: 0.32,
        thumb: 0.54,
        index: 0.5,
        pinch: 0.58,
        openHand: 0.12
      }),
      { objectScore: 0.64, contact: 0.34, hasObject: true, missedFrames: 0 }
    );

    expect(validation.objectInHandScore).toBeGreaterThan(validation.emptyHandScore);
    expect(validation.hasPartialGrip).toBe(true);
  });
});

function analysisFixture(options: {
  gripPercentage?: number;
  confidence?: number;
  objectLockQuality?: number;
  closureScore?: number;
  mode?: GripAnalysis['diagnostics']['mode'];
  state?: GripAnalysis['diagnostics']['state'];
  issueCategory?: GripAnalysis['diagnostics']['issueCategory'];
  contact?: number;
  thumb?: number;
  index?: number;
  middle?: number;
  ring?: number;
  pinky?: number;
  pinch?: number;
  phoneSide?: number;
  hook?: number;
  openHand?: number;
} = {}): GripAnalysis {
  const base = createEmptyAnalysis('fixture');
  const contact = options.contact ?? 0;
  const thumb = options.thumb ?? contact;
  const index = options.index ?? contact;
  const middle = options.middle ?? Math.max(0, contact * 0.7);
  return {
    ...base,
    gripPercentage: options.gripPercentage ?? 0,
    confidence: options.confidence ?? 0,
    contactPoints: contact > 0.18 ? 2 : 0,
    closureScore: options.closureScore ?? 0,
    thumbOpposition: thumb,
    enclosureScore: Math.max(contact, options.phoneSide ?? 0),
    objectLockQuality: options.objectLockQuality ?? 0,
    evidence: {
      ...base.evidence,
      fingerSegmentContactScore: contact,
      visibleContactScore: contact,
      objectLockQuality: options.objectLockQuality ?? 0,
      independentObjectScore: options.objectLockQuality ?? 0,
      temporalLockScore: options.objectLockQuality ?? 0,
      pinchScore: options.pinch ?? 0,
      phoneSideGripScore: options.phoneSide ?? 0,
      hookGripScore: options.hook ?? 0,
      contactRoles: {
        thumb,
        index,
        middle,
        ring: options.ring ?? 0,
        pinky: options.pinky ?? 0,
        palm: contact * 0.4
      },
      modeScores: {
        ...base.evidence.modeScores,
        'open hand': options.openHand ?? 0,
        'pinch grip': options.pinch ?? 0,
        'phone-side grip': options.phoneSide ?? 0,
        'hook grip': options.hook ?? 0,
        'power grip': options.mode === 'power grip' ? 0.7 : 0
      }
    },
    diagnostics: {
      ...base.diagnostics,
      mode: options.mode ?? 'uncertain',
      state: options.state ?? 'Object uncertain',
      issueCategory: options.issueCategory ?? 'none'
    },
    guidance:
      (options.gripPercentage ?? 0) >= 70
        ? 'Strong grip'
        : (options.gripPercentage ?? 0) > 0
          ? 'Improve grip'
          : 'Object not locked'
  };
}
