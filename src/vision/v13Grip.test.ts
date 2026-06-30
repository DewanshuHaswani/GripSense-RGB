import { describe, expect, it } from 'vitest';
import { createEmptyAnalysis } from './gripAnalysis';
import {
  createInitialV13DisplayState,
  stabilizeV13DisplayAnalysis,
  stabilizeV13VisualObject,
  type V13DisplayState
} from './v13Grip';
import type { GripAnalysis, ObjectRegion } from './types';

describe('V13 YOLO stable live mode', () => {
  it('bridges a longer hidden-object miss when the hand is still wrapped around the last stable object', () => {
    const previous = analysisFixture({
      gripPercentage: 82,
      confidence: 0.78,
      objectLockQuality: 0.76,
      closureScore: 0.66,
      mode: 'phone-side grip',
      state: 'Strong hold',
      contact: 0.5,
      thumb: 0.72,
      index: 0.58,
      middle: 0.44,
      phoneSide: 0.78,
      openHand: 0.06
    });
    const currentMiss = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      closureScore: 0.5,
      mode: 'phone-side grip',
      state: 'Hand only',
      contact: 0.1,
      thumb: 0.46,
      index: 0.38,
      middle: 0.24,
      phoneSide: 0.5,
      openHand: 0.18
    });
    const state: V13DisplayState = {
      ...createInitialV13DisplayState(),
      analysis: previous,
      timestamp: 1000,
      lastStableAt: 1000
    };

    const bridged = stabilizeV13DisplayAnalysis(currentMiss, state, 1260, {
      objectScore: 0,
      contact: 0,
      hasObject: false,
      missedFrames: 6
    });

    expect(bridged.analysis.gripPercentage).toBeGreaterThan(24);
    expect(bridged.analysis.gripPercentage).toBeLessThan(previous.gripPercentage);
    expect(bridged.analysis.guidance).not.toBe('Object not locked');
    expect(bridged.analysis.evidence.negativeReasons).toContain('V13 occlusion bridge active');
  });

  it('does not bridge an open empty hand after the object is gone', () => {
    const previous = analysisFixture({
      gripPercentage: 78,
      confidence: 0.72,
      objectLockQuality: 0.7,
      closureScore: 0.6,
      mode: 'power grip',
      state: 'Strong hold',
      contact: 0.48,
      thumb: 0.68,
      index: 0.6,
      middle: 0.5,
      phoneSide: 0.7,
      openHand: 0.08
    });
    const emptyHand = analysisFixture({
      gripPercentage: 0,
      confidence: 0,
      objectLockQuality: 0,
      closureScore: 0.12,
      mode: 'open hand',
      state: 'Hand only',
      contact: 0,
      openHand: 0.96
    });
    const dropped = stabilizeV13DisplayAnalysis(
      emptyHand,
      {
        ...createInitialV13DisplayState(),
        analysis: previous,
        timestamp: 1000,
        lastStableAt: 1000
      },
      1120,
      { objectScore: 0, contact: 0, hasObject: false, missedFrames: 2 }
    );

    expect(dropped.analysis.gripPercentage).toBe(0);
    expect(dropped.analysis.guidance).toBe('Object not locked');
  });

  it('dampens visual object jumps after detector reacquisition', () => {
    const previousObject = objectFixture({ x: 100, y: 120, radiusX: 42, radiusY: 64, confidence: 0.88 });
    const nextObject = objectFixture({ x: 360, y: 310, radiusX: 46, radiusY: 70, confidence: 0.9 });
    const state: V13DisplayState = {
      ...createInitialV13DisplayState(),
      analysis: analysisFixture({ gripPercentage: 76, confidence: 0.72, objectLockQuality: 0.72 }),
      timestamp: 1000,
      lastStableAt: 1000,
      visualObject: previousObject,
      visualTimestamp: 1000
    };

    const stabilized = stabilizeV13VisualObject(nextObject, state, 1120, analysisFixture({
      gripPercentage: 62,
      confidence: 0.6,
      objectLockQuality: 0.6
    }));

    expect(stabilized.object).not.toBeNull();
    expect(stabilized.object!.center.x).toBeLessThan(260);
    expect(stabilized.object!.center.y).toBeLessThan(240);
    expect(stabilized.state.visualMissedFrames).toBe(0);
  });

  it('expires stale visual holds instead of leaving an old oval on screen', () => {
    const state: V13DisplayState = {
      ...createInitialV13DisplayState(),
      analysis: analysisFixture({ gripPercentage: 70, confidence: 0.66, objectLockQuality: 0.66 }),
      timestamp: 1000,
      lastStableAt: 1000,
      visualObject: objectFixture({ x: 180, y: 180 }),
      visualTimestamp: 1000
    };

    const expired = stabilizeV13VisualObject(
      null,
      state,
      1900,
      analysisFixture({ gripPercentage: 0, confidence: 0, objectLockQuality: 0, state: 'Hand only' })
    );

    expect(expired.object).toBeNull();
    expect(expired.state.visualObject).toBeNull();
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

function objectFixture(options: {
  x: number;
  y: number;
  radiusX?: number;
  radiusY?: number;
  confidence?: number;
}): ObjectRegion {
  const radiusX = options.radiusX ?? 40;
  const radiusY = options.radiusY ?? 62;
  return {
    center: { x: options.x, y: options.y },
    radiusX,
    radiusY,
    angle: 0.1,
    confidence: options.confidence ?? 0.8,
    locked: true,
    source: 'detector',
    velocity: { x: 0, y: 0 },
    contour: [
      { x: options.x - radiusX, y: options.y - radiusY },
      { x: options.x + radiusX, y: options.y - radiusY },
      { x: options.x + radiusX, y: options.y + radiusY },
      { x: options.x - radiusX, y: options.y + radiusY }
    ],
    detectorLabel: 'yolo:object',
    detectorScore: options.confidence ?? 0.8
  };
}
