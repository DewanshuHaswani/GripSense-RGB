import { describe, expect, it } from 'vitest';
import { analyzeGripWithRealSense, applyRealSenseDepthGate, normalizeRealSenseDepthSignal, refineRealSenseOfflineTimeline } from './realsense';
import { EMPTY_RFDETR_TRACK, type RfdetrDetection, type RfdetrTimelinePoint } from './rfdetr';
import { createEmptyAnalysis } from './gripAnalysis';
import type { Landmark } from './types';

describe('RealSense grip fusion', () => {
  it('boosts confidence only when stereo depth agrees with RF-DETR contact', () => {
    const result = analyzeGripWithRealSense({
      hand: phoneGripHand(),
      detections: [objectDetection()],
      previousPalm: null,
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true,
      depthSignal: {
        available: true,
        contactDepthScore: 0.82,
        depthSeparationScore: 0.08,
        stereoConfidence: 0.88,
        occlusionScore: 0.14,
        surfaceContinuity: 0.8
      }
    });

    expect(result.object).not.toBeNull();
    expect(result.analysis.confidence).toBeGreaterThan(0.7);
    expect(result.selection.contact).toBeGreaterThan(0.6);
    expect(result.analysis.diagnostics.state).not.toBe('Slip risk');
  });

  it('caps grip when RealSense depth shows the object away from the hand', () => {
    const base = createEmptyAnalysis('Grip detected');
    const analysis = applyRealSenseDepthGate(
      {
        ...base,
        gripPercentage: 88,
        confidence: 0.9,
        objectLockQuality: 0.86,
        diagnostics: { ...base.diagnostics, state: 'Strong hold', recommendation: 'Strong grip' },
        evidence: {
          ...base.evidence,
          fingerSegmentContactScore: 0.82,
          visibleContactScore: 0.8,
          objectLockQuality: 0.86
        }
      },
      {
        available: true,
        contactDepthScore: 0.12,
        depthSeparationScore: 0.82,
        stereoConfidence: 0.76,
        occlusionScore: 0.2,
        surfaceContinuity: 0.36
      },
      true
    );

    expect(analysis.gripPercentage).toBeLessThanOrEqual(24);
    expect(analysis.confidence).toBeLessThanOrEqual(0.42);
    expect(analysis.diagnostics.state).toBe('Slip risk');
    expect(analysis.slipRisk).toBeGreaterThanOrEqual(0.62);
  });

  it('smooths Offline V3 misses with neighboring RealSense depth contact', () => {
    const points: Array<RfdetrTimelinePoint & { time: number; realsenseDepthContact?: number; realsenseStereoConfidence?: number }> = [
      timelinePoint(0, 72, 0.76, 0.7, 0.86),
      timelinePoint(0.2, 16, 0.06, 0.02, 0.04),
      timelinePoint(0.4, 74, 0.8, 0.72, 0.88)
    ];
    const smoothed = refineRealSenseOfflineTimeline(points);

    expect(smoothed[1].grip).toBeGreaterThan(points[1].grip);
    expect(smoothed[1].contact).toBeGreaterThan(points[1].contact);
    expect(smoothed[1].confidence).toBeGreaterThan(points[1].confidence);
  });

  it('normalizes unavailable depth without inventing contact', () => {
    const signal = normalizeRealSenseDepthSignal({ available: false, contactDepthScore: 10, stereoConfidence: 0.7 });

    expect(signal.available).toBe(false);
    expect(signal.contactDepthScore).toBe(0);
    expect(signal.depthSeparationScore).toBe(1);
    expect(signal.stereoConfidence).toBe(0);
    expect(signal.handDepthM).toBeNull();
  });
});

function baseHand(): Landmark[] {
  return Array.from({ length: 21 }, () => ({ x: 300, y: 340 }));
}

function phoneGripHand(): Landmark[] {
  const hand = baseHand();
  hand[0] = { x: 318, y: 382 };
  hand[4] = { x: 306, y: 322 };
  hand[5] = { x: 292, y: 346 };
  hand[8] = { x: 384, y: 306 };
  hand[9] = { x: 320, y: 342 };
  hand[12] = { x: 398, y: 332 };
  hand[13] = { x: 344, y: 352 };
  hand[16] = { x: 394, y: 368 };
  hand[17] = { x: 368, y: 366 };
  hand[20] = { x: 386, y: 390 };
  return hand;
}

function objectDetection(): RfdetrDetection {
  const x = 318;
  const y = 232;
  const width = 80;
  const height = 150;
  return {
    id: 'object-1',
    label: 'cup',
    score: 0.9,
    bbox: { x, y, width, height },
    maskPolygon: [
      { x, y },
      { x: x + width, y },
      { x: x + width, y: y + height },
      { x, y: y + height }
    ],
    maskArea: width * height,
    center: { x: x + width / 2, y: y + height / 2 },
    latencyMs: 18
  };
}

function timelinePoint(time: number, grip: number, objectScore: number, contact: number, stereo: number) {
  return {
    time,
    grip,
    confidence: objectScore,
    objectMatch: objectScore,
    lock: objectScore,
    contact,
    slip: 0.05,
    weak: grip < 42,
    guidance: grip < 42 ? 'Reposition' : 'Improve grip',
    state: grip < 42 ? 'Object uncertain' : 'Grip detected',
    object: 'RealSense object',
    rfdetrObjectScore: objectScore,
    rfdetrContact: contact,
    realsenseDepthContact: contact,
    realsenseStereoConfidence: stereo
  };
}
