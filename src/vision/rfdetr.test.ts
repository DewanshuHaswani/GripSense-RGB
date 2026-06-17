import { describe, expect, it } from 'vitest';
import {
  EMPTY_RFDETR_TRACK,
  analyzeGripWithRfdetr,
  compensateRfdetrResponseForHandMotion,
  isRfdetrResultFresh,
  refineRfdetrOfflineTimeline,
  scaleRfdetrResponseToVideo,
  selectRfdetrGripObject,
  type RfdetrDetection,
  type RfdetrTimelinePoint
} from './rfdetr';
import { palmCenter } from './geometry';
import type { Landmark } from './types';

describe('RF-DETR grip analysis', () => {
  it('lets a near RF-DETR mask enable grip scoring', () => {
    const result = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [objectDetection()],
      previousPalm: { x: 326, y: 346 },
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      persistentSlipScore: 0.02,
      serverAvailable: true
    });

    expect(result.selection.contact).toBeGreaterThan(0.2);
    expect(result.object?.source).toBe('segmenter');
    expect(result.analysis.gripPercentage).toBeGreaterThan(35);
    expect(result.analysis.guidance).not.toBe('Object not locked');
  });

  it('caps a closed hand with no RF-DETR object low', () => {
    const result = analyzeGripWithRfdetr({
      hand: closedGripHand(),
      detections: [],
      previousPalm: null,
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true
    });

    expect(result.analysis.gripPercentage).toBe(0);
    expect(result.analysis.diagnostics.state).toBe('Hand only');
    expect(result.selection.objectScore).toBe(0);
  });

  it('rejects person detections as grip objects', () => {
    const selection = selectRfdetrGripObject([objectDetection({ label: 'person', score: 0.99 })], phoneGripHand(), EMPTY_RFDETR_TRACK);

    expect(selection.detection).toBeNull();
    expect(selection.rejectedPerson).toBe(true);
  });

  it('does not lock a high-confidence RF-DETR object without mask contact', () => {
    const result = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [objectDetection({ label: 'laptop', score: 0.98, x: 500, y: 520, width: 260, height: 180 })],
      previousPalm: null,
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true
    });

    expect(result.selection.contact).toBe(0);
    expect(result.selection.objectScore).toBeLessThan(0.18);
    expect(result.object).toBeNull();
    expect(result.analysis.objectLockQuality).toBe(0);
  });

  it('does not draw a weak offset RF-DETR mask as a locked object', () => {
    const detection = objectDetection({ label: 'cup', score: 0.94, x: 178, y: 250, width: 104, height: 172 });
    const result = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [detection],
      previousPalm: null,
      previousObject: null,
      previousTrack: {
        detectionKey: detection.id,
        center: detection.center,
        confidence: 0.52,
        continuity: 0.9,
        missedFrames: 0,
        lastSeenAt: 940
      },
      now: 1000,
      serverAvailable: true
    });

    expect(result.selection.objectScore).toBeGreaterThan(0.18);
    expect(result.selection.contact).toBeLessThan(0.16);
    expect(result.object).toBeNull();
    expect(result.analysis.gripPercentage).toBe(0);
    expect(result.analysis.diagnostics.state).toBe('Object uncertain');
  });

  it('briefly holds the previous RF-DETR object visual across one missed live frame', () => {
    const first = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [objectDetection()],
      previousPalm: { x: 326, y: 346 },
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true
    });
    const held = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [],
      previousPalm: { x: 326, y: 346 },
      previousObject: first.object,
      previousTrack: first.track,
      now: 1160,
      serverAvailable: true
    });

    expect(held.object).not.toBeNull();
    expect(held.object?.locked).toBe(false);
    expect(held.selection.objectScore).toBeGreaterThan(0.18);
    expect(held.analysis.gripPercentage).toBeLessThanOrEqual(45);
    expect(held.analysis.confidence).toBeLessThan(0.4);
    expect(held.objectIdentity.matched).toBe(false);
    expect(held.analysis.diagnostics.recommendation).toContain('holding the previous object boundary');
  });

  it('motion-compensates the held RF-DETR visual across short moving-hand misses', () => {
    const firstHand = phoneGripHand();
    const movedHand = firstHand.map((point) => ({ ...point, x: point.x + 24, y: point.y - 12 }));
    const first = analyzeGripWithRfdetr({
      hand: firstHand,
      detections: [objectDetection()],
      previousPalm: palmCenter(firstHand),
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true
    });
    const missedOnce = analyzeGripWithRfdetr({
      hand: movedHand,
      detections: [],
      previousPalm: palmCenter(firstHand),
      previousObject: first.object,
      previousTrack: first.track,
      now: 1160,
      serverAvailable: true
    });
    const missedTwice = analyzeGripWithRfdetr({
      hand: movedHand,
      detections: [],
      previousPalm: palmCenter(firstHand),
      previousObject: first.object,
      previousTrack: missedOnce.track,
      now: 1320,
      serverAvailable: true
    });

    expect(missedTwice.object).not.toBeNull();
    expect(missedTwice.object?.center.x).toBeCloseTo((first.object?.center.x ?? 0) + 24, 5);
    expect(missedTwice.object?.center.y).toBeCloseTo((first.object?.center.y ?? 0) - 12, 5);
    expect(missedTwice.object?.locked).toBe(false);
    expect(missedTwice.analysis.confidence).toBeLessThan(0.45);
    expect(missedTwice.analysis.gripPercentage).toBeLessThan(55);
  });

  it('does not lock oversized background masks that drift through the hand area', () => {
    const result = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [objectDetection({ label: 'dining table', score: 0.96, x: 250, y: 300, width: 780, height: 260 })],
      previousPalm: null,
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true
    });

    expect(result.selection.objectScore).toBeLessThan(0.18);
    expect(result.object).toBeNull();
    expect(result.analysis.gripPercentage).toBe(0);
  });

  it('lowers contact and grip quickly when the object drops away', () => {
    const first = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [objectDetection()],
      previousPalm: { x: 326, y: 346 },
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: true
    });
    const dropped = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [objectDetection({ x: 610, y: 90, width: 74, height: 122, id: 'dropped' })],
      previousPalm: { x: 326, y: 346 },
      previousObject: first.object,
      previousTrack: first.track,
      now: 1160,
      serverAvailable: true
    });

    expect(dropped.selection.contact).toBeLessThan(0.16);
    expect(dropped.analysis.gripPercentage).toBeLessThanOrEqual(18);
    expect(dropped.analysis.diagnostics.state).toBe('Object uncertain');
  });

  it('bridges short RF-DETR misses in Offline V2 smoothing', () => {
    const points: RfdetrTimelinePoint[] = [
      timelinePoint(0, 58, 0.66, 0.52),
      timelinePoint(0.2, 8, 0.02, 0.01),
      timelinePoint(0.4, 62, 0.7, 0.54)
    ];
    const smoothed = refineRfdetrOfflineTimeline(points);

    expect(smoothed[1].rfdetrObjectScore).toBeGreaterThan(0.4);
    expect(smoothed[1].contact).toBeGreaterThan(0.3);
    expect(smoothed[1].grip).toBeGreaterThan(points[1].grip);
  });

  it('reports unavailable without fake confidence', () => {
    const result = analyzeGripWithRfdetr({
      hand: phoneGripHand(),
      detections: [],
      previousPalm: null,
      previousObject: null,
      previousTrack: EMPTY_RFDETR_TRACK,
      now: 1000,
      serverAvailable: false,
      unavailableMessage: 'RF-DETR unavailable'
    });

    expect(result.analysis.gripPercentage).toBe(0);
    expect(result.analysis.confidence).toBe(0);
    expect(result.analysis.message).toContain('RF-DETR unavailable');
    expect(result.analysis.diagnostics.issueCategory).toBe('server_unavailable');
  });

  it('rescales RF-DETR detections from request frame size to video coordinates', () => {
    const scaled = scaleRfdetrResponseToVideo(
      {
        detections: [objectDetection({ x: 180, y: 90, width: 120, height: 240 })],
        latencyMs: 18,
        model: 'RF-DETR-Seg Nano',
        device: 'cpu'
      },
      { width: 720, height: 405, sourceWidth: 1440, sourceHeight: 810 }
    );

    expect(scaled.detections[0].bbox).toEqual({ x: 360, y: 180, width: 240, height: 480 });
    expect(scaled.detections[0].center).toEqual({ x: 480, y: 420 });
    expect(scaled.detections[0].maskPolygon[2]).toEqual({ x: 600, y: 660 });
  });

  it('keeps recent RF-DETR detections fresh while the next request is pending', () => {
    expect(
      isRfdetrResultFresh(
        {
          status: 'pending',
          message: 'RF-DETR analyzing live frame.',
          endpoint: 'http://127.0.0.1:7867/api/rfdetr/analyze',
          result: { detections: [objectDetection()], latencyMs: 132 },
          resultPalm: { x: 320, y: 342 },
          receivedAt: 1200,
          lastRequestAt: 1400,
          latencyMs: 132
        },
        1600,
        1500
      )
    ).toBe(true);
  });

  it('motion-compensates recent RF-DETR masks using palm movement', () => {
    const hand = phoneGripHand().map((point) => ({ ...point, x: point.x + 38, y: point.y - 24 }));
    const response = {
      detections: [objectDetection()],
      latencyMs: 132
    };
    const shifted = compensateRfdetrResponseForHandMotion(response, palmCenter(phoneGripHand()), hand);

    expect(shifted.detections[0].center.x).toBeCloseTo(response.detections[0].center.x + 38, 5);
    expect(shifted.detections[0].center.y).toBeCloseTo(response.detections[0].center.y - 24, 5);
  });

  it('does not motion-compensate stale RF-DETR masks after very large hand jumps', () => {
    const hand = phoneGripHand().map((point) => ({ ...point, x: point.x + 500, y: point.y + 40 }));
    const response = {
      detections: [objectDetection()],
      latencyMs: 132
    };
    const shifted = compensateRfdetrResponseForHandMotion(response, palmCenter(phoneGripHand()), hand);

    expect(shifted.detections[0].center).toEqual(response.detections[0].center);
  });
});

function baseHand(): Landmark[] {
  return Array.from({ length: 21 }, () => ({ x: 300, y: 340 }));
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

function objectDetection(overrides: Partial<RfdetrDetection & { x: number; y: number; width: number; height: number }> = {}): RfdetrDetection {
  const x = overrides.x ?? 318;
  const y = overrides.y ?? 232;
  const width = overrides.width ?? 80;
  const height = overrides.height ?? 150;
  return {
    id: overrides.id ?? 'object-1',
    label: overrides.label ?? 'bottle',
    score: overrides.score ?? 0.9,
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

function timelinePoint(time: number, grip: number, objectScore: number, contact: number): RfdetrTimelinePoint & { time: number } {
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
    object: objectScore > 0.2 ? 'RF-DETR object' : '',
    rfdetrObjectScore: objectScore,
    rfdetrContact: contact
  };
}
