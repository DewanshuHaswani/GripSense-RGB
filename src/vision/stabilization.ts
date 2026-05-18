import type { GripAnalysis, GripGuidance, Landmark, ObjectRegion, Point } from './types';
import { clamp, subtract, vectorMagnitude } from './geometry';

type FilterMap = Map<string, OneEuroFilter>;

const SMOOTHED_FIELDS = [
  'gripPercentage',
  'confidence',
  'closureScore',
  'thumbOpposition',
  'enclosureScore',
  'motionCoupling',
  'slipRisk'
] as const;

export class TrackingStabilizer {
  private landmarkFilters: FilterMap = new Map();
  private objectFilters: FilterMap = new Map();
  private analysisFilters: FilterMap = new Map();
  private previousGuidance: GripGuidance = 'Object not locked';
  private previousTimestamp: number | null = null;
  private previousObjectCenter: Point | null = null;
  private slipHistory: number[] = [];

  reset() {
    this.landmarkFilters.clear();
    this.objectFilters.clear();
    this.analysisFilters.clear();
    this.previousGuidance = 'Object not locked';
    this.previousTimestamp = null;
    this.previousObjectCenter = null;
    this.slipHistory = [];
  }

  stabilizeHand(hand: Landmark[] | null, timestamp: number): Landmark[] | null {
    if (!hand) return null;
    const time = timestamp / 1000;
    return hand.map((landmark, index) => ({
      ...landmark,
      x: this.filter(this.landmarkFilters, `hand-${index}-x`, landmark.x, time, 0.8, 0.015),
      y: this.filter(this.landmarkFilters, `hand-${index}-y`, landmark.y, time, 0.8, 0.015)
    }));
  }

  stabilizeObject(object: ObjectRegion | null, timestamp: number): ObjectRegion | null {
    if (!object) return null;
    const time = timestamp / 1000;
    const center = {
      x: this.filter(this.objectFilters, 'object-center-x', object.center.x, time, 0.7, 0.012),
      y: this.filter(this.objectFilters, 'object-center-y', object.center.y, time, 0.7, 0.012)
    };
    const radiusX = this.filter(this.objectFilters, 'object-radius-x', object.radiusX, time, 0.45, 0.01);
    const radiusY = this.filter(this.objectFilters, 'object-radius-y', object.radiusY, time, 0.45, 0.01);
    const angle = this.filter(this.objectFilters, 'object-angle', object.angle, time, 0.55, 0.01);
    const previousCenter = this.previousObjectCenter;
    this.previousObjectCenter = center;

    return {
      ...object,
      center,
      radiusX,
      radiusY,
      angle,
      velocity: {
        x: previousCenter ? center.x - previousCenter.x : object.velocity.x,
        y: previousCenter ? center.y - previousCenter.y : object.velocity.y
      },
      contour: object.contour.map((point, index) => ({
        x: this.filter(this.objectFilters, `contour-${index}-x`, point.x, time, 0.65, 0.01),
        y: this.filter(this.objectFilters, `contour-${index}-y`, point.y, time, 0.65, 0.01)
      }))
    };
  }

  stabilizeAnalysis(analysis: GripAnalysis, timestamp: number): GripAnalysis {
    const time = timestamp / 1000;
    const stabilized = { ...analysis };
    SMOOTHED_FIELDS.forEach((field) => {
      const value = field === 'gripPercentage' ? analysis[field] / 100 : analysis[field];
      const smoothed = this.filter(this.analysisFilters, `analysis-${field}`, value, time, 0.5, 0.02);
      stabilized[field] = field === 'gripPercentage' ? Math.round(smoothed * 100) : clamp(smoothed);
    });

    stabilized.guidance = this.stabilizeGuidance(analysis.guidance, stabilized.gripPercentage, stabilized.slipRisk);
    stabilized.message = analysis.guidance === stabilized.guidance ? analysis.message : messageForGuidance(stabilized.guidance);
    this.previousTimestamp = timestamp;
    return stabilized;
  }

  updatePersistentSlip(handVelocity: Point, object: ObjectRegion | null) {
    const handMotion = vectorMagnitude(handVelocity);
    const objectMotion = object ? vectorMagnitude(object.velocity) : 0;
    const moving = Math.max(handMotion, objectMotion) > 2.8;
    const relativeMotion = object ? vectorMagnitude(subtract(object.velocity, handVelocity)) : 0;
    const instantSlip = moving ? clamp(relativeMotion / Math.max(10, Math.max(handMotion, objectMotion) * 1.9)) : 0;
    this.slipHistory.push(instantSlip);
    if (this.slipHistory.length > 10) this.slipHistory.shift();
    const sustained = this.slipHistory.reduce((sum, value) => sum + value, 0) / Math.max(1, this.slipHistory.length);
    const spikes = this.slipHistory.filter((value) => value > 0.42).length / Math.max(1, this.slipHistory.length);
    return moving ? clamp(sustained * 0.72 + spikes * 0.28) : 0;
  }

  private filter(filters: FilterMap, key: string, value: number, timestamp: number, minCutoff: number, beta: number) {
    let filter = filters.get(key);
    if (!filter) {
      filter = new OneEuroFilter({ minCutoff, beta, dCutoff: 1 });
      filters.set(key, filter);
    }
    return filter.filter(value, timestamp);
  }

  private stabilizeGuidance(next: GripGuidance, score: number, slipRisk: number): GripGuidance {
    if (next === 'Object not locked') {
      this.previousGuidance = next;
      return next;
    }

    const previous = this.previousGuidance;
    if (previous === 'Object not locked') {
      this.previousGuidance = next;
      return next;
    }

    const shouldUpgradeToStrong = next === 'Strong grip' && score >= 78 && slipRisk < 0.3;
    const shouldDropFromStrong = previous === 'Strong grip' && (score < 66 || slipRisk > 0.42);
    const shouldDropToReposition = next === 'Reposition' && score < 38;
    const shouldLeaveReposition = previous === 'Reposition' && score > 52;

    if (shouldUpgradeToStrong || shouldDropFromStrong || shouldDropToReposition || shouldLeaveReposition) {
      this.previousGuidance = next;
      return next;
    }

    if (previous !== next) return previous;
    this.previousGuidance = next;
    return next;
  }
}

class OneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  private xPrevious: number | null = null;
  private dxPrevious = 0;
  private tPrevious: number | null = null;

  constructor({ minCutoff, beta, dCutoff }: { minCutoff: number; beta: number; dCutoff: number }) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  filter(value: number, timestamp: number) {
    if (this.tPrevious === null || this.xPrevious === null) {
      this.tPrevious = timestamp;
      this.xPrevious = value;
      return value;
    }

    const dt = Math.max(1 / 120, timestamp - this.tPrevious);
    const derivative = (value - this.xPrevious) / dt;
    const derivativeAlpha = smoothingAlpha(dt, this.dCutoff);
    const dxHat = exponentialSmooth(derivativeAlpha, derivative, this.dxPrevious);
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const alpha = smoothingAlpha(dt, cutoff);
    const xHat = exponentialSmooth(alpha, value, this.xPrevious);

    this.xPrevious = xHat;
    this.dxPrevious = dxHat;
    this.tPrevious = timestamp;
    return xHat;
  }
}

function smoothingAlpha(dt: number, cutoff: number) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

function exponentialSmooth(alpha: number, value: number, previous: number) {
  return alpha * value + (1 - alpha) * previous;
}

function messageForGuidance(guidance: GripGuidance) {
  if (guidance === 'Strong grip') return 'Grip looks stable. Keep the thumb opposed while the object moves.';
  if (guidance === 'Improve grip') return 'Grip is usable but could improve with wider finger spread and thumb opposition.';
  if (guidance === 'Reposition') return 'Reposition the object deeper between the thumb and fingers.';
  return 'Object not locked. Move an object into the hand area or click it to lock tracking.';
}
