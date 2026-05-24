import type { Landmark, ObjectRegion, Point } from './types';

export const FINGERTIP_INDICES = [4, 8, 12, 16, 20] as const;
export const FINGER_MCP_INDICES = [5, 9, 13, 17] as const;
export const PALM_INDICES = [0, 5, 9, 13, 17] as const;

export function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function averagePoint(points: Point[]): Point {
  if (!points.length) return { x: 0, y: 0 };
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length
  };
}

export function subtract(a: Point, b: Point): Point {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function vectorMagnitude(vector: Point) {
  return Math.hypot(vector.x, vector.y);
}

export function normalize(vector: Point): Point {
  const magnitude = vectorMagnitude(vector);
  if (magnitude < 0.001) return { x: 0, y: 0 };
  return { x: vector.x / magnitude, y: vector.y / magnitude };
}

export function dot(a: Point, b: Point) {
  return a.x * b.x + a.y * b.y;
}

export function cross(a: Point, b: Point) {
  return a.x * b.y - a.y * b.x;
}

export function handSize(hand: Landmark[]) {
  if (hand.length < 21) return 1;
  const palmWidth = distance(hand[5], hand[17]);
  const wristToMiddle = distance(hand[0], hand[9]);
  return Math.max(1, palmWidth * 1.25, wristToMiddle * 1.45);
}

export function palmCenter(hand: Landmark[]) {
  return averagePoint(PALM_INDICES.map((index) => hand[index]).filter(Boolean));
}

export function fingertipPoints(hand: Landmark[]) {
  return FINGERTIP_INDICES.map((index) => hand[index]).filter(Boolean);
}

export function handOrientation(hand: Landmark[]) {
  if (hand.length < 21) return { angle: 0, handednessSign: 1 };
  const wrist = hand[0];
  const indexMcp = hand[5];
  const pinkyMcp = hand[17];
  const middleMcp = hand[9];
  const palmAxis = normalize(subtract(middleMcp, wrist));
  const knuckleAxis = normalize(subtract(pinkyMcp, indexMcp));
  const handednessSign = cross(palmAxis, knuckleAxis) >= 0 ? 1 : -1;
  return {
    angle: Math.atan2(palmAxis.y, palmAxis.x),
    handednessSign
  };
}

export function toHandLocal(point: Point, hand: Landmark[], origin = palmCenter(hand)): Point {
  const { angle, handednessSign } = handOrientation(hand);
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  const dx = point.x - origin.x;
  const dy = point.y - origin.y;
  return {
    x: (dx * cos - dy * sin) * handednessSign,
    y: dx * sin + dy * cos
  };
}

export function ellipsePoint(region: ObjectRegion, theta: number): Point {
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const angleCos = Math.cos(region.angle);
  const angleSin = Math.sin(region.angle);
  const localX = region.radiusX * cos;
  const localY = region.radiusY * sin;
  return {
    x: region.center.x + localX * angleCos - localY * angleSin,
    y: region.center.y + localX * angleSin + localY * angleCos
  };
}

export function distanceToEllipseBoundary(point: Point, region: ObjectRegion) {
  const angleCos = Math.cos(-region.angle);
  const angleSin = Math.sin(-region.angle);
  const dx = point.x - region.center.x;
  const dy = point.y - region.center.y;
  const localX = dx * angleCos - dy * angleSin;
  const localY = dx * angleSin + dy * angleCos;
  const theta = Math.atan2(localY / region.radiusY, localX / region.radiusX);
  const boundary = {
    x: region.radiusX * Math.cos(theta),
    y: region.radiusY * Math.sin(theta)
  };
  return Math.hypot(localX - boundary.x, localY - boundary.y);
}

export function pointsToPixelSpace(hand: Landmark[], width: number, height: number): Landmark[] {
  return hand.map((point) => ({
    x: point.x * width,
    y: point.y * height,
    z: point.z,
    visibility: point.visibility
  }));
}
