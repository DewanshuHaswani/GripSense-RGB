import type { GripAnalysis, Landmark, ObjectRegion } from './types';
import { FINGER_MCP_INDICES, FINGERTIP_INDICES, PALM_INDICES, palmCenter } from './geometry';

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17]
] as const;

export function drawTrackingOverlay(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  mirrored: boolean,
  hand: Landmark[] | null,
  object: ObjectRegion | null,
  analysis: GripAnalysis
) {
  context.clearRect(0, 0, width, height);
  context.save();
  if (mirrored) {
    context.translate(width, 0);
    context.scale(-1, 1);
  }

  if (object) drawObject(context, object, analysis);
  if (hand) drawHand(context, hand, analysis);

  context.restore();
  drawHud(context, width, analysis);
}

function drawHand(context: CanvasRenderingContext2D, hand: Landmark[], analysis: GripAnalysis) {
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.lineWidth = 4;
  context.strokeStyle = 'rgba(135, 232, 255, 0.74)';
  HAND_CONNECTIONS.forEach(([from, to]) => {
    context.beginPath();
    context.moveTo(hand[from].x, hand[from].y);
    context.lineTo(hand[to].x, hand[to].y);
    context.stroke();
  });

  context.fillStyle = 'rgba(158, 246, 178, 0.98)';
  FINGERTIP_INDICES.forEach((index) => drawPoint(context, hand[index].x, hand[index].y, 6));
  context.fillStyle = 'rgba(255, 255, 255, 0.64)';
  FINGER_MCP_INDICES.forEach((index) => drawPoint(context, hand[index].x, hand[index].y, 4));

  const palm = palmCenter(hand);
  const palmRadius = 9 + analysis.closureScore * 5;
  context.fillStyle = 'rgba(255, 209, 102, 0.92)';
  drawPoint(context, palm.x, palm.y, palmRadius);

  context.strokeStyle = 'rgba(255, 209, 102, 0.24)';
  context.beginPath();
  PALM_INDICES.forEach((index, pointIndex) => {
    const point = hand[index];
    if (pointIndex === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  });
  context.closePath();
  context.stroke();
}

function drawObject(context: CanvasRenderingContext2D, object: ObjectRegion, analysis: GripAnalysis) {
  const stateColor =
    analysis.guidance === 'Strong grip'
      ? '74, 222, 128'
      : analysis.guidance === 'Improve grip'
        ? '250, 204, 21'
        : '248, 113, 113';

  context.save();
  context.translate(object.center.x, object.center.y);
  context.rotate(object.angle);
  context.strokeStyle = `rgba(${stateColor}, 0.92)`;
  context.fillStyle = `rgba(${stateColor}, 0.08)`;
  context.lineWidth = 4;
  context.beginPath();
  context.ellipse(0, 0, object.radiusX, object.radiusY, 0, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.restore();

  context.setLineDash([8, 9]);
  context.strokeStyle = `rgba(${stateColor}, 0.38)`;
  context.lineWidth = 2;
  context.beginPath();
  object.contour.forEach((point, index) => {
    if (index === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  });
  context.closePath();
  context.stroke();
  context.setLineDash([]);

  analysis.recommendedGripPoints.forEach((point) => {
    context.fillStyle =
      point.label === 'thumb'
        ? 'rgba(96, 165, 250, 0.96)'
        : point.label === 'finger'
          ? 'rgba(52, 211, 153, 0.96)'
          : 'rgba(255, 255, 255, 0.86)';
    context.strokeStyle = 'rgba(8, 13, 20, 0.85)';
    context.lineWidth = 3;
    context.beginPath();
    context.arc(point.x, point.y, 9 + point.score * 4, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
}

function drawHud(context: CanvasRenderingContext2D, width: number, analysis: GripAnalysis) {
  context.save();
  context.font = '600 16px Inter, system-ui, sans-serif';
  context.textAlign = 'center';
  context.fillStyle = 'rgba(226, 232, 240, 0.9)';
  context.fillText(`${analysis.guidance} · ${analysis.gripPercentage}%`, width / 2, 34);
  context.restore();
}

function drawPoint(context: CanvasRenderingContext2D, x: number, y: number, radius: number) {
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
}
