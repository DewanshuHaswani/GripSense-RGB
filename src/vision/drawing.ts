import type { ObjectProfileCandidate } from './objectProfile';
import type { DetectedObjectBox, GripAnalysis, Landmark, ObjectRegion } from './types';
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
  analysis: GripAnalysis,
  candidates: ObjectProfileCandidate[] = [],
  selectedProfileId: string | null = null,
  baseCandidates: Array<DetectedObjectBox & { candidateId: string }> = [],
  selectedBaseId: string | null = null,
  showObjectLabels = true
) {
  context.clearRect(0, 0, width, height);
  context.save();
  if (mirrored) {
    context.translate(width, 0);
    context.scale(-1, 1);
  }

  if (baseCandidates.length && !selectedProfileId) {
    drawBaseCandidates(context, baseCandidates, selectedBaseId, mirrored, showObjectLabels);
  }
  if (candidates.length && !selectedBaseId) {
    drawProfileCandidates(context, candidates, selectedProfileId, mirrored, showObjectLabels);
  }
  if (object) drawObject(context, object, analysis);
  if (hand) drawHand(context, hand, analysis);

  context.restore();
  drawHud(context, width, analysis);
}

function drawBaseCandidates(
  context: CanvasRenderingContext2D,
  candidates: Array<DetectedObjectBox & { candidateId: string }>,
  selectedBaseId: string | null,
  mirrored: boolean,
  showObjectLabels: boolean
) {
  context.save();
  context.font = '800 13px Inter, system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  const visibleCandidates = selectedBaseId ? candidates.filter((candidate) => candidate.candidateId === selectedBaseId) : candidates.slice(0, 8);
  visibleCandidates.forEach((candidate) => {
    const center = {
      x: candidate.box.x + candidate.box.width / 2,
      y: candidate.box.y + candidate.box.height / 2
    };
    const selected = selectedBaseId === candidate.candidateId;
    const color = selected ? '34, 211, 238' : '167, 243, 208';
    context.strokeStyle = `rgba(${color}, ${selected ? 0.94 : 0.58})`;
    context.fillStyle = `rgba(${color}, ${selected ? 0.14 : 0.07})`;
    context.lineWidth = selected ? 4 : 2;
    roundRect(context, candidate.box.x, candidate.box.y, candidate.box.width, candidate.box.height, 10);
    context.fill();
    context.stroke();

    const label = idFromBaseCandidateId(candidate.candidateId);
    const text = showObjectLabels && candidate.label !== 'unknown' ? `${label} ${candidate.label}` : label;
    context.beginPath();
    context.arc(center.x, center.y, selected ? 12 : 9, 0, Math.PI * 2);
    context.fillStyle = `rgba(${color}, 0.94)`;
    context.fill();
    context.strokeStyle = 'rgba(2, 6, 12, 0.82)';
    context.lineWidth = 3;
    context.stroke();
    const labelWidth = showObjectLabels ? Math.max(36, context.measureText(text).width + 18) : 36;
    context.fillStyle = 'rgba(2, 6, 12, 0.72)';
    roundRect(context, center.x - labelWidth / 2, center.y - 34, labelWidth, 23, 7);
    context.fill();
    context.fillStyle = '#f8fafc';
    drawReadableText(context, text, center.x, center.y - 22, mirrored);
  });
  context.restore();
}

function drawProfileCandidates(
  context: CanvasRenderingContext2D,
  candidates: ObjectProfileCandidate[],
  selectedProfileId: string | null,
  mirrored: boolean,
  showObjectLabels: boolean
) {
  context.save();
  context.font = '800 14px Inter, system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  const visibleCandidates = selectedProfileId
    ? candidates.filter((candidate) => candidate.profileId === selectedProfileId)
    : candidates.slice(0, 8);
  visibleCandidates.forEach((candidate, index) => {
    const selected = selectedProfileId === candidate.profileId;
    const matched = candidate.matched;
    const color = selected ? '34, 211, 238' : matched ? '134, 239, 172' : '250, 204, 21';
    const radius = selected ? 13 : 10;
    context.beginPath();
    context.arc(candidate.center.x, candidate.center.y, radius + 7, 0, Math.PI * 2);
    context.fillStyle = `rgba(${color}, 0.14)`;
    context.fill();
    context.beginPath();
    context.arc(candidate.center.x, candidate.center.y, radius, 0, Math.PI * 2);
    context.fillStyle = `rgba(${color}, 0.92)`;
    context.fill();
    context.lineWidth = selected ? 4 : 3;
    context.strokeStyle = 'rgba(2, 6, 12, 0.82)';
    context.stroke();

    const label = selectedProfileId ? 'O1' : `O${index + 1}`;
    const text = showObjectLabels ? `${label} ${candidate.name}` : label;
    const labelX = candidate.center.x;
    const labelY = candidate.center.y - radius - 20;
    const labelWidth = showObjectLabels ? Math.max(36, context.measureText(text).width + 18) : 36;
    context.fillStyle = 'rgba(2, 6, 12, 0.72)';
    roundRect(context, labelX - labelWidth / 2, labelY - 12, labelWidth, 24, 7);
    context.fill();
    context.fillStyle = selected ? '#67e8f9' : '#f8fafc';
    drawReadableText(context, text, labelX, labelY + 1, mirrored);
  });
  context.restore();
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

function roundRect(context: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(x + width - radius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + radius);
  context.lineTo(x + width, y + height - radius);
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  context.lineTo(x + radius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function drawReadableText(context: CanvasRenderingContext2D, text: string, x: number, y: number, mirrored: boolean) {
  if (!mirrored) {
    context.fillText(text, x, y);
    return;
  }
  context.save();
  context.translate(x, y);
  context.scale(-1, 1);
  context.fillText(text, 0, 0);
  context.restore();
}

function idFromBaseCandidateId(id: string) {
  const legacyIndex = Number(id.replace('base-', ''));
  if (Number.isFinite(legacyIndex)) return `B${legacyIndex + 1}`;
  const stableId = Number(id.replace('base-track-', ''));
  return Number.isFinite(stableId) ? `B${stableId}` : 'B?';
}
