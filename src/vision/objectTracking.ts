import type { AlgorithmVersion, DetectedObjectBox, Landmark, ObjectRegion, Point } from './types';
import { averagePoint, clamp, distance, ellipsePoint, fingertipPoints, handSize, palmCenter, subtract } from './geometry';

type TrackerOptions = {
  video: HTMLVideoElement;
  hand: Landmark[] | null;
  previous: ObjectRegion | null;
  manualPoint: Point | null;
  manualScale?: number;
  locked: boolean;
  detectorBox?: DetectedObjectBox | null;
  algorithmVersion?: AlgorithmVersion;
};

const SAMPLE_SIZE = 96;
let sampleCanvas: HTMLCanvasElement | null = null;
let sampleContext: CanvasRenderingContext2D | null = null;

export function inferObjectRegion({
  video,
  hand,
  previous,
  manualPoint,
  manualScale = 1,
  locked,
  detectorBox,
  algorithmVersion = 'v1'
}: TrackerOptions): ObjectRegion | null {
  if (!hand || hand.length < 21 || !video.videoWidth || !video.videoHeight) return null;

  const palm = palmCenter(hand);
  const tips = fingertipPoints(hand);
  const size = handSize(hand);
  const tipCenter = averagePoint([hand[4], hand[8], hand[12], hand[16]]);
  const graspCenter = averagePoint([palm, tipCenter, hand[8], hand[12], hand[16]]);
  const detectorCandidate = sanitizeDetectorBox(detectorBox, palm, tips, size);
  const manualOrDetector = Boolean(manualPoint || detectorCandidate);
  const openHandScore = computeOpenHandScore(hand, size, palm);
  const autoCenter = manualPoint ?? previous?.center ?? graspCenter;
  const radiusSeed = clamp(distance(palm, tipCenter) / Math.max(1, size), 0.18, 0.48);
  const detectorRadiusX = detectorCandidate ? clamp(detectorCandidate.box.width * 0.5, size * 0.22, size * 0.78) : null;
  const detectorRadiusY = detectorCandidate ? clamp(detectorCandidate.box.height * 0.5, size * 0.18, size * 0.92) : null;
  const radiusX = (detectorRadiusX ?? Math.max(30, size * (manualPoint ? 0.3 : radiusSeed * 0.9))) * (manualPoint ? manualScale : 1);
  const radiusY = (detectorRadiusY ?? Math.max(24, size * (manualPoint ? 0.5 : radiusSeed * 1.18))) * (manualPoint ? manualScale : 1);
  const detectorCenter = detectorCandidate
    ? { x: detectorCandidate.box.x + detectorCandidate.box.width / 2, y: detectorCandidate.box.y + detectorCandidate.box.height / 2 }
    : null;
  const rawCenter = detectorCenter ?? autoCenter;
  const inferredCenter =
    manualPoint || distance(rawCenter, graspCenter) < size * 0.72
      ? rawCenter
      : {
          x: rawCenter.x * 0.38 + graspCenter.x * 0.62,
          y: rawCenter.y * 0.38 + graspCenter.y * 0.62
        };
  const previousDrift = previous ? distance(inferredCenter, previous.center) / Math.max(1, size) : 0;
  const objectFirstAlgorithm = algorithmVersion === 'v2' || algorithmVersion === 'v3';
  const previousStable =
    previous?.locked &&
    (previous.lockAgeFrames ?? 0) > 6 &&
    (previous.independentEvidenceScore ?? 0) > 0.42 &&
    (previous.tightness ?? 0) > 0.38;
  const center =
    previousStable && !manualOrDetector && previousDrift > 0.42
      ? {
          x: previous!.center.x * 0.7 + inferredCenter.x * 0.3,
          y: previous!.center.y * 0.7 + inferredCenter.y * 0.3
        }
      : inferredCenter;
  const relativeDriftScore = clamp(previous ? distance(center, previous.center) / Math.max(1, size * 0.62) : 0);
  const imageEvidence = sampleRegionEvidence(video, center, Math.max(radiusX, radiusY));
  const strongVisualObject = imageEvidence.edgeEnergy > 0.42 || imageEvidence.colorVariance > 0.46;
  const previousIndependentEvidence = previous?.independentEvidenceScore ?? 0;
  const stablePreviousAuto =
    previous?.source === 'automatic' &&
    previousIndependentEvidence > 0.52 &&
    (previous.lockAgeFrames ?? 0) > 8 &&
    (previous.tightness ?? 0) > 0.48;
  const staleAutomaticLock =
    !locked &&
    !manualPoint &&
    !detectorCandidate &&
    previous?.source === 'automatic' &&
    (previous.lockAgeFrames ?? 0) > 8;
  if (openHandScore > 0.62 && !strongVisualObject && (!previous || staleAutomaticLock)) return null;
  if (
    objectFirstAlgorithm &&
    !manualOrDetector &&
    !stablePreviousAuto &&
    (openHandScore > 0.52 || !strongVisualObject)
  ) {
    return null;
  }

  const nearHand = tips.some((tip) => distance(tip, center) < size * 0.78) && distance(center, palm) < size * 0.84;
  const lockConfidence = manualPoint ? 0.92 : locked && previous ? 0.72 : 0;
  const aspectRatio = Math.max(radiusX, radiusY) / Math.max(1, Math.min(radiusX, radiusY));
  const phoneLike = aspectRatio > 1.35 && imageEvidence.edgeEnergy > 0.22;
  const tightness = clamp(1 - distance(center, graspCenter) / Math.max(1, size * 0.95));
  const independentEvidenceScore = clamp(
    Math.max(
      manualPoint ? 0.96 : 0,
      detectorCandidate ? 0.82 : 0,
      imageEvidence.edgeEnergy * 0.58 + imageEvidence.colorVariance * 0.28 + (phoneLike ? 0.14 : 0),
      previousIndependentEvidence * (distance(center, previous?.center ?? center) < size * 0.28 ? 0.9 : 0)
    ) +
      (tightness > 0.52 ? 0.05 : 0)
  );
  const confidence = clamp(
    lockConfidence +
      (detectorCandidate ? 0.22 : 0) +
      (nearHand ? 0.24 : 0) +
      tightness * 0.2 +
      imageEvidence.edgeEnergy * 0.24 +
      imageEvidence.colorVariance * 0.1
  );
  if (
    objectFirstAlgorithm &&
    !manualOrDetector &&
    !stablePreviousAuto &&
    (independentEvidenceScore < 0.42 || tightness < 0.36 || !nearHand || relativeDriftScore > 0.72)
  ) {
    return null;
  }

  const objectLocked = Boolean(manualPoint || detectorCandidate || locked || confidence > 0.48);
  if (!objectLocked && confidence < 0.35) return null;

  const angle = Math.atan2(hand[12].y - hand[4].y, hand[12].x - hand[4].x);
  const velocity = previous ? subtract(center, previous.center) : { x: 0, y: 0 };
  const contour = Array.from({ length: 28 }, (_item, index) => ellipsePoint(
    {
      center,
      radiusX,
      radiusY,
      angle,
      confidence,
      locked: objectLocked,
      source: manualPoint ? 'manual' : detectorCandidate ? 'detector' : 'automatic',
      velocity,
      contour: [],
      shape: phoneLike ? 'phone-like' : aspectRatio > 1.18 ? 'ellipse' : 'unknown',
      aspectRatio,
      tightness,
      lockAgeFrames: objectLocked ? (previous?.lockAgeFrames ?? 0) + 1 : 0,
      manuallyAdjusted: Boolean(manualPoint),
      visualEdgeScore: imageEvidence.edgeEnergy,
      visualTextureScore: imageEvidence.colorVariance,
      independentEvidenceScore,
      relativeDriftScore,
      detectorLabel: detectorCandidate?.label,
      detectorScore: detectorCandidate?.score
    },
    (index / 28) * Math.PI * 2
  ));

  return {
    center,
    radiusX,
    radiusY,
    angle,
    confidence,
    locked: objectLocked,
    source: manualPoint ? 'manual' : detectorCandidate ? 'detector' : 'automatic',
    velocity,
    contour,
    shape: phoneLike ? 'phone-like' : aspectRatio > 1.18 ? 'ellipse' : 'unknown',
    aspectRatio,
    tightness,
    lockAgeFrames: objectLocked ? (previous?.lockAgeFrames ?? 0) + 1 : 0,
    manuallyAdjusted: Boolean(manualPoint),
    visualEdgeScore: imageEvidence.edgeEnergy,
    visualTextureScore: imageEvidence.colorVariance,
    independentEvidenceScore,
    relativeDriftScore,
    detectorLabel: detectorCandidate?.label,
    detectorScore: detectorCandidate?.score
  };
}

function sanitizeDetectorBox(detection: DetectedObjectBox | null | undefined, palm: Point, tips: Point[], size: number) {
  if (!detection) return null;
  const box = detection.box;
  const center = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  const closeToHand = tips.some((tip) => distance(tip, center) < size * 1.05) || distance(center, palm) < size * 0.95;
  const tooHuge = box.width * box.height > size * size * 4.4 || Math.max(box.width, box.height) > size * 2.35;
  const tooTiny = box.width < size * 0.14 || box.height < size * 0.14;
  if (!closeToHand || tooHuge || tooTiny) return null;
  return detection;
}

function computeOpenHandScore(hand: Landmark[], size: number, palm: Point) {
  const extendedTips = [8, 12, 16, 20].map((index) => clamp((distance(hand[index], palm) - size * 0.34) / Math.max(1, size * 0.48)));
  const thumbExtended = clamp((distance(hand[4], palm) - size * 0.22) / Math.max(1, size * 0.38));
  const spread = clamp(distance(hand[8], hand[20]) / Math.max(1, size * 0.86));
  return clamp((extendedTips.reduce((sum, value) => sum + value, 0) / extendedTips.length) * 0.62 + thumbExtended * 0.18 + spread * 0.2);
}

function sampleRegionEvidence(video: HTMLVideoElement, center: Point, radius: number) {
  if (!sampleCanvas) {
    sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = SAMPLE_SIZE;
    sampleCanvas.height = SAMPLE_SIZE;
    sampleContext = sampleCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!sampleContext) return { edgeEnergy: 0, colorVariance: 0 };

  const sourceSize = Math.max(24, radius * 2.4);
  const sx = clamp(center.x - sourceSize / 2, 0, Math.max(0, video.videoWidth - sourceSize));
  const sy = clamp(center.y - sourceSize / 2, 0, Math.max(0, video.videoHeight - sourceSize));
  sampleContext.drawImage(video, sx, sy, sourceSize, sourceSize, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  const { data } = sampleContext.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  let gradientTotal = 0;
  let colorTotal = 0;
  let colorSquaredTotal = 0;
  let samples = 0;

  for (let y = 1; y < SAMPLE_SIZE - 1; y += 3) {
    for (let x = 1; x < SAMPLE_SIZE - 1; x += 3) {
      const index = (y * SAMPLE_SIZE + x) * 4;
      const right = (y * SAMPLE_SIZE + x + 1) * 4;
      const down = ((y + 1) * SAMPLE_SIZE + x) * 4;
      const luminance = data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722;
      const luminanceRight = data[right] * 0.2126 + data[right + 1] * 0.7152 + data[right + 2] * 0.0722;
      const luminanceDown = data[down] * 0.2126 + data[down + 1] * 0.7152 + data[down + 2] * 0.0722;
      const chroma = Math.max(data[index], data[index + 1], data[index + 2]) - Math.min(data[index], data[index + 1], data[index + 2]);
      gradientTotal += Math.abs(luminance - luminanceRight) + Math.abs(luminance - luminanceDown);
      colorTotal += chroma;
      colorSquaredTotal += chroma * chroma;
      samples += 1;
    }
  }

  const meanColor = colorTotal / Math.max(1, samples);
  const variance = colorSquaredTotal / Math.max(1, samples) - meanColor * meanColor;
  return {
    edgeEnergy: clamp(gradientTotal / Math.max(1, samples) / 62),
    colorVariance: clamp(Math.sqrt(Math.max(0, variance)) / 55)
  };
}
