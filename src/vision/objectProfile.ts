import type { ObjectRegion, Point } from './types';
import { clamp } from './geometry';

export type TrainingQualityLabel = 'Rejected' | 'Needs more angles' | 'Mask too loose' | 'Good view' | 'Ready to train';

export type ObjectDescriptor = {
  vector: number[];
  quality: number;
  qualityLabel: TrainingQualityLabel;
  reasons: string[];
  maskCoverage: number;
  foregroundContrast: number;
  edgeStrength: number;
  textureStrength: number;
  aspectRatio: number;
};

export type ObjectTrainingSampleV2 = {
  id: string;
  imageDataUrl: string;
  descriptor: ObjectDescriptor;
  cropBounds: { x: number; y: number; size: number };
  objectRegion: {
    center: Point;
    radiusX: number;
    radiusY: number;
    angle: number;
    shape: ObjectRegion['shape'];
  };
  quality: number;
  qualityLabel: TrainingQualityLabel;
  createdAt: number;
  source?: 'camera' | 'upload' | 'locked-crop';
  sourceName?: string;
};

export type ObjectProfileV2 = {
  id: string;
  name: string;
  enabled: boolean;
  samples: ObjectTrainingSampleV2[];
  descriptor: number[];
  descriptorVariance: number;
  minTrainingQuality: number;
  recommendedViewCount: number;
  createdAt: number;
  updatedAt: number;
};

export type ObjectProfileMatch = {
  profileId: string;
  name: string;
  score: number;
  matched: boolean;
} | null;

export type TrainObjectProfileResult =
  | { ok: true; profile: ObjectProfileV2; label: TrainingQualityLabel; message: string }
  | { ok: false; label: TrainingQualityLabel; message: string };

export type ObjectDescriptorProvider = {
  describe(video: HTMLVideoElement, object: ObjectRegion): ObjectDescriptor | null;
  createSample(video: HTMLVideoElement, object: ObjectRegion): ObjectTrainingSampleV2 | null;
};

export const DESCRIPTOR_SIZE = 48;
export const THUMBNAIL_SIZE = 144;
export const MIN_SAMPLE_QUALITY = 0.56;
export const RECOMMENDED_VIEW_COUNT = 3;
export const OBJECT_MATCH_THRESHOLD = 0.62;

export const browserObjectDescriptorProvider: ObjectDescriptorProvider = {
  describe: describeObjectPatch,
  createSample: createBrowserObjectTrainingSample
};

export function createBrowserObjectTrainingSample(
  video: HTMLVideoElement,
  object: ObjectRegion
): ObjectTrainingSampleV2 | null {
  const descriptor = describeObjectPatch(video, object);
  const imageDataUrl = createObjectThumbnail(video, object);
  const cropBounds = cropBoundsFor(video, object);
  if (!descriptor || !imageDataUrl || !cropBounds) return null;
  return {
    id: crypto.randomUUID(),
    imageDataUrl,
    descriptor,
    cropBounds,
    objectRegion: {
      center: object.center,
      radiusX: object.radiusX,
      radiusY: object.radiusY,
      angle: object.angle,
      shape: object.shape
    },
    quality: descriptor.quality,
    qualityLabel: descriptor.qualityLabel,
    createdAt: Date.now(),
    source: 'locked-crop'
  };
}

export function createCanvasObjectTrainingSample(
  canvas: HTMLCanvasElement,
  source: ObjectTrainingSampleV2['source'] = 'camera',
  sourceName?: string
): ObjectTrainingSampleV2 | null {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context || !canvas.width || !canvas.height) return null;
  const size = Math.min(canvas.width, canvas.height);
  const square = document.createElement('canvas');
  square.width = DESCRIPTOR_SIZE;
  square.height = DESCRIPTOR_SIZE;
  const squareContext = square.getContext('2d', { willReadFrequently: true });
  if (!squareContext) return null;
  const sourceX = Math.max(0, (canvas.width - size) / 2);
  const sourceY = Math.max(0, (canvas.height - size) / 2);
  squareContext.drawImage(canvas, sourceX, sourceY, size, size, 0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE);
  const object = {
    center: { x: DESCRIPTOR_SIZE / 2, y: DESCRIPTOR_SIZE / 2 },
    radiusX: DESCRIPTOR_SIZE * 0.36,
    radiusY: DESCRIPTOR_SIZE * 0.36,
    angle: 0,
    shape: 'unknown' as ObjectRegion['shape']
  };
  const descriptor = describeImageData(squareContext.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), object);
  if (!descriptor) return null;

  const thumbnail = document.createElement('canvas');
  thumbnail.width = THUMBNAIL_SIZE;
  thumbnail.height = THUMBNAIL_SIZE;
  const thumbnailContext = thumbnail.getContext('2d');
  if (!thumbnailContext) return null;
  thumbnailContext.drawImage(canvas, sourceX, sourceY, size, size, 0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
  return {
    id: crypto.randomUUID(),
    imageDataUrl: thumbnail.toDataURL('image/jpeg', 0.84),
    descriptor,
    cropBounds: { x: sourceX, y: sourceY, size },
    objectRegion: object,
    quality: descriptor.quality,
    qualityLabel: descriptor.qualityLabel,
    createdAt: Date.now(),
    source,
    sourceName
  };
}

export function describeObjectPatch(video: HTMLVideoElement, object: ObjectRegion): ObjectDescriptor | null {
  const canvas = renderObjectPatch(video, object, DESCRIPTOR_SIZE);
  if (!canvas) return null;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return null;
  return describeImageData(context.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), object);
}

export function describeImageData(imageData: ImageData, object: Pick<ObjectRegion, 'radiusX' | 'radiusY'>): ObjectDescriptor | null {
  const { data, width, height } = imageData;
  const hueBins = new Array(8).fill(0);
  const saturationBins = new Array(4).fill(0);
  const valueBins = new Array(4).fill(0);
  const edgeBins = new Array(8).fill(0);
  const gridBins = new Array(4).fill(0);
  const insideLum: number[] = [];
  const borderLum: number[] = [];
  let foregroundPixels = 0;
  let edgeTotal = 0;
  let textureTotal = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = (y * width + x) * 4;
      const alpha = data[index + 3] / 255;
      const luminance = luminanceAt(data, index);
      if (alpha < 0.08) {
        if (x < 3 || y < 3 || x > width - 4 || y > height - 4) borderLum.push(luminance);
        continue;
      }

      const r = data[index] / 255;
      const g = data[index + 1] / 255;
      const b = data[index + 2] / 255;
      const hsv = rgbToHsv(r, g, b);
      hueBins[Math.min(7, Math.floor(hsv.h * 8))] += alpha;
      saturationBins[Math.min(3, Math.floor(hsv.s * 4))] += alpha;
      valueBins[Math.min(3, Math.floor(hsv.v * 4))] += alpha;

      const lumLeft = luminanceAt(data, (y * width + x - 1) * 4);
      const lumRight = luminanceAt(data, (y * width + x + 1) * 4);
      const lumUp = luminanceAt(data, ((y - 1) * width + x) * 4);
      const lumDown = luminanceAt(data, ((y + 1) * width + x) * 4);
      const gx = lumRight - lumLeft;
      const gy = lumDown - lumUp;
      const magnitude = Math.hypot(gx, gy);
      const angle = (Math.atan2(gy, gx) + Math.PI) / (Math.PI * 2);
      edgeBins[Math.min(7, Math.floor(angle * 8))] += magnitude * alpha;
      edgeTotal += magnitude * alpha;

      const gridIndex = (x < width / 2 ? 0 : 1) + (y < height / 2 ? 0 : 2);
      gridBins[gridIndex] += luminance * alpha;
      insideLum.push(luminance);
      textureTotal += Math.abs(luminance - average([lumLeft, lumRight, lumUp, lumDown]));
      foregroundPixels += 1;
    }
  }

  const maskCoverage = foregroundPixels / Math.max(1, width * height);
  if (foregroundPixels < width * height * 0.08) return null;
  const foregroundMean = average(insideLum);
  const borderMean = borderLum.length ? average(borderLum) : foregroundMean;
  const foregroundContrast = clamp(Math.abs(foregroundMean - borderMean) / 100);
  const edgeStrength = clamp(edgeTotal / Math.max(1, foregroundPixels) / 72);
  const textureStrength = clamp(textureTotal / Math.max(1, foregroundPixels) / 34);
  const aspectRatio = Math.max(object.radiusX, object.radiusY) / Math.max(1, Math.min(object.radiusX, object.radiusY));

  normalizeBins(hueBins);
  normalizeBins(saturationBins);
  normalizeBins(valueBins);
  normalizeBins(edgeBins);
  normalizeBins(gridBins);

  const quality = scoreSampleQuality({ maskCoverage, foregroundContrast, edgeStrength, textureStrength, aspectRatio });
  const qualityLabel = labelSampleQuality(quality, maskCoverage);
  const vector = [
    ...hueBins,
    ...saturationBins,
    ...valueBins,
    ...edgeBins,
    ...gridBins,
    maskCoverage,
    foregroundContrast,
    edgeStrength,
    textureStrength,
    clamp(aspectRatio / 3)
  ];

  return {
    vector,
    quality,
    qualityLabel,
    reasons: qualityReasons({ quality, maskCoverage, foregroundContrast, edgeStrength, textureStrength }),
    maskCoverage,
    foregroundContrast,
    edgeStrength,
    textureStrength,
    aspectRatio
  };
}

export function trainingReadiness(samples: ObjectTrainingSampleV2[]) {
  const goodSamples = samples.filter((sample) => sample.quality >= MIN_SAMPLE_QUALITY);
  if (!samples.length) {
    return { ready: false, label: 'Needs more angles' as TrainingQualityLabel, message: 'Capture 3 good masked views.' };
  }
  if (samples.some((sample) => sample.qualityLabel === 'Mask too loose')) {
    return { ready: goodSamples.length >= 1, label: 'Mask too loose' as TrainingQualityLabel, message: 'Some views look loose. You can train, but add tighter object-only images for better matching.' };
  }
  if (goodSamples.length < RECOMMENDED_VIEW_COUNT) {
    return {
      ready: true,
      label: 'Needs more angles' as TrainingQualityLabel,
      message: `You can train now. Add ${RECOMMENDED_VIEW_COUNT - goodSamples.length} more good view${RECOMMENDED_VIEW_COUNT - goodSamples.length === 1 ? '' : 's'} for a stronger profile.`
    };
  }
  return { ready: true, label: 'Ready to train' as TrainingQualityLabel, message: 'Ready to train this object profile.' };
}

export function trainObjectProfileV2(
  name: string,
  samples: ObjectTrainingSampleV2[],
  existingId?: string
): TrainObjectProfileResult {
  const trimmed = name.trim();
  if (!trimmed) return { ok: false, label: 'Rejected', message: 'Give the object a name before training.' };
  if (!samples.length) return { ok: false, label: 'Needs more angles', message: 'Add at least one image before training.' };
  const readiness = trainingReadiness(samples);
  const goodSamples = samples.filter((sample) => sample.quality >= MIN_SAMPLE_QUALITY);
  const trainingSamples = goodSamples.length ? goodSamples : samples;
  const descriptor = averageDescriptor(trainingSamples.map((sample) => sample.descriptor.vector));
  return {
    ok: true,
    label: readiness.label,
    message: `${trimmed} trained successfully with ${samples.length} image${samples.length === 1 ? '' : 's'}. ${readiness.message}`,
    profile: {
      id: existingId ?? crypto.randomUUID(),
      name: trimmed,
      enabled: true,
      samples,
      descriptor,
      descriptorVariance: descriptorVariance(trainingSamples.map((sample) => sample.descriptor.vector), descriptor),
      minTrainingQuality: Math.min(...samples.map((sample) => sample.quality)),
      recommendedViewCount: RECOMMENDED_VIEW_COUNT,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }
  };
}

export function matchObjectProfiles(
  descriptor: ObjectDescriptor | null,
  profiles: ObjectProfileV2[]
): ObjectProfileMatch {
  if (!descriptor || !profiles.length || descriptor.quality < 0.36) return null;
  const ranked = profiles
    .map((profile) => {
      const distance = descriptorDistance(descriptor.vector, profile.descriptor);
      const tolerance = clamp(0.48 + profile.descriptorVariance * 1.6, 0.48, 0.78);
      const qualityFactor = clamp(descriptor.quality * 0.68 + profile.minTrainingQuality * 0.32, 0.35, 1);
      const score = clamp((1 - distance / tolerance) * qualityFactor);
      return {
        profileId: profile.id,
        name: profile.name,
        score,
        matched: score >= OBJECT_MATCH_THRESHOLD
      };
    })
    .sort((a, b) => b.score - a.score);
  return ranked[0] ?? null;
}

export function createObjectThumbnail(video: HTMLVideoElement, object: ObjectRegion) {
  const canvas = renderObjectPatch(video, object, THUMBNAIL_SIZE);
  return canvas?.toDataURL('image/jpeg', 0.82) ?? null;
}

export function renderObjectPatch(video: HTMLVideoElement, object: ObjectRegion, size: number) {
  if (!video.videoWidth || !video.videoHeight) return null;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  const cropBounds = cropBoundsFor(video, object);
  if (!context || !cropBounds) return null;

  context.clearRect(0, 0, size, size);
  context.save();
  context.beginPath();
  context.ellipse(size / 2, size / 2, size * 0.38, size * 0.38, object.angle, 0, Math.PI * 2);
  context.clip();
  context.drawImage(video, cropBounds.x, cropBounds.y, cropBounds.size, cropBounds.size, 0, 0, size, size);
  context.restore();
  return canvas;
}

export function cropBoundsFor(video: HTMLVideoElement, object: ObjectRegion) {
  if (!video.videoWidth || !video.videoHeight) return null;
  const cropRadius = Math.max(object.radiusX, object.radiusY) * 1.35;
  const size = Math.max(16, Math.min(cropRadius * 2, video.videoWidth, video.videoHeight));
  return {
    x: Math.min(Math.max(0, object.center.x - size / 2), Math.max(0, video.videoWidth - size)),
    y: Math.min(Math.max(0, object.center.y - size / 2), Math.max(0, video.videoHeight - size)),
    size
  };
}

export function averageDescriptor(descriptors: number[][]) {
  if (!descriptors.length) return [];
  return descriptors[0].map((_value, index) => average(descriptors.map((descriptor) => descriptor[index] ?? 0)));
}

export function descriptorDistance(a: number[], b: number[]) {
  const length = Math.max(a.length, b.length, 1);
  let total = 0;
  for (let index = 0; index < length; index += 1) {
    total += Math.abs((a[index] ?? 0) - (b[index] ?? 0));
  }
  return total / length;
}

function descriptorVariance(descriptors: number[][], centroid: number[]) {
  if (!descriptors.length) return 0;
  return average(descriptors.map((descriptor) => descriptorDistance(descriptor, centroid)));
}

function scoreSampleQuality(scores: {
  maskCoverage: number;
  foregroundContrast: number;
  edgeStrength: number;
  textureStrength: number;
  aspectRatio: number;
}) {
  const coverageScore =
    scores.maskCoverage < 0.16 || scores.maskCoverage > 0.72
      ? 0.18
      : scores.maskCoverage > 0.28 && scores.maskCoverage < 0.58
        ? 1
        : 0.62;
  const aspectScore = scores.aspectRatio > 4.2 ? 0.3 : 1;
  return clamp(
    coverageScore * 0.26 +
      scores.foregroundContrast * 0.22 +
      scores.edgeStrength * 0.24 +
      scores.textureStrength * 0.12 +
      aspectScore * 0.16
  );
}

function labelSampleQuality(quality: number, maskCoverage: number): TrainingQualityLabel {
  if (maskCoverage < 0.16 || maskCoverage > 0.72) return 'Mask too loose';
  if (quality < 0.38) return 'Rejected';
  if (quality < MIN_SAMPLE_QUALITY) return 'Needs more angles';
  return 'Good view';
}

function qualityReasons(scores: {
  quality: number;
  maskCoverage: number;
  foregroundContrast: number;
  edgeStrength: number;
  textureStrength: number;
}) {
  const reasons: string[] = [];
  if (scores.maskCoverage < 0.16) reasons.push('object crop is too small');
  if (scores.maskCoverage > 0.72) reasons.push('mask includes too much background');
  if (scores.foregroundContrast < 0.12) reasons.push('foreground/background contrast is low');
  if (scores.edgeStrength < 0.08) reasons.push('object edges are weak or blurry');
  if (scores.textureStrength > 0.16) reasons.push('texture is useful');
  if (scores.quality >= MIN_SAMPLE_QUALITY) reasons.push('view is usable');
  return reasons;
}

function normalizeBins(values: number[]) {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (total <= 0) return;
  for (let index = 0; index < values.length; index += 1) {
    values[index] = values[index] / total;
  }
}

function luminanceAt(data: Uint8ClampedArray, index: number) {
  return data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722;
}

function rgbToHsv(r: number, g: number, b: number) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) h = ((g - b) / delta) % 6;
    else if (max === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    h /= 6;
    if (h < 0) h += 1;
  }
  return {
    h,
    s: max === 0 ? 0 : delta / max,
    v: max
  };
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}
