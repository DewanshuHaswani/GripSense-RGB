import type { Landmark, ObjectRegion, Point } from './types';
import { averagePoint, clamp, distance, ellipsePoint, fingertipPoints, handSize, palmCenter } from './geometry';

export type TrainingQualityLabel = 'Rejected' | 'Needs more angles' | 'Mask too loose' | 'Good view' | 'Ready to train';
export type TrainingViewRole = 'front' | 'side' | 'rotated' | 'in-hand' | 'alone' | 'negative';
export type ProfileStrength = 'weak profile' | 'good profile' | 'robust profile';

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
  viewRole?: TrainingViewRole;
  descriptorVariants?: number[][];
};

export type CanvasObjectMaskOptions = {
  cropBounds: { x: number; y: number; size: number };
  maskScale: number;
  maskShape: 'ellipse' | 'rect';
  source?: ObjectTrainingSampleV2['source'];
  sourceName?: string;
  viewRole?: TrainingViewRole;
};

export type ObjectProfileV2 = {
  id: string;
  name: string;
  enabled: boolean;
  samples: ObjectTrainingSampleV2[];
  descriptor: number[];
  exemplarDescriptors?: number[][];
  descriptorVariance: number;
  minTrainingQuality: number;
  recommendedViewCount: number;
  negativeDescriptor?: number[];
  strength?: ProfileStrength;
  coverageScore?: number;
  createdAt: number;
  updatedAt: number;
};

export type ObjectProfileMatch = {
  profileId: string;
  name: string;
  score: number;
  matched: boolean;
} | null;

export type ObjectProfileCandidate = {
  candidateId: string;
  profileId: string;
  name: string;
  score: number;
  matched: boolean;
  center: Point;
  radiusX: number;
  radiusY: number;
  aspectRatio: number;
  descriptorQuality: number;
  scanRank?: number;
};

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
export const OBJECT_SEARCH_THRESHOLD = 0.58;
export const OBJECT_MATCH_MARGIN = 0.07;
export const V5_OBJECT_MATCH_THRESHOLD = 0.56;
export const V5_OBJECT_SEARCH_THRESHOLD = 0.52;

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
    source: 'locked-crop',
    viewRole: 'in-hand'
  };
}

export function createCanvasObjectTrainingSample(
  canvas: HTMLCanvasElement,
  source: ObjectTrainingSampleV2['source'] = 'camera',
  sourceName?: string
): ObjectTrainingSampleV2 | null {
  const size = Math.min(canvas.width, canvas.height);
  return createMaskedCanvasObjectTrainingSample(canvas, {
    cropBounds: {
      x: Math.max(0, (canvas.width - size) / 2),
      y: Math.max(0, (canvas.height - size) / 2),
      size
    },
    maskScale: 0.86,
    maskShape: 'ellipse',
    source,
    sourceName
  });
}

export function createMaskedCanvasObjectTrainingSample(
  canvas: HTMLCanvasElement,
  options: CanvasObjectMaskOptions
): ObjectTrainingSampleV2 | null {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context || !canvas.width || !canvas.height) return null;
  const size = clampCropSize(options.cropBounds.size, canvas);
  const sourceX = clamp(options.cropBounds.x, 0, Math.max(0, canvas.width - size));
  const sourceY = clamp(options.cropBounds.y, 0, Math.max(0, canvas.height - size));
  const maskScale = clamp(options.maskScale, 0.35, 1);
  const square = document.createElement('canvas');
  square.width = DESCRIPTOR_SIZE;
  square.height = DESCRIPTOR_SIZE;
  const squareContext = square.getContext('2d', { willReadFrequently: true });
  if (!squareContext) return null;
  squareContext.drawImage(canvas, sourceX, sourceY, size, size, 0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE);
  applyObjectMask(squareContext, DESCRIPTOR_SIZE, maskScale, options.maskShape);
  const object = {
    center: { x: DESCRIPTOR_SIZE / 2, y: DESCRIPTOR_SIZE / 2 },
    radiusX: DESCRIPTOR_SIZE * 0.43 * maskScale,
    radiusY: DESCRIPTOR_SIZE * 0.43 * maskScale,
    angle: 0,
    shape: (options.maskShape === 'ellipse' ? 'ellipse' : 'unknown') as ObjectRegion['shape']
  };
  const descriptor = describeImageData(squareContext.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), object);
  if (!descriptor) return null;
  const descriptorVariants = buildAugmentedDescriptors(square, object);

  const thumbnail = document.createElement('canvas');
  thumbnail.width = THUMBNAIL_SIZE;
  thumbnail.height = THUMBNAIL_SIZE;
  const thumbnailContext = thumbnail.getContext('2d');
  if (!thumbnailContext) return null;
  thumbnailContext.drawImage(canvas, sourceX, sourceY, size, size, 0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
  applyObjectMask(thumbnailContext, THUMBNAIL_SIZE, maskScale, options.maskShape);
  return {
    id: crypto.randomUUID(),
    imageDataUrl: thumbnail.toDataURL('image/png'),
    descriptor,
    cropBounds: { x: sourceX, y: sourceY, size },
    objectRegion: object,
    quality: descriptor.quality,
    qualityLabel: descriptor.qualityLabel,
    createdAt: Date.now(),
    source: options.source,
    sourceName: options.sourceName,
    viewRole: options.viewRole,
    descriptorVariants
  };
}

function applyObjectMask(
  context: CanvasRenderingContext2D,
  size: number,
  maskScale: number,
  maskShape: CanvasObjectMaskOptions['maskShape']
) {
  context.save();
  context.globalCompositeOperation = 'destination-in';
  context.beginPath();
  const inset = (size * (1 - maskScale)) / 2;
  if (maskShape === 'rect') {
    const radius = Math.max(4, size * 0.08);
    roundedRect(context, inset, inset, size - inset * 2, size - inset * 2, radius);
  } else {
    context.ellipse(size / 2, size / 2, (size * maskScale) / 2, (size * maskScale) / 2, 0, 0, Math.PI * 2);
  }
  context.fill();
  context.restore();
}

function buildAugmentedDescriptors(source: HTMLCanvasElement, object: Pick<ObjectRegion, 'radiusX' | 'radiusY'>) {
  const variants = [
    { flip: true, brightness: 1, contrast: 1 },
    { flip: false, brightness: 1.1, contrast: 1.04 },
    { flip: false, brightness: 0.9, contrast: 1.08 },
    { flip: false, brightness: 1, contrast: 0.9 }
  ];
  return variants
    .map((variant) => {
      const canvas = document.createElement('canvas');
      canvas.width = DESCRIPTOR_SIZE;
      canvas.height = DESCRIPTOR_SIZE;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) return null;
      context.save();
      if (variant.flip) {
        context.translate(DESCRIPTOR_SIZE, 0);
        context.scale(-1, 1);
      }
      context.drawImage(source, 0, 0);
      context.restore();
      const imageData = context.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE);
      adjustImageData(imageData, variant.brightness, variant.contrast);
      context.putImageData(imageData, 0, 0);
      return describeImageData(context.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), object)?.vector ?? null;
    })
    .filter((vector): vector is number[] => Boolean(vector));
}

function adjustImageData(imageData: ImageData, brightness: number, contrast: number) {
  const { data } = imageData;
  for (let index = 0; index < data.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const centered = data[index + channel] - 128;
      data[index + channel] = clamp(centered * contrast + 128 * brightness, 0, 255);
    }
  }
}

function roundedRect(context: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
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

function clampCropSize(size: number, canvas: HTMLCanvasElement) {
  return clamp(size, Math.min(32, canvas.width, canvas.height), Math.min(canvas.width, canvas.height));
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
  const ringBins = new Array(3).fill(0);
  const rgbBins = new Array(6).fill(0);
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
      rgbBins[0] += r * alpha;
      rgbBins[1] += g * alpha;
      rgbBins[2] += b * alpha;
      rgbBins[3] += Math.abs(r - g) * alpha;
      rgbBins[4] += Math.abs(g - b) * alpha;
      rgbBins[5] += Math.abs(b - r) * alpha;

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
      const dx = x / width - 0.5;
      const dy = y / height - 0.5;
      const ringIndex = Math.min(2, Math.floor(Math.hypot(dx, dy) * 4.8));
      ringBins[ringIndex] += luminance * alpha;
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
  normalizeBins(ringBins);
  normalizeBins(rgbBins);

  const quality = scoreSampleQuality({ maskCoverage, foregroundContrast, edgeStrength, textureStrength, aspectRatio });
  const qualityLabel = labelSampleQuality(quality, maskCoverage);
  const vector = [
    ...hueBins,
    ...saturationBins,
    ...valueBins,
    ...edgeBins,
    ...gridBins,
    ...ringBins,
    ...rgbBins,
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
  const positiveSamples = samples.filter((sample) => sample.viewRole !== 'negative');
  const goodSamples = positiveSamples.filter((sample) => sample.quality >= MIN_SAMPLE_QUALITY);
  const coverage = trainingCoverage(samples);
  const strength = profileStrength(samples);
  if (!samples.length) {
    return { ready: false, label: 'Needs more angles' as TrainingQualityLabel, message: 'Capture 3 good masked views.' };
  }
  if (!positiveSamples.length) {
    return { ready: false, label: 'Needs more angles' as TrainingQualityLabel, message: 'Add at least one positive object view. Negative examples help later, but cannot define the object.' };
  }
  if (positiveSamples.some((sample) => sample.qualityLabel === 'Mask too loose')) {
    return { ready: goodSamples.length >= 1, label: 'Mask too loose' as TrainingQualityLabel, message: 'Some views look loose. You can train, but add tighter object-only images for better matching.' };
  }
  if (goodSamples.length < RECOMMENDED_VIEW_COUNT) {
    return {
      ready: true,
      label: 'Needs more angles' as TrainingQualityLabel,
      message: `You can train now. Add ${RECOMMENDED_VIEW_COUNT - goodSamples.length} more good view${RECOMMENDED_VIEW_COUNT - goodSamples.length === 1 ? '' : 's'} for a stronger profile.`
    };
  }
  return {
    ready: true,
    label: 'Ready to train' as TrainingQualityLabel,
    message: `Ready to train. Coverage: ${Math.round(coverage * 100)}%. Expected result: ${strength}.`
  };
}

export function trainingCoverage(samples: ObjectTrainingSampleV2[]) {
  const roles = new Set(samples.map((sample) => sample.viewRole).filter(Boolean));
  const roleScore =
    ['front', 'side', 'rotated', 'in-hand', 'alone'].filter((role) => roles.has(role as TrainingViewRole)).length / 5;
  const negativeScore = roles.has('negative') ? 1 : 0;
  const goodPositive = samples.filter((sample) => sample.viewRole !== 'negative' && sample.quality >= MIN_SAMPLE_QUALITY).length;
  const countScore = clamp(goodPositive / 5);
  return clamp(roleScore * 0.58 + negativeScore * 0.16 + countScore * 0.26);
}

export function profileStrength(samples: ObjectTrainingSampleV2[]): ProfileStrength {
  const coverage = trainingCoverage(samples);
  if (coverage >= 0.76) return 'robust profile';
  if (coverage >= 0.46) return 'good profile';
  return 'weak profile';
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
  const positiveSamples = samples.filter((sample) => sample.viewRole !== 'negative');
  const negativeSamples = samples.filter((sample) => sample.viewRole === 'negative');
  if (!positiveSamples.length) return { ok: false, label: 'Needs more angles', message: 'Add at least one positive object image before training.' };
  const goodSamples = positiveSamples.filter((sample) => sample.quality >= MIN_SAMPLE_QUALITY);
  const trainingSamples = goodSamples.length ? goodSamples : positiveSamples;
  const descriptorVectors = trainingSamples.flatMap((sample) => [sample.descriptor.vector, ...(sample.descriptorVariants ?? [])]);
  const descriptor = averageDescriptor(descriptorVectors);
  const negativeVectors = negativeSamples.flatMap((sample) => [sample.descriptor.vector, ...(sample.descriptorVariants ?? [])]);
  const coverageScore = trainingCoverage(samples);
  const strength = profileStrength(samples);
  return {
    ok: true,
    label: readiness.label,
    message: `${trimmed} trained successfully with ${samples.length} image${samples.length === 1 ? '' : 's'} using augmented descriptors. Profile strength: ${strength}. ${readiness.message}`,
    profile: {
      id: existingId ?? crypto.randomUUID(),
      name: trimmed,
      enabled: true,
      samples,
      descriptor,
      exemplarDescriptors: descriptorVectors.slice(0, 96).map((vector) => [...vector]),
      descriptorVariance: descriptorVariance(descriptorVectors, descriptor),
      minTrainingQuality: Math.min(...samples.map((sample) => sample.quality)),
      recommendedViewCount: RECOMMENDED_VIEW_COUNT,
      negativeDescriptor: negativeVectors.length ? averageDescriptor(negativeVectors) : undefined,
      strength,
      coverageScore,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }
  };
}

export function matchObjectProfiles(
  descriptor: ObjectDescriptor | null,
  profiles: ObjectProfileV2[],
  options: { threshold?: number; margin?: number; useExemplars?: boolean } = {}
): ObjectProfileMatch {
  if (!descriptor || !profiles.length || descriptor.quality < 0.36) return null;
  const threshold = options.threshold ?? OBJECT_MATCH_THRESHOLD;
  const margin = options.margin ?? OBJECT_MATCH_MARGIN;
  const ranked = profiles
    .map((profile) => {
      const distance = descriptorDistance(descriptor.vector, profile.descriptor);
      const exemplarDistance =
        options.useExemplars && profile.exemplarDescriptors?.length
          ? Math.min(...profile.exemplarDescriptors.map((vector) => descriptorDistance(descriptor.vector, vector)))
          : distance;
      const blendedDistance = Math.min(distance * 0.72 + exemplarDistance * 0.28, exemplarDistance * 1.08, distance);
      const tolerance = clamp(0.48 + profile.descriptorVariance * 1.6, 0.48, 0.78);
      const qualityFactor = clamp(descriptor.quality * 0.68 + profile.minTrainingQuality * 0.32, 0.35, 1);
      const strengthBoost =
        profile.strength === 'robust profile' ? 0.05 : profile.strength === 'good profile' ? 0.025 : 0;
      const positiveScore = clamp((1 - blendedDistance / tolerance) * qualityFactor + strengthBoost);
      const negativeDistance = profile.negativeDescriptor ? descriptorDistance(descriptor.vector, profile.negativeDescriptor) : 1;
      const negativePenalty = profile.negativeDescriptor ? clamp((1 - negativeDistance / 0.52) * 0.38) : 0;
      const coverageBoost = clamp((profile.coverageScore ?? 0.35) * 0.08, 0, 0.08);
      const score = clamp(positiveScore + coverageBoost - negativePenalty);
      return {
        profileId: profile.id,
        name: profile.name,
        score,
        matched: score >= threshold
      };
    })
    .sort((a, b) => b.score - a.score);
  const best = ranked[0];
  if (!best) return null;
  const runnerUp = ranked[1];
  const ambiguous =
    Boolean(runnerUp) &&
    best.score - (runnerUp?.score ?? 0) < margin &&
    (runnerUp?.score ?? 0) > threshold * 0.82;
  return {
    ...best,
    matched: best.matched && !ambiguous
  };
}

export function findObjectProfileCandidates(
  video: HTMLVideoElement,
  profiles: ObjectProfileV2[],
  hand: Landmark[] | null,
  limitOrOptions: number | { limit?: number; scanMode?: 'hand-corridor' | 'all-frame'; targetProfileId?: string | null; relaxed?: boolean } = 6
): ObjectProfileCandidate[] {
  const options =
    typeof limitOrOptions === 'number'
      ? { limit: limitOrOptions, scanMode: 'hand-corridor' as const, targetProfileId: null, relaxed: false }
      : {
          limit: limitOrOptions.limit ?? 6,
          scanMode: limitOrOptions.scanMode ?? 'hand-corridor',
          targetProfileId: limitOrOptions.targetProfileId ?? null,
          relaxed: limitOrOptions.relaxed ?? false
        };
  const searchableProfiles = options.targetProfileId ? profiles.filter((profile) => profile.id === options.targetProfileId) : profiles;
  if (!searchableProfiles.length || !video.videoWidth || !video.videoHeight) return [];
  const centers = candidateCenters(video, hand, options.scanMode);
  const sizes = candidateSizes(video, hand, options.scanMode);
  const candidatesByProfile = new Map<string, ObjectProfileCandidate>();
  const threshold = options.relaxed ? V5_OBJECT_MATCH_THRESHOLD : OBJECT_MATCH_THRESHOLD;
  const searchThreshold = options.relaxed ? V5_OBJECT_SEARCH_THRESHOLD : OBJECT_SEARCH_THRESHOLD;
  let scanRank = 0;

  for (const center of centers) {
    for (const size of sizes) {
      scanRank += 1;
      const descriptor = describeVideoSquare(video, center, size);
      if (!descriptor || descriptor.quality < 0.28) continue;
      const match = matchObjectProfiles(descriptor, searchableProfiles, {
        threshold,
        margin: options.relaxed ? OBJECT_MATCH_MARGIN * 0.75 : OBJECT_MATCH_MARGIN,
        useExemplars: options.relaxed
      });
      if (!match) continue;
      const profile = searchableProfiles.find((item) => item.id === match.profileId);
      if (!profile) continue;
      const aspectRatio = averageProfileAspectRatio(profile);
      const candidate = {
        candidateId: `${match.profileId}-${scanRank}`,
        profileId: match.profileId,
        name: match.name,
        score: match.score,
        matched: match.score >= searchThreshold,
        center,
        radiusX: size * 0.26,
        radiusY: size * 0.26 * Math.min(2.6, Math.max(1, aspectRatio)),
        aspectRatio,
        descriptorQuality: descriptor.quality,
        scanRank
      };
      const existing = candidatesByProfile.get(match.profileId);
      if (!existing || candidate.score > existing.score) {
        candidatesByProfile.set(match.profileId, candidate);
      }
    }
  }

  return Array.from(candidatesByProfile.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, options.limit);
}

export function objectRegionFromProfileCandidate(candidate: ObjectProfileCandidate, previous: ObjectRegion | null): ObjectRegion {
  const radiusX = Math.max(18, candidate.radiusX);
  const radiusY = Math.max(18, candidate.radiusY);
  const confidence = clamp(candidate.score * 0.88 + candidate.descriptorQuality * 0.12);
  const detectorLabel = `profile:${candidate.name}`;
  const sameProfileAsPrevious = previous?.detectorLabel === detectorLabel;
  const region: ObjectRegion = {
    center: candidate.center,
    radiusX,
    radiusY,
    angle: sameProfileAsPrevious ? previous?.angle ?? 0 : 0,
    confidence,
    locked: candidate.matched,
    source: 'automatic',
    velocity: sameProfileAsPrevious && previous ? { x: candidate.center.x - previous.center.x, y: candidate.center.y - previous.center.y } : { x: 0, y: 0 },
    contour: [],
    shape: candidate.aspectRatio > 1.35 ? 'phone-like' : candidate.aspectRatio > 1.12 ? 'ellipse' : 'unknown',
    aspectRatio: candidate.aspectRatio,
    tightness: 0.72,
    lockAgeFrames: candidate.matched ? (sameProfileAsPrevious ? previous?.lockAgeFrames ?? 0 : 0) + 1 : 0,
    manuallyAdjusted: false,
    visualEdgeScore: candidate.descriptorQuality,
    visualTextureScore: candidate.descriptorQuality,
    independentEvidenceScore: confidence,
    relativeDriftScore: sameProfileAsPrevious && previous ? clamp(distance(candidate.center, previous.center) / Math.max(1, Math.max(radiusX, radiusY) * 1.8)) : 0,
    detectorLabel,
    detectorScore: candidate.score
  };
  region.contour = Array.from({ length: 28 }, (_item, index) => ellipsePoint(region, (index / 28) * Math.PI * 2));
  return region;
}

function describeVideoSquare(video: HTMLVideoElement, center: Point, size: number): ObjectDescriptor | null {
  const cropSize = clamp(size, 24, Math.min(video.videoWidth, video.videoHeight));
  const sourceX = clamp(center.x - cropSize / 2, 0, Math.max(0, video.videoWidth - cropSize));
  const sourceY = clamp(center.y - cropSize / 2, 0, Math.max(0, video.videoHeight - cropSize));
  const canvas = document.createElement('canvas');
  canvas.width = DESCRIPTOR_SIZE;
  canvas.height = DESCRIPTOR_SIZE;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(video, sourceX, sourceY, cropSize, cropSize, 0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE);
  const object = {
    radiusX: DESCRIPTOR_SIZE * 0.38,
    radiusY: DESCRIPTOR_SIZE * 0.38
  };
  return describeImageData(context.getImageData(0, 0, DESCRIPTOR_SIZE, DESCRIPTOR_SIZE), object);
}

function candidateCenters(video: HTMLVideoElement, hand: Landmark[] | null, scanMode: 'hand-corridor' | 'all-frame' = 'hand-corridor') {
  if (scanMode === 'all-frame') {
    const xs = [0.22, 0.38, 0.5, 0.62, 0.78];
    const ys = [0.24, 0.38, 0.52, 0.66, 0.8];
    const grid = xs.flatMap((x) => ys.map((y) => ({ x: video.videoWidth * x, y: video.videoHeight * y })));
    if (!hand?.length) return grid;
    const palm = palmCenter(hand);
    const tips = fingertipPoints(hand);
    return [
      averagePoint([palm, averagePoint(tips), hand[8], hand[12], hand[16]]),
      ...tips,
      ...grid
    ].map((point) => ({ x: clamp(point.x, 0, video.videoWidth), y: clamp(point.y, 0, video.videoHeight) }));
  }
  if (!hand?.length) {
    return [
      { x: video.videoWidth * 0.5, y: video.videoHeight * 0.5 },
      { x: video.videoWidth * 0.38, y: video.videoHeight * 0.5 },
      { x: video.videoWidth * 0.62, y: video.videoHeight * 0.5 }
    ];
  }
  const palm = palmCenter(hand);
  const tips = fingertipPoints(hand);
  const tipCenter = averagePoint(tips);
  const size = handSize(hand);
  const base = averagePoint([palm, tipCenter, hand[8], hand[12], hand[16]]);
  const offsets = [
    { x: 0, y: 0 },
    { x: size * 0.18, y: 0 },
    { x: -size * 0.18, y: 0 },
    { x: 0, y: size * 0.18 },
    { x: 0, y: -size * 0.18 },
    { x: size * 0.3, y: -size * 0.1 },
    { x: -size * 0.3, y: -size * 0.1 }
  ];
  return offsets.map((offset) => ({
    x: clamp(base.x + offset.x, 0, video.videoWidth),
    y: clamp(base.y + offset.y, 0, video.videoHeight)
  }));
}

function candidateSizes(video: HTMLVideoElement, hand: Landmark[] | null, scanMode: 'hand-corridor' | 'all-frame' = 'hand-corridor') {
  const base = hand ? handSize(hand) : Math.min(video.videoWidth, video.videoHeight) * 0.26;
  const multipliers = scanMode === 'all-frame' ? [0.42, 0.58, 0.78, 1.02, 1.28] : [0.55, 0.78, 1.02];
  return multipliers.map((size) => clamp(base * size, 42, Math.min(video.videoWidth, video.videoHeight) * 0.72));
}

function averageProfileAspectRatio(profile: ObjectProfileV2) {
  const values = profile.samples.map((sample) => sample.descriptor.aspectRatio).filter((value) => Number.isFinite(value) && value > 0);
  return values.length ? average(values) : 1;
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
