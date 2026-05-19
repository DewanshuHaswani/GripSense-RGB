import { describe, expect, it } from 'vitest';
import {
  matchObjectProfiles,
  trainingReadiness,
  trainObjectProfileV2,
  type TrainingQualityLabel,
  type ObjectTrainingSampleV2
} from './objectProfile';

describe('object profile v2', () => {
  it('trains a profile from three good multi-view samples', () => {
    const samples = [
      sample('a', descriptor([0.64, 0.12, 0.08, 0.16])),
      sample('b', descriptor([0.6, 0.14, 0.1, 0.16])),
      sample('c', descriptor([0.62, 0.1, 0.1, 0.18]))
    ];
    const result = trainObjectProfileV2('Phone', samples);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.profile.name).toBe('Phone');
      expect(result.profile.samples).toHaveLength(3);
      expect(result.profile.minTrainingQuality).toBeGreaterThan(0.6);
    }
  });

  it('rejects training when not enough good views are present', () => {
    const result = trainObjectProfileV2('Remote', [
      sample('a', descriptor([0.4, 0.2, 0.2, 0.2])),
      sample('b', descriptor([0.42, 0.18, 0.2, 0.2]))
    ]);
    expect(result.ok).toBe(false);
    expect(result.message).toContain('more good view');
  });

  it('reports mask quality problems before training', () => {
    const readiness = trainingReadiness([
      sample('a', descriptor([0.4, 0.2, 0.2, 0.2]), 0.7, 'Mask too loose')
    ]);
    expect(readiness.ready).toBe(false);
    expect(readiness.label).toBe('Mask too loose');
  });

  it('matches a similar object descriptor against the trained profile', () => {
    const result = trainObjectProfileV2('Mug', [
      sample('a', descriptor([0.18, 0.62, 0.1, 0.1])),
      sample('b', descriptor([0.2, 0.58, 0.12, 0.1])),
      sample('c', descriptor([0.16, 0.64, 0.1, 0.1]))
    ]);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const match = matchObjectProfiles(descriptor([0.19, 0.6, 0.11, 0.1]), [result.profile]);
    expect(match?.matched).toBe(true);
    expect(match?.score).toBeGreaterThan(0.62);
  });

  it('does not match a different object descriptor', () => {
    const result = trainObjectProfileV2('Bottle', [
      sample('a', descriptor([0.7, 0.1, 0.1, 0.1])),
      sample('b', descriptor([0.68, 0.12, 0.1, 0.1])),
      sample('c', descriptor([0.72, 0.08, 0.1, 0.1]))
    ]);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const match = matchObjectProfiles(descriptor([0.08, 0.1, 0.72, 0.1]), [result.profile]);
    expect(match?.matched).toBe(false);
  });
});

function descriptor(vector: number[]) {
  return {
    vector,
    quality: 0.76,
    qualityLabel: 'Good view' as const,
    reasons: ['view is usable'],
    maskCoverage: 0.42,
    foregroundContrast: 0.5,
    edgeStrength: 0.55,
    textureStrength: 0.22,
    aspectRatio: 1.8
  };
}

function sample(
  id: string,
  objectDescriptor: ReturnType<typeof descriptor>,
  quality = objectDescriptor.quality,
  qualityLabel: TrainingQualityLabel = objectDescriptor.qualityLabel
): ObjectTrainingSampleV2 {
  return {
    id,
    imageDataUrl: 'data:image/jpeg;base64,test',
    descriptor: { ...objectDescriptor, quality, qualityLabel },
    cropBounds: { x: 0, y: 0, size: 100 },
    objectRegion: {
      center: { x: 50, y: 50 },
      radiusX: 24,
      radiusY: 38,
      angle: 0,
      shape: 'ellipse'
    },
    quality,
    qualityLabel,
    createdAt: Date.now()
  };
}
