import { describe, expect, it } from 'vitest';
import { isManualLockAnchorStale, shouldRejectSkinOnlyManualLock } from './objectTracking';

describe('object tracking', () => {
  it('keeps a fresh manual lock anchor near the current grasp', () => {
    const stale = isManualLockAnchorStale(
      { x: 100, y: 100 },
      { x: 124, y: 110 },
      { source: 'manual' },
      true,
      80
    );

    expect(stale).toBe(false);
  });

  it('treats an old manual click as stale after the grasp moves away', () => {
    const stale = isManualLockAnchorStale(
      { x: 100, y: 100 },
      { x: 220, y: 180 },
      { source: 'manual' },
      true,
      80
    );

    expect(stale).toBe(true);
  });

  it('does not mark detector or automatic locks as stale manual anchors', () => {
    const stale = isManualLockAnchorStale(
      { x: 100, y: 100 },
      { x: 220, y: 180 },
      { source: 'automatic' },
      true,
      80
    );

    expect(stale).toBe(false);
  });

  it('rejects a manual lock that looks like skin-only palm area', () => {
    expect(
      shouldRejectSkinOnlyManualLock({
        objectFirstAlgorithm: true,
        hasManualPoint: true,
        hasDetectorCandidate: false,
        skinSimilarity: 0.86,
        edgeEnergy: 0.18,
        colorVariance: 0.14,
        centerDistanceFromPalm: 32,
        handSizeValue: 90
      })
    ).toBe(true);
  });

  it('keeps a manual lock when the region has independent object evidence', () => {
    expect(
      shouldRejectSkinOnlyManualLock({
        objectFirstAlgorithm: true,
        hasManualPoint: true,
        hasDetectorCandidate: false,
        skinSimilarity: 0.4,
        edgeEnergy: 0.48,
        colorVariance: 0.36,
        centerDistanceFromPalm: 32,
        handSizeValue: 90
      })
    ).toBe(false);
  });
});
