import { describe, expect, it } from 'vitest';
import { EMPTY_TEMPORAL_IDENTITY, temporalIdentityToMatch, updateTemporalIdentity } from './temporalIdentity';

describe('temporal identity', () => {
  it('requires repeated matches before becoming stable', () => {
    const candidate = {
      candidateId: 'bottle-1',
      profileId: 'bottle',
      name: 'Bottle',
      score: 0.82,
      matched: true,
      center: { x: 100, y: 100 },
      radiusX: 20,
      radiusY: 40,
      aspectRatio: 2,
      descriptorQuality: 0.76
    };

    const one = updateTemporalIdentity(EMPTY_TEMPORAL_IDENTITY, candidate);
    const two = updateTemporalIdentity(one, candidate);
    const three = updateTemporalIdentity(two, candidate);

    expect(one.stable).toBe(false);
    expect(two.stable).toBe(false);
    expect(three.stable).toBe(true);
    expect(temporalIdentityToMatch(three)?.matched).toBe(true);
  });

  it('decays instead of instantly dropping on a short miss', () => {
    const stable = {
      profileId: 'bottle',
      name: 'Bottle',
      score: 0.82,
      streak: 4,
      missed: 0,
      stable: true
    };

    const next = updateTemporalIdentity(stable, null);
    expect(next.score).toBeLessThan(stable.score);
    expect(next.stable).toBe(false);
  });
});
