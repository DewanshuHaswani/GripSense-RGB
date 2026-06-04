import { describe, expect, it } from 'vitest';
import { classifyStableMotion } from './stabilization';
import type { MotionState } from './types';

function sample(state: MotionState, slipRisk: number, coupling = 0.88) {
  return {
    state,
    slipRisk,
    coupling,
    handSpeed: 0,
    objectSpeed: 0,
    timestamp: 0
  };
}

describe('motion stabilization', () => {
  it('keeps small jitter in idle state', () => {
    const history = [
      sample('idle', 0.04),
      sample('moving-with-hand', 0.08),
      sample('idle', 0.05),
      sample('idle', 0.04),
      sample('uncertain', 0.09),
      sample('idle', 0.03),
      sample('idle', 0.04),
      sample('idle', 0.05)
    ];

    expect(classifyStableMotion(history, 'idle')).toBe('idle');
  });

  it('requires sustained coupling before moving-with-hand', () => {
    const history = [
      sample('idle', 0.08),
      sample('moving-with-hand', 0.16, 0.78),
      sample('moving-with-hand', 0.18, 0.76),
      sample('moving-with-hand', 0.2, 0.72),
      sample('moving-with-hand', 0.16, 0.74),
      sample('idle', 0.08)
    ];

    expect(classifyStableMotion(history, 'idle')).toBe('moving-with-hand');
  });

  it('switches to slipping only after sustained slip evidence', () => {
    const history = [
      sample('moving-with-hand', 0.18, 0.72),
      sample('slipping', 0.56, 0.34),
      sample('slipping', 0.62, 0.28),
      sample('slipping', 0.58, 0.32),
      sample('slipping', 0.66, 0.3),
      sample('uncertain', 0.5, 0.38)
    ];

    expect(classifyStableMotion(history, 'moving-with-hand')).toBe('slipping');
  });
});
