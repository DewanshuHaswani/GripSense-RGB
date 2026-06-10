import type { AlgorithmVersion, GripMode } from './types';

export type GripScoringConfig = {
  contactRoleWeights: Record<'thumb' | 'index' | 'middle' | 'ring' | 'pinky' | 'palm', number>;
  modeWeights: Record<Exclude<GripMode, 'open hand' | 'uncertain'>, Record<string, number>>;
  confidenceWeights: Record<AlgorithmVersion, Record<string, number>>;
};

export const DEFAULT_GRIP_SCORING_CONFIG: GripScoringConfig = {
  contactRoleWeights: {
    thumb: 0.22,
    index: 0.2,
    middle: 0.18,
    ring: 0.14,
    pinky: 0.1,
    palm: 0.16
  },
  modeWeights: {
    'phone-side grip': {
      phoneSideGripScore: 0.28,
      fingerCurlScore: 0.18,
      fingerSegmentContactScore: 0.16,
      thumbSupportScore: 0.12,
      occlusionResilienceScore: 0.14,
      motionStabilityScore: 0.12
    },
    'pinch grip': {
      pinchScore: 0.34,
      thumbSupportScore: 0.2,
      visibleContactScore: 0.14,
      fingerSegmentContactScore: 0.12,
      motionStabilityScore: 0.2
    },
    'power grip': {
      powerGripScore: 0.3,
      palmObjectContainmentScore: 0.2,
      fingerCurlScore: 0.18,
      fingerSegmentContactScore: 0.16,
      motionStabilityScore: 0.16
    },
    'hook grip': {
      hookGripScore: 0.34,
      fingerCurlScore: 0.22,
      fingerSegmentContactScore: 0.2,
      motionStabilityScore: 0.14,
      palmObjectContainmentScore: 0.1
    }
  },
  confidenceWeights: {
    v1: {
      objectLockQuality: 0.42,
      gripPercentage: 0.16,
      motionCoupling: 0.12,
      closureScore: 0.12,
      bestModeScore: 0.1,
      calibration: 0.08,
      weakCalibration: -0.05
    },
    v2: {
      objectLockQuality: 0.24,
      independentObjectScore: 0.24,
      temporalLockScore: 0.1,
      gripPercentage: 0.12,
      motionCoupling: 0.1,
      closureScore: 0.08,
      bestModeScore: 0.08,
      identityMatch: 0.08,
      identityMiss: -0.08,
      calibration: 0.04,
      weakCalibration: -0.06
    },
    v3: {
      objectLockQuality: 0.24,
      independentObjectScore: 0.24,
      temporalLockScore: 0.1,
      gripPercentage: 0.12,
      motionCoupling: 0.1,
      closureScore: 0.08,
      bestModeScore: 0.08,
      identityMatch: 0.08,
      identityMiss: -0.08,
      calibration: 0.04,
      weakCalibration: -0.06
    },
    v4: {
      objectLockQuality: 0.22,
      independentObjectScore: 0.22,
      temporalLockScore: 0.14,
      gripPercentage: 0.1,
      motionCoupling: 0.1,
      closureScore: 0.08,
      bestModeScore: 0.08,
      identityMatch: 0.12,
      identityMiss: -0.12,
      calibration: 0.04,
      weakCalibration: -0.06
    },
    v5: {
      objectLockQuality: 0.18,
      independentObjectScore: 0.18,
      temporalLockScore: 0.16,
      gripPercentage: 0.08,
      motionCoupling: 0.08,
      closureScore: 0.06,
      bestModeScore: 0.06,
      identityMatch: 0.18,
      identityMiss: -0.16,
      calibration: 0.03,
      weakCalibration: -0.07
    },
    v6: {
      objectLockQuality: 0.2,
      independentObjectScore: 0.2,
      temporalLockScore: 0.18,
      gripPercentage: 0.07,
      motionCoupling: 0.08,
      closureScore: 0.08,
      bestModeScore: 0.08,
      identityMatch: 0.14,
      identityMiss: -0.1,
      calibration: 0.03,
      weakCalibration: -0.06
    },
    v7: {
      objectLockQuality: 0.24,
      independentObjectScore: 0.24,
      temporalLockScore: 0.1,
      gripPercentage: 0.12,
      motionCoupling: 0.1,
      closureScore: 0.08,
      bestModeScore: 0.08,
      identityMatch: 0.08,
      identityMiss: -0.08,
      calibration: 0.04,
      weakCalibration: -0.06
    }
  }
};
