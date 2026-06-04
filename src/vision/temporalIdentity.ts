import type { ObjectProfileCandidate, ObjectProfileMatch } from './objectProfile';
import { clamp } from './geometry';

export type TemporalIdentityState = {
  profileId: string | null;
  name: string | null;
  score: number;
  streak: number;
  missed: number;
  stable: boolean;
};

export const EMPTY_TEMPORAL_IDENTITY: TemporalIdentityState = {
  profileId: null,
  name: null,
  score: 0,
  streak: 0,
  missed: 0,
  stable: false
};

export function updateTemporalIdentity(
  previous: TemporalIdentityState,
  candidate: ObjectProfileCandidate | null,
  options = { stableFrames: 3, decay: 0.72, threshold: 0.62 }
): TemporalIdentityState {
  if (!candidate || candidate.score < options.threshold) {
    const missed = previous.missed + 1;
    const score = clamp(previous.score * options.decay);
    return {
      ...previous,
      score,
      missed,
      streak: missed > 2 ? 0 : previous.streak,
      stable: missed <= 2 && previous.stable && score >= options.threshold
    };
  }

  const same = previous.profileId === candidate.profileId;
  const streak = same ? previous.streak + 1 : 1;
  const score = same ? clamp(previous.score * 0.58 + candidate.score * 0.42) : candidate.score;
  return {
    profileId: candidate.profileId,
    name: candidate.name,
    score,
    streak,
    missed: 0,
    stable: streak >= options.stableFrames && score >= options.threshold
  };
}

export function temporalIdentityToMatch(identity: TemporalIdentityState): ObjectProfileMatch {
  if (!identity.profileId || !identity.name) return null;
  return {
    profileId: identity.profileId,
    name: identity.name,
    score: identity.score,
    matched: identity.stable
  };
}
