import { describe, expect, it } from 'vitest';
import { chooseOfflineMaxEvidenceMode, offlineMaxEvidenceLabel } from './offlineMax';

describe('Offline Max evidence selection', () => {
  it('prefers D455 RGB-D when enough frames have reliable stereo depth evidence', () => {
    const mode = chooseOfflineMaxEvidenceMode([
      { realsenseDepthContact: 0.62, realsenseStereoConfidence: 0.74 },
      { realsenseDepthContact: 0.51, realsenseStereoConfidence: 0.68 },
      { realsenseDepthContact: 0.08, realsenseStereoConfidence: 0.12 },
      { realsenseDepthSeparation: 0.33, realsenseStereoConfidence: 0.58 }
    ]);

    expect(mode).toBe('d455-rgbd');
    expect(offlineMaxEvidenceLabel(mode)).toBe('D455 RGB-D');
  });

  it('falls back to RGB/RF-DETR when D455 depth is missing or too sparse', () => {
    const mode = chooseOfflineMaxEvidenceMode([
      { realsenseDepthContact: 0, realsenseStereoConfidence: 0 },
      { realsenseDepthContact: 0.24, realsenseStereoConfidence: 0.2 },
      { realsenseDepthContact: 0.04, realsenseStereoConfidence: 0.12 }
    ]);

    expect(mode).toBe('rgb-rfdetr');
    expect(offlineMaxEvidenceLabel(mode)).toBe('RGB/RF-DETR');
  });
});
