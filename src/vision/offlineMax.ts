export type OfflineMaxEvidenceMode = 'd455-rgbd' | 'rgb-rfdetr';

export type OfflineMaxEvidencePoint = {
  realsenseDepthContact?: number;
  realsenseDepthSeparation?: number;
  realsenseStereoConfidence?: number;
};

export function chooseOfflineMaxEvidenceMode(points: OfflineMaxEvidencePoint[]): OfflineMaxEvidenceMode {
  if (!points.length) return 'rgb-rfdetr';

  const reliableDepthFrames = points.filter((point) => {
    const stereo = point.realsenseStereoConfidence ?? 0;
    const contact = point.realsenseDepthContact ?? 0;
    const separation = point.realsenseDepthSeparation ?? 0;
    return stereo >= 0.38 && (contact >= 0.18 || separation >= 0.18);
  }).length;

  const requiredFrames = Math.max(2, Math.ceil(points.length * 0.08));
  return reliableDepthFrames >= requiredFrames ? 'd455-rgbd' : 'rgb-rfdetr';
}

export function offlineMaxEvidenceLabel(mode: OfflineMaxEvidenceMode) {
  return mode === 'd455-rgbd' ? 'D455 RGB-D' : 'RGB/RF-DETR';
}
