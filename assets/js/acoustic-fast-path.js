export const ACOUSTIC_FAST_PATH_VERSION = "acoustic-social-1";

const PITCH_POINTS = [
  [85, 0.05],
  [130, 0.16],
  [165, 0.36],
  [180, 0.5],
  [220, 0.75],
  [260, 0.9],
  [310, 0.97],
  [380, 0.72],
  [450, 0.55],
];

const F2_POINTS = [
  [900, 0.08],
  [1300, 0.24],
  [1600, 0.44],
  [2000, 0.64],
  [2400, 0.84],
  [2800, 0.95],
  [3400, 0.96],
];

const F3_POINTS = [
  [1800, 0.12],
  [2400, 0.34],
  [2800, 0.58],
  [3200, 0.8],
  [3600, 0.94],
  [4200, 0.96],
];

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function interpolate(points, rawValue) {
  const value = Number(rawValue);
  if (!Number.isFinite(value)) return NaN;
  if (value <= points[0][0]) return points[0][1];
  for (let index = 1; index < points.length; index += 1) {
    const [nextX, nextY] = points[index];
    const [prevX, prevY] = points[index - 1];
    if (value <= nextX) {
      const ratio = (value - prevX) / Math.max(1e-9, nextX - prevX);
      return prevY + ((nextY - prevY) * ratio);
    }
  }
  return points[points.length - 1][1];
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function weightedAverage(entries) {
  const available = entries.filter(([value]) => Number.isFinite(value));
  const totalWeight = available.reduce((sum, [, weight]) => sum + weight, 0);
  if (!totalWeight) return NaN;
  return available.reduce((sum, [value, weight]) => sum + (value * weight), 0) / totalWeight;
}

function frameIsUsable(store, index, minPitchConfidence) {
  const voiced = store?.voiced?.[index];
  if (voiced === false) return false;
  const confidence = Number(store?.pitchConfidence?.[index]);
  return !Number.isFinite(confidence) || confidence >= minPitchConfidence;
}

function frontEnergyScore(store, minPitchConfidence) {
  const sums = [0, 0, 0];
  let count = 0;
  const energy = Array.isArray(store?.energy) ? store.energy : [];
  for (let index = 0; index < energy.length; index += 1) {
    if (!frameIsUsable(store, index, minPitchConfidence)) continue;
    const frame = energy[index];
    if (!Array.isArray(frame) || frame.length < 3 || !frame.every(Number.isFinite)) continue;
    sums[0] += Math.max(0, frame[0]);
    sums[1] += Math.max(0, frame[1]);
    sums[2] += Math.max(0, frame[2]);
    count += 1;
  }
  const total = sums[0] + sums[1] + sums[2];
  if (!count || total <= 0) return NaN;
  const front = ((sums[1] / total) * 0.58) + ((sums[2] / total) * 0.42);
  return interpolate([
    [0.15, 0.12],
    [0.35, 0.28],
    [0.55, 0.5],
    [0.75, 0.76],
    [0.95, 0.94],
  ], front);
}

export function estimateAcousticPresentation(store, {
  minPitchConfidence = 0.6,
} = {}) {
  const pitches = [];
  const f2Values = [];
  const f3Values = [];
  const processed = Array.isArray(store?.pitchProcessed) ? store.pitchProcessed : [];
  const formants = Array.isArray(store?.formants) ? store.formants : [];
  const length = Math.max(processed.length, formants.length);
  let usableFrames = 0;

  for (let index = 0; index < length; index += 1) {
    if (!frameIsUsable(store, index, minPitchConfidence)) continue;
    usableFrames += 1;
    const pitch = Number(processed[index]);
    if (Number.isFinite(pitch)) pitches.push(pitch);
    const formant = formants[index];
    if (!Array.isArray(formant)) continue;
    const f2 = Number(formant[1]);
    const f3 = Number(formant[2]);
    if (Number.isFinite(f2)) f2Values.push(f2);
    if (Number.isFinite(f3)) f3Values.push(f3);
  }

  const pitchMedian = median(pitches);
  const f2Median = median(f2Values);
  const f3Median = median(f3Values);
  const pitchScore = interpolate(PITCH_POINTS, pitchMedian);
  const f2Score = interpolate(F2_POINTS, f2Median);
  const f3Score = interpolate(F3_POINTS, f3Median);
  const frontScore = frontEnergyScore(store, minPitchConfidence);
  const resonanceScore = weightedAverage([
    [f2Score, 0.55],
    [f3Score, 0.25],
    [frontScore, 0.2],
  ]);
  const rawScore = weightedAverage([
    [pitchScore, 0.5],
    [resonanceScore, 0.5],
  ]);
  const frameSec = Number(store?.frameSec) > 0 ? Number(store.frameSec) : 0.05;
  const voicedDurationSec = pitches.length * frameSec;
  const formantCoverage = usableFrames > 0
    ? Math.max(f2Values.length, f3Values.length) / usableFrames
    : 0;
  const durationQuality = clamp01(voicedDurationSec / 5);
  const coverageQuality = clamp01(formantCoverage / 0.4);
  const quality = (durationQuality * 0.55) + (coverageQuality * 0.45);
  const evidenceStrength = 0.72 + (0.28 * quality);
  const feminine = Number.isFinite(rawScore)
    ? clamp01(0.5 + ((rawScore - 0.5) * evidenceStrength))
    : 0.5;

  return {
    feminine,
    masculine: 1 - feminine,
    quality,
    ready: Number.isFinite(pitchScore)
      && Number.isFinite(resonanceScore)
      && voicedDurationSec >= 0.6
      && formantCoverage >= 0.12,
    source: ACOUSTIC_FAST_PATH_VERSION,
    diagnostics: {
      f2Median,
      f3Median,
      formantCoverage,
      pitchMedian,
      resonanceScore,
      voicedDurationSec,
    },
  };
}
