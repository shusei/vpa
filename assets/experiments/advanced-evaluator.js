import { evaluateVoiceAgeV2 } from "./voice-age-v2.js";

const VERSION = "advanced-beta-1";

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

function finite(value) {
  // Export sanitization converts unavailable numeric values (NaN/undefined)
  // to null. Null must remain unavailable instead of being coerced to zero,
  // otherwise the same recording receives a different score after export.
  if (value == null || value === "") return NaN;
  return Number.isFinite(Number(value)) ? Number(value) : NaN;
}

function interpolate(points, rawValue) {
  const value = finite(rawValue);
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

function qualityScore(analysis, formantCoverage) {
  const snr = finite(analysis?.volume?.snr);
  const voicedRatio = finite(analysis?.summary?.voicedRatio);
  const duration = finite(analysis?.offlineSamples?.duration);
  const snrScore = interpolate([
    [0, 0.05],
    [8, 0.18],
    [12, 0.48],
    [20, 0.86],
    [28, 1],
  ], snr);
  const voicedScore = interpolate([
    [0, 0.05],
    [0.2, 0.18],
    [0.5, 0.62],
    [0.75, 1],
  ], voicedRatio);
  const coverageScore = interpolate([
    [0, 0.05],
    [0.18, 0.3],
    [0.32, 0.72],
    [0.6, 1],
  ], formantCoverage);
  const durationScore = interpolate([
    [0, 0.05],
    [2, 0.28],
    [5, 0.78],
    [8, 1],
  ], duration);
  const values = [snrScore, voicedScore, coverageScore, durationScore]
    .filter(Number.isFinite)
    .map((value) => Math.max(0.01, value));
  if (!values.length) return 0;
  return Math.pow(values.reduce((product, value) => product * value, 1), 1 / values.length);
}

function intonationScore(analysis) {
  const intonation = analysis?.advanced?.intonation;
  const minHz = finite(intonation?.minHz);
  const maxHz = finite(intonation?.maxHz);
  if (!(minHz > 0) || !(maxHz > minHz)) return NaN;
  const semitoneRange = 12 * Math.log2(maxHz / minHz);
  return interpolate([
    [0, 0.15],
    [2, 0.34],
    [4, 0.58],
    [7, 0.82],
    [11, 0.9],
    [16, 0.76],
    [24, 0.55],
  ], semitoneRange);
}

function resonanceScore(analysis) {
  const advanced = analysis?.advanced;
  const f2 = advanced?.formants?.f2;
  const f3 = advanced?.formants?.f3;
  const f2Score = interpolate(F2_POINTS, f2?.median);
  const f3Score = interpolate(F3_POINTS, f3?.median);
  const energy = advanced?.energyPct;
  const mask = finite(energy?.mask);
  const head = finite(energy?.head);
  const frontEnergy = Number.isFinite(mask) && Number.isFinite(head)
    ? clamp01((mask * 0.58) + (head * 0.42))
    : NaN;
  const frontScore = interpolate([
    [0.15, 0.12],
    [0.35, 0.28],
    [0.55, 0.5],
    [0.75, 0.76],
    [0.95, 0.94],
  ], frontEnergy);
  const weighted = [
    [f2Score, 0.55],
    [f3Score, 0.25],
    [frontScore, 0.2],
  ].filter(([value]) => Number.isFinite(value));
  const totalWeight = weighted.reduce((sum, [, weight]) => sum + weight, 0);
  const score = totalWeight > 0
    ? weighted.reduce((sum, [value, weight]) => sum + (value * weight), 0) / totalWeight
    : NaN;
  const coverageValues = [finite(f2?.coverage), finite(f3?.coverage)].filter(Number.isFinite);
  const coverage = coverageValues.length
    ? coverageValues.reduce((sum, value) => sum + value, 0) / coverageValues.length
    : NaN;
  return {
    coverage,
    f2: f2Score,
    f3: f3Score,
    front: frontScore,
    score,
  };
}

function archetypeFor(analysis, components, contradiction) {
  const advanced = analysis?.advanced;
  const breathiness = finite(advanced?.breathinessAvg);
  const tilt = finite(advanced?.tiltAvg);
  if (contradiction) return "falsettoExplorer";
  if (advanced?.brightnessKey === "sweet" && breathiness >= 0.25) return "airySweet";
  if (components.resonance >= 0.72 && components.pitch >= 0.68) return "brightForward";
  if (components.pitch < 0.42 && Number.isFinite(tilt) && tilt >= 4.5) return "matureWarm";
  if (components.intonation >= 0.76 && components.strict >= 0.55) return "livelyExpressive";
  if (Math.abs(components.strict - 0.5) <= 0.08) return "neutralClear";
  return "balancedIntelligent";
}

function confidenceKey(score) {
  if (score >= 0.78) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

export function evaluateAdvancedExperience(analysis) {
  const model = clamp01(analysis?.probabilities?.feminine);
  const pitch = interpolate(PITCH_POINTS, analysis?.pitch?.stats?.med);
  const resonance = resonanceScore(analysis);
  const intonation = intonationScore(analysis);
  const quality = qualityScore(analysis, resonance.coverage);
  const required = {
    model: Number.isFinite(finite(analysis?.probabilities?.feminine)),
    pitch: Number.isFinite(pitch),
    resonance: Number.isFinite(resonance.score) && finite(resonance.coverage) >= 0.18,
  };
  const insufficientReasons = Object.entries(required)
    .filter(([, available]) => !available)
    .map(([key]) => key);
  if (quality < 0.25) insufficientReasons.push("quality");

  const availableComponents = [
    [model, 0.3],
    [resonance.score, 0.45],
    [pitch, 0.15],
    [intonation, 0.1],
  ].filter(([value]) => Number.isFinite(value));
  const totalWeight = availableComponents.reduce((sum, [, weight]) => sum + weight, 0);
  const weightedScore = totalWeight > 0
    ? availableComponents.reduce((sum, [value, weight]) => sum + (value * weight), 0) / totalWeight
    : 0.5;
  const contradiction = (pitch >= 0.72 && resonance.score < 0.52)
    || (model >= 0.82 && resonance.score < 0.46);
  const contradictionGap = contradiction
    ? Math.max(0, Math.max(pitch, model) - resonance.score)
    : 0;
  const contradictionPenalty = Math.min(0.14, contradictionGap * 0.2);
  const evidenceStrength = 0.55 + (0.45 * quality);
  const strict = clamp01(0.5 + (((weightedScore - contradictionPenalty) - 0.5) * evidenceStrength));
  const components = {
    intonation: Number.isFinite(intonation) ? clamp01(intonation) : 0.5,
    model,
    pitch: Number.isFinite(pitch) ? clamp01(pitch) : 0.5,
    quality: clamp01(quality),
    resonance: Number.isFinite(resonance.score) ? clamp01(resonance.score) : 0.5,
    strict,
  };
  const ready = insufficientReasons.length === 0;
  const voiceAge = evaluateVoiceAgeV2(analysis, components);
  const voiceQuality = analysis?.offlineSamples?.extensions?.["voice-age-v2"] || null;
  const archetypeKey = archetypeFor(analysis, components, contradiction);
  let insightKey = "balancedGrowth";
  if (!ready) insightKey = "insufficient";
  else if (contradiction) insightKey = "falsettoContrast";
  else if (components.resonance + 0.12 < components.pitch) insightKey = "resonanceOpportunity";
  else if (components.pitch + 0.12 < components.resonance) insightKey = "pitchOpportunity";
  else if (components.quality < 0.62) insightKey = "consistencyOpportunity";
  else if (strict >= 0.76) insightKey = "strongIntegration";

  return {
    archetypeKey,
    components,
    confidence: {
      key: confidenceKey(quality),
      score: clamp01(quality),
    },
    contradiction,
    insightKey,
    insufficientReasons,
    ready,
    score: Math.round(strict * 100),
    version: VERSION,
    voiceAge,
    voiceQuality,
  };
}

export const advancedEvaluatorInternals = {
  F2_POINTS,
  F3_POINTS,
  PITCH_POINTS,
  interpolate,
};
