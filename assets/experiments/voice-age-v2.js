const VERSION = "voice-age-impression-2.0.0-research";
const CALIBRATION = "research-preview-1";

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function finite(value) {
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

function confidenceKey(score) {
  if (score >= 0.78) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function voiceQualityFrom(analysis) {
  return analysis?.offlineSamples?.extensions?.["voice-age-v2"] || null;
}

function ageBand(youthfulness) {
  if (youthfulness >= 0.8) {
    return { bandKey: "veryYouthful", min: 18, max: 26 };
  }
  if (youthfulness >= 0.65) {
    return { bandKey: "youthful", min: 23, max: 33 };
  }
  if (youthfulness >= 0.48) {
    return { bandKey: "balanced", min: 30, max: 42 };
  }
  if (youthfulness < 0.3) {
    return { bandKey: "grounded", min: 48, max: 65 };
  }
  return { bandKey: "mature", min: 39, max: 53 };
}

function brightnessScore(analysis) {
  return interpolate([
    [1800, 0.15],
    [2400, 0.32],
    [2800, 0.55],
    [3200, 0.75],
    [3600, 0.9],
    [4200, 0.96],
  ], analysis?.advanced?.formants?.f3?.median);
}

function breathinessScore(analysis) {
  const breathiness = finite(analysis?.advanced?.breathinessAvg);
  return Number.isFinite(breathiness)
    ? 1 - Math.min(1, Math.abs(breathiness - 0.2) / 0.42)
    : NaN;
}

function evidenceEntry(key, value, weight) {
  return Number.isFinite(value) ? { key, value: clamp01(value), weight } : null;
}

function connectedSpeechEvidence(analysis, components, quality) {
  const cpp = quality?.metrics?.cpp;
  const hnr = quality?.metrics?.hnr;
  const reasons = [];
  if (!cpp?.reliable || !Number.isFinite(finite(cpp?.valueDb))) reasons.push("cpp");
  if (!hnr?.reliable || !Number.isFinite(finite(hnr?.valueDb))) reasons.push("hnr");
  const evidence = [
    evidenceEntry("cpp", interpolate([
      [0, 0.05],
      [4, 0.18],
      [8, 0.46],
      [12, 0.7],
      [16, 0.88],
      [22, 0.97],
    ], cpp?.valueDb), 0.24),
    evidenceEntry("hnr", interpolate([
      [-5, 0.04],
      [3, 0.16],
      [8, 0.38],
      [14, 0.66],
      [20, 0.85],
      [28, 0.96],
    ], hnr?.valueDb), 0.2),
    evidenceEntry("intonation", components?.intonation, 0.18),
    evidenceEntry("pitch", components?.pitch, 0.15),
    evidenceEntry("brightness", brightnessScore(analysis), 0.13),
    evidenceEntry("breathiness", breathinessScore(analysis), 0.1),
  ].filter(Boolean);
  return { evidence, reasons };
}

function sustainedVowelEvidence(analysis, components, quality) {
  const cpp = quality?.metrics?.cpp;
  const hnr = quality?.metrics?.hnr;
  const jitter = quality?.metrics?.jitterLocal;
  const shimmer = quality?.metrics?.shimmerLocal;
  const reasons = [];
  if (!cpp?.reliable || !Number.isFinite(finite(cpp?.valueDb))) reasons.push("cpp");
  if (!hnr?.reliable || !Number.isFinite(finite(hnr?.valueDb))) reasons.push("hnr");
  if (!jitter?.reliable || !Number.isFinite(finite(jitter?.valuePct))) reasons.push("jitter");
  if (!shimmer?.reliable || !Number.isFinite(finite(shimmer?.valuePct))) reasons.push("shimmer");
  const evidence = [
    evidenceEntry("cpp", interpolate([
      [0, 0.05],
      [4, 0.18],
      [8, 0.46],
      [12, 0.7],
      [16, 0.88],
      [22, 0.97],
    ], cpp?.valueDb), 0.22),
    evidenceEntry("hnr", interpolate([
      [-5, 0.04],
      [3, 0.16],
      [8, 0.38],
      [14, 0.66],
      [20, 0.85],
      [28, 0.96],
    ], hnr?.valueDb), 0.2),
    evidenceEntry("jitter", interpolate([
      [0, 1],
      [0.5, 0.9],
      [1, 0.72],
      [2, 0.43],
      [4, 0.16],
      [8, 0.04],
    ], jitter?.valuePct), 0.18),
    evidenceEntry("shimmer", interpolate([
      [0, 1],
      [2, 0.9],
      [4, 0.74],
      [7, 0.46],
      [12, 0.17],
      [20, 0.04],
    ], shimmer?.valuePct), 0.16),
    evidenceEntry("pitch", components?.pitch, 0.14),
    evidenceEntry("brightness", brightnessScore(analysis), 0.1),
  ].filter(Boolean);
  return { evidence, reasons };
}

function unavailable(reasons, quality, evidence = []) {
  const confidenceScore = Math.min(0.49, clamp01(quality?.quality?.score) * 0.7);
  return {
    bandKey: "unavailable",
    calibration: CALIBRATION,
    confidenceKey: confidenceKey(confidenceScore),
    confidenceScore,
    evidence,
    max: null,
    min: null,
    ready: false,
    reasons: unique(reasons),
    sampleType: quality?.sampleType || "unknown",
    version: VERSION,
    youthfulness: null,
  };
}

export function evaluateVoiceAgeV2(analysis, components) {
  const quality = voiceQualityFrom(analysis);
  if (!quality) return unavailable(["metrics"], null);
  const reasons = [...(quality?.quality?.reasons || [])];
  if (!quality?.quality?.ready) reasons.push("quality");
  const result = quality.sampleType === "sustainedVowel"
    ? sustainedVowelEvidence(analysis, components, quality)
    : connectedSpeechEvidence(analysis, components, quality);
  reasons.push(...result.reasons);
  const totalWeight = result.evidence.reduce((sum, item) => sum + item.weight, 0);
  if (result.evidence.length < 4 || totalWeight < 0.7) reasons.push("evidence");
  const youthfulness = totalWeight > 0
    ? result.evidence.reduce((sum, item) => sum + (item.value * item.weight), 0) / totalWeight
    : NaN;
  const coverage = finite(quality?.voicing?.coverage);
  const evidenceCoverage = result.evidence.length / 6;
  const calibrationCap = quality.sampleType === "sustainedVowel" ? 0.68 : 0.72;
  const confidenceScore = Math.min(
    calibrationCap,
    clamp01(quality?.quality?.score)
      * (0.65 + (0.35 * clamp01(coverage)))
      * (0.7 + (0.3 * evidenceCoverage)),
  );
  if (confidenceScore < 0.5) reasons.push("confidence");
  if (reasons.length || !Number.isFinite(youthfulness)) {
    return unavailable(reasons, quality, result.evidence);
  }
  return {
    ...ageBand(youthfulness),
    calibration: CALIBRATION,
    confidenceKey: confidenceKey(confidenceScore),
    confidenceScore,
    evidence: result.evidence,
    ready: true,
    reasons: [],
    sampleType: quality.sampleType,
    version: VERSION,
    youthfulness: clamp01(youthfulness),
  };
}

export const voiceAgeV2Internals = {
  CALIBRATION,
  VERSION,
  ageBand,
  interpolate,
};
