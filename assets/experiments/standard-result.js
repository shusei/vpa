const REQUIRED_RESULT_COUNT = 3;
const COMPONENT_KEYS = [
  "intonation",
  "model",
  "pitch",
  "quality",
  "resonance",
  "strict",
];

function median(values) {
  const sorted = values
    .map(Number)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (!sorted.length) return NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function majority(values, fallback) {
  const counts = new Map();
  values.forEach((value) => {
    if (value === undefined || value === null || value === "") return;
    counts.set(value, (counts.get(value) || 0) + 1);
  });
  const selectedCount = Math.max(0, ...counts.values());
  const selected = [...counts.entries()]
    .filter(([, count]) => count === selectedCount)
    .map(([value]) => value);
  return selected.length === 1 ? selected[0] : fallback;
}

function confidenceKey(score) {
  if (score >= 0.78) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

function stabilityKey(spread) {
  if (spread <= 5) return "high";
  if (spread <= 12) return "medium";
  return "low";
}

function aggregateVoiceAge(results, reference) {
  const ages = results.map((result) => result.voiceAge);
  const confidenceScore = median(ages.map((age) => age?.confidenceScore));
  const version = ages.every((age) => age?.version === reference.voiceAge?.version)
    ? reference.voiceAge?.version
    : "voice-age-mixed";
  if (ages.some((age) => !age?.ready)) {
    return {
      bandKey: "unavailable",
      calibration: reference.voiceAge?.calibration || "unknown",
      confidenceKey: confidenceKey(confidenceScore),
      confidenceScore,
      evidence: [],
      max: null,
      min: null,
      ready: false,
      reasons: [...new Set(ages.flatMap((age) => age?.reasons || ["metrics"]))],
      sampleType: majority(
        ages.map((age) => age?.sampleType),
        reference.voiceAge?.sampleType || "unknown",
      ),
      version,
      youthfulness: null,
    };
  }
  return {
    bandKey: majority(
      ages.map((age) => age.bandKey),
      reference.voiceAge.bandKey,
    ),
    calibration: majority(
      ages.map((age) => age.calibration),
      reference.voiceAge.calibration,
    ),
    confidenceKey: confidenceKey(confidenceScore),
    confidenceScore,
    evidence: [],
    max: Math.round(median(ages.map((age) => age.max))),
    min: Math.round(median(ages.map((age) => age.min))),
    ready: true,
    reasons: [],
    sampleType: majority(
      ages.map((age) => age.sampleType),
      reference.voiceAge.sampleType,
    ),
    version,
    youthfulness: median(ages.map((age) => age.youthfulness)),
  };
}

export function aggregateStandardResults(results) {
  if (!Array.isArray(results) || results.length !== REQUIRED_RESULT_COUNT) {
    throw new TypeError("standard test requires exactly three results");
  }
  if (results.some((result) => !result?.ready)) {
    throw new TypeError("standard test requires three ready results");
  }

  const scores = results.map((result) => Math.round(Number(result.score)));
  const score = Math.round(median(scores));
  const reference = [...results].sort((left, right) => (
    Math.abs(Number(left.score) - score) - Math.abs(Number(right.score) - score)
  ))[0];
  const components = {};
  COMPONENT_KEYS.forEach((key) => {
    components[key] = median(results.map((result) => result.components?.[key]));
  });
  const confidenceScore = median(results.map((result) => result.confidence?.score));
  const pitchHz = median(results.map((result) => result.pitchHz));
  const spread = Math.max(...scores) - Math.min(...scores);
  const version = results.every((result) => result.version === reference.version)
    ? `${reference.version}.standard3`
    : "advanced-standard3";

  return {
    archetypeKey: majority(
      results.map((result) => result.archetypeKey),
      reference.archetypeKey,
    ),
    components,
    confidence: {
      key: confidenceKey(confidenceScore),
      score: confidenceScore,
    },
    contradiction: results.filter((result) => result.contradiction).length >= 2,
    insightKey: majority(
      results.map((result) => result.insightKey),
      reference.insightKey,
    ),
    insufficientReasons: [],
    pitchHz,
    ready: true,
    score,
    standard: {
      scores,
      spread,
      stabilityKey: stabilityKey(spread),
    },
    version,
    voiceAge: aggregateVoiceAge(results, reference),
  };
}

export const standardResultInternals = {
  aggregateVoiceAge,
  confidenceKey,
  majority,
  median,
  stabilityKey,
};
