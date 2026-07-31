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
    ready: true,
    score,
    standard: {
      scores,
      spread,
      stabilityKey: stabilityKey(spread),
    },
    version,
    voiceAge: {
      bandKey: majority(
        results.map((result) => result.voiceAge?.bandKey),
        reference.voiceAge.bandKey,
      ),
      confidenceKey: majority(
        results.map((result) => result.voiceAge?.confidenceKey),
        reference.voiceAge.confidenceKey,
      ),
      max: Math.round(median(results.map((result) => result.voiceAge?.max))),
      min: Math.round(median(results.map((result) => result.voiceAge?.min))),
      youthfulness: median(results.map((result) => result.voiceAge?.youthfulness)),
    },
  };
}

export const standardResultInternals = {
  confidenceKey,
  majority,
  median,
  stabilityKey,
};
