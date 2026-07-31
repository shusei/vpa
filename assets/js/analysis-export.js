export function cloneOfflineFeatureStore(offlineFeatureStore) {
  const frameSec = offlineFeatureStore.frameSec;
  const pitchRaw = Array.from(offlineFeatureStore.pitchRaw);
  const pitchProcessed = Array.from(offlineFeatureStore.pitchProcessed);
  const pitchConfidence = Array.from(offlineFeatureStore.pitchConfidence);
  const db = Array.from(offlineFeatureStore.db);
  const voiced = Array.from(offlineFeatureStore.voiced);
  const formants = offlineFeatureStore.formants.map((triple) => Array.isArray(triple) ? triple.slice() : [NaN, NaN, NaN]);
  const tilt = Array.from(offlineFeatureStore.tilt);
  const breathiness = Array.from(offlineFeatureStore.breathiness);
  const energy = offlineFeatureStore.energy.map((triple) => Array.isArray(triple) ? triple.slice() : [NaN, NaN, NaN]);
  const zcr = Array.from(offlineFeatureStore.zcr);
  const extensions = cloneValue(offlineFeatureStore.extensions || {});
  const duration = Number.isFinite(frameSec) ? frameSec * pitchProcessed.length : NaN;
  const result = {
    frameSec,
    pitchRaw,
    pitchProcessed,
    pitchConfidence,
    db,
    voiced,
    formants,
    tilt,
    breathiness,
    energy,
    zcr,
    duration,
  };
  if (Object.keys(extensions).length) result.extensions = extensions;
  return result;
}

function cloneValue(value) {
  if (Array.isArray(value)) return value.map((item) => cloneValue(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, cloneValue(item)]),
    );
  }
  return value;
}

export function sanitizeForJson(value) {
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeForJson(item));
  }
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, val] of Object.entries(value)) {
      out[key] = sanitizeForJson(val);
    }
    return out;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (value === undefined) return null;
  return value;
}

export function prepareAnalysisExport(payload) {
  return sanitizeForJson(payload);
}
