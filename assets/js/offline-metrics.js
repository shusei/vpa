export function offlineExtractStreamMetrics(float32, sr, append = false, deps) {
  const {
    arrays,
    offlineFeatureStore,
    resetPitchPostState,
    pitchPostState,
    psOfflineNoiseTracker,
    startAutoRangeSession,
    maybeEnableAdvancedPitch,
    applyDbCalibration,
    runPitchDetection,
    estimateSpectralFeatures,
    appendPitchSample,
    intervalMs,
  } = deps;

  const { psHz, psHzSmooth, psDb, psVoiced, psConfidence } = arrays;

  try {
    if (!append) {
      psHz.length = 0;
      psHzSmooth.length = 0;
      psDb.length = 0;
      psVoiced.length = 0;
      psConfidence.length = 0;
      resetOfflineFeatureStore(offlineFeatureStore);
      resetPitchPostState(pitchPostState);
      psOfflineNoiseTracker.reset();
      startAutoRangeSession({ preserveRange: false });
    }
    if (!append) maybeEnableAdvancedPitch("offline", { allowRetry: true });
    const step = Math.max(1, Math.floor((intervalMs / 1000) * sr));
    const frame = Math.min(Math.floor(0.08 * sr), 8192);
    offlineFeatureStore.frameSec = step / sr;
    for (let i = 0; i + frame <= float32.length; i += step) {
      const seg = float32.subarray(i, i + frame);
      const rawDb = 20 * Math.log10(Math.max(rms(seg, 0, seg.length), 1e-6)) + 100;
      const { value: db } = applyDbCalibration(rawDb);
      const wasVoiced = psVoiced.length ? psVoiced[psVoiced.length - 1] : false;
      let hz = null;
      let spectral = null;
      const gate = psOfflineNoiseTracker.shouldDetect(db, wasVoiced);
      if (gate.detect) {
        const candHz = runPitchDetection(seg, sr, { context: "offline" });
        if (candHz != null) {
          hz = candHz;
          spectral = estimateSpectralFeatures(seg, sr);
        } else {
          psOfflineNoiseTracker.capture(db);
        }
      } else {
        psOfflineNoiseTracker.capture(db);
      }
      psDb.push(db);
      const { processed, voiced, confidence } = appendPitchSample(
        hz ?? null,
        { db, ambientDb: gate.ambient, spectral },
        { dtMs: Math.max(1, Math.round((offlineFeatureStore.frameSec || (intervalMs / 1000)) * 1000)) },
      );
      offlineFeatureStore.pitchRaw.push(Number.isFinite(hz) ? hz : NaN);
      offlineFeatureStore.pitchProcessed.push(Number.isFinite(processed) ? processed : NaN);
      offlineFeatureStore.pitchConfidence.push(confidence);
      offlineFeatureStore.voiced.push(Boolean(voiced));
      offlineFeatureStore.db.push(db);
      if (spectral) {
        offlineFeatureStore.formants.push([spectral.f1 ?? NaN, spectral.f2 ?? NaN, spectral.f3 ?? NaN]);
        offlineFeatureStore.tilt.push(spectral.tilt ?? NaN);
        offlineFeatureStore.breathiness.push(spectral.breathiness ?? NaN);
        offlineFeatureStore.energy.push([spectral.energy?.low ?? NaN, spectral.energy?.mid ?? NaN, spectral.energy?.high ?? NaN]);
        offlineFeatureStore.zcr.push(spectral.zcr ?? NaN);
      } else {
        offlineFeatureStore.formants.push([NaN, NaN, NaN]);
        offlineFeatureStore.tilt.push(NaN);
        offlineFeatureStore.breathiness.push(NaN);
        offlineFeatureStore.energy.push([NaN, NaN, NaN]);
        offlineFeatureStore.zcr.push(NaN);
      }
    }
  } catch (e) {
    console.error("[offlineExtractStreamMetrics]", e);
  }
}

export function resetOfflineFeatureStore(store) {
  store.frameSec = 0;
  store.pitchRaw.length = 0;
  store.pitchProcessed.length = 0;
  store.pitchConfidence.length = 0;
  store.db.length = 0;
  store.voiced.length = 0;
  store.formants.length = 0;
  store.tilt.length = 0;
  store.breathiness.length = 0;
  store.energy.length = 0;
  store.zcr.length = 0;
  store.extensions = {};
}

function rms(arr, a, b) {
  let s = 0;
  for (let i = a; i < b; i++) {
    const v = arr[i];
    s += v * v;
  }
  return Math.sqrt(s / Math.max(1, b - a));
}
