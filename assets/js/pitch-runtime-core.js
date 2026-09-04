export function createPitchRuntimeCore(deps) {
  const {
    clamp01,
    clampPitchRange,
    CONFIDENCE_INCLUDE_THRESHOLD,
    createPitchPostState,
    createPitchProfileController,
    createPitchStrategyController,
    createRealtimePitchStreamController,
    DEFAULT_PITCH_RANGE,
    fmt1,
    log,
    makeNoiseTracker,
    PITCH_PROFILE_DEFAULT,
    PITCH_RANGE_HARD,
    PS_INTERVAL_MS,
    PS_MAX_HZ,
    PS_MIN_HZ,
    resetPitchPostState,
    resetRealtimePanels,
    setRealtimePanelsActive,
    sharedAppendPitchSample,
    sharedDetectPitchACF,
    sharedDetectPitchYinLite,
    sharedOfflineExtractStreamMetrics,
    sharedResetOfflineFeatureStore,
    sharedTrimYinBuffers,
    t,
    VOICE_PRESETS,
  } = deps;

  // Pitch Stream 狀態
  let psHz = [], psHzSmooth = [], psDb = [], psVoiced = [], psConfidence = [];
  const psRealtimeNoiseTracker = makeNoiseTracker();
  const psOfflineNoiseTracker = makeNoiseTracker();
  const INTONATION_RAW_KEY = "vpa:intonationShowRaw";

  const pitchPostState = createPitchPostState();
  let showIntonationRawPoints = true;

  const offlineFeatureStore = {
    frameSec: 0,
    pitchRaw: [],
    pitchProcessed: [],
    pitchConfidence: [],
    db: [],
    voiced: [],
    formants: [],
    tilt: [],
    breathiness: [],
    energy: [],
    zcr: [],
    extensions: {},
  };

  const yinBuffers = {
    x: new Float32Array(0),
    diff: new Float32Array(0),
    cmndf: new Float32Array(0),
  };

  const acfBuffers = {
    x: new Float32Array(0),
  };

  function loadIntonationRawPreference() {
    if (typeof window === "undefined" || !window.localStorage) return true;
    try {
      const raw = window.localStorage.getItem(INTONATION_RAW_KEY);
      if (raw == null) return true;
      return raw === "true" || raw === "1";
    } catch {
      return true;
    }
  }

  function saveIntonationRawPreference(flag) {
    showIntonationRawPoints = Boolean(flag);
    if (typeof window === "undefined" || !window.localStorage) return;
    try {
      window.localStorage.setItem(INTONATION_RAW_KEY, showIntonationRawPoints ? "true" : "false");
    } catch { }
  }

  showIntonationRawPoints = loadIntonationRawPreference();

  const pitchProfileController = createPitchProfileController({
    clampPitchRange,
    CONFIDENCE_INCLUDE_THRESHOLD,
    DEFAULT_PITCH_RANGE,
    getOctaveCorrectedCount: () => Number(pitchPostState.counters?.octaveCorrected || 0),
    PITCH_PROFILE_DEFAULT,
    PITCH_RANGE_HARD,
    PS_INTERVAL_MS,
    VOICE_PRESETS,
  });
  pitchProfileController.initPitchRangeControls();

  function detectPitchACF(input, sr) {
    return sharedDetectPitchACF(acfBuffers, input, sr, pitchProfileController.getPitchDetectorRange());
  }

  function detectPitchYinLite(input, sr) {
    return sharedDetectPitchYinLite(yinBuffers, input, sr, pitchProfileController.getPitchDetectorRange(), clamp01);
  }

  const pitchStrategies = {
    acf: { key: "acf", label: "ACF", detect: detectPitchACF },
    yin: { key: "yin", label: "YIN-lite", detect: detectPitchYinLite },
  };

  function trimYinBuffers() {
    sharedTrimYinBuffers(yinBuffers);
  }

  const pitchStrategyController = createPitchStrategyController({
    log,
    pitchStrategies,
    trimYinBuffers,
  });
  pitchStrategyController.initializePitchStrategy();

  function appendPitchSample(rawHz, meta = {}, opts = {}) {
    const frameMs = Number.isFinite(opts?.dtMs) ? opts.dtMs : PS_INTERVAL_MS;
    const result = sharedAppendPitchSample(rawHz, meta, {
      state: pitchPostState,
      arrays: {
        raw: psHz,
        smooth: psHzSmooth,
        voiced: psVoiced,
        confidence: psConfidence,
      },
      getRange: () => pitchProfileController.getPitchDetectorRange(),
      frameMs,
    });
    pitchProfileController.handleAutoRangeFrame(result, { dtMs: frameMs });
    return result;
  }

  function createRealtimeController(runtimeDeps) {
    const {
      applyDbCalibration,
      describeResonanceFromEnergy,
      diagnostics,
      dom,
      estimateSpectralFeatures,
      normalizeResonanceBands,
      onPitchState,
    } = runtimeDeps;

    return createRealtimePitchStreamController({
      appendPitchSample,
      applyDbCalibration,
      arrays: {
        psDb,
        psHz,
        psHzSmooth,
        psVoiced,
        psConfidence,
      },
      describeResonanceFromEnergy,
      diagnostics,
      dom,
      estimateSpectralFeatures,
      fmt1,
      maybeEnableAdvancedPitch: (context, options = {}) => pitchStrategyController.maybeEnableAdvancedPitch(context, options),
      normalizeResonanceBands,
      onPitchState,
      pitchPostState,
      psRealtimeNoiseTracker,
      PS_INTERVAL_MS,
      PS_MAX_HZ,
      PS_MIN_HZ,
      resetPitchPostState,
      resetRealtimePanels,
      runPitchDetection: (input, sr, options = {}) => pitchStrategyController.runPitchDetection(input, sr, options),
      setRealtimePanelsActive,
      startAutoRangeSession: (options) => pitchProfileController.startAutoRangeSession(options),
      t,
    });
  }

  function offlineExtractStreamMetrics(float32, sr, append = false, runtimeDeps) {
    const {
      applyDbCalibration,
      estimateSpectralFeatures,
    } = runtimeDeps;
    sharedOfflineExtractStreamMetrics(float32, sr, append, {
      arrays: {
        psHz,
        psHzSmooth,
        psDb,
        psVoiced,
        psConfidence,
      },
      offlineFeatureStore,
      resetPitchPostState,
      pitchPostState,
      psOfflineNoiseTracker,
      startAutoRangeSession: (options) => pitchProfileController.startAutoRangeSession(options),
      maybeEnableAdvancedPitch: (context, options = {}) => pitchStrategyController.maybeEnableAdvancedPitch(context, options),
      applyDbCalibration,
      runPitchDetection: (input, sampleRate, options = {}) => pitchStrategyController.runPitchDetection(input, sampleRate, options),
      estimateSpectralFeatures,
      appendPitchSample,
      intervalMs: PS_INTERVAL_MS,
    });
  }

  function resetOfflineFeatureStore() {
    sharedResetOfflineFeatureStore(offlineFeatureStore);
  }

  return {
    acfBuffers,
    appendPitchSample,
    createRealtimeController,
    detectPitchACF,
    detectPitchYinLite,
    getPitchDetectorRange: () => pitchProfileController.getPitchDetectorRange(),
    getShowIntonationRawPoints: () => showIntonationRawPoints,
    offlineExtractStreamMetrics,
    offlineFeatureStore,
    pitchPostState,
    psConfidence,
    psDb,
    psHz,
    psHzSmooth,
    psVoiced,
    resetOfflineFeatureStore,
    saveIntonationRawPreference,
    setAnalysisExtensions: (value) => {
      offlineFeatureStore.extensions = value && typeof value === "object" ? value : {};
    },
    setShowIntonationRawPoints: (value) => { showIntonationRawPoints = !!value; },
  };
}
