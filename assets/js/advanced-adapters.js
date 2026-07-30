export function createAdvancedAdapters(deps) {
  const {
    CONFIDENCE_INCLUDE_THRESHOLD,
    EPS,
    PS_MAX_HZ,
    PS_MIN_HZ,
    acfBuffers,
    averageEnergyFn,
    averageFiniteFn,
    bandOfFn,
    buildEligibleFrameMaskFn,
    buildFormantTrendDisplayFn,
    categorizeBreathinessFn,
    categorizeBrightnessFn,
    categorizeTiltFn,
    describeResonanceFromEnergyFn,
    detectPitchACFFn,
    detectVoiceLeaningFn,
    drawIntonationCurveFn,
    escapeAttrFn,
    getPitchDetectorRange,
    getShowIntonationRawPoints,
    isDivergentFn,
    makeFormantHintFn,
    normalizeResonanceBandsFn,
    saveIntonationRawPreference,
    setShowIntonationRawPoints,
    setupIntonationLegendFn,
    summarizeBreathinessFn,
    summarizeFormantTrendsFn,
    t,
    analyzeConnectedSpeechFn,
    analyzeIntonationFn,
    analyzeSpeechRateFn,
    analyzeVowelFocusFn,
    estimateSpectralFeaturesFn,
  } = deps;

  const FORMANT_CONFIDENCE_THRESHOLD = CONFIDENCE_INCLUDE_THRESHOLD;
  const FORMANT_MAX_GAP_FRAMES = 8;

  function averageFinite(arr, mask) {
    return averageFiniteFn(arr, mask);
  }

  function averageEnergy(arr, info = {}) {
    return averageEnergyFn(arr, info);
  }

  function summarizeBreathiness(arr, info = {}, hopSec) {
    return summarizeBreathinessFn(arr, info, hopSec);
  }

  function buildEligibleFrameMask(store, { minConfidence = FORMANT_CONFIDENCE_THRESHOLD, maxGapFrames = FORMANT_MAX_GAP_FRAMES } = {}) {
    return buildEligibleFrameMaskFn(store, { minConfidence, maxGapFrames });
  }

  function categorizeBrightness({ f3Stats, tilt, breath, leaning } = {}) {
    return categorizeBrightnessFn({ f3Stats, tilt, breath, leaning });
  }

  function detectVoiceLeaning(pf, pm) {
    return detectVoiceLeaningFn(pf, pm);
  }

  function normalizeResonanceBands(energy) {
    return normalizeResonanceBandsFn(energy);
  }

  function describeResonanceFromEnergy(energy) {
    return describeResonanceFromEnergyFn(energy);
  }

  function categorizeTilt(tilt) {
    return categorizeTiltFn(tilt);
  }

  function categorizeBreathiness(val, ctx = {}) {
    return categorizeBreathinessFn(val, ctx);
  }

  function makeFormantHint(label, value, low, high) {
    return makeFormantHintFn(label, value, low, high);
  }

  function summarizeFormantTrends(store, statsBundle, options = {}) {
    return summarizeFormantTrendsFn(store, statsBundle, options);
  }

  function buildFormantTrendDisplay(trendKey, coverage, hasAggregate) {
    return buildFormantTrendDisplayFn(trendKey, coverage, hasAggregate);
  }

  function analyzeVowelFocus(store, maskInfo) {
    return analyzeVowelFocusFn(store, maskInfo);
  }

  function analyzeSpeechRate(store) {
    return analyzeSpeechRateFn(store);
  }

  function analyzeConnectedSpeech(voicedArr, hopSec) {
    return analyzeConnectedSpeechFn(voicedArr, hopSec);
  }

  function analyzeIntonation(data, hopSec) {
    return analyzeIntonationFn(data, hopSec);
  }

  function drawIntonationCurve(canvas, intonation) {
    return drawIntonationCurveFn(canvas, intonation, {
      EPS,
      PS_MIN_HZ,
      PS_MAX_HZ,
      showIntonationRawPoints: getShowIntonationRawPoints(),
    });
  }

  function setupIntonationLegend(intonation) {
    return setupIntonationLegendFn(intonation, {
      getShowIntonationRawPoints,
      setShowIntonationRawPoints: (value) => { setShowIntonationRawPoints(!!value); },
      saveIntonationRawPreference,
      drawIntonationCurve: (canvas, data) => drawIntonationCurve(canvas, data),
    });
  }

  function estimateSpectralFeatures(frame, sr) {
    return estimateSpectralFeaturesFn(
      frame,
      sr,
      getPitchDetectorRange(),
      detectPitchACFFn,
      acfBuffers,
      EPS,
    );
  }

  function bandOf(medHz) {
    return bandOfFn(medHz, { t, PS_MAX_HZ });
  }

  function isDivergent(medHz, pf, pm) {
    return isDivergentFn(medHz, pf, pm);
  }

  function escapeAttr(value) {
    return escapeAttrFn(value);
  }

  return {
    averageFinite,
    averageEnergy,
    summarizeBreathiness,
    buildEligibleFrameMask,
    categorizeBrightness,
    detectVoiceLeaning,
    normalizeResonanceBands,
    describeResonanceFromEnergy,
    categorizeTilt,
    categorizeBreathiness,
    makeFormantHint,
    summarizeFormantTrends,
    buildFormantTrendDisplay,
    analyzeVowelFocus,
    analyzeSpeechRate,
    analyzeConnectedSpeech,
    analyzeIntonation,
    drawIntonationCurve,
    setupIntonationLegend,
    estimateSpectralFeatures,
    bandOf,
    isDivergent,
    escapeAttr,
    FORMANT_CONFIDENCE_THRESHOLD,
    FORMANT_MAX_GAP_FRAMES,
  };
}
