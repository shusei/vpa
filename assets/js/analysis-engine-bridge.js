export function createAnalysisEngineBridge(deps) {
  const {
    MODEL_ID,
    clamp01,
    fmtSec,
    getClf,
    getCurrentDevice,
    isAnalysisActive,
    isOOMError,
    log,
    meter,
    microYield,
    mixChannelDataToMono,
    pipeline,
    render,
    setClf,
    setCurrentDevice,
    setStatus,
    sharedAnalyzeStreamed,
    sharedAnalyzeWhole,
    sharedDecodeSmartToFloat32,
    sharedEnsurePipeline,
    sharedPickStreamStrategy,
    sharedRunStreamedWithWindow,
    startHeartbeat,
    stopHeartbeat,
    t,
    toMap,
  } = deps;

  async function decodeSmartToFloat32(blobOrFile, targetSR) {
    return sharedDecodeSmartToFloat32(blobOrFile, targetSR, {
      setStatus,
      t,
      log,
      mixChannelDataToMono,
    });
  }

  let pipelinePromise = null;

  async function loadPipeline({ reportProgress = true } = {}) {
    const existing = getClf();
    if (existing) return existing;
    if (pipelinePromise) {
      if (reportProgress) setStatus(t("status.modelLoading"), true);
      return pipelinePromise;
    }

    pipelinePromise = sharedEnsurePipeline({
      getClf,
      setClf,
      setCurrentDevice,
    }, {
      pipeline,
      modelId: MODEL_ID,
      t,
      setStatus: reportProgress ? setStatus : () => { },
    });
    try {
      return await pipelinePromise;
    } finally {
      pipelinePromise = null;
    }
  }

  function ensurePipeline() {
    return loadPipeline();
  }

  async function preloadPipeline() {
    try {
      return await loadPipeline({ reportProgress: false });
    } catch (error) {
      console.warn("[model-preload] background preload failed; analysis will retry.", error);
      return null;
    }
  }

  function pickStreamStrategy(durationSec) {
    return sharedPickStreamStrategy(durationSec, { currentDevice: getCurrentDevice(), t });
  }

  async function runStreamedWithWindow(model, float32, sr, durationSec, WIN_S, HOP_S, reason, token) {
    return sharedRunStreamedWithWindow({
      model,
      float32,
      sr,
      durationSec,
      WIN_S,
      HOP_S,
      reason,
      token,
    }, {
      isAnalysisActive,
      startHeartbeat,
      stopHeartbeat,
      setStatus,
      t,
      fmtSec,
      toMap,
      render,
      clamp01,
      microYield,
    });
  }

  async function analyzeStreamed(float32, sr, durationSec, reason = t("status.streamingDefaultReason"), token) {
    return sharedAnalyzeStreamed({ float32, sr, durationSec, reason, token }, {
      isAnalysisActive,
      ensurePipeline,
      meter,
      pickStreamStrategy,
      runStreamedWithWindow,
      isOOMError,
      setStatus,
      t,
    });
  }

  async function analyzeWhole(float32, sr, durationSec, token) {
    return sharedAnalyzeWhole({ float32, sr, durationSec, token }, {
      isAnalysisActive,
      ensurePipeline,
      meter,
      startHeartbeat,
      stopHeartbeat,
      setStatus,
      t,
      fmtSec,
      toMap,
      render,
      isOOMError,
      analyzeStreamed,
    });
  }

  return {
    analyzeStreamed,
    analyzeWhole,
    decodeSmartToFloat32,
    preloadPipeline,
  };
}
