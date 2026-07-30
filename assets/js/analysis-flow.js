export function createAnalysisFlowController(deps) {
  const {
    analyzeStreamed,
    analyzeWhole,
    decodeSmartToFloat32,
    finishAnalysisRun,
    finishStreamStats,
    fmtSec,
    isAnalysisActive,
    MAX_WHOLE_SEC,
    maybeApplyAdaptiveVAD,
    microYield,
    notifyInferenceListeners,
    offlineExtractStreamMetrics,
    setPlaybackSource,
    setStatus,
    startAnalysisRun,
    t,
    TARGET_SR,
    updatePlaybackAvailability,
    WARN_LONG_SEC,
  } = deps;

  async function handleFileOrBlob(fileOrBlob, source = "upload") {
    const token = startAnalysisRun(source);
    let decoded = null;
    try {
      setPlaybackSource(fileOrBlob);
      updatePlaybackAvailability();

      setStatus(t("status.decoding"), true);
      decoded = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
      if (!isAnalysisActive(token)) return;
      let { float32, sr, durationSec } = decoded;

      // 離線抽樣（供 Statistics / 簡評）。先對原始音檔做一次。
      offlineExtractStreamMetrics(float32, sr, /*append*/false);

      if (durationSec > WARN_LONG_SEC) {
        setStatus(t("status.warnLong", { duration: fmtSec(durationSec) }), true);
        await microYield();
        if (!isAnalysisActive(token)) return;
      }

      // VAD（只選段）
      const vad = maybeApplyAdaptiveVAD(float32, sr);
      if (vad && vad.used) {
        const reducedRatio = 1 - (vad.keptSec / durationSec);
        float32 = vad.arr; durationSec = vad.keptSec;
        setStatus(t("status.vadApplied", { ratio: Math.round(reducedRatio * 100), duration: fmtSec(durationSec) }), true);
        // 針對「有效語音」再抽樣一次，提升代表性
        offlineExtractStreamMetrics(float32, sr, /*append*/true);
        await microYield();
        if (!isAnalysisActive(token)) return;
      }

      if (!isAnalysisActive(token)) return;

      if (durationSec <= MAX_WHOLE_SEC) {
        await analyzeWhole(float32, sr, durationSec, token);
      } else {
        await analyzeStreamed(
          float32,
          sr,
          durationSec,
          t("status.streamingSwitch", { limit: MAX_WHOLE_SEC }),
          token,
        );
      }

      // 顯示統計（錄音/上傳皆會有）
      if (!isAnalysisActive(token)) return;
      finishStreamStats();
    } catch (e) {
      console.error("[handleFileOrBlob]", e);
      if (isAnalysisActive(token)) {
        setStatus(t("status.errorPrefix", { message: e?.message || t("status.decodeFailure") }));
      }
      notifyInferenceListeners(0, 0);
    } finally {
      if (decoded) decoded.float32 = null;
      decoded = null;
      finishAnalysisRun(token);
    }
  }

  return {
    handleFileOrBlob,
  };
}
