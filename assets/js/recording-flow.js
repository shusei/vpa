export function createRecordingFlowController(deps) {
  const {
    dismissOnboardTip,
    diagnostics = null,
    createMediaRecorder = (stream, mimeType) => new MediaRecorder(stream, mimeType ? { mimeType } : undefined),
    getMicCaptureInfo = () => null,
    handleFileOrBlob,
    pickSupportedMime,
    preparePitchStream = () => null,
    prepareAnalysis = () => null,
    refreshAvailability,
    requestMicStream,
    onStateChange = () => { },
    setStatus,
    startPitchStream,
    startRecordingTimer,
    stopPitchStream,
    stopPlayback,
    stopRecordingTimer,
    t,
    MEDIA_RECORDER_DATA_TIMEOUT_MS,
  } = deps;

  let mediaRecorder = null;
  let chunks = [];
  let currentSessionId = 0;

  async function startRecording({ sessionId = 0, source = "professional" } = {}) {
    if (typeof MediaRecorder === "undefined") {
      setStatus(t("status.recordUnsupported"), false);
      onStateChange("error", { error: new Error("record-unsupported"), sessionId, source });
      return false;
    }
    stopPlayback();
    const pitchPreparation = preparePitchStream({ sessionId, source });
    diagnostics?.record("recording.microphone.request", { sessionId, source });
    let stream;
    try {
      stream = await requestMicStream();
      diagnostics?.recordStream("recording.microphone.ready", stream, { sessionId, source });
    } catch (err) {
      console.error("[startRecording] getUserMedia failed", err);
      diagnostics?.recordError("recording.microphone.error", err, { sessionId, source });
      await stopPitchStream({ sessionId, source });
      if (err?.message === "record-unsupported") {
        setStatus(t("status.recordUnsupported"), false);
      } else {
        setStatus(t("status.recordFailed"));
      }
      onStateChange("error", { error: err, sessionId, source });
      return false;
    }
    dismissOnboardTip(true);
    chunks = [];
    currentSessionId = sessionId;
    let captureInfo;
    let mimeType;
    let sessionRecorder;
    try {
      captureInfo = getMicCaptureInfo(stream);
      mimeType = pickSupportedMime();
      sessionRecorder = createMediaRecorder(stream, mimeType);
    } catch (error) {
      stream.getTracks().forEach(t => t.stop());
      await stopPitchStream({ sessionId, source });
      diagnostics?.recordError("recording.media-recorder.create-error", error, { sessionId, source });
      onStateChange("error", { error, sessionId, source });
      throw error;
    }
    mediaRecorder = sessionRecorder;
    let finalDataReady = false;
    let finalDataPromise = null;
    let resolveFinalData = null;
    let rejectFinalData = null;
    let finalDataTimer = null;

    const clearFinalDataTimer = () => {
      if (finalDataTimer !== null) {
        clearTimeout(finalDataTimer);
        finalDataTimer = null;
      }
    };

    const resolveFinalDataPromise = () => {
      const resolver = resolveFinalData;
      clearFinalDataTimer();
      finalDataPromise = null;
      resolveFinalData = null;
      rejectFinalData = null;
      if (typeof resolver === "function") resolver();
    };

    const rejectFinalDataPromise = (error) => {
      const rejecter = rejectFinalData;
      clearFinalDataTimer();
      finalDataPromise = null;
      resolveFinalData = null;
      rejectFinalData = null;
      if (typeof rejecter === "function") rejecter(error);
    };

    const waitForFinalData = () => {
      if (finalDataReady) {
        return Promise.resolve();
      }
      if (finalDataPromise) {
        return finalDataPromise;
      }
      finalDataPromise = new Promise((resolve, reject) => {
        resolveFinalData = resolve;
        rejectFinalData = reject;
      });
      finalDataTimer = setTimeout(() => {
        const timeoutError = new Error("Timed out while waiting for recording data.");
        timeoutError.name = "MediaRecorderTimeoutError";
        rejectFinalDataPromise(timeoutError);
      }, MEDIA_RECORDER_DATA_TIMEOUT_MS);
      return finalDataPromise;
    };

    const markFinalDataReady = () => {
      if (finalDataReady) return;
      finalDataReady = true;
      resolveFinalDataPromise();
    };

    sessionRecorder.ondataavailable = (ev) => {
      if (ev.data?.size) chunks.push(ev.data);
      if (sessionRecorder.state === "inactive") {
        markFinalDataReady();
      }
    };
    sessionRecorder.onerror = (event) => {
      const error = event?.error || new Error("MediaRecorder error.");
      diagnostics?.recordError("recording.media-recorder.error", error, { sessionId, source });
    };
    sessionRecorder.onstop = async () => {
      stopRecordingTimer();

      try {
        await waitForFinalData();
      } catch (waitErr) {
        console.error("[onstop] waiting for data failed", waitErr);
        diagnostics?.recordError("recording.final-data.error", waitErr, { sessionId, source });
        await stopPitchStream({ sessionId, source });
        chunks.length = 0;
        setStatus(t(waitErr?.name === "MediaRecorderTimeoutError" ? "status.recordProcessingTimeout" : "status.recordProcessingFailed"));
        stream.getTracks().forEach(t => t.stop());
        diagnostics?.recordStream("recording.microphone.stopped", stream, { sessionId, source });
        onStateChange("error", { error: waitErr, sessionId, source });
        return;
      }

      await stopPitchStream({ sessionId, source });                 // 停止即時圖，但保留資料做統計
      onStateChange("analyzing", { sessionId, source });
      try {
        const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
        diagnostics?.record("recording.analysis.begin", {
          bytes: blob.size,
          mimeType: blob.type,
          sessionId,
          source,
        });
        const analysisCompleted = await handleFileOrBlob(blob, "recording");
        chunks.length = 0;
        if (analysisCompleted === false) {
          const error = new Error("Recording analysis did not complete.");
          diagnostics?.recordError("recording.analysis.incomplete", error, { sessionId, source });
          onStateChange("error", { error, sessionId, source });
          return;
        }
        diagnostics?.record("recording.analysis.end", { sessionId, source });
        onStateChange("idle", { pitchState: "inactive", sessionId, source });
      } catch (e) {
        console.error("[onstop]", e);
        diagnostics?.recordError("recording.analysis.error", e, { sessionId, source });
        setStatus(t("status.recordProcessingFailed"));
        chunks.length = 0;
        onStateChange("error", { error: e, sessionId, source });
      } finally {
        stream.getTracks().forEach(t => t.stop());
        diagnostics?.recordStream("recording.microphone.stopped", stream, { sessionId, source });
      }
    };

    document.body.classList.add("recording");
    document.querySelector(".container")?.classList.add("recording");
    const recordingStatusKey = captureInfo?.processingActive
      ? "status.recordingProcessed"
      : (captureInfo && !captureInfo.verified ? "status.recordingUnverified" : "status.recording");
    setStatus(t(recordingStatusKey));
    startRecordingTimer();
    try {
      sessionRecorder.start();
      diagnostics?.record("recording.media-recorder.start", {
        mimeType: sessionRecorder.mimeType || mimeType || "",
        sessionId,
        source,
        state: sessionRecorder.state,
      });
      onStateChange("recording", { sessionId, source });
      void Promise.resolve()
        .then(() => prepareAnalysis())
        .catch((error) => console.warn("[model-preload] unable to start preload.", error));
    } catch (err) {
      document.body.classList.remove("recording");
      document.querySelector(".container")?.classList.remove("recording");
      stream.getTracks().forEach(t => t.stop());
      stopRecordingTimer();
      await stopPitchStream({ sessionId, source });
      diagnostics?.recordError("recording.media-recorder.start-error", err, { sessionId, source });
      onStateChange("error", { error: err, sessionId, source });
      throw err;
    }

    // 啟動 Pitch Stream
    await startPitchStream(stream, {
      preparation: pitchPreparation,
      sessionId,
      source,
    });
    return true;
  }

  async function stopRecording({ sessionId = currentSessionId } = {}) {
    stopRecordingTimer();
    try {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        setStatus(t("status.processingAudio"), true);
        mediaRecorder.stop();
        diagnostics?.record("recording.media-recorder.stop", {
          sessionId,
          state: mediaRecorder.state,
        });
      } else {
        return false;
      }
    } catch (err) {
      diagnostics?.recordError("recording.media-recorder.stop-error", err, { sessionId });
      throw err;
    } finally {
      document.body.classList.remove("recording");
      document.querySelector(".container")?.classList.remove("recording");
      refreshAvailability();
    }
    return true;
  }

  return {
    startRecording,
    stopRecording,
  };
}
