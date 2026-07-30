export function createRecordingFlowController(deps) {
  const {
    dismissOnboardTip,
    handleFileOrBlob,
    pickSupportedMime,
    refreshAvailability,
    requestMicStream,
    setBusy,
    setIsRecording,
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

  function getMediaRecorder() {
    return mediaRecorder;
  }

  async function startRecording() {
    if (typeof MediaRecorder === "undefined") { setStatus(t("status.recordUnsupported"), false); return; }
    stopPlayback();
    let stream;
    try {
      stream = await requestMicStream();
    } catch (err) {
      console.error("[startRecording] getUserMedia failed", err);
      if (err?.message === "record-unsupported") {
        setStatus(t("status.recordUnsupported"), false);
      } else {
        setStatus(t("status.recordFailed"));
      }
      return;
    }
    dismissOnboardTip(true);
    chunks = [];
    const mimeType = pickSupportedMime();
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
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

    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data?.size) chunks.push(ev.data);
      if (mediaRecorder?.state === "inactive") {
        markFinalDataReady();
      }
    };
    mediaRecorder.onstop = async () => {
      stopRecordingTimer();
      const resetBusyState = () => {
        setBusy(false);
        refreshAvailability();
      };

      try {
        await waitForFinalData();
      } catch (waitErr) {
        console.error("[onstop] waiting for data failed", waitErr);
        stopPitchStream();
        chunks.length = 0;
        resetBusyState();
        setStatus(t(waitErr?.name === "MediaRecorderTimeoutError" ? "status.recordProcessingTimeout" : "status.recordProcessingFailed"));
        stream.getTracks().forEach(t => t.stop());
        return;
      }

      stopPitchStream();                 // 停止即時圖，但保留資料做統計
      try {
        const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
        await handleFileOrBlob(blob, "recording");      // 分析完成後會呼叫 finishStreamStats()
        chunks.length = 0;
      } catch (e) {
        console.error("[onstop]", e);
        setStatus(t("status.recordProcessingFailed"));
        chunks.length = 0;
        resetBusyState();
      } finally {
        stream.getTracks().forEach(t => t.stop());
      }
    };

    document.body.classList.add("recording");
    document.querySelector(".container")?.classList.add("recording");
    setStatus(t("status.recording"));
    startRecordingTimer();
    setIsRecording(true);
    refreshAvailability();
    try {
      mediaRecorder.start();
    } catch (err) {
      setIsRecording(false);
      refreshAvailability();
      document.body.classList.remove("recording");
      document.querySelector(".container")?.classList.remove("recording");
      stream.getTracks().forEach(t => t.stop());
      stopRecordingTimer();
      throw err;
    }

    // 啟動 Pitch Stream
    startPitchStream(stream);
  }

  async function stopRecording() {
    stopRecordingTimer();
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      setBusy(true);
      setIsRecording(false);
      refreshAvailability();
      setStatus(t("status.processingAudio"), true);
      try {
        mediaRecorder.stop();
      } catch (err) {
        setBusy(false);
        refreshAvailability();
        throw err;
      }
    }
    document.body.classList.remove("recording");
    document.querySelector(".container")?.classList.remove("recording");
  }

  return {
    getMediaRecorder,
    startRecording,
    stopRecording,
  };
}
