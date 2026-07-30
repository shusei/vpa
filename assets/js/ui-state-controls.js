export function createUIStateControls(deps) {
  const {
    fileInput,
    getBusy,
    getHasPlaybackSource,
    getIsRecording,
    getPlayBtn,
    recordBtn,
    setStatusTimer,
    statusTimerReset,
    toggleStatusTimer,
    uploadFab,
    warmupCard,
    warmupOpenKey,
  } = deps;

  const RECORDING_TIMER_INTERVAL_MS = 250;
  let recordingTimerStartMs = 0;
  let recordingTimerInterval = null;

  function formatRecordingTimer(ms) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  function nowMs() {
    try {
      if (typeof performance !== "undefined" && performance?.now) {
        return performance.now();
      }
    } catch { }
    return Date.now();
  }

  function startRecordingTimer() {
    if (recordingTimerInterval !== null) {
      clearInterval(recordingTimerInterval);
    }
    toggleStatusTimer(true);
    setStatusTimer(statusTimerReset);
    recordingTimerStartMs = nowMs();
    recordingTimerInterval = setInterval(() => {
      const elapsedMs = nowMs() - recordingTimerStartMs;
      setStatusTimer(formatRecordingTimer(elapsedMs));
    }, RECORDING_TIMER_INTERVAL_MS);
  }

  function stopRecordingTimer() {
    if (recordingTimerInterval !== null) {
      clearInterval(recordingTimerInterval);
      recordingTimerInterval = null;
    }
    toggleStatusTimer(false);
  }

  function updateUploadAvailability() {
    const disable = getIsRecording();
    if (fileInput) {
      fileInput.disabled = disable;
    }
    if (uploadFab) {
      if (disable) {
        uploadFab.setAttribute("disabled", "true");
        uploadFab.setAttribute("aria-disabled", "true");
      } else {
        uploadFab.removeAttribute("disabled");
        uploadFab.removeAttribute("aria-disabled");
      }
    }
  }

  function updatePlaybackAvailability() {
    const playBtn = getPlayBtn();
    if (!playBtn) return;
    const hasSource = getHasPlaybackSource();
    const disable = !hasSource || getIsRecording() || getBusy();
    if (disable) {
      playBtn.setAttribute("disabled", "true");
      playBtn.setAttribute("aria-disabled", "true");
    } else {
      playBtn.removeAttribute("disabled");
      playBtn.removeAttribute("aria-disabled");
    }
  }

  function updateRecordAvailability() {
    if (!recordBtn) return;
    const disable = getBusy() && !getIsRecording();
    if (disable) {
      recordBtn.setAttribute("disabled", "true");
      recordBtn.setAttribute("aria-disabled", "true");
    } else {
      recordBtn.removeAttribute("disabled");
      recordBtn.removeAttribute("aria-disabled");
    }
  }

  function initWarmupCard() {
    if (!warmupCard) return;
    let defaultOpen = true;
    try {
      const raw = localStorage.getItem(warmupOpenKey);
      if (raw === "1") defaultOpen = true;
      else if (raw === "0") defaultOpen = false;
    } catch { }
    warmupCard.open = defaultOpen;
    warmupCard.setAttribute("aria-expanded", warmupCard.open ? "true" : "false");
    warmupCard.addEventListener("toggle", () => {
      warmupCard.setAttribute("aria-expanded", warmupCard.open ? "true" : "false");
      try {
        localStorage.setItem(warmupOpenKey, warmupCard.open ? "1" : "0");
      } catch { }
    });
  }

  function refreshAvailability() {
    updateUploadAvailability();
    updatePlaybackAvailability();
    updateRecordAvailability();
  }

  return {
    initWarmupCard,
    refreshAvailability,
    startRecordingTimer,
    stopRecordingTimer,
    updatePlaybackAvailability,
    updateRecordAvailability,
    updateUploadAvailability,
  };
}
