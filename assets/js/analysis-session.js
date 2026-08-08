export function createAnalysisSessionController() {
  let clf = null;
  let busy = false;
  let heartbeatTimer = null;
  let currentDevice = "wasm";
  let isRecording = false;
  let analysisSeq = 0;
  let activeAnalysisToken = 0;
  let lastPf = 0;
  let lastPm = 0;
  const inferenceListeners = new Set();

  function onInferenceDone(cb) {
    if (typeof cb !== "function") return () => { };
    inferenceListeners.add(cb);
    return () => inferenceListeners.delete(cb);
  }

  function notifyInferenceListeners(pf, pm, analysis = null, presentation = null) {
    if (!inferenceListeners.size) return;
    const payload = {
      analysis,
      pf: Number.isFinite(pf) ? pf : 0,
      pm: Number.isFinite(pm) ? pm : 0,
      presentation,
    };
    inferenceListeners.forEach((listener) => {
      try {
        listener(payload);
      } catch (err) {
        console.warn("[practice] listener error", err);
      }
    });
  }

  function startAnalysisRun(onStart) {
    analysisSeq += 1;
    activeAnalysisToken = analysisSeq;
    busy = true;
    if (typeof onStart === "function") onStart();
    return activeAnalysisToken;
  }

  function isAnalysisActive(token) {
    return token === activeAnalysisToken;
  }

  function finishAnalysisRun(token, onFinish) {
    if (!isAnalysisActive(token)) return false;
    busy = false;
    if (typeof onFinish === "function") onFinish();
    return true;
  }

  function getAnalysisSeq() {
    return analysisSeq;
  }

  function getClf() {
    return clf;
  }

  function setClf(value) {
    clf = value;
  }

  function getCurrentDevice() {
    return currentDevice;
  }

  function setCurrentDevice(value) {
    currentDevice = value;
  }

  function getBusy() {
    return busy;
  }

  function setBusy(value) {
    busy = !!value;
  }

  function getHeartbeatTimer() {
    return heartbeatTimer;
  }

  function setHeartbeatTimer(value) {
    heartbeatTimer = value;
  }

  function getIsRecording() {
    return isRecording;
  }

  function setIsRecording(value) {
    isRecording = !!value;
  }

  function getLastPf() {
    return lastPf;
  }

  function getLastPm() {
    return lastPm;
  }

  function setLastScores(pf, pm) {
    lastPf = pf;
    lastPm = pm;
  }

  function resetLastScores() {
    lastPf = 0;
    lastPm = 0;
  }

  return {
    finishAnalysisRun,
    getAnalysisSeq,
    getBusy,
    getClf,
    getCurrentDevice,
    getHeartbeatTimer,
    getIsRecording,
    getLastPf,
    getLastPm,
    isAnalysisActive,
    notifyInferenceListeners,
    onInferenceDone,
    resetLastScores,
    setBusy,
    setClf,
    setCurrentDevice,
    setHeartbeatTimer,
    setIsRecording,
    setLastScores,
    startAnalysisRun,
  };
}
