export async function ensurePipeline(state, deps) {
  const {
    getClf,
    setClf,
    setCurrentDevice,
  } = state;
  const {
    pipeline,
    modelId,
    t,
    setStatus,
  } = deps;

  const existing = getClf();
  if (existing) return existing;
  setStatus(t("status.modelLoading"), true);

  const progress_callback = (p) => {
    if (!p) return;
    let pct = null;
    if (typeof p.loadedBytes === "number" && typeof p.totalBytes === "number" && p.totalBytes > 0) {
      pct = p.loadedBytes / p.totalBytes;
    } else if (typeof p.progress === "number" && isFinite(p.progress)) {
      pct = p.progress;
    }
    const label = p.status || t("status.modelDownloading");
    if (pct == null) setStatus(`${label}…`, true);
    else setStatus(`${label} ${Math.min(99, Math.max(0, Math.floor(pct * 100)))}% …`, true);
  };

  const device = (typeof navigator !== "undefined" && navigator.gpu) ? "webgpu" : "wasm";
  const created = await pipeline("audio-classification", modelId, { progress_callback, device });
  setClf(created);
  setCurrentDevice(device);
  setStatus(t("status.modelReady", { device }));
  return created;
}
