import { EPS } from "./constants.js";

export async function analyzeWhole({ float32, sr, durationSec, token }, deps) {
  const {
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
  } = deps;

  if (!isAnalysisActive(token)) return;
  const model = await ensurePipeline();
  if (!isAnalysisActive(token)) return;
  meter?.classList.remove("hidden");

  const started = performance.now();
  startHeartbeat(() => {
    if (!isAnalysisActive(token)) return;
    const elapsed = (performance.now() - started) / 1000;
    setStatus(t("status.analyzeWhole", { duration: fmtSec(durationSec), elapsed: fmtSec(elapsed) }), true);
  });

  try {
    const res = await model(float32, { sampling_rate: sr, topk: 2 });
    if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
    const map = toMap(res);
    render(map.female || 0, map.male || 0);
    setStatus(t("status.analyzeWholeDone"));
  } catch (err) {
    if (isOOMError(err)) {
      console.warn("[analyzeWhole] OOM → switch to streamed mode…");
      stopHeartbeat();
      if (isAnalysisActive(token)) {
        await analyzeStreamed(float32, sr, durationSec, t("status.analyzeWholeOOM"), token);
      }
      return;
    }
    console.error("[analyzeWhole]", err);
    if (isAnalysisActive(token)) {
      setStatus(t("status.analyzeWholeFailed"));
    }
  } finally {
    stopHeartbeat();
  }
}

export async function analyzeStreamed({ float32, sr, durationSec, reason, token }, deps) {
  const {
    isAnalysisActive,
    ensurePipeline,
    meter,
    pickStreamStrategy,
    runStreamedWithWindow,
    isOOMError,
    setStatus,
    t,
  } = deps;

  if (!isAnalysisActive(token)) return;
  const model = await ensurePipeline();
  if (!isAnalysisActive(token)) return;
  meter?.classList.remove("hidden");

  const strategy = pickStreamStrategy(durationSec);
  const reasonBits = [reason, strategy.label].filter(Boolean);
  const reasonLabel = reasonBits.join("｜");

  let lastErr = null;
  for (const winSec of strategy.wins) {
    try {
      await runStreamedWithWindow(model, float32, sr, durationSec, winSec, strategy.hop, reasonLabel || reason, token);
      if (!isAnalysisActive(token)) return;
      return;
    } catch (e) {
      lastErr = e;
      if (isOOMError(e)) {
        console.warn(`[streamed] OOM at win=${winSec}s → downshift`);
        continue;
      }
      console.error(`[streamed] error at win=${winSec}s`, e);
      break;
    }
  }
  console.error("[analyzeStreamed] failed", lastErr);
  if (isAnalysisActive(token)) {
    setStatus(t("status.analyzeStreamFailed"));
  }
}

export async function runStreamedWithWindow({ model, float32, sr, durationSec, WIN_S, HOP_S, reason, token }, deps) {
  const {
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
  } = deps;

  if (!isAnalysisActive(token)) return;
  const win = Math.max(1, Math.floor(WIN_S * sr));
  const hop = Math.max(1, Math.floor(HOP_S * sr));

  const chunks = [];
  for (let s = 0; s < float32.length; s += hop) {
    const e = Math.min(s + win, float32.length);
    if (e - s < Math.floor(0.5 * sr)) break;
    chunks.push([s, e]);
    if (e === float32.length) break;
  }
  if (!chunks.length) chunks.push([0, Math.min(win, float32.length)]);

  let avgMs = 0;
  let processedSec = 0;
  let logitSum = 0;
  let wSum = 0;

  const started = performance.now();
  startHeartbeat(() => {
    if (!isAnalysisActive(token)) return;
    const elapsed = (performance.now() - started) / 1000;
    const pct = processedSec > 0 ? Math.min(99, Math.round((processedSec / durationSec) * 100)) : 0;
    setStatus(t("status.analyzeStream", { win: WIN_S, step: HOP_S, reason, progress: pct, elapsed: fmtSec(elapsed) }), true);
  });

  try {
    for (let i = 0; i < chunks.length; i++) {
      if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
      const [s0, s1] = chunks[i];
      const seg = float32.subarray(s0, s1);
      const dur = (s1 - s0) / sr;

      const t0 = performance.now();
      const out = await model(seg, { sampling_rate: sr, topk: 2 });
      if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
      const dt = performance.now() - t0;
      avgMs = avgMs === 0 ? dt : (avgMs * 0.65 + dt * 0.35);

      const map = toMap(out);
      const pf = clamp01(map.female || EPS);
      const pm = clamp01(map.male || EPS);
      const logit = Math.log(pf) - Math.log(pm);

      logitSum += logit * dur;
      wSum += dur;

      const logitAvg = logitSum / Math.max(wSum, EPS);
      const pf_now = 1 / (1 + Math.exp(-logitAvg));
      const pm_now = 1 - pf_now;
      render(pf_now, pm_now);

      processedSec = Math.min(durationSec, (s1 / sr));
      const remain = chunks.length - i - 1;
      const etaSec = (remain * (avgMs / 1000));
      const pct = Math.round(((i + 1) / chunks.length) * 100);
      setStatus(t("status.analyzeStreamChunk", {
        win: WIN_S,
        current: i + 1,
        total: chunks.length,
        progress: pct,
        done: fmtSec(processedSec),
        totalDuration: fmtSec(durationSec),
        eta: fmtSec(etaSec),
      }), true);
      await microYield();
      if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
    }
    const logitAvg = logitSum / Math.max(wSum, EPS);
    const pf = 1 / (1 + Math.exp(-logitAvg));
    const pm = 1 - pf;
    render(pf, pm);
    if (isAnalysisActive(token)) {
      setStatus(t("status.analyzeStreamDone"));
    }
  } finally {
    stopHeartbeat();
  }
}
