export function createRealtimePitchStreamController(deps) {
  const {
    appendPitchSample,
    applyDbCalibration,
    arrays,
    describeResonanceFromEnergy,
    diagnostics = null,
    dom,
    estimateSpectralFeatures,
    fmt1,
    maybeEnableAdvancedPitch,
    normalizeResonanceBands,
    onPitchState = () => { },
    pitchPostState,
    psRealtimeNoiseTracker,
    PS_INTERVAL_MS,
    PS_MAX_HZ,
    PS_MIN_HZ,
    resetPitchPostState,
    resetRealtimePanels,
    runPitchDetection,
    setRealtimePanelsActive,
    startAutoRangeSession,
    t,
  } = deps;

  const {
    psDb,
    psHz,
    psHzSmooth,
    psVoiced,
    psConfidence,
  } = arrays;

  const {
    pitchWrap,
    pitchCanvas,
    pitchNowEl,
    bandNowEl,
    volNowEl,
    formantWrap,
    f1NowEl,
    f2NowEl,
    f3NowEl,
    breathNowEl,
    resonanceNowEl,
    tiltNowEl,
    resBarChest,
    resBarMask,
    resBarHead,
    resValChest,
    resValMask,
    resValHead,
  } = dom;

  let psCtx = null;
  let psSrc = null;
  let psProc = null;
  let psRAF = null;
  let psRunning = false;
  let psGeneration = 0;
  let psOwnerSessionId = null;
  let psContextSessionId = null;
  let psResizeHandler = null;
  let psTransition = Promise.resolve();
  let psStartController = null;

  function trace(type, detail = {}) {
    diagnostics?.record(type, detail);
  }

  function traceError(type, error, detail = {}) {
    diagnostics?.recordError(type, error, detail);
  }

  function publishPitchState(state, detail = {}) {
    try {
      onPitchState(state, detail);
    } catch (error) {
      console.warn("[realtime-pitch] state listener failed", error);
    }
  }

  function bandLabel(hz) {
    if (!hz) return "—";
    if (hz < 85) return t("pitchBands.bandLow");
    if (hz < 165) return t("pitchBands.bandBlue");
    if (hz < 180) return t("pitchBands.bandNeutral");
    if (hz < 310) return t("pitchBands.bandPink");
    if (hz < 450) return t("pitchBands.bandHigh");
    if (hz <= PS_MAX_HZ) return t("pitchBands.bandFalsetto");
    return t("pitchBands.bandUnknown");
  }

  function updateRealtimeMonitor(features) {
    try {
      if (!formantWrap) return;
      if (!features) {
        resetRealtimePanels();
        return;
      }
      const { f1, f2, f3, breathiness, tilt, energy } = features;
      if (f1NowEl) f1NowEl.textContent = Number.isFinite(f1) ? `${Math.round(f1)} Hz` : "— Hz";
      if (f2NowEl) f2NowEl.textContent = Number.isFinite(f2) ? `${Math.round(f2)} Hz` : "— Hz";
      if (f3NowEl) f3NowEl.textContent = Number.isFinite(f3) ? `${Math.round(f3)} Hz` : "— Hz";
      if (breathNowEl) breathNowEl.textContent = Number.isFinite(breathiness)
        ? `${Math.round(breathiness * 100)}%`
        : "—";

      const desc = describeResonanceFromEnergy(energy);
      if (resonanceNowEl) resonanceNowEl.textContent = desc.label || "—";
      if (tiltNowEl) tiltNowEl.textContent = Number.isFinite(tilt)
        ? t("realtime.resonance.tiltValue", { value: fmt1(tilt) })
        : t("realtime.resonance.tiltPlaceholder");

      const pct = desc.pct || normalizeResonanceBands(energy);
      const chestPct = Math.max(0, Math.min(1, pct?.chest ?? 0));
      const maskPct = Math.max(0, Math.min(1, pct?.mask ?? 0));
      const headPct = Math.max(0, Math.min(1, pct?.head ?? 0));

      if (resBarChest) { resBarChest.style.flexGrow = Math.max(chestPct, 0.001); resBarChest.style.flexBasis = `${(chestPct * 100).toFixed(1)}%`; }
      if (resBarMask) { resBarMask.style.flexGrow = Math.max(maskPct, 0.001); resBarMask.style.flexBasis = `${(maskPct * 100).toFixed(1)}%`; }
      if (resBarHead) { resBarHead.style.flexGrow = Math.max(headPct, 0.001); resBarHead.style.flexBasis = `${(headPct * 100).toFixed(1)}%`; }
      if (resValChest) resValChest.textContent = t("realtime.resonance.chest", { value: Math.round(chestPct * 100) });
      if (resValMask) resValMask.textContent = t("realtime.resonance.mask", { value: Math.round(maskPct * 100) });
      if (resValHead) resValHead.textContent = t("realtime.resonance.head", { value: Math.round(headPct * 100) });
    } catch (e) { console.error("[updateRealtimeMonitor]", e); }
  }

  function startDrawLoop(meta = {}) {
    const drawGen = psGeneration;
    const ctx = pitchCanvas.getContext("2d");
    const DPR = Math.max(1, window.devicePixelRatio || 1);
    let frameCount = 0;
    function resize() {
      const r = pitchCanvas.getBoundingClientRect();
      pitchCanvas.width = Math.max(600, Math.round(r.width * DPR));
      pitchCanvas.height = Math.round(r.height * DPR);
      diagnostics?.recordPanel("pitch.canvas.resize", {
        generation: drawGen,
        sessionId: meta.sessionId,
      });
    }
    resize();
    psResizeHandler = resize;
    addEventListener("resize", psResizeHandler);

    function yOf(hz) {
      const h = pitchCanvas.height;
      const clamped = Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz));
      return h - ((clamped - PS_MIN_HZ) / (PS_MAX_HZ - PS_MIN_HZ)) * h;
    }
    function drawBands() {
      const styles = getComputedStyle(document.documentElement);
      const isDark = document.documentElement.getAttribute("data-faction") === "dark";
      const cGray = styles.getPropertyValue("--band-gray")?.trim() || (isDark ? "#1b212c" : "#eef0f4");
      const cBlue = styles.getPropertyValue("--band-blue")?.trim() || (isDark ? "#15325b" : "#d0ebff");
      const cPink = styles.getPropertyValue("--band-pink")?.trim() || (isDark ? "#4a1c28" : "#ffd6e0");
      const cLilac = styles.getPropertyValue("--band-lilac")?.trim() || (isDark ? "#321c47" : "#e8ddff");
      const cGrid = styles.getPropertyValue("--chart-grid")?.trim() || (isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,.08)");
      const w = pitchCanvas.width, h = pitchCanvas.height;

      // 區帶：灰(50–85) / 藍(85–165) / 灰(165–180) / 粉(180–310) / 灰(310–450) / 淡紫(450–600)
      ctx.fillStyle = cGray; ctx.fillRect(0, yOf(85), w, h - yOf(85));
      ctx.fillStyle = cBlue; ctx.fillRect(0, yOf(165), w, yOf(85) - yOf(165));
      ctx.fillStyle = cGray; ctx.fillRect(0, yOf(180), w, yOf(165) - yOf(180));
      ctx.fillStyle = cPink; ctx.fillRect(0, yOf(310), w, yOf(180) - yOf(310));
      ctx.fillStyle = cGray; ctx.fillRect(0, yOf(450), w, yOf(310) - yOf(450));
      ctx.fillStyle = cLilac; ctx.fillRect(0, 0, w, yOf(450));

      // 網格線
      ctx.strokeStyle = cGrid; ctx.lineWidth = 1 * DPR;
      [50, 85, 165, 180, 310, 450, PS_MAX_HZ].forEach(f => { const y = yOf(f); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); });
    }

    function draw() {
      if (drawGen !== psGeneration) return;
      if (!psRunning && psHzSmooth.length === 0) { psRAF = requestAnimationFrame(draw); return; }

      const expectedH = Math.round(pitchCanvas.clientHeight * DPR);
      if (pitchCanvas.height === 0 && expectedH > 0) {
        resize();
      }

      const w = pitchCanvas.width, h = pitchCanvas.height;
      if (h === 0) {
        psRAF = requestAnimationFrame(draw);
        return;
      }

      ctx.clearRect(0, 0, w, h);
      drawBands();

      const styles = getComputedStyle(document.documentElement);
      const isDark = document.documentElement.getAttribute("data-faction") === "dark";
      ctx.lineWidth = 2 * DPR;
      ctx.strokeStyle = styles.getPropertyValue("--stream-ink")?.trim() || (isDark ? "#38bdf8" : "#111827");

      // 往右跑：最右是最新
      const stepX = 3 * DPR;
      const maxN = Math.floor(w / stepX) - 2;
      const n = Math.min(psHzSmooth.length, maxN);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const hz = psHzSmooth[psHzSmooth.length - n + i] ?? psHz[psHz.length - n + i];
        const x = w - (n - i) * stepX;
        if (hz == null) continue;
        const y = yOf(hz);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const axisColor = styles.getPropertyValue("--stream-axis")?.trim() || (isDark ? "#cbd5e1" : "#4b5563");
      const axisFont = (styles.getPropertyValue("--font-ui") || "sans-serif").trim() || "sans-serif";
      const axisFontSize = 11 * DPR;
      const axisTicks = [PS_MAX_HZ, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50];
      const tickLen = 6 * DPR;
      const leftX = 8 * DPR;
      const rightX = w - 8 * DPR;
      const labelHalf = axisFontSize * 0.6;

      ctx.save();
      ctx.fillStyle = axisColor;
      ctx.strokeStyle = axisColor;
      ctx.lineWidth = 1 * DPR;
      ctx.font = `${axisFontSize}px ${axisFont}`;
      ctx.textBaseline = "middle";

      axisTicks.forEach((hz) => {
        const y = yOf(hz);
        const textY = Math.min(Math.max(y, labelHalf), h - labelHalf);

        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(tickLen, y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(w, y);
        ctx.lineTo(w - tickLen, y);
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.fillText(`${hz} Hz`, leftX, textY);
        ctx.textAlign = "right";
        ctx.fillText(`${hz} Hz`, rightX, textY);
      });

      ctx.restore();

      frameCount += 1;
      if (frameCount === 1 || frameCount % 60 === 0) {
        diagnostics?.recordPanel("pitch.raf.frame", {
          frameCount,
          generation: drawGen,
          sessionId: meta.sessionId,
        });
      }

      psRAF = requestAnimationFrame(draw);
    }
    trace("pitch.raf.start", { generation: drawGen, sessionId: meta.sessionId });
    draw();
  }

  function detachPitchResources() {
    psRunning = false;

    if (psRAF) {
      cancelAnimationFrame(psRAF);
      trace("pitch.raf.stop", { generation: psGeneration });
      psRAF = null;
    }
    if (psResizeHandler) {
      removeEventListener("resize", psResizeHandler);
      psResizeHandler = null;
    }

    const resources = {
      proc: psProc,
      src: psSrc,
    };
    psProc = null;
    psSrc = null;
    return resources;
  }

  async function releasePitchResources({ proc, src }) {
    if (proc) {
      proc.onaudioprocess = null;
      try { proc.disconnect(); } catch { }
      trace("pitch.processor.disconnect");
    }
    if (src) {
      try { src.disconnect(); } catch { }
      trace("pitch.source.disconnect");
    }
  }

  function enqueuePitchTransition(task) {
    const pending = psTransition
      .catch(() => { })
      .then(task);
    psTransition = pending.catch(() => { });
    return pending;
  }

  function createPitchContext(reason, meta = {}) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) throw new Error("Web Audio is unavailable.");
    const context = new Ctx();
    trace("pitch.context.create", {
      reason,
      sampleRate: context.sampleRate,
      sessionId: meta.sessionId,
      state: context.state,
    });
    context.addEventListener?.("statechange", () => {
      trace("pitch.context.statechange", {
        sessionId: psContextSessionId,
        state: context.state,
      });
    });
    return context;
  }

  function resumePitchContext(context, reason, meta = {}) {
    trace("pitch.context.resume.before", {
      reason,
      sessionId: meta.sessionId,
      state: context.state,
    });
    let resumeResult;
    try {
      resumeResult = context.resume();
    } catch (error) {
      traceError("pitch.context.resume.error", error, {
        reason,
        sessionId: meta.sessionId,
        state: context.state,
      });
      const rejected = Promise.reject(error);
      rejected.catch(() => { });
      return rejected;
    }
    const pending = Promise.resolve(resumeResult).then(() => {
      if (context.state !== "running") {
        throw new Error(`AudioContext remained ${context.state} after resume().`);
      }
      trace("pitch.context.resume.after", {
        reason,
        sessionId: meta.sessionId,
        state: context.state,
      });
      return context;
    }).catch((error) => {
      traceError("pitch.context.resume.error", error, {
        reason,
        sessionId: meta.sessionId,
        state: context.state,
      });
      throw error;
    });
    pending.catch(() => { });
    return pending;
  }

  function prepareForUserGesture(meta = {}) {
    try {
      psContextSessionId = meta.sessionId ?? null;
      if (!psCtx || psCtx.state === "closed") {
        psCtx = createPitchContext("user-gesture", meta);
      }
      const context = psCtx;
      const promise = context.state === "running"
        ? Promise.resolve(context)
        : resumePitchContext(context, "user-gesture", meta);
      trace("pitch.prepare", {
        sessionId: meta.sessionId,
        source: meta.source,
        state: context.state,
      });
      return { context, promise, sessionId: meta.sessionId };
    } catch (error) {
      traceError("pitch.prepare.error", error, { sessionId: meta.sessionId });
      const promise = Promise.reject(error);
      promise.catch(() => { });
      return { context: null, promise, sessionId: meta.sessionId };
    }
  }

  function waitForPitchContext(promise, context, signal) {
    // A stopped session must not hold the lifecycle queue behind a pending resume.
    const pending = Promise.resolve(promise);
    const contextSessionId = psContextSessionId;
    pending.then(async () => {
      if (signal?.aborted && psCtx === context && psContextSessionId === contextSessionId && psOwnerSessionId === null && !psRunning && context.state === "running") {
        try { await context.suspend(); } catch (error) { traceError("pitch.context.suspend.error", error); }
      }
    }, () => { });
    if (!signal) return pending;
    return new Promise((resolve, reject) => {
      const abort = () => reject(new DOMException("Pitch session stopped.", "AbortError"));
      if (signal.aborted) { abort(); return; }
      signal.addEventListener("abort", abort, { once: true });
      pending.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
    });
  }

  async function getRunningPitchContext(preparation = null, meta = {}, signal) {
    let lastError = null;

    if (preparation?.context && preparation.context === psCtx) {
      try {
        const preparedContext = await waitForPitchContext(preparation.promise, preparation.context, signal);
        if (preparedContext.state === "running") return preparedContext;
      } catch (error) {
        if (signal?.aborted) throw error;
        lastError = error;
      }
    }

    for (let attempt = 0; attempt < 2; attempt += 1) {
      signal?.throwIfAborted();
      if (!psCtx || psCtx.state === "closed") {
        psCtx = createPitchContext(attempt === 0 ? "start" : "resume-retry", meta);
      }
      const context = psCtx;
      try {
        if (context.state !== "running") {
          await waitForPitchContext(resumePitchContext(context, "start", meta), context, signal);
        }
        return context;
      } catch (error) {
        if (signal?.aborted) throw error;
        lastError = error;
        if (psCtx === context) psCtx = null;
        try { await context.close(); } catch { }
        trace("pitch.context.close", {
          attempt,
          sessionId: meta.sessionId,
          state: context.state,
        });
      }
    }
    throw lastError || new Error("Unable to start realtime audio analysis.");
  }

  async function activatePitchStream(userMediaStream, gen, options = {}) {
    const { preparation = null, sessionId, source, signal } = options;
    if (gen !== psGeneration) return false;
    publishPitchState("starting", { sessionId, source });
    const previousResources = detachPitchResources();
    setRealtimePanelsActive(false);
    await releasePitchResources(previousResources);
    if (gen !== psGeneration) return false;

    let localCtx = null;
    let localSrc = null;
    let localProc = null;
    try {
      psHz.length = 0; psHzSmooth.length = 0; psDb.length = 0; psVoiced.length = 0; psConfidence.length = 0;
      resetPitchPostState(pitchPostState);
      psRealtimeNoiseTracker.reset();
      startAutoRangeSession({ preserveRange: false });

      maybeEnableAdvancedPitch("realtime", { allowRetry: true });

      localCtx = await getRunningPitchContext(preparation, { sessionId, source }, signal);

      if (gen !== psGeneration) {
        await releasePitchResources({ proc: localProc, src: localSrc });
        return false;
      }

      localSrc = localCtx.createMediaStreamSource(userMediaStream);
      localProc = localCtx.createScriptProcessor(2048, 1, 1);
      const sampleRate = localCtx.sampleRate;
      trace("pitch.graph.create", {
        contextState: localCtx.state,
        generation: gen,
        sampleRate,
        sessionId,
        source,
      });

      let lastTick = 0;
      let processorCallbackCount = 0;
      let pitchSampleCount = 0;
      localProc.onaudioprocess = (ev) => {
        if (gen !== psGeneration) return;
        processorCallbackCount += 1;
        if (processorCallbackCount === 1 || processorCallbackCount % 25 === 0) {
          trace("pitch.processor.callback", {
            count: processorCallbackCount,
            generation: gen,
            sessionId,
          });
        }
        const input = ev.inputBuffer.getChannelData(0);
        const rms = Math.sqrt(input.reduce((a, v) => a + v * v, 0) / Math.max(1, input.length));
        const rawDb = 20 * Math.log10(Math.max(rms, 1e-6)) + 100; // 相對 dB
        const { value: db } = applyDbCalibration(rawDb);
        const wasVoiced = psVoiced.length ? psVoiced[psVoiced.length - 1] : false;
        let hz = null;
        let spectral = null;
        const gate = psRealtimeNoiseTracker.shouldDetect(db, wasVoiced);
        if (gate.detect) {
          const candHz = runPitchDetection(input, sampleRate, { context: "realtime" });
          if (candHz != null) {
            hz = candHz;
            spectral = estimateSpectralFeatures(input, sampleRate);
          } else {
            psRealtimeNoiseTracker.capture(db);
          }
        } else {
          psRealtimeNoiseTracker.capture(db);
        }
        const now = performance.now();
        if (now - lastTick >= PS_INTERVAL_MS) {
          psDb.push(db);
          const { processed } = appendPitchSample(
            hz ?? null,
            { db, ambientDb: gate.ambient, spectral },
            { dtMs: PS_INTERVAL_MS },
          );
          const displayHz = Number.isFinite(processed)
            ? processed
            : (Number.isFinite(hz) ? hz : null);
          const maxN = Math.round(15000 / PS_INTERVAL_MS); // 保留約 15 秒
          if (psDb.length > maxN) {
            psDb.shift(); psHz.shift(); psHzSmooth.shift(); psVoiced.shift(); psConfidence.shift();
          }
          lastTick = now;

          if (pitchNowEl) {
            pitchNowEl.textContent = Number.isFinite(displayHz) ? `${displayHz.toFixed(1)}Hz` : "— Hz";
          }
          if (volNowEl) volNowEl.textContent = `${db.toFixed(1)} dB`;
          if (bandNowEl) bandNowEl.textContent = bandLabel(displayHz);
          updateRealtimeMonitor(spectral);
          if (Number.isFinite(displayHz)) {
            pitchSampleCount += 1;
            if (pitchSampleCount === 1) {
              publishPitchState("sampling", { sessionId, source });
            }
            if (pitchSampleCount === 1 || pitchSampleCount % 10 === 0) {
              trace("pitch.sample", {
                db,
                hz: displayHz,
                sampleCount: pitchSampleCount,
                sessionId,
              });
            }
          }
        }
      };

      localSrc.connect(localProc); localProc.connect(localCtx.destination);

      psCtx = localCtx;
      psSrc = localSrc;
      psProc = localProc;
      psRunning = true;

      setRealtimePanelsActive(true);
      diagnostics?.recordPanel("pitch.panel.active", { generation: gen, sessionId });
      startDrawLoop({ sessionId });
      publishPitchState("active", { sessionId, source });
      return true;
    } catch (e) {
      if (gen === psGeneration) console.error("[startPitchStream]", e);
      traceError("pitch.stream.error", e, { generation: gen, sessionId, source });
      if (psSrc === localSrc && psProc === localProc) {
        await releasePitchResources(detachPitchResources());
      } else {
        await releasePitchResources({ proc: localProc, src: localSrc });
      }
      if (gen === psGeneration) setRealtimePanelsActive(false);
      if (gen === psGeneration) {
        psOwnerSessionId = null;
        publishPitchState("error", { error: e, sessionId, source });
      }
      return false;
    }
  }

  function startPitchStream(userMediaStream, options = {}) {
    if (!pitchWrap || !pitchCanvas) return;
    psStartController?.abort();
    psStartController = new AbortController();
    const signal = psStartController.signal;
    psOwnerSessionId = options.sessionId ?? null;
    psContextSessionId = options.sessionId ?? psContextSessionId;
    const gen = ++psGeneration;
    diagnostics?.recordStream("pitch.stream.start", userMediaStream, {
      generation: gen,
      sessionId: options.sessionId,
      source: options.source,
    });
    return enqueuePitchTransition(() => activatePitchStream(userMediaStream, gen, { ...options, signal }));
  }

  function stopPitchStream(options = {}) {
    if (
      options.sessionId != null
      && psOwnerSessionId != null
      && options.sessionId !== psOwnerSessionId
    ) {
      trace("pitch.stream.stop-stale", {
        activeSessionId: psOwnerSessionId,
        ignoredSessionId: options.sessionId,
      });
      return Promise.resolve(false);
    }
    psStartController?.abort();
    psStartController = null;
    const gen = ++psGeneration;
    return enqueuePitchTransition(async () => {
      if (gen !== psGeneration) return false;
      await releasePitchResources(detachPitchResources());
      if (psCtx?.state === "running" && typeof psCtx.suspend === "function") {
        trace("pitch.context.suspend.before", {
          sessionId: options.sessionId,
          state: psCtx.state,
        });
        try {
          await psCtx.suspend();
          trace("pitch.context.suspend.after", {
            sessionId: options.sessionId,
            state: psCtx.state,
          });
        } catch (error) {
          traceError("pitch.context.suspend.error", error, { sessionId: options.sessionId });
        }
      }
      if (gen === psGeneration) {
        psOwnerSessionId = null;
        setRealtimePanelsActive(false);
        diagnostics?.recordPanel("pitch.panel.inactive", { generation: gen, sessionId: options.sessionId });
        publishPitchState("inactive", { sessionId: options.sessionId, source: options.source });
      }
      return true;
    });
  }

  return {
    prepareForUserGesture,
    startPitchStream,
    stopPitchStream,
  };
}
