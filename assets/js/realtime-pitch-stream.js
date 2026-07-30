export function createRealtimePitchStreamController(deps) {
  const {
    appendPitchSample,
    applyDbCalibration,
    arrays,
    describeResonanceFromEnergy,
    dom,
    estimateSpectralFeatures,
    fmt1,
    maybeEnableAdvancedPitch,
    normalizeResonanceBands,
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

  function startDrawLoop() {
    const ctx = pitchCanvas.getContext("2d");
    const DPR = Math.max(1, window.devicePixelRatio || 1);
    function resize() {
      const r = pitchCanvas.getBoundingClientRect();
      pitchCanvas.width = Math.max(600, Math.round(r.width * DPR));
      pitchCanvas.height = Math.round(r.height * DPR);
    }
    resize(); addEventListener("resize", resize);

    function yOf(hz) {
      const h = pitchCanvas.height;
      const clamped = Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz));
      return h - ((clamped - PS_MIN_HZ) / (PS_MAX_HZ - PS_MIN_HZ)) * h;
    }
    function drawBands() {
      const styles = getComputedStyle(document.documentElement);
      const cGray = styles.getPropertyValue("--band-gray") || "#ddd";
      const cBlue = styles.getPropertyValue("--band-blue") || "#bfe7ff";
      const cPink = styles.getPropertyValue("--band-pink") || "#ffd1dc";
      const cLilac = styles.getPropertyValue("--band-lilac") || "#e2d5ff";
      const w = pitchCanvas.width, h = pitchCanvas.height;

      // 區帶：灰(50–85) / 藍(85–165) / 灰(165–180) / 粉(180–310) / 灰(310–450) / 淡紫(450–600)
      ctx.fillStyle = cGray; ctx.fillRect(0, yOf(85), w, h - yOf(85));
      ctx.fillStyle = cBlue; ctx.fillRect(0, yOf(165), w, yOf(85) - yOf(165));
      ctx.fillStyle = cGray; ctx.fillRect(0, yOf(180), w, yOf(165) - yOf(180));
      ctx.fillStyle = cPink; ctx.fillRect(0, yOf(310), w, yOf(180) - yOf(310));
      ctx.fillStyle = cGray; ctx.fillRect(0, yOf(450), w, yOf(310) - yOf(450));
      ctx.fillStyle = cLilac; ctx.fillRect(0, 0, w, yOf(450));

      // 網格線
      ctx.strokeStyle = "rgba(0,0,0,.08)"; ctx.lineWidth = 1 * DPR;
      [50, 85, 165, 180, 310, 450, PS_MAX_HZ].forEach(f => { const y = yOf(f); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); });
    }

    function draw() {
      if (!psRunning && psHzSmooth.length === 0) { psRAF = requestAnimationFrame(draw); return; }
      const w = pitchCanvas.width, h = pitchCanvas.height;
      ctx.clearRect(0, 0, w, h);
      drawBands();

      const styles = getComputedStyle(document.documentElement);
      ctx.lineWidth = 2 * DPR;
      ctx.strokeStyle = styles.getPropertyValue("--stream-ink") || "#222";

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

      const axisColor = styles.getPropertyValue("--stream-axis") || styles.getPropertyValue("--muted") || "rgba(0,0,0,.5)";
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

      psRAF = requestAnimationFrame(draw);
    }
    draw();
  }

  function startPitchStream(userMediaStream) {
    try {
      if (!pitchWrap || !pitchCanvas) return;
      psHz.length = 0; psHzSmooth.length = 0; psDb.length = 0; psVoiced.length = 0; psConfidence.length = 0;
      resetPitchPostState(pitchPostState);
      psRealtimeNoiseTracker.reset();
      startAutoRangeSession({ preserveRange: false });

      maybeEnableAdvancedPitch("realtime", { allowRetry: true });

      const Ctx = window.AudioContext || window.webkitAudioContext;
      psCtx = new Ctx();
      psSrc = psCtx.createMediaStreamSource(userMediaStream);
      psProc = psCtx.createScriptProcessor(2048, 1, 1);
      const sampleRate = psCtx.sampleRate;

      setRealtimePanelsActive(true);

      let lastTick = 0;
      psProc.onaudioprocess = (ev) => {
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
        }
      };

      psSrc.connect(psProc); psProc.connect(psCtx.destination);
      psRunning = true;
      startDrawLoop();
    } catch (e) { console.error("[startPitchStream]", e); }
  }

  function stopPitchStream() {
    try {
      psRunning = false;
      if (psRAF) { cancelAnimationFrame(psRAF); psRAF = null; }
      psProc?.disconnect(); psSrc?.disconnect();
      psCtx?.close();
    } catch { } finally {
      psProc = null; psSrc = null; psCtx = null;
      setRealtimePanelsActive(false);
    }
  }

  return {
    startPitchStream,
    stopPitchStream,
  };
}
