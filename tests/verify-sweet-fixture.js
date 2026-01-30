(async () => {
  const PASS = (name, detail = "") => console.log(`${name} -> PASS${detail ? " " + detail : ""}`);
  const FAIL = (name, detail = "") => console.log(`${name} -> FAIL${detail ? " " + detail : ""}`);
  const resolveUrl = (path) => {
    try {
      return new URL(path, import.meta.url);
    } catch (err) {
      return path;
    }
  };
  const loadJson = async (path) => {
    const url = resolveUrl(path);
    if (typeof window === "undefined" && typeof url === "object" && url && url.protocol === "file:") {
      const fs = await import("node:fs/promises");
      const text = await fs.readFile(url, "utf8");
      return JSON.parse(text);
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    return res.json();
  };
  try {
    const data = await loadJson("../fixtures/analysis/sweet_feminine.json");
    const store = data.offlineSamples || {};
    const {
      PS_INTERVAL_MS,
      DEFAULT_PITCH_RANGE,
      clampPitchRange,
      createPitchPostState,
      resetPitchPostState,
      appendPitchSample,
      makeNoiseTracker,
      filterPitchForStats,
      makeStats,
      computeIntonationMetrics,
      CONFIDENCE_INCLUDE_THRESHOLD,
      CONFIDENCE_VOICED_THRESHOLD,
      EPSILON,
    } = await import("../src/pitch-shared.js");
    const { renderStatsCard } = await import("../src/ui/stats-card.js");

    const formants = Array.isArray(store.formants) ? store.formants : [];
    const dbSeries = Array.isArray(store.db) ? store.db : [];
    const rawPitch = Array.isArray(store.pitchRaw)
      ? store.pitchRaw
      : Array.isArray(store.pitch)
        ? store.pitch
        : Array.isArray(store.pitchProcessed)
          ? store.pitchProcessed
          : [];
    const breathiness = Array.isArray(store.breathiness) ? store.breathiness : [];
    const zcr = Array.isArray(store.zcr) ? store.zcr : [];
    const energy = Array.isArray(store.energy) ? store.energy : [];
    const frameSec = Number.isFinite(store.frameSec) && store.frameSec > 0 ? store.frameSec : (PS_INTERVAL_MS / 1000);
    const frameMs = frameSec * 1000;
    const total = Math.max(dbSeries.length, rawPitch.length, breathiness.length, zcr.length, energy.length);
    if (!total) throw new Error("fixture missing offline samples");

    const buildSpectral = (i) => {
      const spec = {};
      const breath = breathiness[i];
      if (Number.isFinite(breath)) spec.breathiness = breath;
      const z = zcr[i];
      if (Number.isFinite(z)) spec.zcr = z;
      const e = energy[i];
      if (Array.isArray(e)) {
        const [low, mid, high] = e;
        spec.energy = {
          low: Number.isFinite(low) ? low : NaN,
          mid: Number.isFinite(mid) ? mid : NaN,
          high: Number.isFinite(high) ? high : NaN,
        };
      }
      return Object.keys(spec).length ? spec : null;
    };

    const rangeConfig = data?.pitch?.detectorRange || data?.pitch?.range || DEFAULT_PITCH_RANGE;
    const range = clampPitchRange(rangeConfig);
    const state = createPitchPostState();
    resetPitchPostState(state);
    const arrays = { raw: [], smooth: [], voiced: [], confidence: [] };
    const tracker = makeNoiseTracker();
    let lastVoiced = false;
    for (let i = 0; i < total; i++) {
      const db = dbSeries[i] ?? NaN;
      const gate = tracker.shouldDetect(db, lastVoiced);
      let candHz = null;
      if (gate.detect) {
        const raw = rawPitch[i];
        if (Number.isFinite(raw)) candHz = raw;
        else tracker.capture(db);
      } else {
        tracker.capture(db);
      }
      const spectral = buildSpectral(i);
      const { voiced } = appendPitchSample(candHz, {
        db,
        ambientDb: gate.ambient ?? NaN,
        spectral,
      }, {
        state,
        arrays,
        range,
        frameMs,
      });
      lastVoiced = Boolean(voiced);
    }

    const buildEligibleMask = (flags, confArr, {
      minConfidence = 0.6,
      maxGapFrames = 8,
    } = {}) => {
      const n = Math.min(flags.length, confArr.length);
      const mask = new Array(n).fill(false);
      for (let i = 0; i < n; i++) {
        mask[i] = Boolean(flags[i]) && (confArr[i] ?? 0) >= minConfidence;
      }
      if (maxGapFrames > 0 && mask.length) {
        let gapStart = -1;
        for (let i = 0; i <= n; i++) {
          const flag = i < n ? mask[i] : true;
          if (!flag) {
            if (gapStart < 0) gapStart = i;
          } else if (gapStart >= 0) {
            const gapLen = i - gapStart;
            const prev = gapStart > 0 ? mask[gapStart - 1] : false;
            const next = i < n ? mask[i] : false;
            if (prev && next && gapLen <= maxGapFrames) {
              for (let j = gapStart; j < i; j++) mask[j] = true;
            }
            gapStart = -1;
          }
        }
        const dilation = Math.max(1, Math.floor(maxGapFrames / 2));
        if (dilation > 0) {
          const expanded = mask.slice();
          for (let i = 0; i < n; i++) {
            if (!mask[i]) continue;
            for (let d = 1; d <= dilation; d++) {
              if (i - d >= 0) expanded[i - d] = true;
              if (i + d < n) expanded[i + d] = true;
            }
          }
          return expanded;
        }
      }
      return mask;
    };
    const mask = buildEligibleMask(arrays.voiced, arrays.confidence);
    const eligibleCount = mask.reduce((acc, flag) => acc + (flag ? 1 : 0), 0);
    const formantCoverage = [0, 1, 2].map((idx) => {
      if (!eligibleCount) return 0;
      let count = 0;
      const limit = Math.min(formants.length, mask.length);
      for (let i = 0; i < limit; i++) {
        if (!mask[i]) continue;
        const triple = formants[i] || [];
        const val = triple[idx];
        if (Number.isFinite(val)) count += 1;
      }
      return count / eligibleCount;
    });
    const coverageOk = formantCoverage.every((val) => Number.isFinite(val) && val >= -1e-6 && val <= 1 + 1e-6);
    (coverageOk ? PASS : FAIL)("FORMANT_COVERAGE in [0,1]", `(f1=${formantCoverage[0].toFixed(3)}, f2=${formantCoverage[1].toFixed(3)}, f3=${formantCoverage[2].toFixed(3)})`);

    const octaveCorrected = Number(state.counters?.octaveCorrected) || 0;
    const hardMute = Number(state.counters?.hardMute) || 0;
    (octaveCorrected <= 5 ? PASS : FAIL)("OCTAVE_CORRECTED <= 5", `value=${octaveCorrected}`);
    (hardMute <= 8 ? PASS : FAIL)("HARD_MUTE <= 8", `value=${hardMute}`);

    const included = [];
    for (let i = 0; i < arrays.smooth.length; i++) {
      const hz = arrays.smooth[i];
      const conf = arrays.confidence[i] ?? 0;
      if (Number.isFinite(hz) && conf >= CONFIDENCE_INCLUDE_THRESHOLD) {
        included.push(hz);
      }
    }
    const stable = filterPitchForStats(included);
    const stats = makeStats(stable.length ? stable : included);
    const spread = Number.isFinite(stats?.p95) && Number.isFinite(stats?.p05)
      ? (stats.p95 - stats.p05)
      : NaN;
    const totalFrames = arrays.voiced.length;
    const totalVoicedSec = eligibleCount * frameSec;
    const wideThreshold = Math.max(90, 60 * Math.sqrt(Math.max(totalVoicedSec, EPSILON) / 5));
    const stabilityOk = !Number.isFinite(spread) || spread <= wideThreshold + 1e-6;
    (stabilityOk ? PASS : FAIL)("STABILITY != \"wide\"", `spread=${Number.isFinite(spread) ? spread.toFixed(3) : "NaN"}, threshold=${wideThreshold.toFixed(3)}`);

    const voicedRatio = totalFrames ? (eligibleCount / totalFrames) : 0;
    (voicedRatio >= 0.85 ? PASS : FAIL)("VOICED_RATIO >= 0.85", `value=${voicedRatio.toFixed(3)}`);

    const summarizeBreathiness = (arr, maskArr, hopSeconds) => {
      if (!Array.isArray(arr) || !arr.length) return { avg: NaN, count: 0 };
      const limit = maskArr ? Math.min(arr.length, maskArr.length) : arr.length;
      const step = Number.isFinite(hopSeconds) && hopSeconds > 0 ? hopSeconds : (PS_INTERVAL_MS / 1000);
      const tau = 0.2;
      const alpha = 1 - Math.exp(-step / Math.max(0.08, tau));
      let ema = null;
      let sum = 0;
      let count = 0;
      for (let i = 0; i < limit; i++) {
        if (maskArr && !maskArr[i]) continue;
        let val = arr[i];
        if (!Number.isFinite(val)) continue;
        val = Math.max(0, Math.min(1, val));
        if (ema == null) ema = val;
        else ema = ema + alpha * (val - ema);
        sum += ema;
        count++;
      }
      if (!count) return { avg: NaN, count: 0 };
      return { avg: Math.max(0, Math.min(1, sum / count)), count };
    };

    const breathSummary = summarizeBreathiness(breathiness, mask, frameSec);
    const dbVals = dbSeries.filter((v) => Number.isFinite(v));
    const dbStats = makeStats(dbVals);
    const sortedDb = dbVals.slice().sort((a, b) => a - b);
    const p10Idx = Math.max(0, Math.min(sortedDb.length - 1, Math.floor((sortedDb.length - 1) * 0.10)));
    const envDb = sortedDb[p10Idx] ?? NaN;
    const snr = Number.isFinite(dbStats?.med) && Number.isFinite(envDb) ? (dbStats.med - envDb) : NaN;

    const categorizeBreathiness = (val, { snrValue } = {}) => {
      if (!Number.isFinite(val)) return { key: "insufficient", label: "資料不足" };
      const styleEligible = Number.isFinite(snrValue) ? snrValue > 20 : false;
      let key = "airy";
      if (val < 0.08) key = "dense";
      else if (val <= 0.18) key = "balanced";
      else if (val <= 0.28) key = "airy";
      else if (val <= 0.45) key = styleEligible ? "style" : "airy";
      else key = styleEligible ? "style" : "tooAiry";
      let label = "";
      if (key === "balanced") label = "平衡";
      else if (key === "style") label = "偏多（風格）";
      else if (key === "dense") label = "偏實聲";
      else if (key === "airy") label = "偏氣聲";
      else if (key === "tooAiry") label = "氣聲過多";
      else label = "資料不足";
      return { key, label };
    };

    const breathInfo = categorizeBreathiness(breathSummary.avg, { snrValue: snr });
    const breathOk = ["balanced", "style"].includes(breathInfo.key);
    (breathOk ? PASS : FAIL)("BREATHINESS_LABEL ∈ {平衡/風格}", `avg=${Number.isFinite(breathSummary.avg) ? breathSummary.avg.toFixed(3) : "NaN"}, label=${breathInfo.label}`);

    const energyPct = data?.advanced?.energyPct || {};
    const chestPct = Number(energyPct.chest);
    const maskPct = Number(energyPct.mask);
    const headPct = Number(energyPct.head);
    const pctValues = [chestPct, maskPct, headPct];
    const energySum = pctValues.reduce((acc, val) => Number.isFinite(val) ? acc + val : acc, 0);
    const energyValid = pctValues.every((val) => Number.isFinite(val));
    (energyValid && Math.abs(energySum - 1) <= 1e-6 ? PASS : FAIL)("RESONANCE_ENERGY_SUM == 1", `sum=${energyValid ? energySum.toFixed(6) : "NaN"}`);
    const targets = [0.19899625304760646, 0.33760414274166645, 0.4633996042107273];
    const withinTolerance = energyValid && pctValues.every((val, idx) => Math.abs(val - targets[idx]) <= 0.01);
    (withinTolerance ? PASS : FAIL)("RESONANCE_ENERGY_MATCH", `values=${energyValid ? pctValues.map((v) => v.toFixed(4)).join(",") : "NaN"}`);

    const intonation = computeIntonationMetrics({
      processed: arrays.smooth,
      raw: arrays.raw,
      confidence: arrays.confidence,
      voiced: arrays.voiced,
    }, frameSec, {
      confidenceThreshold: CONFIDENCE_INCLUDE_THRESHOLD,
      voicedThreshold: CONFIDENCE_VOICED_THRESHOLD,
    });
    if (intonation && Number.isFinite(intonation.range)) {
      console.log(`INTONATION_RANGE -> ${intonation.range.toFixed(2)} Hz`);
    }



    console.log("[Verify] Loading fixture data...");
    // Force render to UI to verify visuals
    const advSummary = data.advanced;

    // Check if UI elements exist
    const meterEl = document.getElementById("meter");
    console.log("[Verify] #meter element found:", !!meterEl);

    let statsEl = document.getElementById("streamStats");
    console.log("[Verify] #streamStats element found (before):", !!statsEl);

    try {
      console.log("[Verify] Calling renderStatsCard...");

      // Construct data bundle
      const bundle = {
        psHzSmooth: new Array(100).fill(230),
        psConfidence: new Array(100).fill(1),
        psDb: new Array(100).fill(70),
        psVoiced: new Array(100).fill(true),
        lastPf: data.probabilities?.feminine ?? 0.9,
        lastPm: data.probabilities?.masculine ?? 0.1,
        offlineFeatureStore: data.offlineSamples, // Pass the offline samples!
        advSummary: data.advanced // Explicitly pass it too
      };

      // Render
      renderStatsCard(bundle);

      // Re-check
      statsEl = document.getElementById("streamStats");
      console.log("[Verify] #streamStats element found (after):", !!statsEl);

      if (statsEl) {
        statsEl.scrollIntoView({ behavior: "smooth", block: "center" });
        console.log("[Verify] SUCCESS: UI Updated with Fixture Data. Check the stats card for 'Liaison' and 'Vowel Focus'.");
        console.log("[Verify] Advanced Summary Data:", advSummary);
      } else {
        console.error("[Verify] FAILED: renderStatsCard did not create #streamStats container.");
        // Manual fallback to debug
        if (meterEl) {
          const debugDiv = document.createElement("div");
          debugDiv.style.border = "2px solid red";
          debugDiv.style.padding = "10px";
          debugDiv.style.color = "red";
          debugDiv.innerHTML = "<h3>Debug: renderStatsCard failed or container missing</h3><p>Check console for details.</p>";
          meterEl.insertAdjacentElement("afterend", debugDiv);
          console.log("[Verify] Created manual debug element.");
        }
      }
    } catch (renderErr) {
      console.error("[Verify] Error during renderStatsCard:", renderErr);
    }

  } catch (err) {
    FAIL("verify-sweet-fixture", err?.message || String(err));
    console.error("[Verify] Script failed:", err);
  }
})();
