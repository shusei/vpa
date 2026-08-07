export function finishStreamStats(deps) {
  const {
    analysisSeq,
    bandOf,
    buildEligibleFrameMask,
    buildFocusInsights,
    cloneOfflineFeatureStore,
    computeAdvancedSummary,
    CONFIDENCE_INCLUDE_THRESHOLD,
    currentDevice,
    describeResonanceFromEnergy,
    drawIntonationCurve,
    EPS,
    escapeAttr,
    filterPitchForStats,
    fmt1,
    FORMANT_CONFIDENCE_THRESHOLD,
    FORMANT_MAX_GAP_FRAMES,
    getSummaryText,
    isDivergent,
    lastPf,
    lastPm,
    logPostProcessingDiagnostics,
    makeStats,
    notifyInferenceListeners,
    offlineFeatureStore,
    openPracticeCategory,
    PITCH_COUNTER_KEYS,
    percentileSorted,
    pitchPostState,
    PS_INTERVAL_MS,
    psConfidence,
    psDb,
    psHz,
    psHzSmooth,
    psVoiced,
    renderAdvancedSummary,
    renderFocusBlock,
    resizeIntonationCanvas,
    setLatestAnalysisExport,
    setupAdvancedSection,
    setupIntonationLegend,
    summaryString,
    summaryText,
    t,
    VOLUME_DISPLAY_MODE,
    wireAdvancedIntonation,
  } = deps;

  const activeSummaryText = (typeof getSummaryText === "function" ? getSummaryText() : null) || summaryText || {};

  try {
    const statsEl = document.getElementById("streamStats");
    const pfVal = Number.isFinite(lastPf) ? lastPf : 0;
    const pmVal = Number.isFinite(lastPm) ? lastPm : 0;
    notifyInferenceListeners(pfVal, pmVal);
    if (!statsEl) return;

    const headerHTML = `
      <div class="insight-header">
        <span class="badge">${activeSummaryText?.badge || t("summary.badge")}</span>
        <div class="tags"></div>
      </div>
    `;

    // 僅對有聲點統計；若沒有資料就清空
    const voicedHzRaw = [];
    for (let i = 0; i < psHzSmooth.length; i++) {
      const val = psHzSmooth[i];
      const conf = psConfidence[i] ?? 0;
      if (Number.isFinite(val) && conf >= CONFIDENCE_INCLUDE_THRESHOLD) voicedHzRaw.push(val);
    }
    const vols = psDb.slice();
    if (!voicedHzRaw.length && !vols.length) {
      statsEl.innerHTML = "";
      setLatestAnalysisExport(null);
      return;
    }

    const stableVoicedHz = filterPitchForStats(voicedHzRaw);
    const voicedHz = stableVoicedHz.length ? stableVoicedHz : voicedHzRaw;
    const pitchStats = makeStats(voicedHz);
    const volStats = makeStats(vols);
    const volsSorted = vols.slice().sort((a, b) => a - b);
    const envDb = percentileSorted(volsSorted, 10); // 10th 近似環境底噪
    const snr = Number.isFinite(volStats.med) && Number.isFinite(envDb) ? (volStats.med - envDb) : NaN;

    // ====== 簡評（可一眼看懂）======
    const band = bandOf(pitchStats.med);                 // 常見音高區（依 Median）
    const spread = (pitchStats.p95 - pitchStats.p05);    // 變化幅度
    const store = offlineFeatureStore || {};
    const maskInfo = buildEligibleFrameMask(store, {
      minConfidence: FORMANT_CONFIDENCE_THRESHOLD,
      maxGapFrames: FORMANT_MAX_GAP_FRAMES,
    });
    let eligibleMask = Array.isArray(maskInfo?.mask) && maskInfo.mask.length ? maskInfo.mask : null;
    let eligibleCount = 0;
    if ((!eligibleMask || !maskInfo?.count) && Array.isArray(store.voiced) && store.voiced.length) {
      eligibleMask = store.voiced.map(Boolean);
    }
    if (eligibleMask) {
      const limit = Math.min(eligibleMask.length, psVoiced.length);
      for (let i = 0; i < limit; i++) {
        if (eligibleMask[i]) eligibleCount++;
      }
    }
    const voicedCount = eligibleCount;
    const frameSec = Number.isFinite(offlineFeatureStore.frameSec) && offlineFeatureStore.frameSec > 0
      ? offlineFeatureStore.frameSec
      : (PS_INTERVAL_MS / 1000);
    const totalVoicedSec = voicedCount * frameSec;
    let stabilityKey = "steady";
    if (isFinite(spread)) {
      const wideThreshold = Math.max(90, 60 * Math.sqrt(Math.max(totalVoicedSec, EPS) / 5));
      if (spread > wideThreshold) stabilityKey = "wide";
      else if (spread >= 40) stabilityKey = "moderate";
    }
    const stabilityLabel = isFinite(spread)
      ? (activeSummaryText?.stability?.[stabilityKey] || t(`summary.stability.${stabilityKey}`))
      : "—";

    let snrKey = null;
    if (isFinite(snr)) {
      snrKey = snr >= 20 ? "quiet" : snr >= 12 ? "ok" : "noisy";
    }
    const snrLabel = snrKey
      ? (activeSummaryText?.snrTags?.[snrKey] || t(`summary.snrTags.${snrKey}`))
      : "—";

    let volSigmaKey = null;
    if (isFinite(volStats.sd)) {
      volSigmaKey = volStats.sd < 6 ? "steady" : volStats.sd <= 12 ? "moderate" : "wide";
    }
    const volSigmaLabel = volSigmaKey
      ? (activeSummaryText?.volumeVariation?.[volSigmaKey] || t(`summary.volumeVariation.${volSigmaKey}`))
      : "—";

    // 指標分歧（模型傾向 vs 音高常見區）
    const diverge = isDivergent(pitchStats.med, lastPf, lastPm);
    const divergeBadge = diverge
      ? (activeSummaryText?.divergenceBadge || t("summary.divergenceBadge"))
      : "";

    // 取樣覆蓋率（錄音期間有聲點比例）
    const voicedRatio = psVoiced.length ? (voicedCount / psVoiced.length) : NaN;
    let voicedHintKey = null;
    if (!isFinite(voicedRatio) || voicedRatio < 0.25) voicedHintKey = "low";
    else if (voicedRatio < 0.5) voicedHintKey = "medium";

    const pfDisplay = (lastPf * 100).toFixed(1);
    const pmDisplay = (lastPm * 100).toFixed(1);
    const snrDisplay = isFinite(snr) ? `${fmt1(snr)} dB` : "—";
    const trendLabel = lastPf >= lastPm
      ? t("realtime.meter.feminine")
      : t("realtime.meter.masculine");
    const voicedHintLabel = voicedHintKey
      ? t(`summary.voicedHint.${voicedHintKey}`)
      : null;

    const advSummary = computeAdvancedSummary();
    const focusInsights = buildFocusInsights({
      band,
      stabilityKey,
      stabilityLabel,
      spread,
      snr,
      snrDisplay,
      snrKey,
      snrLabel,
      diverge,
      trendLabel,
      advSummary,
      voicedHintKey,
      voicedHintLabel,
    });
    const focusHTML = renderFocusBlock(focusInsights);

    const divergeNote = diverge
      ? t("summary.divergenceNoteHtml", { band, trend: trendLabel })
      : "";

    const envNote = (isFinite(snr) && snr < 12)
      ? (activeSummaryText?.envNoteHtml || t("summary.envNoteHtml"))
      : "";

    const voicedNote = voicedHintKey
      ? `<p class="subline" style="margin:4px 0 0">${t(`summary.voicedHint.${voicedHintKey}`)}</p>`
      : "";

    const volumeModeNoteText = activeSummaryText?.volumeRelativeNote || t("summary.volumeRelativeNote");

    const statsLabels = activeSummaryText?.statsLabels || {};
    const statsHints = activeSummaryText?.statsHints || {};
    const statsRows = [
      { key: "pitchAvg", value: `${fmt1(pitchStats.avg)}Hz` },
      { key: "pitchMed", value: `${fmt1(pitchStats.med)}Hz` },
      { key: "pitchHigh", value: `${fmt1(pitchStats.p95)}Hz` },
      { key: "pitchLow", value: `${fmt1(pitchStats.p05)}Hz` },
      { key: "pitchSpread", value: `${fmt1(spread)}Hz` },
      { key: "volumeAvg", value: `${fmt1(volStats.avg)}dB (${fmt1(volStats.sd)}dB)` },
      { key: "volumeMed", value: `${fmt1(volStats.med)}dB (${fmt1(volStats.sd)}dB)` },
      { key: "volumeHigh", value: `${fmt1(volStats.p95)}dB` },
      { key: "volumeLow", value: `${fmt1(volStats.p05)}dB` },
    ];
    const statsTable = statsRows.map(({ key, value }) => {
      const label = statsLabels[key] || t(`summary.statsLabels.${key}`);
      const hint = statsHints[key] || t(`summary.statsHints.${key}`);
      return { key, label, value, hint };
    });
    const statsRowsHtml = statsTable.map(({ label, value, hint }) => {
      const titleAttr = hint ? ` title="${escapeAttr(hint)}"` : "";
      return `<div class="kv"${titleAttr}><div class="k">${label}</div><div class="v">${value}</div></div>`;
    }).join("");
    const envLabel = statsLabels.env || t("summary.statsLabels.env");
    const statsIntro = summaryString("statsIntro", { sigma: volSigmaLabel });
    const volumeModeNoteHtml = volumeModeNoteText
      ? `<p class="subline" style="margin:4px 0 0">${volumeModeNoteText}</p>`
      : "";
    const statsHTML = `
      <div class="stats-grid">
        ${statsRowsHtml}
      </div>
      <div class="kv" style="margin-top:10px"><div class="k">${envLabel}</div><div class="v">${fmt1(envDb)}dB</div></div>
      <p class="subline" style="margin:8px 0 0">${statsIntro}</p>
      ${volumeModeNoteHtml}
    `;
    logPostProcessingDiagnostics(pitchPostState, {
      spread,
      intonationRange: advSummary?.intonation?.range ?? NaN,
    });
    const advancedHTML = renderAdvancedSummary(advSummary, { band });

    statsEl.innerHTML = headerHTML + focusHTML + divergeNote + envNote + voicedNote + statsHTML + advancedHTML;

    const advRoot = statsEl.querySelector(".advanced-section");
    if (advRoot) setupAdvancedSection(advRoot);
    wireAdvancedIntonation(advRoot, advSummary);

    // ----- Intonation 曲線：展開才畫，resize 會重畫（清理舊監聽） -----
    if (typeof window.__advIntonationOnResize === "function") {
      window.removeEventListener("resize", window.__advIntonationOnResize);
      window.__advIntonationOnResize = null;
    }

    const det = advRoot?.querySelector('details[data-adv="intonation"]');
    function drawIntonationNow() {
      const canvas = advRoot?.querySelector("#intonationCanvas");
      if (!canvas || !advSummary) return;
      resizeIntonationCanvas(canvas); // 這個你已經加過
      if (Array.isArray(advSummary.intonation?.points) && advSummary.intonation.points.length) {
        try { drawIntonationCurve(canvas, advSummary.intonation); } catch { }
      } else {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      try { setupIntonationLegend(advSummary.intonation); } catch { }
    }

    // details 打開時再畫一次，確保不是 0 寬
    if (det) {
      det.addEventListener("toggle", () => { if (det.open) drawIntonationNow(); });
      // 如果預設就是開的，立刻畫一次
      if (det.open) drawIntonationNow();
    }

    // 窗口尺寸改變時重畫：先清舊的，再綁新的，避免越綁越多
    window.__advIntonationOnResize = () => {
      if (det?.open) drawIntonationNow();
    };
    window.addEventListener("resize", window.__advIntonationOnResize, { passive: true });



    const focusButtons = statsEl.querySelectorAll(".focus-cta");
    for (const button of focusButtons) {
      button.addEventListener("click", (event) => {
        const target = event.currentTarget instanceof HTMLElement
          ? event.currentTarget
          : null;
        if (!target) return;
        const cat = target.getAttribute("data-practice");
        if (cat) {
          openPracticeCategory(cat);
        }
      });
    }

    // 標籤列
    const tags = statsEl.querySelector(".tags");
    const pitchTagLabel = summaryString("tags.pitchBand", { band });
    const noiseTagLabel = summaryString("tags.noise", { noise: fmt1(envDb) });
    let resonanceTagLabel = null;
    let speechRateTagLabel = null;
    let breathinessTagLabel = null;
    let brightnessTagLabel = null;
    if (advSummary) {
      // 動態重計算 resonance，避免推論時凍結的中文殘留
      const resonanceTagText = (() => {
        if (advSummary.energyPct) {
          try {
            const desc = deps.describeResonanceFromEnergy?.({
              low: advSummary.energyPct.chest ?? 0.33,
              mid: advSummary.energyPct.mask ?? 0.33,
              high: advSummary.energyPct.head ?? 0.34,
              total: 1,
              validCount: 10,
              coverage: 0.9,
            });
            if (desc?.display || desc?.label) return desc.display || desc.label;
          } catch { }
        }
        return advSummary.resonanceDisplay || advSummary.resonanceLabel;
      })();
      if (resonanceTagText) resonanceTagLabel = summaryString("tags.resonance", { label: resonanceTagText });
      const spKey = advSummary.speechRate?.key;
      const spLabel = spKey && spKey !== "insufficient" ? t(`analysis.speechRate.${spKey}.label`) : (advSummary.speechRateLabel || advSummary.speechRate?.label);
      if (spLabel) speechRateTagLabel = summaryString("tags.speechRate", { label: spLabel });

      const brKey = advSummary.breathinessKey;
      const brLabel = brKey ? t(`analysis.breathiness.${brKey}.label`) : advSummary.breathinessLabel;
      if (brLabel) breathinessTagLabel = summaryString("tags.breathiness", { label: brLabel });

      const bgKey = advSummary.brightnessKey;
      const bgLabel = bgKey ? t(`analysis.brightness.${bgKey}.label`) : advSummary.brightnessLabel;
      if (bgLabel) brightnessTagLabel = summaryString("tags.brightness", { label: bgLabel });
    }

    const summaryTags = {
      pitchBand: pitchTagLabel,
      noise: noiseTagLabel,
      resonance: resonanceTagLabel,
      speechRate: speechRateTagLabel,
      breathiness: breathinessTagLabel,
      brightness: brightnessTagLabel,
    };

    if (tags) {
      let tagHTML = `
        <span class="tag">${pitchTagLabel}</span>
        <span class="tag">${noiseTagLabel}</span>
      `;
      if (resonanceTagLabel) tagHTML += `<span class="tag">${resonanceTagLabel}</span>`;
      if (speechRateTagLabel) tagHTML += `<span class="tag">${speechRateTagLabel}</span>`;
      if (breathinessTagLabel) tagHTML += `<span class="tag">${breathinessTagLabel}</span>`;
      if (brightnessTagLabel) tagHTML += `<span class="tag">${brightnessTagLabel}</span>`;
      tags.innerHTML = tagHTML;
    }

    if (advSummary?.intonation) {
      const canvas = document.getElementById("intonationCanvas");
      if (canvas) {
        if (Array.isArray(advSummary.intonation.points) && advSummary.intonation.points.length) {
          drawIntonationCurve(canvas, advSummary.intonation);
        } else {
          const ctx = canvas.getContext("2d");
          if (ctx) { ctx.clearRect(0, 0, canvas.width, canvas.height); }
        }
      }
      setupIntonationLegend(advSummary.intonation);
    }

    const focusExport = {
      heading: focusInsights.heading,
      empty: focusInsights.empty,
      items: Array.isArray(focusInsights.items)
        ? focusInsights.items.map((item) => ({
          key: item.key,
          title: item.title,
          severity: item.severity,
          severityLabel: item.severityLabel,
          practiceCategory: item.practiceCategory ?? null,
          ctaLabel: item.ctaLabel,
        }))
        : [],
    };
    const payload = {
      analysisId: analysisSeq,
      generatedAt: new Date().toISOString(),
      locale: document.documentElement?.getAttribute("lang") || "zh-Hant",
      device: currentDevice,
      probabilities: { feminine: lastPf, masculine: lastPm },
      pitch: {
        stats: pitchStats,
        band,
        spreadHz: spread,
        stability: { key: stabilityKey, label: stabilityLabel },
        samples: {
          totalRaw: psHz.length,
          confident: voicedHzRaw.length,
          finiteFiltered: voicedHz.length,
          confidenceThreshold: CONFIDENCE_INCLUDE_THRESHOLD,
        },
        postProcess: {
          counters: PITCH_COUNTER_KEYS.reduce((acc, key) => {
            acc[key] = Number(pitchPostState.counters?.[key] ?? 0);
            return acc;
          }, {}),
        },
      },
      volume: {
        stats: volStats,
        envDb,
        snr,
        snrKey,
        snrLabel,
        variation: { key: volSigmaKey, label: volSigmaLabel },
        samples: { total: vols.length },
        mode: VOLUME_DISPLAY_MODE,
        calibrated: false,
        note: volumeModeNoteText,
      },
      summary: {
        focus: focusExport,
        diverge,
        divergeBadge,
        divergeNoteHtml: divergeNote,
        envNoteHtml: envNote,
        voicedRatio,
        voicedHintKey,
        voicedHintLabel,
        snrDisplay,
        tags: summaryTags,
        statsTable,
        envLabel,
        envDb,
        volumeNote: volumeModeNoteText,
        volumeMode: VOLUME_DISPLAY_MODE,
      },
      advanced: advSummary,
      offlineSamples: cloneOfflineFeatureStore(),
      realtimeStream: {
        intervalMs: PS_INTERVAL_MS,
        pitchRaw: Array.from(psHz),
        pitchSmooth: Array.from(psHzSmooth),
        pitchConfidence: Array.from(psConfidence),
        volumeDb: Array.from(psDb),
        voiced: Array.from(psVoiced),
      },
    };
    setLatestAnalysisExport(payload);
  } catch (e) { console.error("[finishStreamStats]", e); }
}
