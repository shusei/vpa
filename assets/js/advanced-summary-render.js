import {
  categorizeTilt,
  categorizeBreathiness,
  categorizeBrightness,
  makeFormantHint,
  describeResonanceFromEnergy,
  buildFormantTrendDisplay,
} from "./advanced-metrics.js";

export function createAdvancedSummaryRenderer(deps) {
  const {
    BASELINES,
    escapeAttr,
    escapeHtml,
    fmt1,
    formatBaselineRange,
    getAdvancedMode,
    getDetailsOpen,
    getSummaryText,
    renderGauge,
    summaryString,
    t,
  } = deps;

function resolveIntonationData(summary) {
  const S = summary || {};

  // 抽出 y 值，支援 number / [t, y] / {hz|f0|y}
  const takeY = (p) => {
    if (typeof p === "number") return p;
    if (Array.isArray(p)) return Number(p[1]);
    if (p && typeof p === "object") return Number(p.hz ?? p.f0 ?? p.y);
    return NaN;
  };

  const rawPts = Array.isArray(S.points) ? S.points
    : Array.isArray(S.rawPoints) ? S.rawPoints
      : [];
  const ys = rawPts.map(takeY).filter(Number.isFinite);

  const rangeHz =
    Number.isFinite(S.rangeHz) ? Number(S.rangeHz)
      : Number.isFinite(S.range) ? Number(S.range)
        : ys.length >= 2 ? Math.max(...ys) - Math.min(...ys)
          : NaN;

  const slopeLabel = S.slopeLabel || S.trendLabel || S.trend || null;
  const slopeHint = S.slopeHint || S.trendHint || "";
  const rangeLabel = (typeof S.rangeLabel === "string" && S.rangeLabel) ? S.rangeLabel : null;

  let rangeDisplay = S.rangeDisplay || null;
  if (!rangeDisplay && Number.isFinite(rangeHz)) {
    try {
      rangeDisplay = summaryString("rangeDisplayHz", { value: Math.round(rangeHz) });
    } catch {
      rangeDisplay = `${Math.round(rangeHz)} Hz`;
    }
  }
  if (!rangeDisplay && rangeLabel) rangeDisplay = rangeLabel;

  const points = rawPts.length; // 傳回數量，不是原陣列
  return { points, rangeHz, rangeDisplay, slopeLabel, slopeHint };
}

function beginnerText(key, fallback, params) {
  const path = `summary.beginnerHighlights.${key}`;
  try {
    const raw = t(path, params);
    if (raw && raw !== path) return raw;
  } catch { }
  return fallback;
}

function renderBeginnerHighlights(summary, context = {}) {
  const heading = beginnerText("heading", "Highlights");
  const empty = beginnerText("empty", "Record a longer clip to unlock coaching tips.");
  const cards = [];
  const bandLabel = context.band || context.bandLabel || null;
  if (bandLabel && bandLabel !== "—") {
    cards.push({
      key: "pitch",
      title: beginnerText("items.pitch.title", "Pitch focus"),
      value: bandLabel,
      tip: beginnerText("items.pitch.tip", "Stay near this pitch range and finish each sentence with a gentle lift.", { value: bandLabel }),
    });
  }
  const resonanceLabel = summary?.resonanceDisplay || summary?.resonanceLabel || null;
  if (resonanceLabel && resonanceLabel !== "—") {
    cards.push({
      key: "resonance",
      title: beginnerText("items.resonance.title", "Resonance focus"),
      value: resonanceLabel,
      tip: beginnerText("items.resonance.tip", "Recall this resonance placement with a soft hum, then speak while keeping that feel.", { value: resonanceLabel }),
    });
  }
  const speech = summary?.speechRate || null;
  if (speech && speech.label && speech.key !== "insufficient") {
    cards.push({
      key: "speech",
      title: beginnerText("items.speech.title", "Pacing focus"),
      value: speech.label,
      tip: beginnerText("items.speech.tip", "Keep this pacing and tap a steady beat so every phrase lands on the same pulse.", { value: speech.label }),
    });
  }

  if (!cards.length) {
    if (!heading && !empty) return "";
    return `
      <div class="beginner-summary">
        <h3 class="beginner-summary__title">${escapeHtml(heading)}</h3>
        <p class="beginner-summary__empty">${escapeHtml(empty)}</p>
      </div>
    `;
  }

  const cardsHtml = cards.map((card) => `
    <div class="adv-card beginner-summary__card" data-highlight="${escapeAttr(card.key)}">
      <div class="k">${escapeHtml(card.title)}</div>
      <div class="v">${escapeHtml(card.value)}</div>
      <div class="hint">${escapeHtml(card.tip)}</div>
    </div>
  `).join("");

  return `
    <div class="beginner-summary">
      <h3 class="beginner-summary__title">${escapeHtml(heading)}</h3>
      <div class="beginner-summary__grid">
        ${cardsHtml}
      </div>
    </div>
  `;
}

function renderAdvancedSummary(summary, context = {}) {
  const summaryText = getSummaryText?.() || {};
  if (!summary) {
    return `<div class="advanced-section"><div class="note">${t("analysis.advanced.insufficient")}</div></div>`;
  }

  // Build safe intonation display object to avoid undefined access
  const __Iraw = (typeof resolveIntonationData === "function") ? resolveIntonationData(summary) : null;
  const intonationTrendKey = summary.intonation?.trendKey || summary.intonation?.trend || summary.intonation?.slopeKey;
  const slopeLabel = intonationTrendKey
    ? (t(`analysis.intonation.slope.${intonationTrendKey}.label`) || __Iraw?.slopeLabel || summary.intonation?.slopeLabel || "—")
    : (__Iraw?.slopeLabel || summary.intonation?.slopeLabel || "—");
  const slopeHint = intonationTrendKey
    ? (t(`analysis.intonation.slope.${intonationTrendKey}.hint`) || __Iraw?.slopeHint || summary.intonation?.slopeHint || "")
    : (__Iraw?.slopeHint || summary.intonation?.slopeHint || "");

  const intonationRangeKey = summary.intonation?.rangeKey;
  const rangeLabel = intonationRangeKey
    ? (t(`analysis.intonation.range.${intonationRangeKey}.label`) || __Iraw?.rangeLabel || "—")
    : (__Iraw?.rangeLabel || "—");
  const rangeHint = intonationRangeKey
    ? (t(`analysis.intonation.range.${intonationRangeKey}.hint`) || summary.intonation?.rangeHint || "")
    : (summary.intonation?.rangeHint || "");

  const I = {
    rangeHz: __Iraw?.rangeHz ?? null,
    rangeLabel,
    rangeDisplay: __Iraw?.rangeDisplay ?? rangeLabel ?? "—",
    slopeLabel,
    slopeHint
  };

  // 共鳴比例
  const chestPct = Math.round((summary.energyPct?.chest ?? 0.33) * 100);
  const maskPct = Math.round((summary.energyPct?.mask ?? 0.33) * 100);
  const headPct = Math.round((summary.energyPct?.head ?? 0.34) * 100);
  const advCopy = summaryText?.advanced || {};
  const advFormantCards = advCopy.formantCards || {};
  const advIntonationCards = advCopy.intonationCards || {};
  const advVowelCards = advCopy.vowelCards || {};
  const labelFormantF1 = advFormantCards.f1 || t("summary.advanced.formantCards.f1") || t("realtime.formants.f1Label") || "F1";
  const labelFormantF2 = advFormantCards.f2 || t("summary.advanced.formantCards.f2") || t("realtime.formants.f2Label") || "F2";
  const labelFormantF3 = advFormantCards.f3 || t("summary.advanced.formantCards.f3") || t("realtime.formants.f3Label") || "F3";
  const labelFormantTilt = advFormantCards.tilt || t("summary.advanced.formantCards.tilt") || t("analysis.tilt.title") || "Spectral Tilt";
  const labelFormantBright = advFormantCards.brightness || t("summary.advanced.formantCards.brightness") || t("analysis.brightness.title") || "Brightness";
  const labelResonance = advCopy.resonanceTitle || t("realtime.resonance.label") || t("analysis.resonanceBalance.title") || "Resonance balance";
  const labelIntonationTrend = advIntonationCards.trend || t("summary.advanced.intonationCards.trend") || t("analysis.advanced.intonationCards.trend") || t("analysis.intonation.trend") || "Trend";
  const labelIntonationRange = advIntonationCards.range || t("summary.advanced.intonationCards.range") || t("analysis.advanced.intonationCards.range") || t("analysis.intonation.range") || "Range";
  const labelSpeechRate = advIntonationCards.speechRate || t("summary.advanced.intonationCards.speechRate") || t("analysis.advanced.intonationCards.speechRate") || t("analysis.speechRate.title") || "Speech rate";
  const chestLabel = t("realtime.resonance.chest", { value: chestPct }) || `Chest ${chestPct}%`;
  const maskLabel = t("realtime.resonance.mask", { value: maskPct }) || `Mask ${maskPct}%`;
  const headLabel = t("realtime.resonance.head", { value: headPct }) || `Head ${headPct}%`;

  // Formants (100% 動態經由 makeFormantHint 與 buildFormantTrendDisplay 重新翻譯)
  const f1 = Number(summary.formants?.f1?.median);
  const f2 = Number(summary.formants?.f2?.median);
  const f3 = Number(summary.formants?.f3?.median);
  const f1Trend = summary.formants?.f1?.trend || "insufficient";
  const f2Trend = summary.formants?.f2?.trend || "insufficient";
  const f3Trend = summary.formants?.f3?.trend || "insufficient";
  const f1Cov = summary.formants?.f1?.coverage;
  const f2Cov = summary.formants?.f2?.coverage;
  const f3Cov = summary.formants?.f3?.coverage;

  const f1Val = Number.isFinite(f1) ? fmt1(f1) : buildFormantTrendDisplay(f1Trend, f1Cov, Number.isFinite(f1Cov));
  const f2Val = Number.isFinite(f2) ? fmt1(f2) : buildFormantTrendDisplay(f2Trend, f2Cov, Number.isFinite(f2Cov));
  const f3Val = Number.isFinite(f3) ? fmt1(f3) : buildFormantTrendDisplay(f3Trend, f3Cov, Number.isFinite(f3Cov));
  const f1Hint = Number.isFinite(f1) ? makeFormantHint("F1", f1, 170, 420) : makeFormantHint("F1", NaN, 170, 420);
  const f2Hint = Number.isFinite(f2) ? makeFormantHint("F2", f2, 1450, 2750) : makeFormantHint("F2", NaN, 1450, 2750);
  const f3Hint = Number.isFinite(f3) ? makeFormantHint("F3", f3, 2400, 3400) : makeFormantHint("F3", NaN, 2400, 3400);

  // 其它指標 (100% 動態調用 categorizer 重新翻譯 Label 與 Hint)
  const tiltAvg = Number(summary.tiltAvg);
  const tiltInfo = Number.isFinite(tiltAvg) ? categorizeTilt(tiltAvg) : { label: summary.tiltLabel || "—", hint: summary.tiltHint || "" };
  const tiltLabel = tiltInfo.label || summary.tiltLabel || "—";
  const tiltHint = tiltInfo.hint || summary.tiltHint || "";

  const breathRatio = Number(summary.breathinessAvg);
  const breathPct = Number.isFinite(breathRatio) ? breathRatio * 100 : NaN;
  const breathInfo = Number.isFinite(breathRatio) ? categorizeBreathiness(breathRatio, { tilt: tiltAvg, brightnessKey: summary.brightnessKey }) : { label: summary.breathinessLabel || "—", hint: summary.breathinessHint || "" };
  const breathinessLabel = breathInfo.label || summary.breathinessLabel || "—";
  const breathinessHint = breathInfo.hint || summary.breathinessHint || "";

  const brightnessInfo = categorizeBrightness({ f3Stats: Number.isFinite(f3) ? { med: f3 } : null, tilt: tiltAvg, breath: breathRatio });
  const brightnessDisplay = brightnessInfo.label || summary.brightnessLabel || "—";
  const brightnessHint = brightnessInfo.hint || summary.brightnessHint || "";

  const energyAvg = summary.energyPct ? {
    low: summary.energyPct.chest ?? 0.33,
    mid: summary.energyPct.mask ?? 0.33,
    high: summary.energyPct.head ?? 0.34,
    total: 1,
    validCount: 10,
    coverage: 0.9,
  } : null;
  const resonanceInfo = energyAvg ? describeResonanceFromEnergy(energyAvg) : { label: summary.resonanceLabel, display: summary.resonanceDisplay, hint: summary.resonanceHint };
  const resonanceDisplay = resonanceInfo.display || resonanceInfo.label || summary.resonanceDisplay || summary.resonanceLabel || "—";
  const resonanceHint = resonanceInfo.hint || summary.resonanceHint || "";

  const speechSyll = Number(summary.speechRate?.syllPerSec);
  const speechWpm = Number(summary.speechRate?.wordsPerMin);
  const speechRateKey = summary.speechRate?.key;
  const speechRateHint = speechRateKey && speechRateKey !== "insufficient"
    ? t(`analysis.speechRate.${speechRateKey}.hint`)
    : (summary.speechRateHint || "");

  const liaisonRatio = Number(summary.liaisonRatio);
  const liaisonPct = Number.isFinite(liaisonRatio) ? liaisonRatio * 100 : NaN;
  const liaisonKey = Number.isFinite(liaisonRatio) ? (liaisonRatio >= 0.7 ? "strong" : liaisonRatio >= 0.4 ? "medium" : "weak") : null;
  const liaisonInfo = liaisonKey ? { label: t(`analysis.liaison.${liaisonKey}.label`), hint: t(`analysis.liaison.${liaisonKey}.hint`) } : { label: "—", hint: "" };
  const liaisonDisplay = Number.isFinite(liaisonPct) ? (liaisonInfo.label + " · " + summaryString("percentSuffix", { value: Math.round(liaisonPct) })) : "—";
  const liaisonHint = liaisonInfo.hint || summary.liaisonHint || "";

  const vowelRatio = summary.vowelFocusRatio;
  const vowelKey = Number.isFinite(vowelRatio) ? (vowelRatio >= 0.5 ? "strong" : vowelRatio >= 0.3 ? "medium" : "weak") : null;
  const vowelInfo = vowelKey ? { label: t(`analysis.vowelFocus.${vowelKey}.label`), hint: t(`analysis.vowelFocus.${vowelKey}.hint`) } : { label: summary.vowelLabel || "—", hint: summary.vowelHint || "" };
  const vowelDisplay = vowelInfo.label + (Number.isFinite(vowelRatio)
    ? " · " + summaryString("percentSuffix", { value: Math.round(vowelRatio * 100) }) : "");
  const vowelHint = vowelInfo.hint || summary.vowelHint || "";

  const labelBrightness = labelFormantBright;
  const labelBreathiness = advVowelCards.breathiness || t("summary.advanced.vowelCards.breathiness") || t("analysis.breathiness.title") || "Breathiness";
  const labelLiaison = advIntonationCards.liaison || t("summary.advanced.intonationCards.liaison") || t("analysis.liaison.title") || "Liaison";
  const labelVowelFocus = advVowelCards.focus || t("summary.advanced.vowelCards.focus") || t("analysis.vowelFocus.title") || "Vowel focus";

  // 格式化
  const speechRateDisplay = Number.isFinite(speechSyll)
    ? summaryString("speechRateDisplay", { value: fmt1(speechSyll) })
    : "—";
  const speechWpmDisplay = Number.isFinite(speechWpm)
    ? summaryString("speechRateWpm", { value: Math.round(speechWpm) })
    : "";
  const percentSuffix = (value) => summaryString("percentSuffix", { value }) || `${value}%`;
  const breathDisplay = Number.isFinite(breathPct) ? percentSuffix(Math.round(breathPct)) : "—";
  const safeHint = (s) => s ? escapeHtml(s) : "&nbsp;";

  const mode = getAdvancedMode();
  const beginnerHighlights = mode === "beginner" ? renderBeginnerHighlights(summary, context) : "";
  const advToggleLabel = mode === "advanced"
    ? (t("ui.advancedMode.beginner") || "Switch to Beginner")
    : (t("ui.advancedMode.advanced") || "Switch to Advanced");

  // 標題（優先自字典讀取多語系）
  const titleFormant = advCopy.formantTitle || t("summary.advanced.formantTitle") || t("realtime.formantTitle") || "Formant & Resonance";
  const titleIntonation = advCopy.intonationTitle || t("summary.advanced.intonationTitle") || "Intonation & Speech";
  const titleVowel = advCopy.vowelBreathTitle || t("summary.advanced.vowelBreathTitle") || "Vowel & Breathiness";

  return `
    <div class="advanced-section modern-analysis-section" data-mode="${mode}">
      <div class="adv-controls" style="margin-bottom: 12px;">
        <button type="button" class="btn sm ghost" data-adv-toggle aria-pressed="${mode === "advanced"}" aria-label="Toggle Beginner/Advanced">${escapeHtml(advToggleLabel)}</button>
      </div>
      ${beginnerHighlights}

  return {
    resolveIntonationData,
    beginnerText,
    renderBeginnerHighlights,
    renderAdvancedSummary,
  };
}
