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
  const I = {
    rangeHz: __Iraw?.rangeHz ?? null,
    rangeLabel: __Iraw?.rangeLabel ?? summary.intonation?.rangeLabel ?? null,
    rangeDisplay: __Iraw?.rangeDisplay ?? __Iraw?.rangeLabel ?? summary.intonation?.rangeDisplay ?? summary.intonation?.rangeLabel ?? "—",
    slopeLabel: __Iraw?.slopeLabel ?? summary.intonation?.trendLabel ?? summary.intonation?.slopeLabel ?? summary.intonation?.trend ?? "—",
    slopeHint: __Iraw?.slopeHint ?? summary.intonation?.slopeHint ?? summary.intonation?.trendHint ?? ""
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
  const labelFormantTilt = advFormantCards.tilt || t("summary.advanced.formantCards.tilt") || "Spectral Tilt";
  const labelFormantBright = advFormantCards.brightness || t("summary.advanced.formantCards.brightness") || "Brightness";
  const labelResonance = advCopy.resonanceTitle || t("realtime.resonance.label") || "Resonance balance";
  const labelIntonationTrend = advIntonationCards.trend || t("summary.advanced.intonationCards.trend") || t("analysis.advanced.intonationCards.trend") || "Trend";
  const labelIntonationRange = advIntonationCards.range || t("summary.advanced.intonationCards.range") || t("analysis.advanced.intonationCards.range") || "Range";
  const labelSpeechRate = advIntonationCards.speechRate || t("summary.advanced.intonationCards.speechRate") || t("analysis.advanced.intonationCards.speechRate") || "Speech rate";
  const chestLabel = t("realtime.resonance.chest", { value: chestPct }) || `Chest ${chestPct}%`;
  const maskLabel = t("realtime.resonance.mask", { value: maskPct }) || `Mask ${maskPct}%`;
  const headLabel = t("realtime.resonance.head", { value: headPct }) || `Head ${headPct}%`;

  // Formants
  const f1 = Number(summary.formants?.f1?.median);
  const f2 = Number(summary.formants?.f2?.median);
  const f3 = Number(summary.formants?.f3?.median);
  const f1Val = Number.isFinite(f1) ? fmt1(f1) : (summary.formants?.f1?.display || "—");
  const f2Val = Number.isFinite(f2) ? fmt1(f2) : (summary.formants?.f2?.display || "—");
  const f3Val = Number.isFinite(f3) ? fmt1(f3) : (summary.formants?.f3?.display || "—");
  const f1Hint = summary.formants?.f1?.hint || `F1 ${f1Val}Hz`;
  const f2Hint = summary.formants?.f2?.hint || `F2 ${f2Val}Hz`;
  const f3Hint = summary.formants?.f3?.hint || `F3 ${f3Val}Hz`;

  // 其它指標
  const tiltAvg = Number(summary.tiltAvg);
  const breathRatio = Number(summary.breathinessAvg);
  const breathPct = Number.isFinite(breathRatio) ? breathRatio * 100 : NaN;
  const speechSyll = Number(summary.speechRate?.syllPerSec);
  const speechWpm = Number(summary.speechRate?.wordsPerMin);
  const liaisonRatio = Number(summary.liaisonRatio);
  const liaisonPct = Number.isFinite(liaisonRatio) ? liaisonRatio * 100 : NaN;
  const brightnessDisplay = summary.brightnessLabel || "—";
  const brightnessHint = summary.brightnessHint || "";
  const labelBrightness = labelFormantBright;
  const labelBreathiness = advVowelCards.breathiness || t("summary.advanced.vowelCards.breathiness") || "Breathiness";
  const labelLiaison = advIntonationCards.liaison || t("summary.advanced.intonationCards.liaison") || "Liaison";
  const labelVowelFocus = advVowelCards.focus || t("summary.advanced.vowelCards.focus") || "Vowel focus";

  // 格式化
  const speechRateDisplay = Number.isFinite(speechSyll)
    ? summaryString("speechRateDisplay", { value: fmt1(speechSyll) })
    : "—";
  const speechWpmDisplay = Number.isFinite(speechWpm)
    ? summaryString("speechRateWpm", { value: Math.round(speechWpm) })
    : "";
  const percentSuffix = (value) => summaryString("percentSuffix", { value }) || `${value}%`;
  const breathDisplay = Number.isFinite(breathPct) ? percentSuffix(Math.round(breathPct)) : "—";
  const liaisonDisplay = Number.isFinite(liaisonPct) ? percentSuffix(Math.round(liaisonPct)) : "";
  const vowelDisplay = (summary.vowelLabel || "—") + (Number.isFinite(summary.vowelFocusRatio)
    ? " · " + percentSuffix(Math.round(summary.vowelFocusRatio * 100)) : "");
  const safeHint = (s) => s ? escapeHtml(s) : "&nbsp;";

  const mode = getAdvancedMode();
  const beginnerHighlights = mode === "beginner" ? renderBeginnerHighlights(summary, context) : "";
  const advToggleLabel = mode === "advanced"
    ? (t("ui.advancedMode.beginner") || "Switch to Beginner")
    : (t("ui.advancedMode.advanced") || "Switch to Advanced");

  // 標題（有就用，沒有就回退英文）
  const titleFormant = advCopy.formantTitle || t("summary.advanced.formantTitle") || "Formant & Resonance";
  const titleIntonation = advCopy.intonationTitle || t("summary.advanced.intonationTitle") || "Intonation & Speech";
  const titleVowel = advCopy.vowelBreathTitle || t("summary.advanced.vowelBreathTitle") || "Vowel & Breathiness";

  return `
    <div class="advanced-section" data-mode="${mode}">
      <div class="adv-controls">
        <button type="button" class="btn sm ghost" data-adv-toggle aria-pressed="${mode === "advanced"}" aria-label="Toggle Beginner/Advanced">${escapeHtml(advToggleLabel)}</button>
      </div>
      ${beginnerHighlights}

      <!-- Formant & Resonance -->
      <details class="adv-details" data-adv="formant" ${getDetailsOpen("formant", mode === "advanced") ? "open" : ""}>
        <summary>
          <span class="adv-title">${escapeHtml(titleFormant)}</span>
          <span class="adv-baselines">
            <span class="baseline">F1 ${f1Val}Hz (${escapeHtml(formatBaselineRange(BASELINES.f1))})</span>
            <span class="baseline">F2 ${f2Val}Hz (${escapeHtml(formatBaselineRange(BASELINES.f2))})</span>
            <span class="baseline">F3 ${f3Val}Hz (${escapeHtml(formatBaselineRange(BASELINES.f3))})</span>
            <span class="baseline">${escapeHtml(labelResonance)}: ${escapeHtml(summary.resonanceDisplay || summary.resonanceLabel || "—")}</span>
          </span>
        </summary>
        <div class="advanced-grid advanced-grid--four">
          <div class="adv-card" title="${escapeAttr(f1Hint)}">
            <div class="k">${escapeHtml(labelFormantF1)}</div>
            <div class="v">${f1Val}Hz</div>
            ${renderGauge(f1, BASELINES.f1, labelFormantF1)}
            <div class="hint">${safeHint(f1Hint)}</div>
          </div>
          <div class="adv-card" title="${escapeAttr(f2Hint)}">
            <div class="k">${escapeHtml(labelFormantF2)}</div>
            <div class="v">${f2Val}Hz</div>
            ${renderGauge(f2, BASELINES.f2, labelFormantF2)}
            <div class="hint">${safeHint(f2Hint)}</div>
          </div>
          <div class="adv-card" title="${escapeAttr(f3Hint)}">
            <div class="k">${escapeHtml(labelFormantF3)}</div>
            <div class="v">${f3Val}Hz</div>
            ${renderGauge(f3, BASELINES.f3, labelFormantF3)}
            <div class="hint">${safeHint(f3Hint)}</div>
          </div>
          <div class="adv-card" title="${escapeAttr(summary.tiltHint || "")}">
            <div class="k">${escapeHtml(labelFormantTilt)}</div>
            <div class="v">${summary.tiltLabel || "—"}</div>
            ${renderGauge(tiltAvg, BASELINES.tilt, labelFormantTilt)}
            <div class="hint">${safeHint(summary.tiltHint)}</div>
          </div>
        </div>
        <div class="adv-card adv-card--resonance" role="group" aria-label="${escapeAttr(labelResonance)}" title="${escapeAttr(summary.resonanceHint || "")}">
          <div class="k">${escapeHtml(labelResonance)}</div>
          <div class="v resonance-value">${escapeHtml(summary.resonanceDisplay || summary.resonanceLabel || "—")}</div>
          <div class="resonance-bar resonance-bar--card">
            <span class="res-part chest" style="width:${chestPct}%"><span>${escapeHtml(chestLabel)}</span></span>
            <span class="res-part mask" style="width:${maskPct}%"><span>${escapeHtml(maskLabel)}</span></span>
            <span class="res-part head" style="width:${headPct}%"><span>${escapeHtml(headLabel)}</span></span>
          </div>
          <div class="hint">${safeHint(summary.resonanceHint)}</div>
        </div>
      </details>

      <!-- Intonation & Speech -->
  <details class="adv-details" data-adv="intonation" ${getDetailsOpen("intonation", mode === "advanced") ? "open" : ""}>
    <summary>
      <span class="adv-title">${escapeHtml(titleIntonation)}</span>
      <span class="adv-baselines">
        <span class="baseline">${escapeHtml(labelIntonationTrend)}: ${escapeHtml(I.slopeLabel || "—")}</span>
        <span class="baseline">${escapeHtml(labelIntonationRange)}: ${escapeHtml(I.rangeDisplay || "—")}</span>
        <span class="baseline">${escapeHtml(labelSpeechRate)}: ${escapeHtml(speechRateDisplay)}</span>
      </span>
    </summary>

<div class="adv-card">
  <div class="k">${escapeHtml(labelIntonationTrend)}</div>
  <div class="v">${escapeHtml(I.slopeLabel || "—")}</div>
  <div class="hint">${escapeHtml(I.slopeHint || t("analysis.intonation.insufficient.slopeHint") || "")}</div>
</div>

<div class="adv-card">
  <div class="k">${escapeHtml(labelIntonationRange)}</div>
  <div class="v">${escapeHtml(I.rangeDisplay || "—")}</div>
  <div class="hint">${escapeHtml(summary.intonation?.rangeHint || "")}</div>
</div>

<div class="adv-card">
  <div class="k">${escapeHtml(labelSpeechRate)}</div>
  <div class="v">${escapeHtml(speechRateDisplay)} <span class="suffix">${escapeHtml(speechWpmDisplay)}</span></div>
  <div class="hint">${escapeHtml(summary.speechRateHint || "")}</div>
</div>

    <div class="intonation-wrap">
      <canvas id="intonationCanvas" class="intonation-canvas" aria-label="${escapeAttr(t("analysis.advanced.canvasAria") || "Intonation curve")}"></canvas>
      <div class="intonation-legend">
        <label><input id="toggleRawDots" type="checkbox" /> ${escapeHtml(t("analysis.advanced.intonationLegend.show") || "Show raw dots")}</label>
      </div>
    </div>
  </details>

      <!-- Vowel & Breathiness -->
      <details class="adv-details" data-adv="vowel" ${getDetailsOpen("vowel", mode === "advanced") ? "open" : ""}>
        <summary>
          <span class="adv-title">${escapeHtml(titleVowel)}</span>
          <span class="adv-baselines">
            <span class="baseline">${escapeHtml(labelBrightness)}: ${escapeHtml(brightnessDisplay)}</span>
            <span class="baseline">${escapeHtml(labelBreathiness)}: ${escapeHtml(breathDisplay)} (${escapeHtml(formatBaselineRange(BASELINES.breath))})</span>
            <span class="baseline">${escapeHtml(labelVowelFocus)}: ${escapeHtml(vowelDisplay || "—")}</span>
            <span class="baseline">${escapeHtml(labelLiaison)}: ${escapeHtml(liaisonDisplay || "—")} (${escapeHtml(formatBaselineRange(BASELINES.liaison))})</span>
          </span>
        </summary>
        <div class="adv-note">${safeHint(brightnessHint)}</div>
        <div class="advanced-grid advanced-grid--three">
          <div class="adv-card">
            <div class="k">${escapeHtml(labelBrightness)}</div>
            <div class="v">${escapeHtml(brightnessDisplay)}</div>
            <div class="hint">${safeHint(brightnessHint)}</div>
          </div>
          <div class="adv-card" title="${escapeAttr(summary.breathinessHint || "")}">
            <div class="k">${escapeHtml(labelBreathiness)}</div>
            <div class="v">${escapeHtml(summary.breathinessLabel || "—")}</div>
            ${renderGauge(breathPct, BASELINES.breath, labelBreathiness)}
            <div class="hint">${safeHint(summary.breathinessHint)}</div>
          </div>
          <div class="adv-card" title="${escapeAttr(summary.liaisonHint || "")}">
            <div class="k">${escapeHtml(labelLiaison)}</div>
            <div class="v">${escapeHtml(liaisonDisplay || "—")}</div>
            ${renderGauge(liaisonPct, BASELINES.liaison, labelLiaison)}
            <div class="hint">${safeHint(summary.liaisonHint)}</div>
          </div>
        </div>

        <div class="adv-card">
          <div class="k">${escapeHtml(labelVowelFocus)}</div>
          <div class="v">${escapeHtml(vowelDisplay || "—")}</div>
          <div class="hint">${safeHint(summary.vowelHint)}</div>
        </div>
      </details>
    </div>
  `;
}


  return {
    resolveIntonationData,
    beginnerText,
    renderBeginnerHighlights,
    renderAdvancedSummary,
  };
}
