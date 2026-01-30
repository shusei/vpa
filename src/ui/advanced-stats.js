import { t, getLocaleValue } from "../i18n.js";
import { escapeHtml, escapeAttr } from "../utils.js";
import { summaryString, fmt1, renderGauge, formatBaselineRange, BASELINES } from "./stats-helpers.js";
import { getAdvancedMode, getDetailsOpen } from "./ui-state.js";
import { renderBeginnerHighlights } from "../ui.js";
import { PS_MIN_HZ, PS_MAX_HZ } from "../pitch-shared.js";
import { EPS, INTONATION_RAW_KEY } from "../config.js";

let showIntonationRawPoints = true;
try {
  const raw = localStorage.getItem(INTONATION_RAW_KEY);
  if (raw === "0") showIntonationRawPoints = false;
} catch { }

function saveIntonationRawPreference(val) {
  try { localStorage.setItem(INTONATION_RAW_KEY, val ? "1" : "0"); } catch { }
}

export function drawIntonationCurve(canvas, intonation) {
  try {
    const pts = intonation?.points || [];
    const rawPts = intonation?.rawPoints || [];
    const shaded = intonation?.shadedRanges || [];
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.clientWidth || canvas.offsetWidth || canvas.width || 520;
    const height = canvas.clientHeight || canvas.offsetHeight || canvas.height || 140;
    const DPR = Math.max(1, window.devicePixelRatio || 1);

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.max(1, Math.round(width * DPR));
    canvas.height = Math.max(1, Math.round(height * DPR));
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8f8f8";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(0,0,0,.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 18);
    ctx.lineTo(width, height - 18);
    ctx.stroke();
    if (!pts.length) return;
    const minT = pts[0].t;
    const maxT = pts[pts.length - 1].t;
    const tRange = Math.max(maxT - minT, EPS);
    const minHz = Number.isFinite(intonation.minHz) ? intonation.minHz : Math.min(...pts.map(p => p.hz));
    const maxHz = Number.isFinite(intonation.maxHz) ? intonation.maxHz : Math.max(...pts.map(p => p.hz));
    const hzRange = Math.max(maxHz - minHz, 1);
    const projectX = (t) => 10 + ((t - minT) / tRange) * (width - 20);
    const projectY = (hz) => height - 20 - ((hz - minHz) / hzRange) * (height - 40);

    shaded.forEach(({ type, start, end }) => {
      const x0 = projectX(Math.max(minT, start));
      const x1 = projectX(Math.min(maxT, end));
      if (x1 <= x0) return;
      ctx.fillStyle = type === "mute" ? "rgba(110,110,110,0.24)" : "rgba(110,110,110,0.12)";
      ctx.fillRect(x0, 10, x1 - x0, height - 30);
    });

    if (showIntonationRawPoints && rawPts.length) {
      ctx.fillStyle = "rgba(60,60,60,0.22)";
      rawPts.forEach(({ t, hz }) => {
        const x = projectX(t);
        const y = projectY(Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz)));
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    ctx.strokeStyle = "rgba(239,93,168,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let drawing = false;
    pts.forEach((p) => {
      if (!Number.isFinite(p.hz)) { drawing = false; return; }
      const x = projectX(p.t);
      const y = projectY(Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, p.hz)));
      if (!drawing) { ctx.moveTo(x, y); drawing = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  } catch (e) { console.error("[drawIntonationCurve]", e); }
}

export function setupIntonationLegend(intonation) {
  try {
    const legend = document.querySelector(".intonation-legend") || document.getElementById("intonationLegend");
    if (!legend) return;

    const hasRaw = Array.isArray(intonation?.rawPoints) && intonation.rawPoints.length > 0;
    legend.setAttribute("data-has-raw", hasRaw ? "true" : "false");

    const cb = document.getElementById("toggleRawDots");
    const canvas = document.getElementById("intonationCanvas");

    const syncLegend = () => {
      legend.setAttribute("data-show-raw", showIntonationRawPoints ? "true" : "false");
    };

    if (cb) {
      cb.disabled = !hasRaw;
      cb.checked = !!showIntonationRawPoints && hasRaw;
      syncLegend();

      cb.onchange = () => {
        showIntonationRawPoints = !!cb.checked;
        saveIntonationRawPreference(showIntonationRawPoints);
        syncLegend();
        if (canvas) drawIntonationCurve(canvas, intonation || {});
      };
      // Initial draw
      if (canvas) drawIntonationCurve(canvas, intonation || {});
      return;
    }
    syncLegend();
  } catch (err) {
    console.error("[setupIntonationLegend]", err);
  }
}


/**
 * Render advanced analysis summary HTML
 * @param {Object} summary - Advanced analysis summary object
 * @param {Object} context - Additional context (e.g. from stats-card.js)
 */
export function renderAdvancedStatistics(summary, context = {}) {
  if (!summary) {
    return `<div class="advanced-section"><div class="note">${t("analysis.advanced.insufficient")}</div></div>`;
  }

  const {
    formants = {},
    resonanceLabel, resonanceDisplay, resonanceHint,
    energyPct = {},
    tiltAvg, tiltLabel, tiltHint,
    brightnessLabel, brightnessHint, brightnessKey,
    breathinessAvg, breathinessLabel, breathinessHint, breathinessKey,
    speechRate, speechRateLabel, speechRateHint,
    vowelFocusRatio, vowelLabel, vowelHint,
    liaisonRatio, liaisonLabel, liaisonHint,
    intonation: I = {},
  } = summary;

  // Formants
  const f1 = formants.f1?.median;
  const f2 = formants.f2?.median;
  const f3 = formants.f3?.median;
  const f1Val = Number.isFinite(f1) ? Math.round(f1) : "—";
  const f2Val = Number.isFinite(f2) ? Math.round(f2) : "—";
  const f3Val = Number.isFinite(f3) ? Math.round(f3) : "—";
  const f1Hint = formants.f1?.hint || "";
  const f2Hint = formants.f2?.hint || "";
  const f3Hint = formants.f3?.hint || "";

  // Resonance
  const chestPct = Number.isFinite(energyPct.chest) ? Math.round(energyPct.chest * 100) : 0;
  const maskPct = Number.isFinite(energyPct.mask) ? Math.round(energyPct.mask * 100) : 0;
  const headPct = Number.isFinite(energyPct.head) ? Math.round(energyPct.head * 100) : 0;
  const chestLabel = t("realtime.resonance.chest", { value: chestPct });
  const maskLabel = t("realtime.resonance.mask", { value: maskPct });
  const headLabel = t("realtime.resonance.head", { value: headPct });

  // Breathiness & Liaison
  const breathPct = Number.isFinite(breathinessAvg) ? Math.round(breathinessAvg * 100) : NaN;
  const liaisonPct = Number.isFinite(liaisonRatio) ? Math.round(liaisonRatio * 100) : NaN;

  // Speech Rate
  const speechSyll = speechRate?.syllPerSec;
  const speechWpm = speechRate?.wordsPerMin;

  // Labels & Titles
  // Note: We use getLocaleValue/t to fetch localized strings similar to legacy logic
  // But since we don't have global 'advCopy', we rely on 't' keys or fallback.
  const labelFormantF1 = t("summary.advanced.vowelCards.f1") || "F1 (Throat)";
  const labelFormantF2 = t("summary.advanced.vowelCards.f2") || "F2 (Tongue)";
  const labelFormantF3 = t("summary.advanced.vowelCards.f3") || "F3 (Lip/Tip)";
  const labelFormantTilt = t("summary.advanced.vowelCards.tilt") || "Spectral Tilt";
  const labelFormantBright = t("summary.advanced.vowelCards.brightness") || "Brightness";
  const labelResonance = t("summary.advanced.vowelCards.resonance") || "Resonance";
  const labelIntonationTrend = t("summary.advanced.intonationCards.trend") || "Trend";
  const labelIntonationRange = t("summary.advanced.intonationCards.range") || "Range";
  const labelSpeechRate = t("summary.advanced.intonationCards.speed") || "Pace";
  const labelBrightness = labelFormantBright;
  const labelBreathiness = t("summary.advanced.vowelCards.breathiness") || "Breathiness";
  const labelLiaison = t("summary.advanced.intonationCards.liaison") || "Liaison";
  const labelVowelFocus = t("summary.advanced.vowelCards.focus") || "Vowel focus";

  // Format Displays
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
  // Brightness display from categorizer
  const brightnessDisplay = summary.brightnessLabel || "—";

  const mode = getAdvancedMode();
  const beginnerHighlights = mode === "beginner" ? renderBeginnerHighlights(summary, context) : "";
  const advToggleLabel = mode === "advanced"
    ? (t("ui.advancedMode.beginner") || "Switch to Beginner")
    : (t("ui.advancedMode.advanced") || "Switch to Advanced");

  const titleFormant = t("summary.advanced.formantTitle") || "Formant & Resonance";
  const titleIntonation = t("summary.advanced.intonationTitle") || "Intonation & Speech";
  const titleVowel = t("summary.advanced.vowelBreathTitle") || "Vowel & Breathiness";

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
            <span class="baseline">${escapeHtml(labelResonance)}: ${escapeHtml(resonanceDisplay || resonanceLabel || "—")}</span>
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
          <div class="adv-card" title="${escapeAttr(tiltHint || "")}">
            <div class="k">${escapeHtml(labelFormantTilt)}</div>
            <div class="v">${tiltLabel || "—"}</div>
            ${renderGauge(tiltAvg, BASELINES.tilt, labelFormantTilt)}
            <div class="hint">${safeHint(tiltHint)}</div>
          </div>
        </div>
        <div class="adv-card adv-card--resonance" role="group" aria-label="${escapeAttr(labelResonance)}" title="${escapeAttr(resonanceHint || "")}">
          <div class="k">${escapeHtml(labelResonance)}</div>
          <div class="v resonance-value">${escapeHtml(resonanceDisplay || resonanceLabel || "—")}</div>
          <div class="resonance-bar resonance-bar--card">
            <span class="res-part chest" style="width:${chestPct}%"><span>${escapeHtml(chestLabel)}</span></span>
            <span class="res-part mask" style="width:${maskPct}%"><span>${escapeHtml(maskLabel)}</span></span>
            <span class="res-part head" style="width:${headPct}%"><span>${escapeHtml(headLabel)}</span></span>
          </div>
          <div class="hint">${safeHint(resonanceHint)}</div>
        </div>
      </details>

      <!-- Intonation & Speech -->
      <details class="adv-details" data-adv="intonation" ${getDetailsOpen("intonation", mode === "advanced") ? "open" : ""}>
        <summary>
          <span class="adv-title">${escapeHtml(titleIntonation)}</span>
          <span class="adv-baselines">
            <span class="baseline">${escapeHtml(labelIntonationTrend)}: ${escapeHtml(I.slopeLabel || "—")}</span>
            <span class="baseline">${escapeHtml(labelIntonationRange)}: ${escapeHtml(I.rangeLabel || "—")}</span>
            <span class="baseline">${escapeHtml(labelSpeechRate)}: ${escapeHtml(speechRateDisplay)}</span>
          </span>
        </summary>

        <div class="advanced-grid advanced-grid--three">
            <div class="adv-card">
            <div class="k">${escapeHtml(labelIntonationTrend)}</div>
            <div class="v">${escapeHtml(I.slopeLabel || "—")}</div>
            <div class="hint">${escapeHtml(I.hint ? I.hint.split(" ")[0] : "")}</div>
            </div>

            <div class="adv-card">
            <div class="k">${escapeHtml(labelIntonationRange)}</div>
            <div class="v">${escapeHtml(I.rangeLabel || "—")}</div>
            <div class="hint">${escapeHtml(I.rangeHint || "")}</div>
            </div>

            <div class="adv-card">
            <div class="k">${escapeHtml(labelSpeechRate)}</div>
            <div class="v">${escapeHtml(speechRateDisplay)} <span class="suffix">${escapeHtml(speechWpmDisplay)}</span></div>
            <div class="hint">${escapeHtml(speechRateHint || "")}</div>
            </div>
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
          <div class="adv-card" title="${escapeAttr(breathinessHint || "")}">
            <div class="k">${escapeHtml(labelBreathiness)}</div>
            <div class="v">${escapeHtml(breathinessLabel || "—")}</div>
            ${renderGauge(breathPct, BASELINES.breath, labelBreathiness)}
            <div class="hint">${safeHint(breathinessHint)}</div>
          </div>
          <div class="adv-card" title="${escapeAttr(liaisonHint || "")}">
            <div class="k">${escapeHtml(labelLiaison)}</div>
            <div class="v">${escapeHtml(liaisonDisplay || "—")}</div>
            ${renderGauge(liaisonPct, BASELINES.liaison, labelLiaison)}
            <div class="hint">${safeHint(liaisonHint)}</div>
          </div>
        </div>

        <div class="adv-card">
          <div class="k">${escapeHtml(labelVowelFocus)}</div>
          <div class="v">${escapeHtml(vowelDisplay || "—")}</div>
          <div class="hint">${safeHint(vowelHint)}</div>
        </div>
      </details>
    </div>
  `;
}
