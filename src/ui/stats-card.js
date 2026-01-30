import { t } from "../i18n.js";
import { fmt1, escapeAttr, summaryString, bandOf, isDivergent } from "./stats-helpers.js";
import { makeStats, percentileSorted, filterPitchForStats, CONFIDENCE_INCLUDE_THRESHOLD } from "../pitch-shared.js";
import { EPS } from "../analysis/helpers.js";
import { computeAdvancedSummary } from "../analysis/advanced-summary.js";
import { buildFocusInsights, renderFocusBlock } from "./focus-insights.js";
import { renderAdvancedStatistics, setupIntonationLegend } from "./advanced-stats.js";
import { setupAdvancedSection } from "../ui.js";

/**
 * Render complete statistics card after analysis
 * @param {Object} dataBundle - Contains pitch, volume, and analysis data
 */
export function renderStatsCard(dataBundle) {
  const {
    psHzSmooth = [],
    psConfidence = [],
    psDb = [],
    psVoiced = [],
    lastPf = 0,
    lastPm = 0,
    offlineFeatureStore = {},
    advSummary: injectedSummary = null,
  } = dataBundle;

  // Ensure stats container exists
  if (!document.getElementById("streamStats")) {
    createStatsContainer();
  }

  const statsEl = document.getElementById("streamStats");
  if (!statsEl) return;

  // Filter voiced pitch data
  const voicedHzRaw = [];
  for (let i = 0; i < psHzSmooth.length; i++) {
    const val = psHzSmooth[i];
    const conf = psConfidence[i] ?? 0;
    if (Number.isFinite(val) && conf >= CONFIDENCE_INCLUDE_THRESHOLD) {
      voicedHzRaw.push(val);
    }
  }

  const vols = psDb.slice();

  // If no data, clear and return
  if (!voicedHzRaw.length && !vols.length) {
    statsEl.innerHTML = "";
    return;
  }

  // Calculate statistics
  const stableVoicedHz = filterPitchForStats(voicedHzRaw);
  const voicedHz = stableVoicedHz.length ? stableVoicedHz : voicedHzRaw;
  const pitchStats = makeStats(voicedHz);
  const volStats = makeStats(vols);

  // Environment noise (10th percentile)
  const volsSorted = vols.slice().sort((a, b) => a - b);
  const envDb = percentileSorted(volsSorted, 10);

  // SNR calculation
  const snr = Number.isFinite(volStats.med) && Number.isFinite(envDb)
    ? (volStats.med - envDb)
    : NaN;

  // Pitch band and spread
  const band = bandOf(pitchStats.med);
  const spread = pitchStats.p95 - pitchStats.p05;

  // Divergence check
  const diverge = isDivergent(pitchStats.med, lastPf, lastPm);

  // Compute advanced summary if feature store available
  // Compute advanced summary if feature store available
  const advSummary = injectedSummary || (offlineFeatureStore ? computeAdvancedSummary(offlineFeatureStore, lastPf, lastPm) : null);

  // Stability categorization
  const voicedCount = psVoiced.filter(Boolean).length;
  const frameSec = offlineFeatureStore.frameSec || (50 / 1000); // PS_INTERVAL_MS
  const totalVoicedSec = voicedCount * frameSec;
  let stabilityKey = "steady";
  if (isFinite(spread)) {
    const wideThreshold = Math.max(90, 60 * Math.sqrt(Math.max(totalVoicedSec, EPS) / 5));
    if (spread > wideThreshold) stabilityKey = "wide";
    else if (spread >= 40) stabilityKey = "moderate";
  }
  const stabilityLabel = isFinite(spread)
    ? t(`summary.stability.${stabilityKey}`) || "—"
    : "—";

  // SNR categorization
  let snrKey = null;
  if (isFinite(snr)) {
    snrKey = snr >= 20 ? "quiet" : snr >= 12 ? "ok" : "noisy";
  }
  const snrLabel = snrKey ? t(`summary.snrTags.${snrKey}`) || "—" : "—";
  const snrDisplay = isFinite(snr) ? `${fmt1(snr)} dB` : "—";

  // Voiced coverage hint
  const voicedRatio = psVoiced.length ? (voicedCount / psVoiced.length) : NaN;
  let voicedHintKey = null;
  if (!isFinite(voicedRatio) || voicedRatio < 0.25) voicedHintKey = "low";
  else if (voicedRatio < 0.5) voicedHintKey = "medium";
  const voicedHintLabel = voicedHintKey ? t(`summary.voicedHint.${voicedHintKey}`) : null;

  const trendLabel = lastPf >= lastPm
    ? t("realtime.meter.feminine") || "女性化"
    : t("realtime.meter.masculine") || "男性化";

  // Build focus insights
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

  // Build HTML
  const html = buildStatsHTML({
    pitchStats,
    volStats,
    envDb,
    snr,
    band,
    spread,
    diverge,
    lastPf,
    lastPm,
    trendLabel,
    focusInsights,
    voicedHintKey,
    voicedHintLabel,
    advSummary,
  });

  statsEl.innerHTML = html;

  // Wire up interactive elements (e.g. intonation legend)
  if (advSummary && advSummary.intonation) {
    setupIntonationLegend(advSummary.intonation);
  }

  // Re-initialize advanced section toggles since DOM is newly inserted
  setupAdvancedSection(statsEl.querySelector(".advanced-section"));
}

/**
 * Create stats container element
 */
function createStatsContainer() {
  const meter = document.getElementById("meter");
  if (meter) {
    const stats = document.createElement("div");
    stats.id = "streamStats";
    stats.className = "insight";
    stats.innerHTML = "";
    meter.insertAdjacentElement("afterend", stats);
  }
}

/**
 * Build statistics HTML
 */
function buildStatsHTML(data) {
  const {
    pitchStats,
    volStats,
    envDb,
    snr,
    band,
    spread,
    diverge,
    lastPf,
    lastPm,
    trendLabel,
    focusInsights,
    voicedHintKey,
    voicedHintLabel,
    advSummary,
  } = data;

  // Header with badge
  const headerHTML = `
    <div class="insight-header">
      <span class="badge">${t("summary.badge") || "分析摘要"}</span>
      <div class="tags"></div>
    </div>
  `;

  // Focus block
  const focusHTML = renderFocusBlock(focusInsights);

  // Divergence note
  const divergeNote = diverge
    ? `<p class="note warning">${t("summary.divergenceNoteHtml", { band, trend: trendLabel }) || `音高常見於${band}，但模型傾向${trendLabel}，可能需調整共鳴或語調。`}</p>`
    : "";

  // Environment note
  const envNote = (Number.isFinite(snr) && snr < 12)
    ? `<p class="note info">${t("summary.envNoteHtml") || "環境噪音較高，建議在安靜環境重測以獲得更準確結果。"}</p>`
    : "";

  // Voiced note
  const voicedNote = voicedHintKey
    ? `<p class="subline" style="margin:4px 0 0">${voicedHintLabel}</p>`
    : "";

  // Stats table
  const statsRows = [
    { label: t("summary.statsLabels.pitchAvg") || "音高平均", value: `${fmt1(pitchStats.avg)} Hz` },
    { label: t("summary.statsLabels.pitchMed") || "音高中位數", value: `${fmt1(pitchStats.med)} Hz` },
    { label: t("summary.statsLabels.pitchHigh") || "音高上限 (95%)", value: `${fmt1(pitchStats.p95)} Hz` },
    { label: t("summary.statsLabels.pitchLow") || "音高下限 (5%)", value: `${fmt1(pitchStats.p05)} Hz` },
    { label: t("summary.statsLabels.pitchSpread") || "音高變化幅度", value: `${fmt1(spread)} Hz` },
    { label: t("summary.statsLabels.volumeAvg") || "音量平均", value: `${fmt1(volStats.avg)} dB (${fmt1(volStats.sd)} dB)` },
    { label: t("summary.statsLabels.volumeMed") || "音量中位數", value: `${fmt1(volStats.med)} dB (${fmt1(volStats.sd)} dB)` },
    { label: t("summary.statsLabels.volumeHigh") || "音量上限 (95%)", value: `${fmt1(volStats.p95)} dB` },
    { label: t("summary.statsLabels.volumeLow") || "音量下限 (5%)", value: `${fmt1(volStats.p05)} dB` },
  ];

  const statsHTML = `
    <div class="stats-grid">
      ${statsRows.map(({ label, value }) => `
        <div class="kv">
          <div class="k">${label}</div>
          <div class="v">${value}</div>
        </div>
      `).join('')}
    </div>
    <div class="kv" style="margin-top:10px">
      <div class="k">${t("summary.statsLabels.env") || "環境噪音"}</div>
      <div class="v">${fmt1(envDb)} dB</div>
    </div>
    <p class="subline" style="margin:8px 0 0">
      ${t("summary.statsIntro") || "以上為基於音高與音量的統計數據，可作為練習參考。"}
    </p>
    <p class="subline" style="margin:4px 0 0">${t("summary.volumeRelativeNote") || "音量數據相對於麥克風輸入，並非絕對分貝值。"}</p>
  `;

  // Advanced summary section
  const advancedHTML = advSummary ? renderAdvancedStatistics(advSummary, data) : '';

  return headerHTML + focusHTML + divergeNote + envNote + voicedNote + statsHTML + advancedHTML;
}
