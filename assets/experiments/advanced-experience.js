import { onInferenceDone } from "../app.js?v=20260801-multiappopen1";
import { getCurrentLocale, onLocaleChange, t } from "../js/i18n.js";
import { evaluateAdvancedExperience } from "./advanced-evaluator.js";
import {
  buildShareTargets,
  buildShareUrl,
  createShareCardBlob,
  downloadBlob,
  shareWithSystem,
} from "./share-card.js";

let lastAnalysis = null;
let lastResult = null;
let renderedAnalysisId = 0;
const resultListeners = new Set();

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function track(eventName, params = {}) {
  if (typeof window.gtag !== "function") return;
  try {
    window.gtag("event", eventName, {
      event_category: "advanced_experiment",
      ...params,
    });
  } catch (error) {
    console.warn("[advanced-experiment] analytics failed", error);
  }
}

const meter = document.getElementById("meter");
const resultPanel = document.createElement("section");
resultPanel.id = "advancedExperience";
resultPanel.className = "advanced-experience";
resultPanel.setAttribute("aria-live", "polite");
resultPanel.hidden = true;

if (meter) meter.insertAdjacentElement("afterend", resultPanel);

function componentLabel(key) {
  return t(`experiment.advanced.components.${key}`);
}

function renderComponent(key, score, weight) {
  const percent = Math.round(score * 100);
  return `
    <div class="advanced-component">
      <div class="advanced-component__head">
        <span>${escapeHtml(componentLabel(key))}</span>
        <span>${percent}</span>
      </div>
      <div class="advanced-component__track" aria-hidden="true">
        <span style="--advanced-score:${score}"></span>
      </div>
      <small>${escapeHtml(t("experiment.advanced.components.weight", { value: weight }))}</small>
    </div>
  `;
}

function ageValue(result) {
  if (!result?.voiceAge?.ready) {
    return t("experiment.advanced.voiceAge.unavailable");
  }
  return t("experiment.advanced.voiceAge.value", {
    max: result.voiceAge.max,
    min: result.voiceAge.min,
  });
}

function archetypeValue(result) {
  return t(`experiment.advanced.archetypes.${result.archetypeKey}`);
}

function confidenceValue(result) {
  return t(`experiment.advanced.confidence.${result.confidence.key}`);
}

function ageConfidenceValue(result) {
  return t(`experiment.advanced.confidence.${result.voiceAge?.confidenceKey || "low"}`);
}

function sampleTypeValue(result) {
  const sampleType = result.voiceAge?.sampleType || result.voiceQuality?.sampleType || "unknown";
  return t(`experiment.advanced.voiceAgeV2.sampleTypes.${sampleType}`);
}

function metricValue(metric, key) {
  const value = Number(metric?.[key]);
  if (!Number.isFinite(value)) return t("experiment.advanced.voiceAgeV2.notAvailable");
  return key === "valuePct"
    ? t("experiment.advanced.voiceAgeV2.percentValue", { value: value.toFixed(2) })
    : t("experiment.advanced.voiceAgeV2.dbValue", { value: value.toFixed(1) });
}

function voiceMetricMarkup(labelKey, metric, valueKey, usedForAge) {
  const statusKey = usedForAge
    ? "experiment.advanced.voiceAgeV2.used"
    : "experiment.advanced.voiceAgeV2.connectedReference";
  return `
    <article class="voice-age-metric">
      <span>${escapeHtml(t(`experiment.advanced.voiceAgeV2.metrics.${labelKey}`))}</span>
      <strong>${escapeHtml(metricValue(metric, valueKey))}</strong>
      <small>${escapeHtml(t(statusKey))}</small>
    </article>
  `;
}

function voiceAgeEvidenceMarkup(result) {
  const voiceQuality = result.voiceQuality;
  if (!voiceQuality) {
    return `
      <section class="voice-age-evidence voice-age-evidence--unavailable">
        <h3>${escapeHtml(t("experiment.advanced.voiceAgeV2.title"))}</h3>
        <p>${escapeHtml(t("experiment.advanced.voiceAgeV2.rejected"))}</p>
      </section>
    `;
  }
  const metrics = voiceQuality.metrics || {};
  const sustained = voiceQuality.sampleType === "sustainedVowel";
  const rejected = result.voiceAge?.ready
    ? ""
    : `<p class="voice-age-evidence__warning">${escapeHtml(t("experiment.advanced.voiceAgeV2.rejected"))}</p>`;
  return `
    <section class="voice-age-evidence">
      <div class="voice-age-evidence__head">
        <div>
          <span>${escapeHtml(t("experiment.advanced.voiceAgeV2.eyebrow"))}</span>
          <h3>${escapeHtml(t("experiment.advanced.voiceAgeV2.title"))}</h3>
        </div>
        <div class="voice-age-evidence__badges">
          <span>${escapeHtml(t("experiment.advanced.voiceAgeV2.sampleType", {
            value: sampleTypeValue(result),
          }))}</span>
          <span>${escapeHtml(t("experiment.advanced.voiceAgeV2.confidence", {
            value: ageConfidenceValue(result),
          }))}</span>
        </div>
      </div>
      ${rejected}
      <div class="voice-age-evidence__metrics">
        ${voiceMetricMarkup("jitter", metrics.jitterLocal, "valuePct", sustained && metrics.jitterLocal?.reliable)}
        ${voiceMetricMarkup("shimmer", metrics.shimmerLocal, "valuePct", sustained && metrics.shimmerLocal?.reliable)}
        ${voiceMetricMarkup("hnr", metrics.hnr, "valueDb", metrics.hnr?.reliable)}
        ${voiceMetricMarkup("cpp", metrics.cpp, "valueDb", metrics.cpp?.reliable)}
      </div>
      <p class="voice-age-evidence__note">${escapeHtml(t("experiment.advanced.voiceAgeV2.note"))}</p>
      <code>${escapeHtml(result.voiceAge?.version || "voice-age-unavailable")}</code>
    </section>
  `;
}

function statusElement() {
  return resultPanel.querySelector("[data-share-status]");
}

function setShareStatus(messageKey) {
  const status = statusElement();
  if (status) status.textContent = t(messageKey);
}

function shareCaption(result) {
  const key = result.voiceAge?.ready
    ? "experiment.advanced.share.caption"
    : "experiment.advanced.share.captionWithoutAge";
  return t(key, {
    age: ageValue(result),
    archetype: archetypeValue(result),
    score: result.score,
  });
}

function shareLabels(result) {
  return {
    archetypeValue: archetypeValue(result),
    beta: t("experiment.advanced.beta"),
    challenge: t("experiment.advanced.share.challenge", { score: result.score }),
    components: {
      intonation: componentLabel("intonation"),
      model: componentLabel("model"),
      pitch: componentLabel("pitch"),
      resonance: componentLabel("resonance"),
    },
    disclaimer: t("experiment.advanced.share.cardDisclaimer"),
    strictScore: t("experiment.advanced.strictScore"),
    voiceAge: t("experiment.advanced.voiceAge.title"),
    voiceAgeValue: ageValue(result),
  };
}

export async function createResultCard(result = lastResult, { shareUrl = buildShareUrl() } = {}) {
  if (!result) return null;
  return createShareCardBlob({
    labels: shareLabels(result),
    result,
    shareUrl,
    theme: document.documentElement.getAttribute("data-faction") === "light" ? "light" : "dark",
  });
}

async function handleSystemShare(result, { downloadWhenUnsupported = true } = {}) {
  try {
    const blob = await createResultCard(result);
    const response = await shareWithSystem({
      blob,
      caption: shareCaption(result),
      title: t("experiment.advanced.share.title"),
      url: buildShareUrl(),
    });
    if (response.method === "unsupported" && downloadWhenUnsupported) {
      downloadBlob(blob, "vpa-advanced-result.png");
      setShareStatus("experiment.advanced.share.downloaded");
      track("share_card_download", { source: "system_fallback" });
      return;
    }
    if (response.method !== "unsupported") {
      setShareStatus("experiment.advanced.share.shared");
      track("share_success", { method: response.method });
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      setShareStatus("experiment.advanced.share.cancelled");
      return;
    }
    console.error("[advanced-experiment] share failed", error);
    setShareStatus("experiment.advanced.share.failed");
  }
}

async function downloadShareCard(result) {
  try {
    const blob = await createResultCard(result);
    downloadBlob(blob, "vpa-advanced-result.png");
    setShareStatus("experiment.advanced.share.downloaded");
    track("share_card_download", { source: "button" });
  } catch (error) {
    console.error("[advanced-experiment] share card failed", error);
    setShareStatus("experiment.advanced.share.failed");
  }
}

async function copyShareText(result) {
  const text = `${shareCaption(result)}\n${buildShareUrl()}`;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const input = document.createElement("textarea");
    input.value = text;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.append(input);
    input.select();
    document.execCommand("copy");
    input.remove();
  }
  setShareStatus("experiment.advanced.share.copied");
  track("share_copy", { score_band: Math.floor(result.score / 10) * 10 });
}

function openPlatform(platform, result) {
  const targets = buildShareTargets({
    caption: shareCaption(result),
    url: buildShareUrl(),
  });
  const target = targets[platform];
  if (!target) return;
  window.open(target, "_blank", "noopener,noreferrer");
  setShareStatus("experiment.advanced.share.opened");
  track("share_platform_selected", {
    platform,
    score_band: Math.floor(result.score / 10) * 10,
  });
}

function bindShareActions(result) {
  resultPanel.querySelector("[data-share-primary]")?.addEventListener("click", () => {
    handleSystemShare(result);
  });
  resultPanel.querySelector("[data-share-instagram]")?.addEventListener("click", () => {
    handleSystemShare(result);
  });
  resultPanel.querySelector("[data-share-download]")?.addEventListener("click", async () => {
    await downloadShareCard(result);
  });
  resultPanel.querySelector("[data-share-copy]")?.addEventListener("click", () => {
    copyShareText(result);
  });
  resultPanel.querySelectorAll("[data-share-platform]").forEach((button) => {
    button.addEventListener("click", () => {
      openPlatform(button.getAttribute("data-share-platform"), result);
    });
  });
}

export function formatAdvancedResult(result) {
  if (!result) return null;
  return {
    age: ageValue(result),
    archetype: archetypeValue(result),
    caption: shareCaption(result),
    confidence: confidenceValue(result),
    insight: t(`experiment.advanced.insight.${result.insightKey}`),
  };
}

export function onAdvancedResult(listener) {
  if (typeof listener !== "function") return () => { };
  resultListeners.add(listener);
  return () => resultListeners.delete(listener);
}

function notifyResultListeners(analysis, result) {
  resultListeners.forEach((listener) => {
    try {
      listener({ analysis, result });
    } catch (error) {
      console.warn("[advanced-experiment] result listener failed", error);
    }
  });
}

export function renderAnalysis(analysis, { notify = true } = {}) {
  lastAnalysis = analysis;
  lastResult = evaluateAdvancedExperience(analysis);
  const result = lastResult;
  const scoreText = result.ready ? `${result.score}%` : "—";
  const insufficient = result.ready
    ? ""
    : `<p class="advanced-experience__warning">${escapeHtml(t("experiment.advanced.insufficient"))}</p>`;
  const contradiction = result.contradiction
    ? `<span class="advanced-experience__flag">${escapeHtml(t("experiment.advanced.contradiction"))}</span>`
    : "";
  const disabled = result.ready ? "" : "disabled";
  resultPanel.innerHTML = `
    <div class="advanced-experience__head">
      <div>
        <span class="advanced-experience__eyebrow">${escapeHtml(t("experiment.advanced.beta"))}</span>
        <h2>${escapeHtml(t("experiment.advanced.title"))}</h2>
      </div>
      <span class="advanced-experience__confidence">
        ${escapeHtml(t("experiment.advanced.confidence.label"))} ${escapeHtml(confidenceValue(result))}
      </span>
    </div>
    ${insufficient}
    <div class="advanced-experience__hero">
      <div class="advanced-experience__score">
        <strong>${scoreText}</strong>
        <span>${escapeHtml(t("experiment.advanced.strictScore"))}</span>
        ${contradiction}
      </div>
      <div class="advanced-experience__identity">
        <span>${escapeHtml(t("experiment.advanced.voiceAge.title"))}</span>
        <strong>${escapeHtml(ageValue(result))}</strong>
        <span>${escapeHtml(t("experiment.advanced.archetype.title"))}</span>
        <strong>${escapeHtml(archetypeValue(result))}</strong>
      </div>
    </div>
    <div class="advanced-experience__components">
      ${renderComponent("model", result.components.model, 30)}
      ${renderComponent("resonance", result.components.resonance, 45)}
      ${renderComponent("pitch", result.components.pitch, 15)}
      ${renderComponent("intonation", result.components.intonation, 10)}
    </div>
    ${voiceAgeEvidenceMarkup(result)}
    <div class="advanced-experience__insight">
      <span>${escapeHtml(t("experiment.advanced.insight.label"))}</span>
      <strong>${escapeHtml(t(`experiment.advanced.insight.${result.insightKey}`))}</strong>
    </div>
    <p class="advanced-experience__disclaimer">${escapeHtml(t("experiment.advanced.disclaimer"))}</p>
    <div class="advanced-share">
      <button type="button" class="advanced-share__primary" data-share-primary ${disabled}>
        ${escapeHtml(t("experiment.advanced.share.primary"))}
      </button>
      <div class="advanced-share__platforms" aria-label="${escapeHtml(t("experiment.advanced.share.platformAria"))}">
        <button type="button" data-share-platform="x" ${disabled}>X</button>
        <button type="button" data-share-platform="threads" ${disabled}>Threads</button>
        <button type="button" data-share-platform="line" ${disabled}>LINE</button>
        <button type="button" data-share-platform="facebook" ${disabled}>Facebook</button>
        <button type="button" data-share-instagram ${disabled}>IG</button>
      </div>
      <p class="advanced-share__hint">${escapeHtml(t("experiment.advanced.share.platformHint"))}</p>
      <div class="advanced-share__fallbacks">
        <button type="button" data-share-download ${disabled}>${escapeHtml(t("experiment.advanced.share.download"))}</button>
        <button type="button" data-share-copy ${disabled}>${escapeHtml(t("experiment.advanced.share.copy"))}</button>
      </div>
      <p class="advanced-share__status" data-share-status role="status"></p>
    </div>
  `;
  resultPanel.hidden = false;
  bindShareActions(result);
  if (notify) {
    track("advanced_result_view", {
      confidence: result.confidence.key,
      score_band: Math.floor(result.score / 10) * 10,
    });
    track("voice_age_evaluated", {
      confidence: result.voiceAge.confidenceKey,
      ready: result.voiceAge.ready,
      sample_type: result.voiceAge.sampleType,
      version: result.voiceAge.version,
    });
  }
  if (notify) notifyResultListeners(analysis, result);
  return result;
}

async function waitForLatestAnalysis(previousId) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    const analysis = window.vpaLatestAnalysis;
    const analysisId = Number(analysis?.analysisId || 0);
    if (analysis && analysisId > previousId) return analysis;
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
  return null;
}

onInferenceDone(async () => {
  const analysis = await waitForLatestAnalysis(renderedAnalysisId);
  if (!analysis) return;
  renderedAnalysisId = Number(analysis.analysisId || renderedAnalysisId + 1);
  renderAnalysis(analysis);
});

onLocaleChange(() => {
  if (lastAnalysis) renderAnalysis(lastAnalysis, { notify: false });
});

window.vpaAdvancedExperience = {
  createCard: () => createResultCard(lastResult),
  evaluate: evaluateAdvancedExperience,
  formatResult: formatAdvancedResult,
  getLastAnalysis: () => lastAnalysis,
  getLastResult: () => lastResult,
  getLocale: getCurrentLocale,
  renderAnalysis,
};
