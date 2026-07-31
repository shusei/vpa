import { onInferenceDone } from "../app.js";
import { getCurrentLocale, onLocaleChange, t } from "../js/i18n.js";
import { evaluateAdvancedExperience } from "./advanced-evaluator.js";
import {
  buildShareTargets,
  buildShareUrl,
  createShareCardBlob,
  downloadBlob,
  shareWithSystem,
} from "./share-card.js";

const MODE_KEY = "vpa::experiment.analysisMode";
const VALID_MODES = new Set(["basic", "advanced"]);

let currentMode = readMode();
let lastAnalysis = null;
let lastResult = null;
let renderedAnalysisId = 0;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function readMode() {
  try {
    const stored = localStorage.getItem(MODE_KEY);
    if (VALID_MODES.has(stored)) return stored;
  } catch {
    // Storage is optional.
  }
  return "basic";
}

function saveMode(mode) {
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    // Storage is optional.
  }
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

const recordButton = document.getElementById("recordBtn");
const meter = document.getElementById("meter");
const modeSelector = document.createElement("div");
modeSelector.id = "analysisModeExperiment";
modeSelector.className = "analysis-mode-experiment";
modeSelector.setAttribute("role", "group");

const resultPanel = document.createElement("section");
resultPanel.id = "advancedExperience";
resultPanel.className = "advanced-experience";
resultPanel.setAttribute("aria-live", "polite");
resultPanel.hidden = true;

if (recordButton) recordButton.insertAdjacentElement("beforebegin", modeSelector);
if (meter) meter.insertAdjacentElement("afterend", resultPanel);

function renderModeSelector() {
  modeSelector.setAttribute("aria-label", t("experiment.advanced.mode.aria"));
  modeSelector.innerHTML = `
    <span class="analysis-mode-experiment__label">${escapeHtml(t("experiment.advanced.mode.label"))}</span>
    <span class="analysis-mode-experiment__buttons">
      <button type="button" data-experiment-mode="basic" aria-pressed="${currentMode === "basic"}">
        ${escapeHtml(t("experiment.advanced.mode.basic"))}
      </button>
      <button type="button" data-experiment-mode="advanced" aria-pressed="${currentMode === "advanced"}">
        ${escapeHtml(t("experiment.advanced.mode.advanced"))}
      </button>
    </span>
    <span class="analysis-mode-experiment__note">${escapeHtml(t("experiment.advanced.mode.note"))}</span>
  `;
  modeSelector.querySelectorAll("[data-experiment-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      const mode = button.getAttribute("data-experiment-mode");
      if (!VALID_MODES.has(mode) || mode === currentMode) return;
      currentMode = mode;
      saveMode(mode);
      document.documentElement.setAttribute("data-analysis-mode", mode);
      renderModeSelector();
      if (mode === "advanced" && lastAnalysis) renderAnalysis(lastAnalysis);
      else resultPanel.hidden = true;
      track("advanced_mode_selected", { mode });
    });
  });
}

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

function statusElement() {
  return resultPanel.querySelector("[data-share-status]");
}

function setShareStatus(messageKey) {
  const status = statusElement();
  if (status) status.textContent = t(messageKey);
}

function shareCaption(result) {
  return t("experiment.advanced.share.caption", {
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

async function createCard(result) {
  return createShareCardBlob({
    labels: shareLabels(result),
    result,
    shareUrl: buildShareUrl(),
    theme: document.documentElement.getAttribute("data-faction") === "light" ? "light" : "dark",
  });
}

async function handleSystemShare(result, { downloadWhenUnsupported = true } = {}) {
  try {
    const blob = await createCard(result);
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
    const blob = await createCard(result);
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

export function renderAnalysis(analysis) {
  lastAnalysis = analysis;
  lastResult = evaluateAdvancedExperience(analysis);
  if (currentMode !== "advanced") {
    resultPanel.hidden = true;
    return lastResult;
  }
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
      <div class="advanced-share__fallbacks">
        <button type="button" data-share-download ${disabled}>${escapeHtml(t("experiment.advanced.share.download"))}</button>
        <button type="button" data-share-copy ${disabled}>${escapeHtml(t("experiment.advanced.share.copy"))}</button>
      </div>
      <p class="advanced-share__status" data-share-status role="status"></p>
    </div>
  `;
  resultPanel.hidden = false;
  bindShareActions(result);
  track("advanced_result_view", {
    age_band: result.voiceAge.bandKey,
    archetype: result.archetypeKey,
    confidence: result.confidence.key,
    score_band: Math.floor(result.score / 10) * 10,
  });
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
  renderModeSelector();
  if (lastAnalysis && currentMode === "advanced") renderAnalysis(lastAnalysis);
});

document.documentElement.setAttribute("data-analysis-mode", currentMode);
renderModeSelector();

window.vpaAdvancedExperience = {
  createCard: () => lastResult ? createCard(lastResult) : null,
  evaluate: evaluateAdvancedExperience,
  getLastResult: () => lastResult,
  getLocale: getCurrentLocale,
  renderAnalysis,
  setMode(mode) {
    if (!VALID_MODES.has(mode)) return false;
    currentMode = mode;
    saveMode(mode);
    document.documentElement.setAttribute("data-analysis-mode", mode);
    renderModeSelector();
    if (mode === "advanced" && lastAnalysis) renderAnalysis(lastAnalysis);
    else resultPanel.hidden = true;
    return true;
  },
};
