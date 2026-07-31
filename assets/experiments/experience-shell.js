import { recorderCtl } from "../app.js";
import {
  getCurrentLocale,
  onLocaleChange,
  setLocale,
  t,
} from "../js/i18n.js";
import {
  createResultCard,
  formatAdvancedResult,
  onAdvancedResult,
} from "./advanced-experience.js";
import { audioFileFromUrl, shareResultFiles } from "./audio-share.js";
import {
  compareChallenge,
  createChallengeUrl,
  readChallenge,
} from "./challenge-link.js";
import { downloadBlob } from "./share-card.js";

const EXPERIENCE_KEY = "vpa::experiment.experience";
const LATEST_RESULT_KEY = "vpa::quick.latestResult";
const VALID_EXPERIENCES = new Set(["quick", "professional"]);
const QUICK_ANALYSIS_TIMEOUT_MS = 90000;

let currentExperience = readChallenge() ? "quick" : readExperience();
let incomingChallenge = readChallenge();
let latestAnalysis = null;
let latestResult = null;
let quickStage = "idle";
let quickErrorKey = "";
let quickStartedAt = 0;
let timerId = null;
let analysisTimeoutId = null;
let shareOpen = false;
let includeAudio = false;
let shareStatusKey = "";
let currentChallenge = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function readExperience() {
  try {
    const value = localStorage.getItem(EXPERIENCE_KEY);
    if (VALID_EXPERIENCES.has(value)) return value;
  } catch {
    // Storage is optional.
  }
  return "quick";
}

function saveExperience(value) {
  try {
    localStorage.setItem(EXPERIENCE_KEY, value);
  } catch {
    // Storage is optional.
  }
}

function readLatestResult() {
  try {
    const value = JSON.parse(localStorage.getItem(LATEST_RESULT_KEY) || "null");
    if (!value || typeof value !== "object") return null;
    if (!Number.isFinite(Number(value.score))) return null;
    return value;
  } catch {
    return null;
  }
}

function saveLatestResult(result) {
  if (!result?.ready) return;
  try {
    localStorage.setItem(LATEST_RESULT_KEY, JSON.stringify({
      ageMax: result.voiceAge.max,
      ageMin: result.voiceAge.min,
      archetype: result.archetypeKey,
      at: Date.now(),
      score: result.score,
      version: result.version,
    }));
  } catch {
    // Storage is optional.
  }
}

function track(eventName, params = {}) {
  if (typeof window.gtag !== "function") return;
  try {
    window.gtag("event", eventName, {
      event_category: "experience_v1",
      ...params,
    });
  } catch (error) {
    console.warn("[experience] analytics failed", error);
  }
}

const professionalHero = document.querySelector("body > .hero");
const experienceNav = document.createElement("nav");
experienceNav.id = "experienceNav";
experienceNav.className = "experience-nav";
experienceNav.setAttribute("aria-label", t("experiment.experience.aria"));

const quickExperience = document.createElement("main");
quickExperience.id = "quickExperience";
quickExperience.className = "quick-experience";

if (professionalHero) {
  professionalHero.insertAdjacentElement("beforebegin", experienceNav);
  experienceNav.insertAdjacentElement("afterend", quickExperience);
}

function localeOptions() {
  return [
    ["zh-Hant", t("topbar.localeNames.zhHant")],
    ["zh-Hans", t("topbar.localeNames.zhHans")],
    ["en", t("topbar.localeNames.en")],
    ["ja", t("topbar.localeNames.ja")],
  ].map(([value, label]) => `
    <option value="${value}" ${getCurrentLocale() === value ? "selected" : ""}>
      ${escapeHtml(label)}
    </option>
  `).join("");
}

function renderExperienceNav() {
  experienceNav.setAttribute("aria-label", t("experiment.experience.aria"));
  experienceNav.innerHTML = `
    <a class="experience-nav__brand" href="#quickExperience" aria-label="VPA">VPA</a>
    <div class="experience-nav__modes" role="group" aria-label="${escapeHtml(t("experiment.experience.aria"))}">
      <button type="button" data-experience-target="quick" aria-pressed="${currentExperience === "quick"}">
        ${escapeHtml(t("experiment.experience.quick"))}
      </button>
      <button type="button" data-experience-target="professional" aria-pressed="${currentExperience === "professional"}">
        ${escapeHtml(t("experiment.experience.professional"))}
      </button>
    </div>
    <label class="experience-nav__locale">
      <span class="sr-only">${escapeHtml(t("experiment.quick.localeLabel"))}</span>
      <select data-quick-locale aria-label="${escapeHtml(t("experiment.quick.localeLabel"))}">
        ${localeOptions()}
      </select>
    </label>
  `;
  bindCommonControls(experienceNav);
}

function challengeBanner() {
  if (!incomingChallenge) return "";
  return `
    <aside class="quick-challenge-invite">
      <span>${escapeHtml(t("experiment.quick.challenge.received"))}</span>
      <strong>${escapeHtml(t("experiment.quick.challenge.inviteScore", {
        score: incomingChallenge.score,
      }))}</strong>
    </aside>
  `;
}

function latestResultMarkup() {
  const saved = readLatestResult();
  if (!saved) {
    return `<span>${escapeHtml(t("experiment.quick.history.none"))}</span>`;
  }
  return `
    <span>${escapeHtml(t("experiment.quick.history.label"))}</span>
    <strong>${escapeHtml(t("experiment.quick.history.score", { score: saved.score }))}</strong>
  `;
}

function landingMarkup() {
  return `
    <section class="quick-landing" data-quick-stage="idle">
      ${challengeBanner()}
      <div class="quick-landing__hero">
        <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.eyebrow"))}</span>
        <h1>${escapeHtml(t("experiment.quick.title"))}</h1>
        <p>${escapeHtml(t("experiment.quick.subtitle"))}</p>
      </div>
      <article class="quick-prompt">
        <span>${escapeHtml(t("experiment.quick.promptLabel"))}</span>
        <blockquote>${escapeHtml(t("experiment.quick.prompt"))}</blockquote>
        <small>${escapeHtml(t("experiment.quick.promptHint"))}</small>
      </article>
      ${quickErrorKey ? `<p class="quick-error" role="alert">${escapeHtml(t(quickErrorKey))}</p>` : ""}
      <button type="button" class="quick-primary" data-quick-record>
        ${escapeHtml(t("experiment.quick.start"))}
      </button>
      <div class="quick-trust">
        <span>${escapeHtml(t("experiment.quick.trust.local"))}</span>
        <span>${escapeHtml(t("experiment.quick.trust.strict"))}</span>
        <span>${escapeHtml(t("experiment.quick.trust.private"))}</span>
      </div>
      <div class="quick-history">${latestResultMarkup()}</div>
    </section>
  `;
}

function progressMarkup() {
  const isRecording = quickStage === "recording";
  const statusKey = quickStage === "requesting"
    ? "experiment.quick.requesting"
    : isRecording
      ? "experiment.quick.recording"
      : "experiment.quick.analyzing";
  return `
    <section class="quick-progress" data-quick-stage="${quickStage}">
      <div class="quick-progress__orb" aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
      <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.eyebrow"))}</span>
      <h1>${escapeHtml(t(statusKey))}</h1>
      <strong class="quick-progress__timer" data-quick-timer>${isRecording ? "00:00" : "···"}</strong>
      <p>${escapeHtml(t(isRecording ? "experiment.quick.recordingHint" : "experiment.quick.analyzingHint"))}</p>
      ${isRecording ? `
        <button type="button" class="quick-primary quick-primary--stop" data-quick-record>
          ${escapeHtml(t("experiment.quick.stop"))}
        </button>
      ` : ""}
    </section>
  `;
}

function comparisonMarkup(result) {
  const comparison = compareChallenge(incomingChallenge, result);
  if (!comparison) return "";
  return `
    <section class="quick-comparison">
      <span>${escapeHtml(t("experiment.quick.challenge.comparison"))}</span>
      <div>
        <strong>${comparison.opponentScore}</strong>
        <i>VS</i>
        <strong>${comparison.score}</strong>
      </div>
      <p>${escapeHtml(t(`experiment.quick.challenge.${comparison.outcome}`, {
        difference: Math.abs(comparison.difference),
      }))}</p>
    </section>
  `;
}

function shareComposerMarkup(result) {
  if (!shareOpen) return "";
  const audioUrl = recorderCtl.getLastRecordingUrl();
  const audioDisabled = !audioUrl;
  return `
    <section class="quick-share-composer">
      <h2>${escapeHtml(t("experiment.quick.share.title"))}</h2>
      <label class="quick-share-option">
        <input type="checkbox" data-quick-audio ${includeAudio ? "checked" : ""} ${audioDisabled ? "disabled" : ""} />
        <span>
          <strong>${escapeHtml(t("experiment.quick.share.includeAudio"))}</strong>
          <small>${escapeHtml(t("experiment.quick.share.audioDefault"))}</small>
        </span>
      </label>
      ${includeAudio && audioUrl ? `
        <div class="quick-audio-preview">
          <span>${escapeHtml(t("experiment.quick.share.preview"))}</span>
          <audio controls preload="metadata" src="${escapeHtml(audioUrl)}"></audio>
          <p>${escapeHtml(t("experiment.quick.share.audioWarning"))}</p>
        </div>
      ` : ""}
      <button type="button" class="quick-primary" data-quick-system-share ${result.ready ? "" : "disabled"}>
        ${escapeHtml(t("experiment.quick.share.system"))}
      </button>
      <button type="button" class="quick-secondary" data-quick-copy-challenge ${result.ready ? "" : "disabled"}>
        ${escapeHtml(t("experiment.quick.share.copyChallenge"))}
      </button>
      <p class="quick-share-status" role="status">
        ${shareStatusKey ? escapeHtml(t(shareStatusKey)) : ""}
      </p>
    </section>
  `;
}

function resultMarkup() {
  const result = latestResult;
  if (!result) return landingMarkup();
  const formatted = formatAdvancedResult(result);
  const ready = result.ready;
  return `
    <section class="quick-result" data-quick-stage="result">
      ${comparisonMarkup(result)}
      <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.reveal.eyebrow"))}</span>
      <div class="quick-result__score">
        <strong>${ready ? result.score : "—"}</strong>
        <span>${escapeHtml(t("experiment.quick.reveal.score"))}</span>
      </div>
      ${ready ? `
        <div class="quick-result__identity">
          <article>
            <span>${escapeHtml(t("experiment.quick.reveal.age"))}</span>
            <strong>${escapeHtml(formatted.age)}</strong>
          </article>
          <article>
            <span>${escapeHtml(t("experiment.quick.reveal.archetype"))}</span>
            <strong>${escapeHtml(formatted.archetype)}</strong>
          </article>
        </div>
        <article class="quick-result__insight">
          <span>${escapeHtml(t("experiment.quick.reveal.insight"))}</span>
          <strong>${escapeHtml(formatted.insight)}</strong>
        </article>
      ` : `
        <p class="quick-error">${escapeHtml(t("experiment.advanced.insufficient"))}</p>
      `}
      <p class="quick-result__disclaimer">${escapeHtml(t("experiment.advanced.disclaimer"))}</p>
      <div class="quick-result__actions">
        <button type="button" class="quick-primary" data-quick-share ${ready ? "" : "disabled"}>
          ${escapeHtml(t("experiment.quick.share.open"))}
        </button>
        <button type="button" class="quick-secondary" data-quick-retry>
          ${escapeHtml(t("experiment.quick.retry"))}
        </button>
        <button type="button" class="quick-link" data-experience-target="professional">
          ${escapeHtml(t("experiment.quick.viewProfessional"))}
        </button>
      </div>
      ${shareComposerMarkup(result)}
    </section>
  `;
}

function renderQuickExperience() {
  quickExperience.innerHTML = `
    <div class="quick-experience__backdrop" aria-hidden="true"></div>
    <div class="quick-experience__container">
      ${quickStage === "idle"
        ? landingMarkup()
        : quickStage === "result"
          ? resultMarkup()
          : progressMarkup()}
      <footer class="quick-footer">${escapeHtml(t("experiment.quick.footer"))}</footer>
    </div>
  `;
  bindCommonControls(quickExperience);
  bindQuickControls();
  updateQuickTimer();
}

function renderAll() {
  document.documentElement.setAttribute("data-experience", currentExperience);
  renderExperienceNav();
  renderQuickExperience();
}

function bindCommonControls(root) {
  root.querySelectorAll("[data-experience-target]").forEach((button) => {
    button.addEventListener("click", () => {
      setExperience(button.getAttribute("data-experience-target"));
    });
  });
  root.querySelector("[data-quick-locale]")?.addEventListener("change", async (event) => {
    await setLocale(event.target.value);
    track("locale_selected", { locale: event.target.value });
  });
}

function bindQuickControls() {
  quickExperience.querySelectorAll("[data-quick-record]").forEach((button) => {
    button.addEventListener("click", () => toggleQuickRecording());
  });
  quickExperience.querySelector("[data-quick-retry]")?.addEventListener("click", () => {
    resetQuickTest();
  });
  quickExperience.querySelector("[data-quick-share]")?.addEventListener("click", () => {
    shareOpen = !shareOpen;
    includeAudio = false;
    shareStatusKey = "";
    renderQuickExperience();
    track("share_opened", { mode: "quick" });
  });
  quickExperience.querySelector("[data-quick-audio]")?.addEventListener("change", (event) => {
    includeAudio = Boolean(event.target.checked);
    shareStatusKey = "";
    renderQuickExperience();
    track("audio_share_opt_in", { enabled: includeAudio });
  });
  quickExperience.querySelector("[data-quick-system-share]")?.addEventListener("click", () => {
    shareQuickResult();
  });
  quickExperience.querySelector("[data-quick-copy-challenge]")?.addEventListener("click", () => {
    copyChallengeLink();
  });
}

function setExperience(value, { persist = true } = {}) {
  if (!VALID_EXPERIENCES.has(value) || value === currentExperience) return;
  currentExperience = value;
  if (persist) saveExperience(value);
  document.documentElement.setAttribute("data-experience", value);
  renderExperienceNav();
  renderQuickExperience();
  track("experience_mode_selected", { mode: value });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function clearTimers() {
  if (timerId !== null) {
    clearInterval(timerId);
    timerId = null;
  }
  if (analysisTimeoutId !== null) {
    clearTimeout(analysisTimeoutId);
    analysisTimeoutId = null;
  }
}

function resetQuickTest() {
  clearTimers();
  recorderCtl.stopPlayback();
  latestAnalysis = null;
  latestResult = null;
  quickStage = "idle";
  quickErrorKey = "";
  quickStartedAt = 0;
  shareOpen = false;
  includeAudio = false;
  shareStatusKey = "";
  currentChallenge = null;
  renderQuickExperience();
}

function updateQuickTimer() {
  const timer = quickExperience.querySelector("[data-quick-timer]");
  if (!timer || quickStage !== "recording" || !quickStartedAt) return;
  const seconds = Math.max(0, Math.floor((Date.now() - quickStartedAt) / 1000));
  const minutes = Math.floor(seconds / 60);
  timer.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

async function waitForRecordingState(expected, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (recorderCtl.isRecording === expected) return true;
    await new Promise((resolve) => setTimeout(resolve, 60));
  }
  return recorderCtl.isRecording === expected;
}

async function toggleQuickRecording() {
  const recordButton = document.getElementById("recordBtn");
  if (!recordButton) return;
  if (recorderCtl.busy && !recorderCtl.isRecording) return;
  quickErrorKey = "";

  if (recorderCtl.isRecording) {
    quickStage = "analyzing";
    clearTimers();
    renderQuickExperience();
    recordButton.click();
    analysisTimeoutId = setTimeout(() => {
      if (quickStage !== "analyzing") return;
      quickStage = "idle";
      quickErrorKey = "experiment.quick.errors.analysisTimeout";
      renderQuickExperience();
    }, QUICK_ANALYSIS_TIMEOUT_MS);
    return;
  }

  latestAnalysis = null;
  latestResult = null;
  shareOpen = false;
  includeAudio = false;
  quickStage = "requesting";
  renderQuickExperience();
  track("quick_test_started");
  recordButton.click();
  const started = await waitForRecordingState(true, 30000);
  if (!started) {
    quickStage = "idle";
    quickErrorKey = "experiment.quick.errors.recordingFailed";
    renderQuickExperience();
    return;
  }
  quickStartedAt = Date.now();
  quickStage = "recording";
  timerId = setInterval(updateQuickTimer, 250);
  renderQuickExperience();
}

function ensureChallenge() {
  if (!latestResult?.ready) return null;
  if (!currentChallenge) {
    currentChallenge = createChallengeUrl(latestResult);
    track("challenge_created", {
      score_band: Math.floor(latestResult.score / 10) * 10,
    });
  }
  return currentChallenge;
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const input = document.createElement("textarea");
    input.value = value;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.append(input);
    input.select();
    document.execCommand("copy");
    input.remove();
  }
}

async function copyChallengeLink() {
  const challenge = ensureChallenge();
  if (!challenge) return;
  await copyText(challenge.url);
  shareStatusKey = "experiment.quick.share.copied";
  renderQuickExperience();
}

async function shareQuickResult() {
  const challenge = ensureChallenge();
  if (!challenge || !latestResult) return;
  shareStatusKey = "";
  try {
    const formatted = formatAdvancedResult(latestResult);
    const cardUrl = new URL(challenge.url);
    cardUrl.hash = "";
    const cardBlob = await createResultCard(latestResult, { shareUrl: cardUrl.toString() });
    const audioFile = includeAudio
      ? await audioFileFromUrl(recorderCtl.getLastRecordingUrl())
      : null;
    const response = await shareResultFiles({
      audioFile,
      cardBlob,
      caption: formatted.caption,
      title: t("experiment.quick.share.shareTitle"),
      url: challenge.url,
    });
    if (response.method === "unsupported" || response.method === "unsupported-files") {
      downloadBlob(cardBlob, "vpa-result.png");
      if (audioFile) downloadBlob(audioFile, audioFile.name);
      shareStatusKey = audioFile
        ? "experiment.quick.share.downloadedWithAudio"
        : "experiment.quick.share.downloaded";
    } else {
      shareStatusKey = "experiment.quick.share.shared";
    }
    track("share_success", {
      includes_audio: Boolean(audioFile),
      method: response.method,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      shareStatusKey = "experiment.quick.share.cancelled";
    } else {
      console.error("[quick-share] failed", error);
      shareStatusKey = "experiment.quick.share.failed";
    }
  }
  renderQuickExperience();
}

onAdvancedResult(({ analysis, result }) => {
  clearTimers();
  latestAnalysis = analysis;
  latestResult = result;
  quickStage = "result";
  quickErrorKey = "";
  quickStartedAt = 0;
  shareOpen = false;
  includeAudio = false;
  shareStatusKey = "";
  currentChallenge = null;
  saveLatestResult(result);
  renderQuickExperience();
  track("quick_test_completed", {
    ready: result.ready,
    score_band: result.ready ? Math.floor(result.score / 10) * 10 : -1,
  });
  if (incomingChallenge && result.ready) {
    track("challenge_completed", {
      outcome: compareChallenge(incomingChallenge, result)?.outcome || "unknown",
    });
  }
});

onLocaleChange(() => {
  renderAll();
});

document.documentElement.setAttribute("data-experience", currentExperience);
renderAll();

window.vpaExperience = {
  getChallenge: () => incomingChallenge,
  getExperience: () => currentExperience,
  getLatestAnalysis: () => latestAnalysis,
  getLatestResult: () => latestResult,
  resetQuickTest,
  setExperience,
};
