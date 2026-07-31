import { recorderCtl } from "../app.js?v=20260801-multiappopen1";
import { registerDecodedAudioAnalyzer } from "../js/analysis-flow.js";
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
import { shareResultFiles } from "./audio-share.js";
import {
  compareChallenge,
  createChallengeUrl,
  readChallenge,
} from "./challenge-link.js";
import { createDynamicCardController } from "./dynamic-card-controller.js";
import {
  getDailyPromptId,
  getStandardPromptId,
  promptTranslationKey,
  STANDARD_PROMPT_IDS,
  STANDARD_TEST_ID,
} from "./quick-prompts.js";
import {
  buildShareTargets,
  buildShareUrl,
  downloadBlob,
} from "./share-card.js";
import { aggregateStandardResults } from "./standard-result.js";
import { analyzeVoiceQuality } from "./voice-quality-metrics.js";

const EXPERIENCE_KEY = "vpa::experiment.experience";
const LATEST_RESULT_KEY = "vpa::quick.latestResult";
const VALID_EXPERIENCES = new Set(["quick", "professional"]);
const QUICK_ANALYSIS_TIMEOUT_MS = 90000;

let currentExperience = readChallenge() ? "quick" : readExperience();
let incomingChallenge = readChallenge();
let quickTestMode = incomingChallenge?.testMode === "standard" ? "standard" : "daily";
let dailyPromptId = incomingChallenge?.testMode === "daily"
  ? incomingChallenge.promptId
  : getDailyPromptId();
let latestAnalysis = null;
let latestResult = null;
let quickStage = "idle";
let quickErrorKey = "";
let quickStartedAt = 0;
let timerId = null;
let analysisTimeoutId = null;
let shareOpen = false;
let dynamicAudioOptIn = false;
let shareStatusKey = "";
let currentChallenge = null;
let standardStep = 0;
let standardRuns = [];

registerDecodedAudioAnalyzer("voice-age-v2", ({ samples, sampleRate }) => (
  analyzeVoiceQuality(samples, sampleRate, {
    sampleType: currentExperience === "quick" ? "connectedSpeech" : undefined,
  })
));

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
      ageMax: result.voiceAge.ready ? result.voiceAge.max : null,
      ageMin: result.voiceAge.ready ? result.voiceAge.min : null,
      ageVersion: result.voiceAge.version,
      archetype: result.archetypeKey,
      at: Date.now(),
      score: result.score,
      testMode: result.quickTest?.mode || "daily",
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

const dynamicCard = createDynamicCardController({
  createResultCard,
  downloadBlob,
  formatResult: formatAdvancedResult,
  getAudioUrl: () => recorderCtl.getLastRecordingUrl(),
  getShareUrl: () => buildShareUrl(),
  render: () => renderQuickExperience(),
  track,
});

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

function activePromptId() {
  return quickTestMode === "standard"
    ? getStandardPromptId(standardStep)
    : dailyPromptId;
}

function activePromptText() {
  const key = promptTranslationKey(activePromptId());
  return key ? t(key) : t("experiment.quick.prompt");
}

function promptMarkup({ showProgress = false } = {}) {
  const label = showProgress
    ? t("experiment.quick.standard.progress", {
      current: standardStep + 1,
      total: STANDARD_PROMPT_IDS.length,
    })
    : t("experiment.quick.promptLabel");
  return `
    <article class="quick-prompt">
      <span>${escapeHtml(label)}</span>
      <blockquote>${escapeHtml(activePromptText())}</blockquote>
      <small>${escapeHtml(t("experiment.quick.promptHint"))}</small>
    </article>
  `;
}

function dailyLandingMarkup() {
  return `
    <section class="quick-landing" data-quick-stage="idle">
      ${challengeBanner()}
      <div class="quick-landing__hero">
        <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.eyebrow"))}</span>
        <h1>${escapeHtml(t("experiment.quick.title"))}</h1>
        <p>${escapeHtml(t("experiment.quick.subtitle"))}</p>
      </div>
      ${promptMarkup()}
      ${quickErrorKey ? `<p class="quick-error" role="alert">${escapeHtml(t(quickErrorKey))}</p>` : ""}
      <button type="button" class="quick-primary" data-quick-record>
        ${escapeHtml(t("experiment.quick.start"))}
      </button>
      <button type="button" class="quick-link quick-standard-cta" data-quick-standard>
        ${escapeHtml(t("experiment.quick.standard.cta"))}
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

function standardLandingMarkup() {
  return `
    <section class="quick-landing quick-standard" data-quick-stage="idle">
      ${challengeBanner()}
      <div class="quick-landing__hero">
        <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.standard.progress", {
          current: standardStep + 1,
          total: STANDARD_PROMPT_IDS.length,
        }))}</span>
        <h1>${escapeHtml(t("experiment.quick.standard.title"))}</h1>
        <p>${escapeHtml(t("experiment.quick.standard.subtitle"))}</p>
      </div>
      ${promptMarkup({ showProgress: true })}
      ${quickErrorKey ? `<p class="quick-error" role="alert">${escapeHtml(t(quickErrorKey))}</p>` : ""}
      <button type="button" class="quick-primary" data-quick-record>
        ${escapeHtml(t("experiment.quick.start"))}
      </button>
      <button type="button" class="quick-link" data-quick-daily>
        ${escapeHtml(t("experiment.quick.standard.backDaily"))}
      </button>
    </section>
  `;
}

function landingMarkup() {
  return quickTestMode === "standard"
    ? standardLandingMarkup()
    : dailyLandingMarkup();
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
      ${isRecording || quickStage === "requesting"
        ? `<div class="quick-progress__prompt">${escapeHtml(activePromptText())}</div>`
        : ""}
      ${isRecording ? `
        <button type="button" class="quick-primary quick-primary--stop" data-quick-record>
          ${escapeHtml(t("experiment.quick.stop"))}
        </button>
      ` : ""}
    </section>
  `;
}

function standardCheckpointMarkup() {
  const result = standardRuns[standardRuns.length - 1];
  return `
    <section class="quick-standard-checkpoint" data-quick-stage="standard-next">
      <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.standard.progress", {
        current: standardRuns.length,
        total: STANDARD_PROMPT_IDS.length,
      }))}</span>
      <h1>${escapeHtml(t("experiment.quick.standard.stepComplete", {
        current: standardRuns.length,
      }))}</h1>
      <strong>${escapeHtml(t("experiment.quick.standard.stepScore", {
        score: result.score,
      }))}</strong>
      <div class="quick-standard-scores" aria-label="${escapeHtml(t("experiment.quick.standard.scoresLabel"))}">
        ${standardRuns.map((run, index) => `
          <span>${index + 1}<strong>${run.score}%</strong></span>
        `).join("")}
      </div>
      <button type="button" class="quick-primary" data-quick-standard-next>
        ${escapeHtml(t("experiment.quick.standard.next"))}
      </button>
      <button type="button" class="quick-link" data-quick-daily>
        ${escapeHtml(t("experiment.quick.standard.backDaily"))}
      </button>
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
        <strong>${comparison.opponentScore}%</strong>
        <i>VS</i>
        <strong>${comparison.score}%</strong>
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
  const audioDefaultKey = result.quickTest?.mode === "standard"
    ? "experiment.quick.standard.audioDefault"
    : "experiment.quick.share.audioDefault";
  return `
    <section class="quick-share-composer">
      <h2>${escapeHtml(t("experiment.quick.share.title"))}</h2>
      <label class="quick-share-option">
        <input type="checkbox" data-quick-audio ${dynamicAudioOptIn ? "checked" : ""} ${audioDisabled ? "disabled" : ""} />
        <span>
          <strong>${escapeHtml(t("experiment.quick.share.includeAudio"))}</strong>
          <small>${escapeHtml(t(audioDefaultKey))}</small>
        </span>
      </label>
      ${dynamicAudioOptIn && audioUrl ? `
        <p class="quick-share-audio-warning">${escapeHtml(t("experiment.quick.share.audioWarning"))}</p>
        ${dynamicCard.markup(result)}
      ` : `
        <button type="button" class="quick-primary" data-quick-system-share ${result.ready ? "" : "disabled"}>
          ${escapeHtml(t("experiment.quick.share.system"))}
        </button>
      `}
      <button type="button" class="quick-secondary" data-quick-copy-challenge ${result.ready ? "" : "disabled"}>
        ${escapeHtml(t("experiment.quick.share.copyChallenge"))}
      </button>
      <p class="quick-share-status" role="status">
        ${shareStatusKey ? escapeHtml(t(shareStatusKey)) : ""}
      </p>
    </section>
  `;
}

function shareShortcutMarkup() {
  const platforms = ["x", "threads", "line", "facebook", "tiktok"];
  return `
    <section class="quick-share-shortcuts" aria-label="${escapeHtml(t("experiment.quick.share.directTitle"))}">
      <div class="quick-share-shortcuts__copy">
        <strong>${escapeHtml(t("experiment.quick.share.directTitle"))}</strong>
        <small>${escapeHtml(t("experiment.quick.share.directHint"))}</small>
      </div>
      <div class="quick-share-shortcuts__buttons">
        ${platforms.map((platform) => `
          <button type="button" data-quick-platform="${platform}"
            aria-label="${escapeHtml(t("experiment.quick.share.directAria", { platform }))}">
            ${platform === "x" ? "X" : platform === "line" ? "LINE" : platform[0].toUpperCase() + platform.slice(1)}
          </button>
        `).join("")}
      </div>
    </section>
  `;
}

function standardSummaryMarkup(result) {
  if (!result.standard) return "";
  return `
    <section class="quick-standard-summary">
      <div>
        <span>${escapeHtml(t("experiment.quick.standard.stability"))}</span>
        <strong>${escapeHtml(t(`experiment.quick.standard.stabilityValues.${result.standard.stabilityKey}`))}</strong>
        <small>${escapeHtml(t("experiment.quick.standard.spread", {
          spread: result.standard.spread,
        }))}</small>
      </div>
      <div>
        <span>${escapeHtml(t("experiment.quick.standard.scoresLabel"))}</span>
        <strong>${result.standard.scores.map((score) => `${score}%`).join(" / ")}</strong>
      </div>
    </section>
  `;
}

function resultMarkup() {
  const result = latestResult;
  if (!result) return landingMarkup();
  const formatted = formatAdvancedResult(result);
  const ready = result.ready;
  const isStandard = result.quickTest?.mode === "standard";
  const scoreKey = isStandard
    ? "experiment.quick.reveal.standardScore"
    : "experiment.quick.reveal.singleScore";
  const feminine = ready ? Math.max(0, Math.min(100, Math.round(result.score))) : null;
  const masculine = ready ? 100 - feminine : null;
  return `
    <section class="quick-result" data-quick-stage="result">
      ${comparisonMarkup(result)}
      <span class="quick-eyebrow">${escapeHtml(t("experiment.quick.reveal.eyebrow"))}</span>
      <div class="quick-result__score">
        <span class="quick-result__tendency-label">${escapeHtml(t("experiment.quick.reveal.tendency"))}</span>
        ${ready ? `
          <div class="quick-result__tendency" aria-label="${escapeHtml(t("experiment.quick.reveal.tendencyAria", {
            feminine,
            masculine,
          }))}">
            <article class="quick-result__feminine">
              <span>${escapeHtml(t("experiment.quick.reveal.feminine"))}</span>
              <strong>${feminine}%</strong>
            </article>
            <article class="quick-result__masculine">
              <span>${escapeHtml(t("experiment.quick.reveal.masculine"))}</span>
              <strong>${masculine}%</strong>
            </article>
          </div>
          <div class="quick-result__tendency-meter" aria-hidden="true">
            <span style="width:${feminine}%"></span>
            <i style="width:${masculine}%"></i>
          </div>
        ` : `<strong class="quick-result__unavailable">—</strong>`}
        <small>${escapeHtml(t(scoreKey))}</small>
      </div>
      ${ready ? standardSummaryMarkup(result) : ""}
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
      ${ready ? shareShortcutMarkup() : ""}
      <div class="quick-result__actions">
        <button type="button" class="quick-primary" data-quick-share ${ready ? "" : "disabled"}>
          ${escapeHtml(t("experiment.quick.share.open"))}
        </button>
        <button type="button" class="quick-secondary" data-quick-retry>
          ${escapeHtml(t("experiment.quick.retry"))}
        </button>
        ${isStandard ? `
          <button type="button" class="quick-link" data-quick-daily>
            ${escapeHtml(t("experiment.quick.standard.backDaily"))}
          </button>
        ` : `
          <button type="button" class="quick-link quick-standard-cta" data-quick-standard>
            ${escapeHtml(t("experiment.quick.standard.cta"))}
          </button>
        `}
        <button type="button" class="quick-link" data-experience-target="professional">
          ${escapeHtml(t(isStandard
            ? "experiment.quick.standard.viewProfessional"
            : "experiment.quick.viewProfessional"))}
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
          : quickStage === "standard-next"
            ? standardCheckpointMarkup()
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
  quickExperience.querySelectorAll("[data-quick-standard]").forEach((button) => {
    button.addEventListener("click", () => {
      startStandardTest();
    });
  });
  quickExperience.querySelectorAll("[data-quick-daily]").forEach((button) => {
    button.addEventListener("click", () => {
      returnToDailyTest();
    });
  });
  quickExperience.querySelector("[data-quick-standard-next]")?.addEventListener("click", () => {
    continueStandardTest();
  });
  quickExperience.querySelectorAll("[data-quick-platform]").forEach((button) => {
    button.addEventListener("click", () => {
      openQuickPlatform(button.getAttribute("data-quick-platform"));
    });
  });
  quickExperience.querySelector("[data-quick-share]")?.addEventListener("click", () => {
    shareOpen = !shareOpen;
    dynamicCard.reset();
    dynamicAudioOptIn = false;
    shareStatusKey = "";
    renderQuickExperience();
    track("share_opened", { mode: "quick" });
  });
  quickExperience.querySelector("[data-quick-audio]")?.addEventListener("change", (event) => {
    dynamicAudioOptIn = Boolean(event.target.checked);
    dynamicCard.reset();
    shareStatusKey = "";
    renderQuickExperience();
    track("audio_share_opt_in", {
      enabled: dynamicAudioOptIn,
      source: "dynamic_card",
    });
    if (dynamicAudioOptIn) dynamicCard.open();
  });
  quickExperience.querySelector("[data-quick-system-share]")?.addEventListener("click", () => {
    shareQuickResult();
  });
  quickExperience.querySelector("[data-quick-copy-challenge]")?.addEventListener("click", () => {
    copyChallengeLink();
  });
  dynamicCard.bind(quickExperience, latestResult);
}

function setExperience(value, { persist = true } = {}) {
  if (!VALID_EXPERIENCES.has(value) || value === currentExperience) return;
  currentExperience = value;
  if (value === "professional") dynamicCard.reset();
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

function clearIncomingChallenge() {
  incomingChallenge = null;
  if (!location.hash.startsWith("#vpa-challenge=")) return;
  try {
    history.replaceState(null, "", `${location.pathname}${location.search}`);
  } catch {
    // History is optional.
  }
}

function resetQuickTest() {
  clearTimers();
  dynamicCard.reset();
  recorderCtl.stopPlayback();
  latestAnalysis = null;
  latestResult = null;
  quickStage = "idle";
  quickErrorKey = "";
  quickStartedAt = 0;
  shareOpen = false;
  dynamicAudioOptIn = false;
  shareStatusKey = "";
  currentChallenge = null;
  if (quickTestMode === "standard") {
    standardStep = 0;
    standardRuns = [];
  }
  renderQuickExperience();
}

function startStandardTest() {
  if (incomingChallenge?.testMode !== "standard") clearIncomingChallenge();
  quickTestMode = "standard";
  standardStep = 0;
  standardRuns = [];
  resetQuickTest();
  track("standard_test_selected");
}

function returnToDailyTest() {
  if (incomingChallenge?.testMode !== "daily") clearIncomingChallenge();
  quickTestMode = "daily";
  dailyPromptId = incomingChallenge?.promptId || getDailyPromptId();
  standardStep = 0;
  standardRuns = [];
  resetQuickTest();
  track("daily_test_selected", { prompt_id: dailyPromptId });
}

function continueStandardTest() {
  if (quickTestMode !== "standard") return;
  standardStep = standardRuns.length;
  if (!getStandardPromptId(standardStep)) return;
  latestAnalysis = null;
  latestResult = null;
  quickStage = "idle";
  quickErrorKey = "";
  shareOpen = false;
  dynamicAudioOptIn = false;
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
  dynamicAudioOptIn = false;
  quickStage = "requesting";
  renderQuickExperience();
  track("quick_test_started", {
    mode: quickTestMode,
    prompt_id: activePromptId(),
    standard_step: quickTestMode === "standard" ? standardStep + 1 : undefined,
  });
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

async function openQuickPlatform(platform) {
  if (!latestResult?.ready) return;
  track("share_platform_selected", {
    mode: "quick",
    platform,
    score_band: Math.floor(latestResult.score / 10) * 10,
  });
  if (platform === "tiktok") {
    await shareQuickResult({ platform });
    return;
  }
  const shareUrl = buildShareUrl();
  const targets = buildShareTargets({
    caption: formatAdvancedResult(latestResult).caption,
    url: shareUrl,
  });
  const target = targets[platform];
  if (!target) return;
  window.open(target, "_blank", "noopener,noreferrer");
}

async function shareQuickResult({ platform = "system" } = {}) {
  if (!latestResult?.ready) return;
  shareStatusKey = "";
  try {
    const formatted = formatAdvancedResult(latestResult);
    const shareUrl = buildShareUrl();
    const cardBlob = await createResultCard(latestResult, { shareUrl });
    const response = await shareResultFiles({
      cardBlob,
      caption: formatted.caption,
      title: t("experiment.quick.share.shareTitle"),
      url: shareUrl,
    });
    if (response.method === "unsupported" || response.method === "unsupported-files") {
      downloadBlob(cardBlob, "vpa-result.png");
      shareStatusKey = "experiment.quick.share.downloaded";
    } else {
      shareStatusKey = "experiment.quick.share.shared";
    }
    track("share_success", {
      includes_audio: false,
      media: "png",
      method: response.method,
      platform,
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
function finalizeQuickResult(result) {
  dynamicCard.reset();
  latestResult = result;
  quickStage = "result";
  saveLatestResult(result);
  renderQuickExperience();
  track("quick_test_completed", {
    mode: result.quickTest?.mode || "daily",
    ready: result.ready,
    score_band: result.ready ? Math.floor(result.score / 10) * 10 : -1,
  });
  if (incomingChallenge && result.ready) {
    track("challenge_completed", {
      outcome: compareChallenge(incomingChallenge, result)?.outcome || "unknown",
    });
  }
}

onAdvancedResult(({ analysis, result }) => {
  if (quickTestMode === "standard" && quickStage !== "analyzing") return;
  clearTimers();
  latestAnalysis = analysis;
  quickErrorKey = "";
  quickStartedAt = 0;
  shareOpen = false;
  dynamicAudioOptIn = false;
  shareStatusKey = "";
  currentChallenge = null;

  if (quickTestMode === "standard") {
    if (!result.ready) {
      finalizeQuickResult({
        ...result,
        quickTest: {
          mode: "standard",
          promptId: STANDARD_TEST_ID,
        },
      });
      return;
    }
    standardRuns.push(result);
    track("standard_test_step_completed", {
      score_band: Math.floor(result.score / 10) * 10,
      standard_step: standardRuns.length,
    });
    if (standardRuns.length < STANDARD_PROMPT_IDS.length) {
      latestResult = null;
      quickStage = "standard-next";
      renderQuickExperience();
      return;
    }
    const aggregate = aggregateStandardResults(standardRuns);
    finalizeQuickResult({
      ...aggregate,
      quickTest: {
        mode: "standard",
        promptId: STANDARD_TEST_ID,
      },
    });
    return;
  }

  finalizeQuickResult({
    ...result,
    quickTest: {
      mode: "daily",
      promptId: dailyPromptId,
    },
  });
});

onLocaleChange(() => {
  dynamicCard.reset();
  renderAll();
});

document.documentElement.setAttribute("data-experience", currentExperience);
renderAll();

window.vpaExperience = {
  getChallenge: () => incomingChallenge,
  getExperience: () => currentExperience,
  getLatestAnalysis: () => latestAnalysis,
  getLatestResult: () => latestResult,
  getQuickTestMode: () => quickTestMode,
  getStandardRuns: () => [...standardRuns],
  resetQuickTest,
  setExperience,
  startStandardTest,
};
