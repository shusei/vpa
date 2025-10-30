// ===== Transformers pipeline =====
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

import { initI18n, t, getLocaleValue, onLocaleChange } from "./js/i18n.js";

import {
  recordBtn,
  dropZone,
  fileInput,
  uploadFab,
  statusEl,
  meter,
  femaleVal,
  maleVal,
  pitchWrap,
  pitchCanvas,
  pitchNowEl,
  bandNowEl,
  volNowEl,
  formantWrap,
  f1NowEl,
  f2NowEl,
  f3NowEl,
  breathNowEl,
  resonanceNowEl,
  tiltNowEl,
  resBarChest,
  resBarMask,
  resBarHead,
  resValChest,
  resValMask,
  resValHead,
  exportBtn,
} from "./js/dom.js";
import { dismissOnboardTip } from "./js/theme.js";
import {
  MODEL_ID,
  TARGET_SR,
  MAX_WHOLE_SEC,
  WARN_LONG_SEC,
  STREAM_WIN_CAND,
  STREAM_HOP_S,
  EPS,
  VAD_MIN_APPLY_SEC,
  VAD_FRAME_MS,
  VAD_HOP_MS,
  VAD_PAD_MS,
  VAD_MIN_SEG_MS,
  VAD_MIN_VOICED_SEC,
  VAD_SILENCE_RATIO_TO_APPLY,
} from "./js/constants.js";
import {
  setStatus,
  log,
  fmtSec,
  clamp01,
  setRealtimePanelsActive,
  resetRealtimePanels,
  resetMeter,
  isOOMError,
} from "./js/ui.js";
import {
  PS_INTERVAL_MS,
  PS_MIN_HZ,
  PS_MAX_HZ,
  CONFIDENCE_INCLUDE_THRESHOLD,
  CONFIDENCE_VOICED_THRESHOLD,
  VOICE_PRESETS,
  PITCH_PROFILE_DEFAULT,
  PITCH_RANGE_HARD,
  DEFAULT_PITCH_RANGE,
  PITCH_COUNTER_KEYS,
  clampPitchRange,
  createPitchPostState,
  resetPitchPostState,
  appendPitchSample as sharedAppendPitchSample,
  makeNoiseTracker,
  filterPitchForStats,
  makeStats,
  percentileSorted,
  computeIntonationMetrics,
  fmt1,
  logPostProcessingDiagnostics,
} from "./js/pitch-shared.js";

/** 只用遠端（Hugging Face Hub），停用本機 /models 尋址 */
env.allowLocalModels = false;
env.allowRemoteModels = true;
/** 視需要可調整：WASM 執行緒數 */
env.backends.onnx.wasm.numThreads = 1;

await initI18n();
let analysisText = getLocaleValue("analysis");
let summaryText = getLocaleValue("summary");
onLocaleChange(() => {
  analysisText = getLocaleValue("analysis");
  summaryText = getLocaleValue("summary");
  updatePlayerCopy();
});

function labelHint(path) {
  return {
    label: t(`${path}.label`),
    hint: t(`${path}.hint`),
  };
}

function summaryString(path, params) {
  return t(`summary.${path}`, params);
}

// 播放器（動態建立；並在其下方插入統計卡容器）
let playBtn = null,
  audioEl = null,
  lastAudioUrl = null,
  playerHintEl = null,
  replayBtn = null,
  replayHintPrefixEl = null,
  replayHintSuffixEl = null,
  replayHintSpacerNode = null;

function updatePlayerCopy(forcePlaying){
  const isPlaying = forcePlaying ?? (audioEl ? !audioEl.paused : false);
  if (playBtn){
    playBtn.textContent = t(isPlaying ? "player.pause" : "player.play");
    playBtn.setAttribute(
      "aria-label",
      t(isPlaying ? "player.ariaPause" : "player.ariaPlay")
    );
  }
  if (playerHintEl){
    if (replayHintPrefixEl){
      replayHintPrefixEl.textContent = t("player.replayHintPrefix");
    }
    if (replayHintSpacerNode){
      replayHintSpacerNode.textContent = t("player.replayHintSpacer");
    }
    if (replayBtn){
      replayBtn.textContent = t("player.replayHintAction");
      replayBtn.setAttribute("aria-label", t("player.replayHintAria"));
    }
    if (replayHintSuffixEl){
      replayHintSuffixEl.textContent = t("player.replayHintSuffix");
    }
  }
}
ensurePlayerUI();
setupExportButton();

// ===== 狀態 =====
let mediaRecorder = null, chunks = [];
let clf = null, busy = false, heartbeatTimer = null;
let currentDevice = "wasm";
let isRecording = false;

let analysisSeq = 0;
let activeAnalysisToken = 0;

function startAnalysisRun(){
  analysisSeq += 1;
  activeAnalysisToken = analysisSeq;
  busy = true;
  updateUploadAvailability();
  updatePlaybackAvailability();
  updateRecordAvailability();
  return activeAnalysisToken;
}

function isAnalysisActive(token){
  return token === activeAnalysisToken;
}

function finishAnalysisRun(token){
  if (!isAnalysisActive(token)) return false;
  busy = false;
  updateUploadAvailability();
  updatePlaybackAvailability();
  updateRecordAvailability();
  return true;
}

function updateUploadAvailability(){
  const disable = isRecording;
  if (fileInput){
    fileInput.disabled = disable;
  }
  if (uploadFab){
    if (disable){
      uploadFab.setAttribute("disabled", "true");
      uploadFab.setAttribute("aria-disabled", "true");
    } else {
      uploadFab.removeAttribute("disabled");
      uploadFab.removeAttribute("aria-disabled");
    }
  }
}

function updatePlaybackAvailability(){
  if (!playBtn) return;
  const hasSource = !!(audioEl && audioEl.src);
  const disable = !hasSource || isRecording || busy;
  if (disable){
    playBtn.setAttribute("disabled", "true");
    playBtn.setAttribute("aria-disabled", "true");
  } else {
    playBtn.removeAttribute("disabled");
    playBtn.removeAttribute("aria-disabled");
  }
}

function updateRecordAvailability(){
  if (!recordBtn) return;
  const disable = busy && !isRecording;
  if (disable){
    recordBtn.setAttribute("disabled", "true");
    recordBtn.setAttribute("aria-disabled", "true");
  } else {
    recordBtn.removeAttribute("disabled");
    recordBtn.removeAttribute("aria-disabled");
  }
}

updateUploadAvailability();
updatePlaybackAvailability();
updateRecordAvailability();

// Pitch Stream 狀態
let psCtx=null, psSrc=null, psProc=null;
let psRAF=null, psRunning=false;
let psHz=[], psHzSmooth=[], psDb=[], psVoiced=[], psConfidence=[]; // 50ms/點
const psRealtimeNoiseTracker = makeNoiseTracker();
const psOfflineNoiseTracker = makeNoiseTracker();
const PITCH_RANGE_KEY = "vpa:pitchRangeHz";
const PITCH_PROFILE_KEY = "vpa:pitchProfile";
const INTONATION_RAW_KEY = "vpa:intonationShowRaw";
const AUTO_WIDE_RANGE = { min: 70, max: 560 };
const AUTO_SAMPLE_MS = 800;
const AUTO_MIN_VALID_FRAMES = 12;
const AUTO_REEVAL_WINDOW_MS = 2000;
const AUTO_INVALID_RATIO_LIMIT = 0.4;
const AUTO_OCTAVE_SPIKE_LIMIT = 3;
const AUTO_HYSTERESIS_UP = 1.2;
const AUTO_HYSTERESIS_DOWN = 0.85;
const AUTO_NEUTRAL_MEDIAN = (VOICE_PRESETS.neutral.min + VOICE_PRESETS.neutral.max) / 2;

const pitchPostState = createPitchPostState();
let showIntonationRawPoints = true;
let pitchProfileSetting = loadPitchProfileSetting();
let pitchRangeSetting = loadPitchRangeSetting();

const autoRangeState = {
  stage: "idle",
  currentRange: clampPitchRange(VOICE_PRESETS.neutral),
  sampleValues: [],
  sampleDurationMs: 0,
  windowFrames: [],
  windowMs: 0,
  windowInvalidMs: 0,
  octaveEvents: [],
  prevOctaveCount: 0,
  timelineMs: 0,
  lastMedian: null,
  lastUpdateMs: 0,
};

let pitchMinInput = null;
let pitchMaxInput = null;
let pitchRangeResetBtn = null;
let voiceProfileButtons = [];
let voiceSettingsContainer = null;

const offlineFeatureStore = {
  frameSec: 0,
  pitchRaw: [],
  pitchProcessed: [],
  pitchConfidence: [],
  db: [],
  voiced: [],
  formants: [],
  tilt: [],
  breathiness: [],
  energy: [],
  zcr: [],
};

initPitchRangeControls();

const pitchStrategies = {
  acf: { key: "acf", label: "ACF", detect: detectPitchACF },
  yin: { key: "yin", label: "YIN-lite", detect: detectPitchYinLite },
};

function loadIntonationRawPreference(){
  if (typeof window === "undefined" || !window.localStorage) return true;
  try{
    const raw = window.localStorage.getItem(INTONATION_RAW_KEY);
    if (raw == null) return true;
    return raw === "true" || raw === "1";
  }catch{
    return true;
  }
}

function saveIntonationRawPreference(flag){
  showIntonationRawPoints = Boolean(flag);
  if (typeof window === "undefined" || !window.localStorage) return;
  try{
    window.localStorage.setItem(INTONATION_RAW_KEY, showIntonationRawPoints ? "true" : "false");
  }catch{}
}

showIntonationRawPoints = loadIntonationRawPreference();

function loadPitchProfileSetting(){
  try{
    const raw = localStorage.getItem(PITCH_PROFILE_KEY);
    if (!raw) return PITCH_PROFILE_DEFAULT;
    if (raw === "custom") return "custom";
    if (raw in VOICE_PRESETS) return raw;
    return PITCH_PROFILE_DEFAULT;
  }catch{
    return PITCH_PROFILE_DEFAULT;
  }
}

function savePitchProfileSetting(profile){
  pitchProfileSetting = profile;
  if (typeof window === "undefined" || !window.localStorage) return;
  try{ window.localStorage.setItem(PITCH_PROFILE_KEY, profile); }catch{}
}

function loadPitchRangeSetting(){
  try{
    const raw = localStorage.getItem(PITCH_RANGE_KEY);
    if (!raw) return { ...DEFAULT_PITCH_RANGE };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_PITCH_RANGE };
    return clampPitchRange(parsed);
  }catch{
    return { ...DEFAULT_PITCH_RANGE };
  }
}

function savePitchRangeSetting(range){
  pitchRangeSetting = clampPitchRange(range || DEFAULT_PITCH_RANGE);
  if (pitchProfileSetting !== "auto"){
    try{ localStorage.setItem(PITCH_RANGE_KEY, JSON.stringify(pitchRangeSetting)); }catch{}
  }
  updatePitchRangeInputs();
  return pitchRangeSetting;
}

function getPitchProfileDisplayRange(){
  if (pitchProfileSetting === "auto"){
    return clampPitchRange(autoRangeState.currentRange || VOICE_PRESETS.neutral);
  }
  if (pitchProfileSetting && pitchProfileSetting in VOICE_PRESETS){
    const preset = VOICE_PRESETS[pitchProfileSetting];
    if (preset) return clampPitchRange(preset);
  }
  return clampPitchRange(pitchRangeSetting || DEFAULT_PITCH_RANGE);
}

function getPitchDetectorRange(){
  if (pitchProfileSetting === "auto"){
    if (autoRangeState.stage === "bootstrap"){
      return clampPitchRange(AUTO_WIDE_RANGE);
    }
    return clampPitchRange(autoRangeState.currentRange || VOICE_PRESETS.neutral);
  }
  if (pitchProfileSetting && pitchProfileSetting in VOICE_PRESETS){
    const preset = VOICE_PRESETS[pitchProfileSetting];
    if (preset) return clampPitchRange(preset);
  }
  return clampPitchRange(pitchRangeSetting || DEFAULT_PITCH_RANGE);
}

function updatePitchProfileControls(){
  const isAuto = pitchProfileSetting === "auto";
  const activePreset = !isAuto && pitchProfileSetting !== "custom" ? pitchProfileSetting : null;
  voiceProfileButtons.forEach((btn)=>{
    if (!btn) return;
    const profile = btn.dataset?.profile;
    const pressed = profile === "auto" ? isAuto : (profile === activePreset);
    btn.setAttribute("aria-pressed", pressed ? "true" : "false");
  });
  if (voiceSettingsContainer){
    voiceSettingsContainer.classList.toggle("is-auto", isAuto);
  }
  if (pitchMinInput){
    pitchMinInput.disabled = isAuto;
    if (isAuto) pitchMinInput.setAttribute("aria-disabled", "true");
    else pitchMinInput.removeAttribute("aria-disabled");
  }
  if (pitchMaxInput){
    pitchMaxInput.disabled = isAuto;
    if (isAuto) pitchMaxInput.setAttribute("aria-disabled", "true");
    else pitchMaxInput.removeAttribute("aria-disabled");
  }
  if (pitchRangeResetBtn){
    if (isAuto){
      pitchRangeResetBtn.setAttribute("disabled", "true");
      pitchRangeResetBtn.setAttribute("aria-disabled", "true");
    } else {
      pitchRangeResetBtn.removeAttribute("disabled");
      pitchRangeResetBtn.removeAttribute("aria-disabled");
    }
  }
}

function updatePitchRangeInputs(){
  const range = getPitchProfileDisplayRange();
  if (pitchMinInput){ pitchMinInput.value = Math.round(range.min); }
  if (pitchMaxInput){ pitchMaxInput.value = Math.round(range.max); }
}

function refreshVoiceProfileUI(){
  updatePitchProfileControls();
  updatePitchRangeInputs();
}

function applyVoiceProfile(profile){
  if (!profile) return;
  if (profile === "auto"){
    savePitchProfileSetting("auto");
    startAutoRangeSession({ preserveRange: false });
    refreshVoiceProfileUI();
    return;
  }
  if (profile in VOICE_PRESETS && profile !== "auto"){
    const preset = VOICE_PRESETS[profile];
    savePitchProfileSetting(profile);
    autoRangeState.stage = "idle";
    const range = preset ? clampPitchRange(preset) : clampPitchRange(DEFAULT_PITCH_RANGE);
    savePitchRangeSetting(range);
    refreshVoiceProfileUI();
    return;
  }
  if (profile === "custom"){
    savePitchProfileSetting("custom");
    autoRangeState.stage = "idle";
    refreshVoiceProfileUI();
  }
}

function initPitchRangeControls(){
  pitchMinInput = document.getElementById("pitchMinInput");
  pitchMaxInput = document.getElementById("pitchMaxInput");
  pitchRangeResetBtn = document.getElementById("pitchRangeReset");
  voiceSettingsContainer = document.querySelector(".voice-settings");
  voiceProfileButtons = Array.from(document.querySelectorAll("[data-profile]"));

  voiceProfileButtons.forEach((btn)=>{
    btn?.addEventListener("click", ()=>{
      const profile = btn.dataset?.profile;
      if (!profile) return;
      if (profile === pitchProfileSetting && profile !== "auto") return;
      applyVoiceProfile(profile);
    });
  });

  const applyChange = ()=>{
    const minVal = Number(pitchMinInput?.value);
    const maxVal = Number(pitchMaxInput?.value);
    const next = clampPitchRange({
      min: Number.isFinite(minVal) ? minVal : DEFAULT_PITCH_RANGE.min,
      max: Number.isFinite(maxVal) ? maxVal : DEFAULT_PITCH_RANGE.max,
    });
    savePitchProfileSetting("custom");
    autoRangeState.stage = "idle";
    savePitchRangeSetting(next);
    refreshVoiceProfileUI();
  };

  pitchMinInput?.addEventListener("change", applyChange);
  pitchMaxInput?.addEventListener("change", applyChange);
  pitchRangeResetBtn?.addEventListener("click", ()=>{
    applyVoiceProfile("neutral");
  });

  if (pitchProfileSetting === "auto"){
    startAutoRangeSession({ preserveRange: true });
  }
  refreshVoiceProfileUI();
}

function resetAutoWatchdogs(){
  autoRangeState.windowFrames.length = 0;
  autoRangeState.windowMs = 0;
  autoRangeState.windowInvalidMs = 0;
  autoRangeState.octaveEvents.length = 0;
  autoRangeState.prevOctaveCount = Number(pitchPostState.counters?.octaveCorrected || 0);
}

function startAutoRangeSession({ preserveRange = false } = {}){
  if (pitchProfileSetting !== "auto"){
    autoRangeState.stage = "idle";
    autoRangeState.sampleValues.length = 0;
    autoRangeState.sampleDurationMs = 0;
    autoRangeState.timelineMs = 0;
    resetAutoWatchdogs();
    return;
  }
  autoRangeState.stage = "bootstrap";
  autoRangeState.sampleValues.length = 0;
  autoRangeState.sampleDurationMs = 0;
  autoRangeState.timelineMs = 0;
  autoRangeState.lastUpdateMs = 0;
  if (!preserveRange){
    autoRangeState.currentRange = clampPitchRange(VOICE_PRESETS.neutral);
    autoRangeState.lastMedian = null;
  }
  resetAutoWatchdogs();
  updatePitchRangeInputs();
}

function deriveAutoRange(median){
  const rawMin = Math.round(median * 0.6);
  const rawMax = Math.round(median * 1.6);
  const min = Math.max(PITCH_RANGE_HARD.min, Math.min(PITCH_RANGE_HARD.max - 20, rawMin));
  const max = Math.max(min + 20, Math.min(PITCH_RANGE_HARD.max, rawMax));
  return { min, max };
}

function finalizeAutoBootstrap(){
  const values = autoRangeState.sampleValues.slice();
  autoRangeState.sampleValues.length = 0;
  autoRangeState.sampleDurationMs = 0;

  let median = NaN;
  if (values.length >= AUTO_MIN_VALID_FRAMES){
    values.sort((a,b)=>a-b);
    median = values[Math.floor(values.length / 2)];
  }

  let nextMedian = Number.isFinite(median) ? median : AUTO_NEUTRAL_MEDIAN;
  let nextRange = Number.isFinite(median)
    ? deriveAutoRange(nextMedian)
    : clampPitchRange(VOICE_PRESETS.neutral);

  const hadMedian = Number.isFinite(autoRangeState.lastMedian);
  let shouldApply = !hadMedian || !Number.isFinite(median);
  if (!shouldApply && hadMedian){
    const lower = autoRangeState.lastMedian * AUTO_HYSTERESIS_DOWN;
    const upper = autoRangeState.lastMedian * AUTO_HYSTERESIS_UP;
    if (nextMedian < lower || nextMedian > upper){
      shouldApply = true;
    }
  }

  if (shouldApply){
    autoRangeState.currentRange = clampPitchRange(nextRange);
    autoRangeState.lastMedian = nextMedian;
    autoRangeState.lastUpdateMs = autoRangeState.timelineMs;
  }

  autoRangeState.stage = "ready";
  resetAutoWatchdogs();
  updatePitchRangeInputs();
}

function triggerAutoRangeRefresh(){
  if (pitchProfileSetting !== "auto") return;
  if (autoRangeState.stage === "bootstrap") return;
  autoRangeState.stage = "bootstrap";
  autoRangeState.sampleValues.length = 0;
  autoRangeState.sampleDurationMs = 0;
  resetAutoWatchdogs();
}

function handleAutoRangeFrame(result, { dtMs = PS_INTERVAL_MS } = {}){
  if (pitchProfileSetting !== "auto") return;
  if (autoRangeState.stage === "idle") return;

  const dt = Number.isFinite(dtMs) && dtMs > 0 ? dtMs : PS_INTERVAL_MS;
  autoRangeState.timelineMs += dt;

  const processed = Number.isFinite(result?.processed) ? result.processed : NaN;
  const confidence = Number.isFinite(result?.confidence) ? result.confidence : 0;
  const isValid = Number.isFinite(processed) && confidence >= CONFIDENCE_INCLUDE_THRESHOLD;

  if (autoRangeState.stage === "bootstrap"){
    autoRangeState.sampleDurationMs += dt;
    if (isValid){
      autoRangeState.sampleValues.push(processed);
    }
    if (autoRangeState.sampleDurationMs >= AUTO_SAMPLE_MS){
      finalizeAutoBootstrap();
    }
    return;
  }

  autoRangeState.windowFrames.push({ dt, invalid: !isValid });
  autoRangeState.windowMs += dt;
  if (!isValid) autoRangeState.windowInvalidMs += dt;

  while (autoRangeState.windowFrames.length && autoRangeState.windowMs > AUTO_REEVAL_WINDOW_MS){
    const head = autoRangeState.windowFrames.shift();
    autoRangeState.windowMs -= head.dt;
    if (head.invalid) autoRangeState.windowInvalidMs -= head.dt;
    autoRangeState.windowMs = Math.max(0, autoRangeState.windowMs);
    autoRangeState.windowInvalidMs = Math.max(0, autoRangeState.windowInvalidMs);
  }

  if (autoRangeState.windowMs >= AUTO_REEVAL_WINDOW_MS * 0.9){
    const ratio = autoRangeState.windowInvalidMs / Math.max(autoRangeState.windowMs, 1);
    if (ratio > AUTO_INVALID_RATIO_LIMIT){
      triggerAutoRangeRefresh();
      return;
    }
  }

  const currentOctave = Number(pitchPostState.counters?.octaveCorrected || 0);
  if (currentOctave < autoRangeState.prevOctaveCount){
    autoRangeState.prevOctaveCount = currentOctave;
    autoRangeState.octaveEvents.length = 0;
  } else if (currentOctave > autoRangeState.prevOctaveCount){
    const diff = currentOctave - autoRangeState.prevOctaveCount;
    autoRangeState.octaveEvents.push({ time: autoRangeState.timelineMs, count: diff });
    autoRangeState.prevOctaveCount = currentOctave;
  }

  while (autoRangeState.octaveEvents.length && (autoRangeState.timelineMs - autoRangeState.octaveEvents[0].time) > AUTO_REEVAL_WINDOW_MS){
    autoRangeState.octaveEvents.shift();
  }

  const octaveSum = autoRangeState.octaveEvents.reduce((acc, evt)=> acc + (evt?.count || 0), 0);
  if (octaveSum > AUTO_OCTAVE_SPIKE_LIMIT){
    triggerAutoRangeRefresh();
    return;
  }
}

const PITCH_RUNTIME_BASE_BUDGET_MS = 26;
const PITCH_RUNTIME_OVER_BUDGET_LIMIT = 3;
const PITCH_RUNTIME_RECOVERY_MS = 1500;
const PITCH_RUNTIME_OFFLINE_MULTIPLIER = 1.65;
const PITCH_RUNTIME_MIN_BUDGET_MS = 18;
const PITCH_RUNTIME_MAX_BUDGET_MS = 60;
const PITCH_RETRY_MIN_INTERVAL_MS = 3000;
const PITCH_RETRY_COOLDOWN_MS = 20000;
const PITCH_RETRY_ERROR_COOLDOWN_MS = 45000;
const PITCH_RETRY_ERROR_GUARD_MS = 6500;
const PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS = 8000;

const pitchStrategyState = {
  activeKey: null,
  lockUntil: 0,
  lockReason: null,
  lockReasonDetail: null,
  overBudgetStreak: 0,
  lastSwitch: 0,
  lastOverBudget: 0,
  runtimeEwma: 0,
  lastEnableAttempt: 0,
  lockedAt: 0,
  lockDuration: 0,
  lockContext: null,
  autoRetryUntil: 0,
};

const yinBuffers = {
  x: new Float32Array(0),
  diff: new Float32Array(0),
  cmndf: new Float32Array(0),
};

const acfBuffers = {
  x: new Float32Array(0),
};

let trimPitchBuffersTimer = null;
let pitchAutoRetryTimer = null;
const pitchRetryTimers = new Set();

function registerPitchRetryTimer(id){
  if (id == null) return null;
  pitchRetryTimers.add(id);
  pitchAutoRetryTimer = id;
  return id;
}

function releasePitchRetryTimer(id){
  if (id == null) return;
  if (pitchRetryTimers.has(id)){
    pitchRetryTimers.delete(id);
  }
  if (pitchAutoRetryTimer === id){
    pitchAutoRetryTimer = null;
  }
}

function cancelPitchAutoRetryTimers(){
  if (!pitchRetryTimers.size) return;
  const clearFn = typeof clearTimeout === "function" ? clearTimeout : null;
  for (const handle of pitchRetryTimers){
    if (clearFn) clearFn(handle);
  }
  pitchRetryTimers.clear();
  pitchAutoRetryTimer = null;
}

initializePitchStrategy();

function nowMs(){
  try {
    if (typeof performance !== "undefined" && performance?.now) {
      return performance.now();
    }
  } catch {}
  return Date.now();
}

function initializePitchStrategy(){
  try {
    const preferred = selectPreferredPitchStrategy("initial");
    switchPitchStrategy(preferred, "initial");
  } catch (err){
    console.warn("[pitch] initialize failed", err);
    switchPitchStrategy(pitchStrategies.acf, "initial-fallback");
  }
}

function switchPitchStrategy(strategy, reason){
  const next = strategy || pitchStrategies.acf;
  if (pitchStrategyState.activeKey === next.key) return;
  pitchStrategyState.activeKey = next.key;
  pitchStrategyState.overBudgetStreak = 0;
  pitchStrategyState.lastSwitch = nowMs();
  pitchStrategyState.lastOverBudget = 0;
  pitchStrategyState.runtimeEwma = 0;
  if (next.key === "yin"){
    cancelPitchBufferTrimTimer();
    cancelPitchAutoRetryTimers();
  }
  if (reason) {
    log(`[pitch] strategy -> ${next.label} (${next.key}) via ${reason}`);
  } else {
    log(`[pitch] strategy -> ${next.label} (${next.key})`);
  }
}

function getActivePitchStrategy(){
  return pitchStrategies[pitchStrategyState.activeKey] || pitchStrategies.acf;
}

function maybeEnableAdvancedPitch(context, { allowRetry = false, force = false } = {}){
  const active = getActivePitchStrategy();
  if (!force && active.key !== "acf") return;

  const now = nowMs();
  if (!force){
    if (pitchStrategyState.lockUntil && now < pitchStrategyState.lockUntil){
      if (!allowRetry) return;
      const reason = pitchStrategyState.lockReason;
      const lockDuration = pitchStrategyState.lockDuration || PITCH_RETRY_COOLDOWN_MS;
      const elapsed = Math.max(0, now - (pitchStrategyState.lockedAt || 0));
      const ratio = lockDuration > 0 ? elapsed / lockDuration : 1;
      const offlineGrace = context === "offline" && reason === "runtime" && ratio >= 0.35;
      const runtimeGrace = reason === "runtime" && ratio >= 0.5;
      if (!(offlineGrace || runtimeGrace)) return;
    }
    const minInterval = allowRetry ? PITCH_RETRY_MIN_INTERVAL_MS / 2 : PITCH_RETRY_MIN_INTERVAL_MS;
    if (now - pitchStrategyState.lastEnableAttempt < minInterval) return;
  }

  pitchStrategyState.lastEnableAttempt = now;
  const preferred = selectPreferredPitchStrategy(context);
  if (preferred.key === active.key && !force) return;
  switchPitchStrategy(preferred, context ? `${context}-enable` : "enable");
  if (preferred.key === "yin"){
    pitchStrategyState.lockUntil = 0;
    pitchStrategyState.lockReason = null;
    pitchStrategyState.lockReasonDetail = null;
    pitchStrategyState.lockDuration = 0;
    pitchStrategyState.lockedAt = 0;
    pitchStrategyState.lockContext = null;
    pitchStrategyState.autoRetryUntil = 0;
    cancelPitchAutoRetryTimers();
  }
}

function degradePitchStrategy(reason, { cooldownMs, detail, context } = {}){
  if (pitchStrategyState.activeKey === "acf") return;
  const now = nowMs();
  const requestedCooldown = Number.isFinite(cooldownMs)
    ? cooldownMs
    : (reason === "error" ? PITCH_RETRY_ERROR_COOLDOWN_MS : PITCH_RETRY_COOLDOWN_MS);

  clearPitchAutoRetry();

  let timeout;
  if (reason === "error"){
    const guard = Math.max(3500, Math.min(PITCH_RETRY_ERROR_GUARD_MS, requestedCooldown));
    timeout = guard;
    pitchStrategyState.autoRetryUntil = now + Math.max(requestedCooldown, guard + 4000);
  } else {
    timeout = Math.max(PITCH_RETRY_COOLDOWN_MS, requestedCooldown);
    pitchStrategyState.autoRetryUntil = 0;
  }

  pitchStrategyState.lockUntil = Math.max(pitchStrategyState.lockUntil, now + timeout);
  pitchStrategyState.lockReason = reason || "degraded";
  pitchStrategyState.lockReasonDetail = detail || null;
  pitchStrategyState.lockedAt = now;
  pitchStrategyState.lockDuration = timeout;
  pitchStrategyState.lockContext = context || null;
  schedulePitchBufferTrim();
  const logReason = detail ? `${reason || "degraded"}:${detail}` : (reason || "degraded");
  if (reason === "error"){
    schedulePitchAutoRetry("error");
  }
  switchPitchStrategy(pitchStrategies.acf, logReason);
}

function clearPitchAutoRetry({ resetWindow = true } = {}){
  cancelPitchAutoRetryTimers();
  if (resetWindow){
    pitchStrategyState.autoRetryUntil = 0;
  }
}

function schedulePitchAutoRetry(reason){
  if (typeof setTimeout !== "function") return;
  if (!pitchStrategyState.autoRetryUntil) return;

  let pendingTimer = null;

  function scheduleNext(delay){
    if (typeof setTimeout !== "function") return null;
    pendingTimer = registerPitchRetryTimer(setTimeout(attempt, delay));
    return pendingTimer;
  }

  function attempt(){
    if (pendingTimer != null){
      releasePitchRetryTimer(pendingTimer);
      pendingTimer = null;
    }
    const now = nowMs();
    if (pitchStrategyState.lockReason !== reason){
      clearPitchAutoRetry();
      return;
    }
    if (pitchStrategyState.lockUntil && now + 50 < pitchStrategyState.lockUntil){
      const retryDelay = Math.max(
        800,
        Math.min(
          PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS,
          pitchStrategyState.lockUntil - now + 200
        )
      );
      scheduleNext(retryDelay);
      return;
    }

    const contexts = pitchStrategyState.lockContext
      ? [pitchStrategyState.lockContext]
      : ["realtime", "offline"];
    for (const ctx of contexts){
      maybeEnableAdvancedPitch(ctx, { allowRetry: true });
    }

    if (
      pitchStrategyState.activeKey === "acf" &&
      pitchStrategyState.lockReason === reason &&
      pitchStrategyState.autoRetryUntil &&
      now < pitchStrategyState.autoRetryUntil
    ){
      const nextDelay = Math.max(
        1500,
        Math.min(
          PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS,
          pitchStrategyState.autoRetryUntil - now
        )
      );
      scheduleNext(nextDelay);
    } else {
      clearPitchAutoRetry();
    }
  }

  const now = nowMs();
  const guardDelay = pitchStrategyState.lockUntil && pitchStrategyState.lockUntil > now
    ? pitchStrategyState.lockUntil - now + 200
    : 800;
  const initialDelay = Math.max(
    1200,
    Math.min(PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS, guardDelay)
  );
  scheduleNext(initialDelay);
}

function selectPreferredPitchStrategy(context){
  const override = getPitchModeOverride();
  if (override === "force-baseline") return pitchStrategies.acf;
  if (override === "force-advanced") return pitchStrategies.yin;

  const info = estimateDeviceTier();
  if (info.saveData) return pitchStrategies.acf;

  const contextHint = context === "offline" ? 1 : 0;
  const now = nowMs();
  const stillCooling = pitchStrategyState.lockUntil && now < pitchStrategyState.lockUntil;
  const lockDuration = pitchStrategyState.lockDuration || PITCH_RETRY_COOLDOWN_MS;
  const elapsed = Math.max(0, now - (pitchStrategyState.lockedAt || 0));
  const ratio = lockDuration > 0 ? elapsed / lockDuration : 1;
  const runtimePenalty =
    stillCooling &&
    pitchStrategyState.lockReason === "runtime" &&
    ratio < (context === "offline" ? 0.35 : 0.5)
      ? 1
      : 0;
  const allowMobileAdvanced = info.isMobile && info.score >= 7 && !info.lowPowerMode;
  const allowDesktopAdvanced = !info.isMobile && info.score >= 5;

  if ((allowMobileAdvanced || allowDesktopAdvanced) && runtimePenalty === 0){
    return pitchStrategies.yin;
  }

  if ((allowMobileAdvanced || allowDesktopAdvanced) && runtimePenalty && contextHint){
    return pitchStrategies.yin;
  }

  if (!info.isMobile && info.score >= 6 && runtimePenalty === 0) return pitchStrategies.yin;
  return pitchStrategies.acf;
}

function getPitchModeOverride(){
  try {
    if (typeof localStorage === "undefined") return null;
    const val = localStorage.getItem("vpa:pitchMode");
    if (val === "force-baseline" || val === "force-advanced") return val;
  } catch {}
  return null;
}

function estimateDeviceTier(){
  const nav = typeof navigator !== "undefined" ? navigator : {};
  const uaData = nav.userAgentData || null;
  const rawUa = nav.userAgent || "";
  const isMobile = uaData?.mobile ?? /Android|iP(hone|od)|Mobile/i.test(rawUa);
  const isTablet = /iPad|Tablet|Silk|PlayBook|Pixel C|Nexus 7/i.test(rawUa);
  const concurrency = Number.isFinite(nav.hardwareConcurrency) ? nav.hardwareConcurrency : 2;
  const deviceMemory = Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : 0;
  const saveData = !!nav?.connection?.saveData;
  const lowPowerMode = !!nav?.connection?.effectiveType && /2g|slow-2g/.test(nav.connection.effectiveType);
  let score = concurrency;
  if (deviceMemory){ score += deviceMemory; }
  if (!isMobile || isTablet){ score += 2; }
  if (typeof nav.gpu !== "undefined") score += 1.5;
  if (saveData) score -= 2;
  if (lowPowerMode) score -= 1;

  const highEndApple = /iPhone1[2-9]|iPhone2[0-9]|iPad\sPro|AppleCoreMedia.*(M1|M2|M3)/i.test(rawUa);
  if (highEndApple) score += 2;
  const flagshipAndroid = /Pixel\s(7|8|9)|SM-G99|SM-S9|Snapdragon\s8/i.test(rawUa);
  if (flagshipAndroid) score += 2;

  return { isMobile: isMobile && !isTablet, isTablet, score, saveData, lowPowerMode };
}

function schedulePitchBufferTrim(){
  if (typeof setTimeout !== "function") return;
  cancelPitchBufferTrimTimer();
  const now = nowMs();
  const baseDelay = pitchStrategyState.lockUntil
    ? Math.max(0, pitchStrategyState.lockUntil - now)
    : 8000;
  const delay = Math.max(4000, Math.min(baseDelay, 15000));
  trimPitchBuffersTimer = setTimeout(()=>{
    if (getActivePitchStrategy().key !== "acf") return;
    if (pitchStrategyState.lockUntil && nowMs() < pitchStrategyState.lockUntil - 5000) return;
    trimYinBuffers();
  }, delay);
}

function cancelPitchBufferTrimTimer(){
  if (trimPitchBuffersTimer){
    clearTimeout(trimPitchBuffersTimer);
    trimPitchBuffersTimer = null;
  }
}

function trimYinBuffers(){
  yinBuffers.x = new Float32Array(0);
  yinBuffers.diff = new Float32Array(0);
  yinBuffers.cmndf = new Float32Array(0);
}

function trackPitchRuntime(elapsedMs, strategy, context, frameSamples){
  if (!strategy || strategy.key === "acf") return;
  const now = nowMs();
  const frameScale = frameSamples ? Math.min(3, Math.max(0.7, frameSamples / 2048)) : 1;
  const contextMultiplier = context === "offline" ? PITCH_RUNTIME_OFFLINE_MULTIPLIER : 1;
  const dynamicBudget = PITCH_RUNTIME_BASE_BUDGET_MS * frameScale * contextMultiplier;
  pitchStrategyState.runtimeEwma = pitchStrategyState.runtimeEwma
    ? (pitchStrategyState.runtimeEwma * 0.7 + elapsedMs * 0.3)
    : elapsedMs;
  const adaptiveBudget = Math.min(
    PITCH_RUNTIME_MAX_BUDGET_MS,
    Math.max(
      PITCH_RUNTIME_MIN_BUDGET_MS,
      Math.max(dynamicBudget, pitchStrategyState.runtimeEwma * 1.6)
    )
  );

  if (elapsedMs <= adaptiveBudget){
    if (pitchStrategyState.overBudgetStreak && pitchStrategyState.lastOverBudget){
      if (now - pitchStrategyState.lastOverBudget > PITCH_RUNTIME_RECOVERY_MS){
        pitchStrategyState.overBudgetStreak = Math.max(0, pitchStrategyState.overBudgetStreak - 1);
        if (!pitchStrategyState.overBudgetStreak) pitchStrategyState.lastOverBudget = 0;
      }
    }
    return;
  }

  pitchStrategyState.overBudgetStreak += 1;
  pitchStrategyState.lastOverBudget = now;
  if (pitchStrategyState.overBudgetStreak >= PITCH_RUNTIME_OVER_BUDGET_LIMIT){
    const cooldown = context === "offline"
      ? Math.max(8000, PITCH_RETRY_COOLDOWN_MS / 2)
      : PITCH_RETRY_COOLDOWN_MS;
    degradePitchStrategy("runtime", { cooldownMs: cooldown, detail: `${elapsedMs.toFixed(1)}ms`, context });
  }
}

function runPitchDetection(input, sr, { context = "realtime" } = {}){
  const strategy = getActivePitchStrategy();
  const start = nowMs();
  try {
    const hz = strategy.detect(input, sr);
    const elapsed = nowMs() - start;
    trackPitchRuntime(elapsed, strategy, context, input?.length || 0);
    return hz;
  } catch (err){
    console.error(`[pitch] ${strategy.key} failed`, err);
    degradePitchStrategy("error", { cooldownMs: PITCH_RETRY_ERROR_COOLDOWN_MS, context });
    if (strategy.key !== "acf"){
      try {
        return pitchStrategies.acf.detect(input, sr);
      } catch (fallbackErr){
        console.error("[pitch] fallback failed", fallbackErr);
      }
    }
    return null;
  }
}

function ensureYinCapacity(n){
  if (yinBuffers.x.length < n){
    yinBuffers.x = new Float32Array(n);
  }
  if (yinBuffers.diff.length < n+1){
    yinBuffers.diff = new Float32Array(n+1);
  }
  if (yinBuffers.cmndf.length < n+1){
    yinBuffers.cmndf = new Float32Array(n+1);
  }
}

let latestAnalysisExport = null;

// 追蹤最新模型傾向（供簡評用）
let lastPf = 0, lastPm = 0;

// ===== 版本資訊（build 與日期） =====
(async function fillBuildMeta(){
  try{
    const verEl = document.getElementById('ver'); const updEl = document.getElementById('updatedAt');
    if (!verEl && !updEl) return;
    const selfUrl = (import.meta && import.meta.url) ? import.meta.url : 'assets/app.js';
    const res = await fetch(selfUrl, { method:'HEAD', cache:'no-store' });
    let d=null; if(res.ok){ const lm=res.headers.get('last-modified'); if(lm) d=new Date(lm); }
    if(!d || isNaN(d.getTime())) d=new Date();
    const y=d.getFullYear(), m=String(d.getMonth()+1).padStart(2,'0'), day=String(d.getDate()).padStart(2,'0');
    const hh=String(d.getHours()).padStart(2,'0'), mm=String(d.getMinutes()).padStart(2,'0');
    if (updEl) updEl.textContent = `${y}-${m}-${day}`;
    if (verEl) verEl.textContent = `build-${y}${m}${day}-${hh}${mm}`;
  }catch{}
})();

// ===== 事件 =====
recordBtn?.addEventListener("click", async ()=>{
  if (busy && !isRecording) return;
  try{
    if (!mediaRecorder || mediaRecorder.state==="inactive"){
      resetMeter();
      await startRecording();
    } else {
      await stopRecording();
    }
  }catch(err){ console.error("[recordBtn]", err); setStatus(t("status.recordFailed")); }
});
fileInput?.addEventListener("change", async (e)=>{
  if (isRecording){
    setStatus(t("status.uploadWhileRecording"));
    if (e.target) e.target.value = "";
    return;
  }
  try{
    const f = e.target.files?.[0]; if(!f) return;
    dismissOnboardTip(true);
    resetMeter();
    stopPlayback();
    await handleFileOrBlob(f, "upload");
    e.target.value = "";
  }catch(err){ console.error("[fileInput]", err); setStatus(t("status.uploadFailed")); }
});

uploadFab?.addEventListener("click", ()=>{
  if (isRecording) return;
  if (typeof window !== "undefined" && typeof window.scrollTo === "function"){
    try{
      window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
    }catch{
      window.scrollTo(0, 0);
    }
  }
  stopPlayback();
  fileInput?.click();
});

const dropZoneActiveClass = "dropzone-active";

if (dropZone){
  let dropZoneDragDepth = 0;

  const hasFilePayload = (event)=>{
    if (!event?.dataTransfer) return false;
    const types = event.dataTransfer.types;
    if (!types) return false;
    return Array.from(types).includes("Files");
  };

  const clearDropZoneHighlight = ()=>{
    dropZoneDragDepth = 0;
    dropZone.classList.remove(dropZoneActiveClass);
  };

  dropZone.addEventListener("dragenter", (event)=>{
    if (!hasFilePayload(event)) return;
    event.preventDefault();
    dropZoneDragDepth += 1;
    dropZone.classList.add(dropZoneActiveClass);
  });

  dropZone.addEventListener("dragover", (event)=>{
    if (!hasFilePayload(event)) return;
    event.preventDefault();
    if (event.dataTransfer){
      event.dataTransfer.dropEffect = isRecording ? "none" : "copy";
    }
    dropZone.classList.add(dropZoneActiveClass);
  });

  dropZone.addEventListener("dragleave", (event)=>{
    if (!hasFilePayload(event)) return;
    event.preventDefault();
    dropZoneDragDepth = Math.max(0, dropZoneDragDepth - 1);
    if (dropZoneDragDepth === 0){
      dropZone.classList.remove(dropZoneActiveClass);
    }
  });

  dropZone.addEventListener("drop", async (event)=>{
    if (!hasFilePayload(event)) return;
    event.preventDefault();
    clearDropZoneHighlight();
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    if (isRecording){
      setStatus(t("status.uploadWhileRecording"));
      return;
    }
    try{
      dismissOnboardTip(true);
      resetMeter();
      stopPlayback();
      await handleFileOrBlob(file, "upload");
    }catch(err){
      console.error("[dropZone]", err);
      setStatus(t("status.uploadFailed"));
    }
  });

  document.addEventListener("dragend", clearDropZoneHighlight);
  document.addEventListener("drop", clearDropZoneHighlight);
}

// ===== 錄音 =====
function pickSupportedMime(){
  const cands = ["audio/webm;codecs=opus","audio/webm","audio/mp4","audio/ogg"];
  try{ if(typeof MediaRecorder!=="undefined" && MediaRecorder.isTypeSupported){ for(const t of cands) if(MediaRecorder.isTypeSupported(t)) return t; } }catch{}
  return "";
}
async function requestMicStream(){
  const base = { audio: { echoCancellation:false, noiseSuppression:false, autoGainControl:false } };
  const fallback = { audio:true };
  const getUserMedia = navigator?.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices);
  if (!getUserMedia){ throw new Error("record-unsupported"); }
  const disableTrackProcessing = async (stream)=>{
    try{
      const tracks = stream?.getAudioTracks?.() || [];
      await Promise.all(tracks.map(async (track)=>{
        if (!track?.applyConstraints) return;
        try{
          await track.applyConstraints({ echoCancellation:false, noiseSuppression:false, autoGainControl:false });
        }catch(err){ console.warn("[audio track] disable processing failed", err); }
      }));
    }catch(err){ console.warn("[audio track] constraints traversal failed", err); }
  };
  try{
    const stream = await getUserMedia(base);
    await disableTrackProcessing(stream);
    return stream;
  }catch(err){
    console.warn("[getUserMedia] preferred constraints failed", err);
  }
  const fallbackStream = await getUserMedia(fallback);
  await disableTrackProcessing(fallbackStream);
  return fallbackStream;
}

async function startRecording(){
  if (typeof MediaRecorder === "undefined"){ setStatus(t("status.recordUnsupported"), false); return; }
  stopPlayback();
  let stream;
  try{
    stream = await requestMicStream();
  }catch(err){
    console.error("[startRecording] getUserMedia failed", err);
    if (err?.message === "record-unsupported"){
      setStatus(t("status.recordUnsupported"), false);
    } else {
      setStatus(t("status.recordFailed"));
    }
    return;
  }
  dismissOnboardTip(true);
  chunks = [];
  const mimeType = pickSupportedMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  mediaRecorder.ondataavailable = (ev)=>{ if(ev.data?.size) chunks.push(ev.data); };
  mediaRecorder.onstop = async ()=>{
    try{
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      chunks.length = 0;
      stopPitchStream();                 // 停止即時圖，但保留資料做統計
      await handleFileOrBlob(blob, "recording");      // 分析完成後會呼叫 finishStreamStats()
    }catch(e){ console.error("[onstop]", e); setStatus(t("status.recordProcessingFailed")); }
    finally{ stream.getTracks().forEach(t=>t.stop()); }
  };

  document.body.classList.add("recording");
  document.querySelector(".container")?.classList.add("recording");
  setStatus(t("status.recording"));
  isRecording = true;
  updateUploadAvailability();
  updatePlaybackAvailability();
  updateRecordAvailability();
  try {
    mediaRecorder.start();
  } catch (err) {
    isRecording = false;
    updateUploadAvailability();
    updatePlaybackAvailability();
    updateRecordAvailability();
    document.body.classList.remove("recording");
    document.querySelector(".container")?.classList.remove("recording");
    stream.getTracks().forEach(t=>t.stop());
    throw err;
  }

  // 啟動 Pitch Stream
  startPitchStream(stream);
}
async function stopRecording(){
  if (mediaRecorder && mediaRecorder.state!=="inactive"){
    busy = true;
    isRecording = false;
    updateUploadAvailability();
    updatePlaybackAvailability();
    updateRecordAvailability();
    setStatus(t("status.processingAudio"), true);
    try {
      mediaRecorder.stop();
    } catch (err) {
      busy = false;
      updateUploadAvailability();
      updatePlaybackAvailability();
      updateRecordAvailability();
      throw err;
    }
  }
  document.body.classList.remove("recording");
  document.querySelector(".container")?.classList.remove("recording");
}

// ===== 主流程 =====
async function handleFileOrBlob(fileOrBlob, source = "upload"){
  const token = startAnalysisRun(source);
  let decoded = null;
  try{
    setPlaybackSource(fileOrBlob);
    updatePlaybackAvailability();

    setStatus(t("status.decoding"), true);
    decoded = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
    if (!isAnalysisActive(token)) return;
    let { float32, sr, durationSec } = decoded;

    // 離線抽樣（供 Statistics / 簡評）。先對原始音檔做一次。
    offlineExtractStreamMetrics(float32, sr, /*append*/false);

    if (durationSec > WARN_LONG_SEC){
      setStatus(t("status.warnLong", { duration: fmtSec(durationSec) }), true);
      await microYield();
      if (!isAnalysisActive(token)) return;
    }

    // VAD（只選段）
    const vad = maybeApplyAdaptiveVAD(float32, sr);
    if (vad && vad.used){
      const reducedRatio = 1 - (vad.keptSec / durationSec);
      float32 = vad.arr; durationSec = vad.keptSec;
      setStatus(t("status.vadApplied", { ratio: Math.round(reducedRatio*100), duration: fmtSec(durationSec) }), true);
      // 針對「有效語音」再抽樣一次，提升代表性
      offlineExtractStreamMetrics(float32, sr, /*append*/true);
      await microYield();
      if (!isAnalysisActive(token)) return;
    }

    if (!isAnalysisActive(token)) return;

    if (durationSec <= MAX_WHOLE_SEC){
      await analyzeWhole(float32, sr, durationSec, token);
    } else {
      await analyzeStreamed(
        float32,
        sr,
        durationSec,
        t("status.streamingSwitch", { limit: MAX_WHOLE_SEC }),
        token
      );
    }

    // 顯示統計（錄音/上傳皆會有）
    if (!isAnalysisActive(token)) return;
    finishStreamStats();
  }catch(e){
    console.error("[handleFileOrBlob]", e);
    if (isAnalysisActive(token)){
      setStatus(t("status.errorPrefix", { message: e?.message || t("status.decodeFailure") }));
    }
  }finally{
    if (decoded) decoded.float32 = null;
    decoded = null;
    finishAnalysisRun(token);
  }
}

// ===== 解碼策略（WebAudio 為主） =====
async function decodeSmartToFloat32(blobOrFile, targetSR){
  setStatus(t("status.webaudioDecode"), true);
  try {
    return await decodeViaWebAudio(blobOrFile, targetSR);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err ?? "");
    log("[decode] WebAudio decode failed:", message);
    throw new Error(t("status.decodeFailure"));
  }
}
async function decodeViaWebAudio(blobOrFile, targetSR=16000){
  const arrayBuf = await blobOrFile.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let offline = null;
  try{
    let audioBuf;
    try {
      audioBuf = await ctx.decodeAudioData(arrayBuf);
    } catch (err) {
      audioBuf = await new Promise((resolve, reject) => {
        try {
          ctx.decodeAudioData(arrayBuf.slice(0), resolve, reject);
        } catch (legacyErr) {
          reject(legacyErr);
        }
      });
    }
    const mono = ctx.createBuffer(1, audioBuf.length, audioBuf.sampleRate);
    const outCh = mono.getChannelData(0);
    const ch0 = audioBuf.getChannelData(0);
    if (audioBuf.numberOfChannels > 1){
      const ch1 = audioBuf.getChannelData(1);
      for (let i=0;i<ch0.length;i++) outCh[i] = (ch0[i] + ch1[i]) / 2;
    } else { outCh.set(ch0); }

    let out;
    if (audioBuf.sampleRate === targetSR){
      out = outCh.slice(0);
    } else {
      offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
      const src = offline.createBufferSource();
      src.buffer = mono; src.connect(offline.destination); src.start(0);
      const rendered = await offline.startRendering();
      out = rendered.getChannelData(0).slice(0);
    }
    return { float32: out, sr: targetSR, durationSec: out.length / targetSR };
  } finally {
    try{ await ctx.close(); }catch{}
    offline = null;
  }
}

// ===== 模型 =====
async function ensurePipeline(){
  if (clf) return clf;
  setStatus(t("status.modelLoading"), true);
  const progress_callback = (p)=>{
    if (!p) return;
    let pct=null;
    if (typeof p.loadedBytes==='number' && typeof p.totalBytes==='number' && p.totalBytes>0) pct=p.loadedBytes/p.totalBytes;
    else if (typeof p.progress==='number' && isFinite(p.progress)) pct=p.progress;
    const label = p.status || t("status.modelDownloading");
    if (pct==null) setStatus(`${label}…`, true);
    else setStatus(`${label} ${Math.min(99, Math.max(0, Math.floor(pct*100)))}% …`, true);
  };
  const device = (typeof navigator!=='undefined' && navigator.gpu) ? 'webgpu' : 'wasm';
  clf = await pipeline("audio-classification", MODEL_ID, { progress_callback, device });
  currentDevice = device;
  setStatus(t("status.modelReady", { device }));
  return clf;
}

// ===== 分析（整段） =====
async function analyzeWhole(float32, sr, durationSec, token){
  if (!isAnalysisActive(token)) return;
  const model = await ensurePipeline();
  if (!isAnalysisActive(token)) return;
  meter?.classList.remove("hidden");

  const started = performance.now();
  startHeartbeat(()=>{
    if (!isAnalysisActive(token)) return;
    const elapsed=(performance.now()-started)/1000;
    setStatus(t("status.analyzeWhole", { duration: fmtSec(durationSec), elapsed: fmtSec(elapsed) }), true);
  });

  try{
    const res = await model(float32, { sampling_rate: sr, topk: 2 });
    if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
    const map = toMap(res);
    render(map.female||0, map.male||0);
    setStatus(t("status.analyzeWholeDone"));
  }catch(err){
    if (isOOMError(err)){
      console.warn("[analyzeWhole] OOM → switch to streamed mode…");
      stopHeartbeat();
      if (isAnalysisActive(token)){
        await analyzeStreamed(float32, sr, durationSec, t("status.analyzeWholeOOM"), token);
      }
      return;
    }
    console.error("[analyzeWhole]", err);
    if (isAnalysisActive(token)){
      setStatus(t("status.analyzeWholeFailed"));
    }
  }finally{ stopHeartbeat(); }
}

// ===== 分析（串流分段） =====
async function analyzeStreamed(float32, sr, durationSec, reason = t("status.streamingDefaultReason"), token){
  if (!isAnalysisActive(token)) return;
  const model = await ensurePipeline();
  if (!isAnalysisActive(token)) return;
  meter?.classList.remove("hidden");

  const strategy = pickStreamStrategy(durationSec);
  const reasonBits = [reason, strategy.label].filter(Boolean);
  const reasonLabel = reasonBits.join("｜");

  let lastErr=null;
  for (const winSec of strategy.wins){
    try{
      await runStreamedWithWindow(model, float32, sr, durationSec, winSec, strategy.hop, reasonLabel || reason, token);
      if (!isAnalysisActive(token)) return;
      return;
    }catch(e){
      lastErr=e;
      if (isOOMError(e)){ console.warn(`[streamed] OOM at win=${winSec}s → downshift`); continue; }
      else { console.error(`[streamed] error at win=${winSec}s`, e); break; }
    }
  }
  console.error("[analyzeStreamed] failed", lastErr);
  if (isAnalysisActive(token)){
    setStatus(t("status.analyzeStreamFailed"));
  }
}
async function runStreamedWithWindow(model, float32, sr, durationSec, WIN_S, HOP_S, reason, token){
  if (!isAnalysisActive(token)) return;
  const win = Math.max(1, Math.floor(WIN_S * sr));
  const hop = Math.max(1, Math.floor(HOP_S * sr));

  const chunks = [];
  for (let s=0; s<float32.length; s+=hop){
    const e = Math.min(s+win, float32.length);
    if (e - s < Math.floor(0.5*sr)) break;
    chunks.push([s,e]);
    if (e === float32.length) break;
  }
  if (!chunks.length) chunks.push([0, Math.min(win, float32.length)]);

  let avgMs=0, processedSec=0;
  let logitSum=0, wSum=0;

  const started = performance.now();
  startHeartbeat(()=>{
    if (!isAnalysisActive(token)) return;
    const elapsed=(performance.now()-started)/1000;
    const pct = processedSec>0 ? Math.min(99, Math.round((processedSec/durationSec)*100)) : 0;
    setStatus(t("status.analyzeStream", { win: WIN_S, step: HOP_S, reason, progress: pct, elapsed: fmtSec(elapsed) }), true);
  });

  try{
    for (let i=0;i<chunks.length;i++){
      if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
      const [s0,s1] = chunks[i];
      const seg = float32.subarray(s0, s1);
      const dur = (s1 - s0) / sr;

      const t0 = performance.now();
      const out = await model(seg, { sampling_rate: sr, topk: 2 });
      if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
      const dt = performance.now() - t0;
      avgMs = avgMs===0 ? dt : (avgMs*0.65 + dt*0.35);

      const map = toMap(out);
      const pf = clamp01(map.female || EPS);
      const pm = clamp01(map.male   || EPS);
      const logit = Math.log(pf) - Math.log(pm);

      logitSum += logit * dur; wSum += dur;

      const logitAvg = logitSum / Math.max(wSum, EPS);
      const pf_now = 1 / (1 + Math.exp(-logitAvg));
      const pm_now = 1 - pf_now;
      render(pf_now, pm_now);

      processedSec = Math.min(durationSec, (s1 / sr));
      const remain = chunks.length - i - 1;
      const etaSec = (remain * (avgMs/1000));
      const pct = Math.round(((i+1)/chunks.length)*100);
      setStatus(t("status.analyzeStreamChunk", { win: WIN_S, current: i+1, total: chunks.length, progress: pct, done: fmtSec(processedSec), totalDuration: fmtSec(durationSec), eta: fmtSec(etaSec) }), true);
      await microYield();
      if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
    }
    const logitAvg = logitSum / Math.max(wSum, EPS);
    const pf = 1 / (1 + Math.exp(-logitAvg));
    const pm = 1 - pf;
    render(pf, pm);
    if (isAnalysisActive(token)){
      setStatus(t("status.analyzeStreamDone"));
    }
  } finally { stopHeartbeat(); }
}

const STREAM_STRATEGY_DEFAULT = Object.freeze({
  hop: STREAM_HOP_S,
  wins: [...STREAM_WIN_CAND],
  label: ""
});

function pickStreamStrategy(durationSec){
  if (!Number.isFinite(durationSec) || durationSec <= MAX_WHOLE_SEC){
    return STREAM_STRATEGY_DEFAULT;
  }

  const dedupeWins = (wins)=>{
    const seen = new Set();
    const out = [];
    for (const w of wins){
      const key = w.toFixed(2);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(w);
    }
    return out;
  };

  const gpuWins = dedupeWins([18, 12, ...STREAM_WIN_CAND, 4]);
  const gpuWinsLong = dedupeWins([24, 18, 12, ...STREAM_WIN_CAND, 4]);
  const wasmWins = dedupeWins([12, ...STREAM_WIN_CAND, 4]);

  if (currentDevice === "webgpu"){
    if (durationSec >= 600){
    return { hop: 6, wins: gpuWinsLong, label: t("status.strategyGpu6") };
    }
    return { hop: 4, wins: gpuWins, label: t("status.strategyGpu4") };
  }

  if (durationSec >= 420){
    return { hop: 4, wins: wasmWins, label: t("status.strategyCpu4") };
  }

  if (durationSec >= 240){
    return { hop: 3.5, wins: wasmWins, label: t("status.strategyCpu35") };
  }

  return STREAM_STRATEGY_DEFAULT;
}

// ===== 播放器與統計卡容器 =====
function ensurePlayerUI(){
  const container = document.querySelector("main.container");
  if (!container) return;
  if (document.getElementById("playBtn")) return;

  const wrap = document.createElement("div");
  wrap.className = "player";

  const btn = document.createElement("button");
  btn.id="playBtn"; btn.type="button"; btn.disabled=true;

  const hint = document.createElement("div");
  hint.className="hint";

  const hintPrefix = document.createElement("span");
  hintPrefix.className = "hint-prefix";
  const hintSpacer = document.createTextNode("");
  const replayButton = document.createElement("button");
  replayButton.type = "button";
  replayButton.className = "replay-button";
  const hintSuffix = document.createElement("span");
  hintSuffix.className = "hint-suffix";

  replayButton.addEventListener("click", (e)=>{
    e.preventDefault();
    btn.click();
  });

  hint.appendChild(hintPrefix);
  hint.appendChild(hintSpacer);
  hint.appendChild(replayButton);
  hint.appendChild(hintSuffix);

  const audio = document.createElement("audio");
  audio.id="playback"; audio.preload="metadata"; audio.style.display="none";

  wrap.appendChild(btn); wrap.appendChild(hint); wrap.appendChild(audio);

  const tipEl = container.querySelector(".callout");
  if (tipEl) container.insertBefore(wrap, tipEl); else container.appendChild(wrap);

  playBtn = btn;
  audioEl = audio;
  playerHintEl = hint;
  replayBtn = replayButton;
  replayHintPrefixEl = hintPrefix;
  replayHintSuffixEl = hintSuffix;
  replayHintSpacerNode = hintSpacer;
  updatePlayerCopy(false);
  updatePlaybackAvailability();

  playBtn.onclick = async ()=>{
    if (!audioEl.src) return;
    try{
      if (audioEl.paused){ await audioEl.play(); updatePlayerCopy(true); }
      else { audioEl.pause(); updatePlayerCopy(false); }
    }catch(e){ console.error("[audio play]", e); }
  };
  audioEl.onended = ()=>{ updatePlayerCopy(false); };
  audioEl.onpause = ()=>{ updatePlayerCopy(false); };
  audioEl.onplay = ()=>{ updatePlayerCopy(true); };

  // 統計卡容器（插在播放器區塊後）
  if (!document.getElementById("streamStats")){
    const stats = document.createElement("div");
    stats.id = "streamStats";
    stats.className = "insight";
    stats.innerHTML = "";
    wrap.insertAdjacentElement("afterend", stats);
  }
}
function setupExportButton(){
  if (!exportBtn) return;
  exportBtn.addEventListener("click", ()=>{
    try{
      const menu = document.getElementById("themeMenu");
      const gear = document.getElementById("settingsBtn");
      if (menu && !menu.hasAttribute("hidden")){
        menu.setAttribute("hidden", "");
        gear?.setAttribute("aria-expanded", "false");
      }
      exportBtn.blur?.();
      if (!latestAnalysisExport){
        setStatus(t("status.exportUnavailable"));
        return;
      }
      const payload = latestAnalysisExport;
      const json = JSON.stringify(payload, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      link.href = url;
      link.download = `vpa-analysis-${timestamp}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(()=> URL.revokeObjectURL(url), 5000);
      setStatus(t("status.exportReady"));
    }catch(err){
      console.error("[export]", err);
      setStatus(t("status.errorPrefix", { message: err?.message || "export failed" }));
    }
  });
}
function stopPlayback(){
  try{
    if (audioEl && !audioEl.paused){
      audioEl.pause();
      audioEl.currentTime = 0;
    }
  }catch(e){ console.error("[stopPlayback]", e); }
  updatePlayerCopy(false);
}
function setPlaybackSource(blob){
  try{
    if(!audioEl || !playBtn) return;
    if(lastAudioUrl){ try{ URL.revokeObjectURL(lastAudioUrl);}catch{} }
    lastAudioUrl = URL.createObjectURL(blob);
    audioEl.src = lastAudioUrl; audioEl.load();
    updatePlaybackAvailability();
    updatePlayerCopy(false);
  }catch(e){ console.error("[setPlaybackSource]", e); }
}

// ===== Render / Utils =====
function toMap(arr){
  const m={female:0, male:0};
  if (Array.isArray(arr)){
    for (const r of arr){
      if (r && typeof r.label==="string") m[r.label] = (typeof r.score==="number" ? r.score : 0);
    }
  }
  return m;
}
function render(pf, pm){
  // 儀表
  const barF = document.querySelector(".bar.female");
  const barM = document.querySelector(".bar.male");
  if (barF){ barF.style.setProperty("--p", pf??0); barF.setAttribute("aria-valuenow", Math.round(((pf??0)*100))); }
  if (barM){ barM.style.setProperty("--p", pm??0); barM.setAttribute("aria-valuenow", Math.round(((pm??0)*100))); }
  if (femaleVal) femaleVal.textContent = `${((pf??0)*100).toFixed(1)}%`;
  if (maleVal)   maleVal.textContent   = `${((pm??0)*100).toFixed(1)}%`;

  // 記錄供簡評使用
  lastPf = pf ?? 0; lastPm = pm ?? 0;
}
function startHeartbeat(fn){ stopHeartbeat(); heartbeatTimer=setInterval(()=>{ try{ fn(); }catch{} }, 1000); }
function stopHeartbeat(){ if (heartbeatTimer){ clearInterval(heartbeatTimer); heartbeatTimer=null; } }
function microYield(){ return new Promise(r=>setTimeout(r,0)); }

// ===== VAD（只「選段」） =====
function maybeApplyAdaptiveVAD(float32, sr){
  const dur = float32.length / sr;
  if (dur < VAD_MIN_APPLY_SEC) return null;

  const frame = Math.max(1, Math.floor(sr * (VAD_FRAME_MS/1000)));
  const hop   = Math.max(1, Math.floor(sr * (VAD_HOP_MS/1000)));
  const pad   = Math.max(0, Math.floor(sr * (VAD_PAD_MS/1000)));
  const minSeg= Math.max(1, Math.floor(sr * (VAD_MIN_SEG_MS/1000)));

  const energies = [];
  for (let s=0; s+frame <= float32.length; s+=hop){
    let acc=0; for (let i=0;i<frame;i++){ const v=float32[s+i]; acc += v*v; }
    energies.push(acc / frame);
  }
  if (energies.length < 5) return null;

  const thr = Math.max(1e-7, percentile(energies, 20) * 1.5);
  const voicedMask = energies.map(e => e > thr);
  smoothMask(voicedMask, 3);

  const segs = [];
  let i = 0;
  while (i < voicedMask.length){
    while (i < voicedMask.length && !voicedMask[i]) i++;
    if (i >= voicedMask.length) break;
    let j = i; while (j < voicedMask.length && voicedMask[j]) j++;
    const s0 = Math.max(0, i*hop - pad);
    const s1 = Math.min(float32.length, j*hop + frame + pad);
    if ((s1 - s0) >= minSeg) segs.push([s0, s1]);
    i = j;
  }
  if (!segs.length) return null;

  const kept = segs.reduce((a,[s0,s1]) => a + (s1 - s0), 0);
  const keptSec = kept / sr;
  const silenceRatio = 1 - (keptSec / dur);
  if (silenceRatio < VAD_SILENCE_RATIO_TO_APPLY || keptSec < VAD_MIN_VOICED_SEC) return null;

  const out = new Float32Array(kept);
  let offset = 0;
  for (const [s0,s1] of segs){ out.set(float32.subarray(s0, s1), offset); offset += (s1 - s0); }
  return { used:true, arr:out, keptSec, segs };
}
function percentile(arr, p){
  const a = arr.slice().sort((x,y)=>x-y);
  const idx = Math.min(a.length-1, Math.max(0, Math.round((p/100)*(a.length-1))));
  return a[idx];
}
function smoothMask(mask, k=3){
  // 把短 0 洞補成 1
  let count=0;
  for (let i=0;i<=mask.length;i++){
    if (i<mask.length && !mask[i]) count++;
    else { if (count>0 && count<k){ for (let j=i-count; j<i; j++) mask[j]=true; } count=0; }
  }
  // 把短 1 島補成 0
  count=0;
  for (let i=0;i<=mask.length;i++){
    if (i<mask.length && mask[i]) count++;
    else { if (count>0 && count<k){ for (let j=i-count; j<i; j++) mask[j]=false; } count=0; }
  }
}

// ===== Pitch Stream（ACF 音高 + 畫布） =====
function appendPitchSample(rawHz, meta = {}, opts = {}){
  const frameMs = Number.isFinite(opts?.dtMs) ? opts.dtMs : PS_INTERVAL_MS;
  const result = sharedAppendPitchSample(rawHz, meta, {
    state: pitchPostState,
    arrays: {
      raw: psHz,
      smooth: psHzSmooth,
      voiced: psVoiced,
      confidence: psConfidence,
    },
    getRange: getPitchDetectorRange,
    frameMs,
  });
  handleAutoRangeFrame(result, { dtMs: frameMs });
  return result;
}

function startPitchStream(userMediaStream){
  try{
    if (!pitchWrap || !pitchCanvas) return;
    psHz.length=0; psHzSmooth.length=0; psDb.length=0; psVoiced.length=0; psConfidence.length=0;
    resetPitchPostState(pitchPostState);
    psRealtimeNoiseTracker.reset();
    startAutoRangeSession({ preserveRange: false });

    maybeEnableAdvancedPitch("realtime", { allowRetry: true });

    const Ctx = window.AudioContext || window.webkitAudioContext;
    psCtx = new Ctx();
    psSrc = psCtx.createMediaStreamSource(userMediaStream);
    psProc = psCtx.createScriptProcessor(2048, 1, 1);
    const sampleRate = psCtx.sampleRate;

    setRealtimePanelsActive(true);

    let lastTick = 0;
    psProc.onaudioprocess = (ev)=>{
      const input = ev.inputBuffer.getChannelData(0);
      const rms = Math.sqrt(input.reduce((a,v)=>a+v*v,0) / Math.max(1,input.length));
      const db  = 20*Math.log10(Math.max(rms, 1e-6)) + 100; // 相對 dB
      const wasVoiced = psVoiced.length ? psVoiced[psVoiced.length-1] : false;
      let hz = null;
      let spectral = null;
      const gate = psRealtimeNoiseTracker.shouldDetect(db, wasVoiced);
      if (gate.detect){
        const candHz = runPitchDetection(input, sampleRate, { context: "realtime" });
        if (candHz != null){
          hz = candHz;
          spectral = estimateSpectralFeatures(input, sampleRate);
        } else {
          psRealtimeNoiseTracker.capture(db);
        }
      } else {
        psRealtimeNoiseTracker.capture(db);
      }
      const now = performance.now();
      if (now - lastTick >= PS_INTERVAL_MS){
        psDb.push(db);
        const { processed } = appendPitchSample(
          hz ?? null,
          { db, ambientDb: gate.ambient, spectral },
          { dtMs: PS_INTERVAL_MS }
        );
        const displayHz = Number.isFinite(processed)
          ? processed
          : (Number.isFinite(hz) ? hz : null);
        const maxN = Math.round(15000 / PS_INTERVAL_MS); // 保留約 15 秒
        if (psDb.length>maxN){
          psDb.shift(); psHz.shift(); psHzSmooth.shift(); psVoiced.shift(); psConfidence.shift();
        }
        lastTick = now;

        if (pitchNowEl){
          pitchNowEl.textContent = Number.isFinite(displayHz) ? `${displayHz.toFixed(1)}Hz` : "— Hz";
        }
        if (volNowEl)   volNowEl.textContent   = `${db.toFixed(1)} dB`;
        if (bandNowEl)  bandNowEl.textContent  = bandLabel(displayHz);
        updateRealtimeMonitor(spectral);
      }
    };

    psSrc.connect(psProc); psProc.connect(psCtx.destination);
    psRunning = true;
    startDrawLoop();
  }catch(e){ console.error("[startPitchStream]", e); }
}

function updateRealtimeMonitor(features){
  try{
    if (!formantWrap) return;
    if (!features){
      resetRealtimePanels();
      return;
    }
    const { f1, f2, f3, breathiness, tilt, energy } = features;
    if (f1NowEl) f1NowEl.textContent = Number.isFinite(f1) ? `${Math.round(f1)} Hz` : "— Hz";
    if (f2NowEl) f2NowEl.textContent = Number.isFinite(f2) ? `${Math.round(f2)} Hz` : "— Hz";
    if (f3NowEl) f3NowEl.textContent = Number.isFinite(f3) ? `${Math.round(f3)} Hz` : "— Hz";
    if (breathNowEl) breathNowEl.textContent = Number.isFinite(breathiness)
      ? `${Math.round(breathiness*100)}%`
      : "—";

    const desc = describeResonanceFromEnergy(energy);
    if (resonanceNowEl) resonanceNowEl.textContent = desc.label || "—";
    if (tiltNowEl) tiltNowEl.textContent = Number.isFinite(tilt)
      ? t("realtime.resonance.tiltValue", { value: fmt1(tilt) })
      : t("realtime.resonance.tiltPlaceholder");

    const pct = desc.pct || normalizeResonanceBands(energy);
    const chestPct = Math.max(0, Math.min(1, pct?.chest ?? 0));
    const maskPct  = Math.max(0, Math.min(1, pct?.mask ?? 0));
    const headPct  = Math.max(0, Math.min(1, pct?.head ?? 0));

    if (resBarChest){ resBarChest.style.flexGrow = Math.max(chestPct, 0.001); resBarChest.style.flexBasis = `${(chestPct*100).toFixed(1)}%`; }
    if (resBarMask){ resBarMask.style.flexGrow = Math.max(maskPct, 0.001); resBarMask.style.flexBasis = `${(maskPct*100).toFixed(1)}%`; }
    if (resBarHead){ resBarHead.style.flexGrow = Math.max(headPct, 0.001); resBarHead.style.flexBasis = `${(headPct*100).toFixed(1)}%`; }
    if (resValChest) resValChest.textContent = t("realtime.resonance.chest", { value: Math.round(chestPct*100) });
    if (resValMask)  resValMask.textContent  = t("realtime.resonance.mask", { value: Math.round(maskPct*100) });
    if (resValHead)  resValHead.textContent  = t("realtime.resonance.head", { value: Math.round(headPct*100) });
  }catch(e){ console.error("[updateRealtimeMonitor]", e); }
}
function stopPitchStream(){
  try{
    psRunning = false;
    if (psRAF){ cancelAnimationFrame(psRAF); psRAF=null; }
    psProc?.disconnect(); psSrc?.disconnect();
    psCtx?.close();
  }catch{} finally{
    psProc=null; psSrc=null; psCtx=null;
    setRealtimePanelsActive(false);
  }
}
function detectPitchACF(input, sr){
  // 簡化自相關（ACF）+ 降採樣到 ~16k；限制 50–600 Hz
  const ds = Math.max(1, Math.floor(sr / 16000));
  const N  = Math.floor(input.length / ds);
  if (N < 128) return null;
  if (acfBuffers.x.length < N){
    acfBuffers.x = new Float32Array(N);
  }
  const x = acfBuffers.x;
  let mean=0; for (let i=0;i<N;i++){ mean += input[i*ds]; }
  mean /= N;
  let energy=0; for (let i=0;i<N;i++){ const v=input[i*ds]-mean; x[i]=v; energy += v*v; }
  if (energy <= 1e-8) return null;

  const srDS = sr / ds;
  const range = getPitchDetectorRange();
  const lagMin = Math.floor(srDS / range.max);
  const lagMax = Math.floor(srDS / range.min);

  let bestLag=-1, bestR=0;
  for (let lag=lagMin; lag<=lagMax; lag++){
    let num=0, den0=0, den1=0;
    for (let i=0;i<N-lag;i++){
      const a=x[i], b=x[i+lag];
      num += a*b; den0 += a*a; den1 += b*b;
    }
    const r = num / Math.sqrt((den0*den1)+1e-10);
    if (r > bestR){ bestR=r; bestLag=lag; }
  }
  if (bestLag<0 || bestR<0.6) return null;
  const freq = srDS / bestLag;
  if (freq < range.min || freq > range.max) return null;
  return freq;
}
function detectPitchYinLite(input, sr){
  const ds = Math.max(1, Math.floor(sr / 16000));
  const N = Math.floor(input.length / ds);
  if (N < 128) return null;

  ensureYinCapacity(N);
  const x = yinBuffers.x;

  let mean = 0;
  for (let i=0;i<N;i++){ mean += input[i*ds]; }
  mean /= N;

  let energy = 0;
  let peak = 0;
  let absSum = 0;
  for (let i=0;i<N;i++){
    const v = input[i*ds] - mean;
    x[i] = v;
    energy += v*v;
    const abs = Math.abs(v);
    absSum += abs;
    if (abs > peak) peak = abs;
  }
  if (energy <= 1e-8) return null;

  const srDS = sr / ds;
  const range = getPitchDetectorRange();
  const tauMin = Math.max(1, Math.floor(srDS / range.max));
  const tauMax = Math.min(N - 3, Math.floor(srDS / range.min));
  if (tauMax <= tauMin) return null;

  const diff = yinBuffers.diff;
  const cmndf = yinBuffers.cmndf;
  diff[0] = 0;
  cmndf[0] = 1;

  const rms = Math.sqrt(energy / N);
  const avgAbs = absSum / N;
  const safeRms = Math.max(rms, 1e-8);
  const crest = peak > 1e-8 ? peak / safeRms : 0;
  const envelopeRatio = rms > 1e-8 ? avgAbs / safeRms : 0;
  const loudnessFactor = clamp01((rms - 0.0045) / 0.04);
  const crestFactor = clamp01((crest - 1.4) / 3.6);
  const envelopeFactor = clamp01((envelopeRatio - 0.72) / 0.28);
  const periodicity = Math.max(crestFactor, 0.6 * crestFactor + 0.4 * envelopeFactor);
  const threshold = Math.max(
    0.065,
    Math.min(0.19, 0.12 - 0.045 * periodicity + 0.03 * (1 - loudnessFactor))
  );
  let running = 0;
  let bestTau = -1;
  let bestVal = Infinity;
  let lastComputedTau = 0;
  let pendingExtraTau = -1;
  let foundBelowThreshold = false;

  for (let tau = 1; tau <= tauMax; tau++){
    const limit = N - tau;
    let sum = 0;
    for (let i=0;i<limit;i++){
      const delta = x[i] - x[i+tau];
      sum += delta * delta;
    }
    diff[tau] = sum;
    running += sum;
    const val = running ? (sum * tau) / running : 1;
    cmndf[tau] = val;
    lastComputedTau = tau;

    if (tau < tauMin) continue;
    if (val < bestVal){
      bestVal = val;
      bestTau = tau;
    }
    if (!foundBelowThreshold && val < threshold){
      foundBelowThreshold = true;
      pendingExtraTau = tau + 1 <= tauMax ? tau + 1 : -1;
      if (pendingExtraTau === -1){
        break;
      }
    } else if (foundBelowThreshold){
      if (pendingExtraTau === tau){
        pendingExtraTau = -1;
        break;
      }
      if (pendingExtraTau === -1){
        break;
      }
    }
  }

  if (pendingExtraTau > lastComputedTau && pendingExtraTau <= tauMax){
    const limit = N - pendingExtraTau;
    let sum = 0;
    for (let i=0;i<limit;i++){
      const delta = x[i] - x[i+pendingExtraTau];
      sum += delta * delta;
    }
    diff[pendingExtraTau] = sum;
    running += sum;
    const val = running ? (sum * pendingExtraTau) / running : 1;
    cmndf[pendingExtraTau] = val;
    lastComputedTau = pendingExtraTau;
    if (val < bestVal){
      bestVal = val;
      bestTau = pendingExtraTau;
    }
  }

  if (bestTau <= 0){
    for (let tau = tauMin; tau <= lastComputedTau; tau++){
      const val = cmndf[tau];
      if (val < bestVal){
        bestVal = val;
        bestTau = tau;
      }
    }
  }
  if (bestTau <= 0) return null;

  while (bestTau + 1 <= lastComputedTau && cmndf[bestTau + 1] <= cmndf[bestTau]){
    bestTau += 1;
  }

  let refinedTau = bestTau;
  if (bestTau > 1 && bestTau < lastComputedTau){
    const prev = cmndf[bestTau-1];
    const curr = cmndf[bestTau];
    const next = cmndf[bestTau+1];
    const denom = (next + prev - 2*curr);
    if (Number.isFinite(denom) && Math.abs(denom) > 1e-6){
      const offset = 0.5 * (prev - next) / denom;
      if (Number.isFinite(offset)) refinedTau = bestTau + Math.max(-1, Math.min(1, offset));
    }
  }

  const freq = srDS / refinedTau;
  if (freq < range.min || freq > range.max) return null;
  return freq;
}
function bandLabel(hz){
  if (!hz) return "—";
  if (hz < 85) return t("pitchBands.bandLow");
  if (hz < 165) return t("pitchBands.bandBlue");
  if (hz < 180) return t("pitchBands.bandNeutral");
  if (hz < 310) return t("pitchBands.bandPink");
  if (hz < 450) return t("pitchBands.bandHigh");
  if (hz <= PS_MAX_HZ) return t("pitchBands.bandFalsetto");
  return t("pitchBands.bandUnknown");
}
function startDrawLoop(){
  const ctx = pitchCanvas.getContext("2d");
  const DPR = Math.max(1, window.devicePixelRatio||1);
  function resize(){
    const r = pitchCanvas.getBoundingClientRect();
    pitchCanvas.width  = Math.max(600, Math.round(r.width*DPR));
    pitchCanvas.height = Math.round(r.height*DPR);
  }
  resize(); addEventListener("resize", resize);

  function yOf(hz){
    const h = pitchCanvas.height;
    const clamped = Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz));
    return h - ((clamped - PS_MIN_HZ) / (PS_MAX_HZ - PS_MIN_HZ)) * h;
  }
  function drawBands(){
    const styles = getComputedStyle(document.documentElement);
    const cGray = styles.getPropertyValue("--band-gray") || "#ddd";
    const cBlue = styles.getPropertyValue("--band-blue") || "#bfe7ff";
    const cPink = styles.getPropertyValue("--band-pink") || "#ffd1dc";
    const cLilac = styles.getPropertyValue("--band-lilac") || "#e2d5ff";
    const w=pitchCanvas.width, h=pitchCanvas.height;

    // 區帶：灰(50–85) / 藍(85–165) / 灰(165–180) / 粉(180–310) / 灰(310–450) / 淡紫(450–600)
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(85),  w, h - yOf(85));
    ctx.fillStyle = cBlue; ctx.fillRect(0, yOf(165), w, yOf(85)-yOf(165));
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(180), w, yOf(165)-yOf(180));
    ctx.fillStyle = cPink; ctx.fillRect(0, yOf(310), w, yOf(180)-yOf(310));
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(450), w, yOf(310)-yOf(450));
    ctx.fillStyle = cLilac; ctx.fillRect(0, 0,        w, yOf(450));

    // 網格線
    ctx.strokeStyle = "rgba(0,0,0,.08)"; ctx.lineWidth = 1*DPR;
    [50,85,165,180,310,450,PS_MAX_HZ].forEach(f=>{ const y=yOf(f); ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); });
  }

  function draw(){
    if (!psRunning && psHzSmooth.length===0){ psRAF = requestAnimationFrame(draw); return; }
    const w=pitchCanvas.width, h=pitchCanvas.height;
    ctx.clearRect(0,0,w,h);
    drawBands();

    const styles = getComputedStyle(document.documentElement);
    ctx.lineWidth = 2*DPR;
    ctx.strokeStyle = styles.getPropertyValue("--stream-ink") || "#222";

    // 往右跑：最右是最新
    const stepX = 3*DPR;
    const maxN  = Math.floor(w/stepX)-2;
    const n = Math.min(psHzSmooth.length, maxN);
    ctx.beginPath();
    for (let i=0;i<n;i++){
      const hz = psHzSmooth[psHzSmooth.length-n+i] ?? psHz[psHz.length-n+i];
      const x = w - (n-i)*stepX;
      if (hz==null) continue;
      const y = yOf(hz);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();

    const axisColor = styles.getPropertyValue("--stream-axis") || styles.getPropertyValue("--muted") || "rgba(0,0,0,.5)";
    const axisFont = (styles.getPropertyValue("--font-ui") || "sans-serif").trim() || "sans-serif";
    const axisFontSize = 11 * DPR;
    const axisTicks = [PS_MAX_HZ, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50];
    const tickLen = 6 * DPR;
    const leftX = 8 * DPR;
    const rightX = w - 8 * DPR;
    const labelHalf = axisFontSize * 0.6;

    ctx.save();
    ctx.fillStyle = axisColor;
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1 * DPR;
    ctx.font = `${axisFontSize}px ${axisFont}`;
    ctx.textBaseline = "middle";

    axisTicks.forEach((hz)=>{
      const y = yOf(hz);
      const textY = Math.min(Math.max(y, labelHalf), h - labelHalf);

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(tickLen, y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(w, y);
      ctx.lineTo(w - tickLen, y);
      ctx.stroke();

      ctx.textAlign = "left";
      ctx.fillText(`${hz} Hz`, leftX, textY);
      ctx.textAlign = "right";
      ctx.fillText(`${hz} Hz`, rightX, textY);
    });

    ctx.restore();

    psRAF = requestAnimationFrame(draw);
  }
  draw();
}

// ===== 離線抽樣（上傳檔用；錄音也會補） =====
function offlineExtractStreamMetrics(float32, sr, append=false){
  try{
    if(!append){
      psHz.length=0; psHzSmooth.length=0; psDb.length=0; psVoiced.length=0; psConfidence.length=0;
      resetOfflineFeatureStore();
      resetPitchPostState(pitchPostState);
      psOfflineNoiseTracker.reset();
      startAutoRangeSession({ preserveRange: false });
    }
    if (!append) maybeEnableAdvancedPitch("offline", { allowRetry: true });
    const step = Math.max(1, Math.floor((PS_INTERVAL_MS/1000)*sr));
    const frame = Math.min(Math.floor(0.08*sr), 8192); // ~80ms ACF 視窗
    offlineFeatureStore.frameSec = step / sr;
    for(let i=0;i+frame<=float32.length; i+=step){
      const seg = float32.subarray(i, i+frame);
      const db = 20*Math.log10(Math.max(rms(seg,0,seg.length), 1e-6)) + 100;
      const wasVoiced = psVoiced.length ? psVoiced[psVoiced.length-1] : false;
      let hz = null;
      let spectral = null;
      const gate = psOfflineNoiseTracker.shouldDetect(db, wasVoiced);
      if (gate.detect){
        const candHz = runPitchDetection(seg, sr, { context: "offline" });
        if (candHz != null){
          hz = candHz;
          spectral = estimateSpectralFeatures(seg, sr);
        } else {
          psOfflineNoiseTracker.capture(db);
        }
      } else {
        psOfflineNoiseTracker.capture(db);
      }
      psDb.push(db);
      const { processed, voiced, confidence } = appendPitchSample(
        hz ?? null,
        { db, ambientDb: gate.ambient, spectral },
        { dtMs: Math.max(1, Math.round((offlineFeatureStore.frameSec || (PS_INTERVAL_MS/1000)) * 1000)) }
      );
      offlineFeatureStore.pitchRaw.push(Number.isFinite(hz) ? hz : NaN);
      offlineFeatureStore.pitchProcessed.push(Number.isFinite(processed) ? processed : NaN);
      offlineFeatureStore.pitchConfidence.push(confidence);
      offlineFeatureStore.voiced.push(Boolean(voiced));
      offlineFeatureStore.db.push(db);
      if (spectral){
        offlineFeatureStore.formants.push([spectral.f1 ?? NaN, spectral.f2 ?? NaN, spectral.f3 ?? NaN]);
        offlineFeatureStore.tilt.push(spectral.tilt ?? NaN);
        offlineFeatureStore.breathiness.push(spectral.breathiness ?? NaN);
        offlineFeatureStore.energy.push([spectral.energy?.low ?? NaN, spectral.energy?.mid ?? NaN, spectral.energy?.high ?? NaN]);
        offlineFeatureStore.zcr.push(spectral.zcr ?? NaN);
      } else {
        offlineFeatureStore.formants.push([NaN,NaN,NaN]);
        offlineFeatureStore.tilt.push(NaN);
        offlineFeatureStore.breathiness.push(NaN);
        offlineFeatureStore.energy.push([NaN,NaN,NaN]);
        offlineFeatureStore.zcr.push(NaN);
      }
    }
  }catch(e){ console.error("[offlineExtractStreamMetrics]", e); }
}
function resetOfflineFeatureStore(){
  offlineFeatureStore.frameSec = 0;
  offlineFeatureStore.pitchRaw.length = 0;
  offlineFeatureStore.pitchProcessed.length = 0;
  offlineFeatureStore.pitchConfidence.length = 0;
  offlineFeatureStore.db.length = 0;
  offlineFeatureStore.voiced.length = 0;
  offlineFeatureStore.formants.length = 0;
  offlineFeatureStore.tilt.length = 0;
  offlineFeatureStore.breathiness.length = 0;
  offlineFeatureStore.energy.length = 0;
  offlineFeatureStore.zcr.length = 0;
}
function rms(arr, a, b){ let s=0; for(let i=a;i<b;i++){ const v=arr[i]; s += v*v; } return Math.sqrt(s/Math.max(1,b-a)); }

// ===== 統計卡（停止&分析完成後，含「簡評」與分歧提示） =====
function finishStreamStats(){
  try{
    const statsEl = document.getElementById("streamStats");
    if (!statsEl) return;

    const headerHTML = `
      <div class="insight-header">
        <span class="badge">${summaryText?.badge || t("summary.badge")}</span>
        <div class="tags"></div>
      </div>
    `;

    // 僅對有聲點統計；若沒有資料就清空
    const voicedHzRaw = [];
    for (let i=0;i<psHzSmooth.length;i++){
      const val = psHzSmooth[i];
      const conf = psConfidence[i] ?? 0;
      if (Number.isFinite(val) && conf >= CONFIDENCE_INCLUDE_THRESHOLD) voicedHzRaw.push(val);
    }
    const vols     = psDb.slice();
    if (!voicedHzRaw.length && !vols.length){
      statsEl.innerHTML="";
      setLatestAnalysisExport(null);
      return;
    }

    const stableVoicedHz = filterPitchForStats(voicedHzRaw);
    const voicedHz = stableVoicedHz.length ? stableVoicedHz : voicedHzRaw;
    const pitchStats = makeStats(voicedHz);
    const volStats   = makeStats(vols);
    const volsSorted = vols.slice().sort((a,b)=>a-b);
    const envDb      = percentileSorted(volsSorted, 10); // 10th 近似環境底噪
    const snr        = Number.isFinite(volStats.med) && Number.isFinite(envDb) ? (volStats.med - envDb) : NaN;

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
    if ((!eligibleMask || !maskInfo?.count) && Array.isArray(store.voiced) && store.voiced.length){
      eligibleMask = store.voiced.map(Boolean);
    }
    if (eligibleMask){
      const limit = Math.min(eligibleMask.length, psVoiced.length);
      for (let i=0;i<limit;i++){
        if (eligibleMask[i]) eligibleCount++;
      }
    }
    const voicedCount = eligibleCount;
    const frameSec = Number.isFinite(offlineFeatureStore.frameSec) && offlineFeatureStore.frameSec > 0
      ? offlineFeatureStore.frameSec
      : (PS_INTERVAL_MS/1000);
    const totalVoicedSec = voicedCount * frameSec;
    let stabilityKey = "steady";
    if (isFinite(spread)){
      const wideThreshold = Math.max(90, 60 * Math.sqrt(Math.max(totalVoicedSec, EPS) / 5));
      if (spread > wideThreshold) stabilityKey = "wide";
      else if (spread >= 40) stabilityKey = "moderate";
    }
    const stabilityLabel = isFinite(spread)
      ? (summaryText?.stability?.[stabilityKey] || t(`summary.stability.${stabilityKey}`))
      : "—";

    let snrKey = null;
    if (isFinite(snr)){
      snrKey = snr >= 20 ? "quiet" : snr >= 12 ? "ok" : "noisy";
    }
    const snrLabel = snrKey
      ? (summaryText?.snrTags?.[snrKey] || t(`summary.snrTags.${snrKey}`))
      : "—";

    let volSigmaKey = null;
    if (isFinite(volStats.sd)){
      volSigmaKey = volStats.sd < 6 ? "steady" : volStats.sd <= 12 ? "moderate" : "wide";
    }
    const volSigmaLabel = volSigmaKey
      ? (summaryText?.volumeVariation?.[volSigmaKey] || t(`summary.volumeVariation.${volSigmaKey}`))
      : "—";

    // 指標分歧（模型傾向 vs 音高常見區）
    const diverge = isDivergent(pitchStats.med, lastPf, lastPm);
    const divergeBadge = diverge
      ? (summaryText?.divergenceBadge || t("summary.divergenceBadge"))
      : "";

    // 取樣覆蓋率（錄音期間有聲點比例）
    const voicedRatio = psVoiced.length ? (voicedCount / psVoiced.length) : NaN;
    let voicedHintKey = null;
    if (!isFinite(voicedRatio) || voicedRatio < 0.25) voicedHintKey = "low";
    else if (voicedRatio < 0.5) voicedHintKey = "medium";

    const pfDisplay = (lastPf * 100).toFixed(1);
    const pmDisplay = (lastPm * 100).toFixed(1);
    const snrDisplay = isFinite(snr) ? `${fmt1(snr)} dB` : "—";
    const oneLinerLabel = summaryText?.oneLinerLabel || t("summary.oneLinerLabel");
    const oneLinerBody = summaryString("oneLinerTemplate", {
      pf: pfDisplay,
      pm: pmDisplay,
      median: fmt1(pitchStats.med),
      band,
      spread: fmt1(spread),
      stability: stabilityLabel,
      snr: snrDisplay,
      snrTag: snrLabel,
      diverge: divergeBadge,
    });
    const oneLiner = `
      <div class="summary-line">
        <strong>${oneLinerLabel}</strong>
        ${oneLinerBody}
      </div>
    `;

    const trendLabel = lastPf >= lastPm ? t("realtime.meter.feminine") : t("realtime.meter.masculine");
    const divergeNote = diverge
      ? t("summary.divergenceNoteHtml", { band, trend: trendLabel })
      : "";

    const envNote = (isFinite(snr) && snr < 12)
      ? (summaryText?.envNoteHtml || t("summary.envNoteHtml"))
      : "";

    const voicedNote = voicedHintKey
      ? `<p class="subline" style="margin:4px 0 0">${t(`summary.voicedHint.${voicedHintKey}`)}</p>`
      : "";

    const statsLabels = summaryText?.statsLabels || {};
    const statsHints = summaryText?.statsHints || {};
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
    const statsHTML = `
      <div class="stats-grid">
        ${statsRowsHtml}
      </div>
      <div class="kv" style="margin-top:10px"><div class="k">${envLabel}</div><div class="v">${fmt1(envDb)}dB</div></div>
      <p class="subline" style="margin:8px 0 0">${statsIntro}</p>
    `;
    const advSummary = computeAdvancedSummary();
    logPostProcessingDiagnostics(pitchPostState, {
      spread,
      intonationRange: advSummary?.intonation?.range ?? NaN,
    });
    const advancedHTML = renderAdvancedSummary(advSummary);

    statsEl.innerHTML = headerHTML + oneLiner + divergeNote + envNote + voicedNote + statsHTML + advancedHTML;

    // 標籤列
    const tags = statsEl.querySelector(".tags");
    const pitchTagLabel = summaryString("tags.pitchBand", { band });
    const noiseTagLabel = summaryString("tags.noise", { noise: fmt1(envDb) });
    let resonanceTagLabel = null;
    let speechRateTagLabel = null;
    let breathinessTagLabel = null;
    let brightnessTagLabel = null;
    if (advSummary){
      const resonanceTag = advSummary.resonanceDisplay || advSummary.resonanceLabel;
      if (resonanceTag) resonanceTagLabel = summaryString("tags.resonance", { label: resonanceTag });
      if (advSummary.speechRateLabel) speechRateTagLabel = summaryString("tags.speechRate", { label: advSummary.speechRateLabel });
      if (advSummary.breathinessLabel) breathinessTagLabel = summaryString("tags.breathiness", { label: advSummary.breathinessLabel });
      if (advSummary.brightnessLabel) brightnessTagLabel = summaryString("tags.brightness", { label: advSummary.brightnessLabel });
    }

    const summaryTags = {
      pitchBand: pitchTagLabel,
      noise: noiseTagLabel,
      resonance: resonanceTagLabel,
      speechRate: speechRateTagLabel,
      breathiness: breathinessTagLabel,
      brightness: brightnessTagLabel,
    };

    if (tags){
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

    if (advSummary?.intonation){
      const canvas = document.getElementById("intonationCanvas");
      if (canvas){
        if (Array.isArray(advSummary.intonation.points) && advSummary.intonation.points.length){
          drawIntonationCurve(canvas, advSummary.intonation);
        } else {
          const ctx = canvas.getContext("2d");
          if (ctx){ ctx.clearRect(0, 0, canvas.width, canvas.height); }
        }
      }
      setupIntonationLegend(advSummary.intonation);
    }

    const voicedHintLabel = voicedHintKey ? t(`summary.voicedHint.${voicedHintKey}`) : null;
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
          counters: PITCH_COUNTER_KEYS.reduce((acc, key)=>{
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
      },
      summary: {
        oneLinerLabel,
        oneLinerBodyHtml: oneLinerBody,
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
  }catch(e){ console.error("[finishStreamStats]", e); }
}
function setLatestAnalysisExport(payload){
  try{
    if (payload == null){
      latestAnalysisExport = null;
      if (typeof window !== "undefined") window.vpaLatestAnalysis = null;
      return;
    }
    const sanitized = sanitizeForJson(payload);
    latestAnalysisExport = sanitized;
    if (typeof window !== "undefined") window.vpaLatestAnalysis = sanitized;
  }catch(err){
    console.error("[export] capture failed", err);
    latestAnalysisExport = null;
  }
}
function cloneOfflineFeatureStore(){
  const frameSec = offlineFeatureStore.frameSec;
  const pitchRaw = Array.from(offlineFeatureStore.pitchRaw);
  const pitchProcessed = Array.from(offlineFeatureStore.pitchProcessed);
  const pitchConfidence = Array.from(offlineFeatureStore.pitchConfidence);
  const db = Array.from(offlineFeatureStore.db);
  const voiced = Array.from(offlineFeatureStore.voiced);
  const formants = offlineFeatureStore.formants.map((triple)=> Array.isArray(triple) ? triple.slice() : [NaN, NaN, NaN]);
  const tilt = Array.from(offlineFeatureStore.tilt);
  const breathiness = Array.from(offlineFeatureStore.breathiness);
  const energy = offlineFeatureStore.energy.map((triple)=> Array.isArray(triple) ? triple.slice() : [NaN, NaN, NaN]);
  const zcr = Array.from(offlineFeatureStore.zcr);
  const duration = Number.isFinite(frameSec) ? frameSec * pitchProcessed.length : NaN;
  return {
    frameSec,
    pitchRaw,
    pitchProcessed,
    pitchConfidence,
    db,
    voiced,
    formants,
    tilt,
    breathiness,
    energy,
    zcr,
    duration,
  };
}
function sanitizeForJson(value){
  if (Array.isArray(value)){
    return value.map((item)=> sanitizeForJson(item));
  }
  if (value && typeof value === "object"){
    const out = {};
    for (const [key, val] of Object.entries(value)){
      out[key] = sanitizeForJson(val);
    }
    return out;
  }
  if (typeof value === "number"){
    return Number.isFinite(value) ? value : null;
  }
  if (value === undefined) return null;
  return value;
}
function computeAdvancedSummary(){
  const store = offlineFeatureStore;
  const processedPitch = Array.isArray(store.pitchProcessed) ? store.pitchProcessed : (store.pitch || []);
  const rawPitch = Array.isArray(store.pitchRaw) ? store.pitchRaw : processedPitch;
  const pitchConfidence = Array.isArray(store.pitchConfidence) ? store.pitchConfidence : [];
  const n = processedPitch.length;
  const hopSec = store.frameSec || (PS_INTERVAL_MS/1000);
  const duration = hopSec * n;
  if (!n || duration < 0.5) return null;

  const maskInfo = buildEligibleFrameMask(store, {
    minConfidence: FORMANT_CONFIDENCE_THRESHOLD,
    maxGapFrames: FORMANT_MAX_GAP_FRAMES,
  });
  let mask = Array.isArray(maskInfo?.mask) && maskInfo.mask.length ? maskInfo.mask : null;
  let eligibleCount = Number.isFinite(maskInfo?.count) ? maskInfo.count : 0;
  if ((!mask || !eligibleCount) && Array.isArray(store.voiced) && store.voiced.length){
    mask = store.voiced.map(Boolean);
    eligibleCount = mask.reduce((acc, flag)=> acc + (flag ? 1 : 0), 0);
  }

  const formantArr = Array.isArray(store.formants) ? store.formants : [];
  const limit = mask ? Math.min(formantArr.length, mask.length) : formantArr.length;
  const f1Vals=[], f2Vals=[], f3Vals=[];
  for (let i=0;i<limit;i++){
    if (mask && !mask[i]) continue;
    const form = formantArr[i];
    if (!form) continue;
    const [f1,f2,f3] = form;
    if (Number.isFinite(f1)) f1Vals.push(f1);
    if (Number.isFinite(f2)) f2Vals.push(f2);
    if (Number.isFinite(f3)) f3Vals.push(f3);
  }
  const f1Stats = f1Vals.length ? makeStats(f1Vals) : null;
  const f2Stats = f2Vals.length ? makeStats(f2Vals) : null;
  const f3Stats = f3Vals.length ? makeStats(f3Vals) : null;
  const formantSummary = summarizeFormantTrends(store, {
    f1: f1Stats,
    f2: f2Stats,
    f3: f3Stats,
    f1Vals,
    f2Vals,
    f3Vals,
  }, {
    mask,
    eligibleCount,
  });

  const energyAvg = averageEnergy(store.energy, { mask, eligibleCount });
  const resonanceDesc = describeResonanceFromEnergy(energyAvg);
  const tiltAvg = averageFinite(store.tilt, mask);
  const tiltInfo = categorizeTilt(tiltAvg);
  const breathSummary = summarizeBreathiness(store.breathiness, { mask }, hopSec);
  const breathAvg = breathSummary.avg;
  const vols = Array.isArray(store.db) ? store.db.filter(Number.isFinite) : [];
  const volStats = vols.length ? makeStats(vols) : null;
  const envDb = vols.length ? percentileSorted(vols.slice().sort((a,b)=>a-b), 10) : NaN;
  const snrEstimate = Number.isFinite(volStats?.med) && Number.isFinite(envDb) ? (volStats.med - envDb) : NaN;
  const brightnessInfo = categorizeBrightness({ f3Stats, tilt: tiltAvg, breath: breathAvg });
  const breathInfo = categorizeBreathiness(breathAvg, {
    snr: snrEstimate,
    brightnessKey: brightnessInfo.key,
    tilt: tiltAvg,
  });
  const vowelInfo = analyzeVowelFocus(store, { mask, count: eligibleCount });
  const speech = analyzeSpeechRate(store);
  const liaison = analyzeConnectedSpeech(store.voiced, hopSec);
  const intonation = analyzeIntonation({
    processed: processedPitch,
    raw: rawPitch,
    confidence: pitchConfidence,
    voiced: store.voiced,
  }, hopSec);

  return {
    formants: formantSummary,
    resonanceLabel: resonanceDesc.label,
    resonanceDisplay: resonanceDesc.display || resonanceDesc.label,
    resonanceHint: resonanceDesc.hint,
    energyPct: resonanceDesc.pct,
    tiltAvg,
    tiltLabel: tiltInfo.label,
    tiltHint: tiltInfo.hint,
    brightnessLabel: brightnessInfo.label,
    brightnessHint: brightnessInfo.hint,
    brightnessKey: brightnessInfo.key,
    breathinessAvg: breathAvg,
    breathinessLabel: breathInfo.label,
    breathinessHint: breathInfo.hint,
    breathinessKey: breathInfo.key,
    speechRate: speech,
    speechRateLabel: speech.label,
    speechRateHint: speech.hint,
    vowelFocusRatio: vowelInfo.ratio,
    vowelLabel: vowelInfo.label,
    vowelHint: vowelInfo.hint,
    liaisonRatio: liaison.ratio,
    liaisonLabel: liaison.label,
    liaisonHint: liaison.hint,
    intonation,
    snrEstimate,
  };
}

function renderAdvancedSummary(summary){
  if (!summary){
    return `
      <div class="advanced-section">
        <div class="note">${t("analysis.advanced.insufficient")}</div>
      </div>
    `;
  }
  const chestPct = Math.round((summary.energyPct?.chest ?? 0.33)*100);
  const maskPct  = Math.round((summary.energyPct?.mask ?? 0.33)*100);
  const headPct  = Math.round((summary.energyPct?.head ?? 0.34)*100);
  const formants = summary.formants || {};
  const f1Trend = formants.f1 || {};
  const f2Trend = formants.f2 || {};
  const f3Trend = formants.f3 || {};
  const f1Value = f1Trend.display || fmt1(f1Trend.median);
  const f2Value = f2Trend.display || fmt1(f2Trend.median);
  const f3Value = f3Trend.display || fmt1(f3Trend.median);
  const f1Hint = f1Trend.hint || makeFormantHint("F1", NaN, 180, 350);
  const f2Hint = f2Trend.hint || makeFormantHint("F2", NaN, 1600, 2500);
  const f3Hint = f3Trend.hint || makeFormantHint("F3", NaN, 2500, 3200);
  const brightnessDisplay = summary.brightnessLabel || "—";
  const brightnessHint = summary.brightnessHint || "";
  const speechRateDisplay = Number.isFinite(summary.speechRate?.syllPerSec)
    ? summaryString("speechRateDisplay", { value: fmt1(summary.speechRate.syllPerSec) })
    : "—";
  const speechWpmDisplay = Number.isFinite(summary.speechRate?.wordsPerMin)
    ? summaryString("speechRateWpm", { value: Math.round(summary.speechRate.wordsPerMin) })
    : "";
  const percentSuffix = (value) => summaryString("percentSuffix", { value });
  const liaisonDisplay = Number.isFinite(summary.liaisonRatio)
    ? percentSuffix(Math.round(summary.liaisonRatio*100))
    : "";
  const vowelDisplay = summary.vowelLabel + (Number.isFinite(summary.vowelFocusRatio) ? percentSuffix(Math.round(summary.vowelFocusRatio*100)) : "");
  const breathDisplay = summary.breathinessLabel + (Number.isFinite(summary.breathinessAvg) ? percentSuffix(Math.round(summary.breathinessAvg*100)) : "");
  const rangeDisplay = Number.isFinite(summary.intonation?.range)
    ? summaryString("rangeDisplayHz", { value: fmt1(summary.intonation.range) })
    : "—";
  const intonationLabel = summary.intonation?.slopeLabel || "—";
  const intonationHint = summary.intonation?.hint || t("summary.summaryHint");
  const rangeHint = summary.intonation?.rangeHint || "";

  const advancedCopy = summaryText?.advanced || {};
  const formantTitle = advancedCopy.formantTitle || t("summary.advanced.formantTitle");
  const intonationTitle = advancedCopy.intonationTitle || t("summary.advanced.intonationTitle");
  const vowelTitle = advancedCopy.vowelBreathTitle || t("summary.advanced.vowelBreathTitle");
  const formantCards = advancedCopy.formantCards || {};
  const intonationCards = advancedCopy.intonationCards || {};
  const vowelCards = advancedCopy.vowelCards || {};
  const canvasAria = advancedCopy.canvasAria || t("summary.advanced.canvasAria");
  const canvasHint = advancedCopy.intonationCanvasHint || t("summary.advanced.intonationCanvasHint");
  const legendCopy = advancedCopy.intonationLegend || {};
  const legendLine = legendCopy.line || t("summary.advanced.intonationLegend.line");
  const legendDots = legendCopy.dots || t("summary.advanced.intonationLegend.dots");
  const legendShade = legendCopy.shade || t("summary.advanced.intonationLegend.shade");
  const legendShow = legendCopy.show || t("summary.advanced.intonationLegend.show");
  const legendHide = legendCopy.hide || t("summary.advanced.intonationLegend.hide");
  const hasRawPoints = Array.isArray(summary.intonation?.rawPoints) && summary.intonation.rawPoints.length > 0;
  const toggleDisabledAttr = hasRawPoints ? "" : " disabled aria-disabled=\"true\"";
  const toggleStateLabel = showIntonationRawPoints ? legendHide : legendShow;

  const liaisonValue = summary.liaisonLabel + (liaisonDisplay ? liaisonDisplay : "");
  const speechRateHint = speechWpmDisplay ? `${summary.speechRateHint} ${speechWpmDisplay}` : summary.speechRateHint;
  const resonanceTail = summary.tiltLabel && summary.tiltLabel !== "—"
    ? summaryString("resonanceTiltTail", { label: summary.tiltLabel })
    : "";
  const resonanceHintLine = [summary.resonanceHint, resonanceTail].filter(Boolean).join(" ").trim();

  return `
    <div class="advanced-section">
      <h3 class="adv-title">${formantTitle}</h3>
      <div class="advanced-grid advanced-grid--four">
        <div class="adv-card"><div class="k">${formantCards.f1 || t("summary.advanced.formantCards.f1")}</div><div class="v">${f1Value}</div><div class="hint">${f1Hint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.f2 || t("summary.advanced.formantCards.f2")}</div><div class="v">${f2Value}</div><div class="hint">${f2Hint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.f3 || t("summary.advanced.formantCards.f3")}</div><div class="v">${f3Value}</div><div class="hint">${f3Hint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.brightness || t("summary.advanced.formantCards.brightness")}</div><div class="v">${brightnessDisplay}</div><div class="hint">${brightnessHint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.tilt || t("summary.advanced.formantCards.tilt")}</div><div class="v">${fmt1(summary.tiltAvg)} dB</div><div class="hint">${summary.tiltHint}</div></div>
      </div>
      <div class="resonance-summary">
        <div class="resonance-bar resonance-bar--summary">
          <div class="res-part chest" style="flex:${Math.max(summary.energyPct?.chest ?? 0.001, 0.001)}">${chestPct}%</div>
          <div class="res-part mask" style="flex:${Math.max(summary.energyPct?.mask ?? 0.001, 0.001)}">${maskPct}%</div>
          <div class="res-part head" style="flex:${Math.max(summary.energyPct?.head ?? 0.001, 0.001)}">${headPct}%</div>
        </div>
        <p class="subline">${resonanceHintLine}</p>
      </div>
    </div>
    <div class="advanced-section">
      <h3 class="adv-title">${intonationTitle}</h3>
      <canvas id="intonationCanvas" width="520" height="140" aria-label="${canvasAria}" title="${escapeAttr(canvasHint)}"></canvas>
      <div class="intonation-legend" id="intonationLegend" data-show-raw="${showIntonationRawPoints ? "true" : "false"}" data-has-raw="${hasRawPoints ? "true" : "false"}">
        <div class="legend-item"><span class="legend-swatch legend-swatch--line"></span><span>${legendLine}</span></div>
        <button type="button" class="legend-item legend-toggle" id="intonationRawToggle" aria-pressed="${showIntonationRawPoints ? "true" : "false"}"${toggleDisabledAttr}>
          <span class="legend-swatch legend-swatch--dots"></span>
          <span>${legendDots}</span>
          <span class="legend-toggle-state">${toggleStateLabel}</span>
        </button>
        <div class="legend-item legend-item--shade"><span class="legend-swatch legend-swatch--shade"></span><span>${legendShade}</span></div>
      </div>
      <div class="advanced-grid advanced-grid--four">
        <div class="adv-card"><div class="k">${intonationCards.trend || t("summary.advanced.intonationCards.trend")}</div><div class="v">${intonationLabel}</div><div class="hint">${intonationHint}</div></div>
        <div class="adv-card"><div class="k">${intonationCards.range || t("summary.advanced.intonationCards.range")}</div><div class="v">${rangeDisplay}</div><div class="hint">${rangeHint}</div></div>
        <div class="adv-card"><div class="k">${intonationCards.speechRate || t("summary.advanced.intonationCards.speechRate")}</div><div class="v">${speechRateDisplay}</div><div class="hint">${speechRateHint}</div></div>
        <div class="adv-card"><div class="k">${intonationCards.liaison || t("summary.advanced.intonationCards.liaison")}</div><div class="v">${liaisonValue || "—"}</div><div class="hint">${summary.liaisonHint}</div></div>
      </div>
    </div>
    <div class="advanced-section">
      <h3 class="adv-title">${vowelTitle}</h3>
      <div class="advanced-grid advanced-grid--three">
        <div class="adv-card"><div class="k">${vowelCards.focus || t("summary.advanced.vowelCards.focus")}</div><div class="v">${vowelDisplay}</div><div class="hint">${summary.vowelHint}</div></div>
        <div class="adv-card"><div class="k">${vowelCards.breathiness || t("summary.advanced.vowelCards.breathiness")}</div><div class="v">${breathDisplay}</div><div class="hint">${summary.breathinessHint}</div></div>
        <div class="adv-card"><div class="k">${vowelCards.tilt || t("summary.advanced.vowelCards.tilt")}</div><div class="v">${summary.tiltLabel}</div><div class="hint">${summary.tiltHint}</div></div>
      </div>
    </div>
  `;
}

function averageFinite(arr, mask){
  const values = Array.isArray(arr) ? arr : [];
  const maskArray = Array.isArray(mask?.mask) ? mask.mask : (Array.isArray(mask) ? mask : null);
  const limit = maskArray ? Math.min(values.length, maskArray.length) : values.length;
  let sum = 0;
  let count = 0;
  for (let i=0;i<limit;i++){
    if (maskArray && !maskArray[i]) continue;
    const val = values[i];
    if (Number.isFinite(val)){
      sum += val;
      count++;
    }
  }
  if (!count) return NaN;
  return sum / count;
}

function averageEnergy(arr, info = {}){
  if (!Array.isArray(arr) || !arr.length) return { low:0, mid:0, high:0, total:0, coverage:0, validCount:0 };
  const mask = Array.isArray(info.mask) ? info.mask : (Array.isArray(info.mask?.mask) ? info.mask.mask : null);
  const eligible = Number.isFinite(info.eligibleCount) ? info.eligibleCount : (mask ? mask.reduce((acc, flag)=> acc + (flag ? 1 : 0), 0) : arr.length);
  const limit = mask ? Math.min(arr.length, mask.length) : arr.length;
  let low=0, mid=0, high=0, valid=0, considered=0;
  for (let i=0;i<limit;i++){
    if (mask && !mask[i]) continue;
    considered++;
    const v = arr[i];
    if (!Array.isArray(v)) continue;
    const [l,m,h] = v;
    if (!Number.isFinite(l) && !Number.isFinite(m) && !Number.isFinite(h)) continue;
    low += Number.isFinite(l) ? l : 0;
    mid += Number.isFinite(m) ? m : 0;
    high += Number.isFinite(h) ? h : 0;
    valid++;
  }
  if (!valid){
    const baseCoverage = eligible > 0 ? (considered / eligible) : 0;
    return { low:0, mid:0, high:0, total:0, coverage: Math.max(0, Math.min(1, baseCoverage)), validCount:0 };
  }
  const avgLow = low / valid;
  const avgMid = mid / valid;
  const avgHigh = high / valid;
  const baseCoverage = eligible > 0 ? (valid / eligible) : 0;
  return {
    low: avgLow,
    mid: avgMid,
    high: avgHigh,
    total: avgLow + avgMid + avgHigh,
    coverage: Math.max(0, Math.min(1, baseCoverage)),
    validCount: valid,
  };
}

const RESONANCE_PRIOR_WEIGHT = 6;
const RESONANCE_DOMINANCE_DELTA = 0.26;
const RESONANCE_HEAD_MIN = 0.40;
const RESONANCE_MASK_MIN = 0.38;
const RESONANCE_CHEST_MIN = 0.52;
const RESONANCE_MIN_SAMPLES = 18;
const RESONANCE_MIN_COVERAGE = 0.2;
const RESONANCE_COVERAGE_GOOD = 0.35;

const FORMANT_MIN_SAMPLES = 24;
const FORMANT_MIN_COVERAGE = 0.18;
const FORMANT_GOOD_COVERAGE = 0.32;
const FORMANT_MIN_SPREAD = 70;
const FORMANT_CONFIDENCE_THRESHOLD = CONFIDENCE_INCLUDE_THRESHOLD;
const FORMANT_MAX_GAP_FRAMES = 8;

const BREATHINESS_EMA_TAU_SEC = 0.2;
const BRIGHTNESS_F3_LOW = 2400;
const BRIGHTNESS_F3_HIGH = 3400;
const BRIGHTNESS_WARM_Z = -0.7;
const BRIGHTNESS_SPARKLE_Z = 0.4;
const BRIGHTNESS_SWEET_Z = 1.0;
const BRIGHTNESS_TILT_SHARP = -1.5;
const BRIGHTNESS_BREATH_THRESHOLD = 0.45;

function summarizeBreathiness(arr, info = {}, hopSec){
  if (!Array.isArray(arr) || !arr.length) return { avg: NaN, count: 0 };
  const mask = Array.isArray(info.mask) ? info.mask : (Array.isArray(info.mask?.mask) ? info.mask.mask : null);
  const limit = mask ? Math.min(arr.length, mask.length) : arr.length;
  const step = Number.isFinite(hopSec) && hopSec > 0 ? hopSec : (PS_INTERVAL_MS/1000);
  const tau = Math.max(0.08, BREATHINESS_EMA_TAU_SEC);
  const alpha = 1 - Math.exp(-step / tau);
  let ema = null;
  let sum = 0;
  let count = 0;
  for (let i=0;i<limit;i++){
    if (mask && !mask[i]) continue;
    let val = arr[i];
    if (!Number.isFinite(val)) continue;
    val = Math.max(0, Math.min(1, val));
    if (ema == null) ema = val;
    else ema = ema + alpha * (val - ema);
    sum += ema;
    count++;
  }
  if (!count) return { avg: NaN, count: 0 };
  const avg = Math.max(0, Math.min(1, sum / count));
  return { avg, count };
}

function buildEligibleFrameMask(store, { minConfidence = FORMANT_CONFIDENCE_THRESHOLD, maxGapFrames = FORMANT_MAX_GAP_FRAMES } = {}){
  const voiced = Array.isArray(store.voiced) ? store.voiced : [];
  const confidence = Array.isArray(store.pitchConfidence) ? store.pitchConfidence : [];
  const n = Math.min(voiced.length, confidence.length);
  let mask = new Array(n).fill(false);
  for (let i=0;i<n;i++){
    const conf = confidence[i] ?? 0;
    mask[i] = Boolean(voiced[i]) && conf >= minConfidence;
  }
  if (maxGapFrames > 0 && mask.length){
    let gapStart = -1;
    for (let i=0;i<=n;i++){
      const flag = i < n ? mask[i] : true;
      if (!flag){
        if (gapStart < 0) gapStart = i;
      } else if (gapStart >= 0){
        const gapLen = i - gapStart;
        const prev = gapStart > 0 ? mask[gapStart-1] : false;
        const next = i < n ? mask[i] : false;
        if (prev && next && gapLen <= maxGapFrames){
          for (let j=gapStart;j<i;j++) mask[j] = true;
        }
        gapStart = -1;
      }
    }
    const dilation = Math.max(1, Math.floor(maxGapFrames / 2));
    if (dilation > 0){
      const expanded = mask.slice();
      for (let i=0;i<n;i++){
        if (!mask[i]) continue;
        for (let d=1; d<=dilation; d++){
          if (i-d >= 0) expanded[i-d] = true;
          if (i+d < n) expanded[i+d] = true;
        }
      }
      mask = expanded;
    }
  }
  const count = mask.reduce((acc, flag)=> acc + (flag ? 1 : 0), 0);
  return { mask, count, minConfidence, maxGapFrames };
}

function categorizeBrightness({ f3Stats, tilt, breath } = {}){
  const brightnessText = analysisText?.brightness;
  if (!Number.isFinite(f3Stats?.med)){
    const insufficient = brightnessText?.insufficient;
    return {
      label: insufficient?.label || t("analysis.brightness.insufficient.label"),
      hint: insufficient?.hint || t("analysis.brightness.insufficient.hint"),
      key: "insufficient",
    };
  }
  const med = f3Stats.med;
  const center = (BRIGHTNESS_F3_LOW + BRIGHTNESS_F3_HIGH) / 2;
  const span = Math.max(1, (BRIGHTNESS_F3_HIGH - BRIGHTNESS_F3_LOW) / 2);
  const z = (med - center) / span;
  const tiltVal = Number.isFinite(tilt) ? tilt : NaN;
  const breathVal = Number.isFinite(breath) ? breath : NaN;
  let key = "balanced";
  if (z <= BRIGHTNESS_WARM_Z) key = "warm";
  else if (z >= BRIGHTNESS_SWEET_Z){
    const needsRelax = (Number.isFinite(breathVal) && breathVal > BRIGHTNESS_BREATH_THRESHOLD)
      || (Number.isFinite(tiltVal) && tiltVal < BRIGHTNESS_TILT_SHARP);
    key = needsRelax ? "sharp" : "sweet";
  } else if (z >= BRIGHTNESS_SPARKLE_Z){
    key = "sparkle";
  }
  const entry = brightnessText?.[key];
  const label = entry?.label || t(`analysis.brightness.${key}.label`);
  const hint = entry?.hint || t(`analysis.brightness.${key}.hint`);
  return { label, hint, key, zScore: z };
}

function normalizeResonanceBands(energy){
  if (!energy) return { chest: NaN, mask: NaN, head: NaN, total: 0, coverage: 0 };
  const low = Math.max(0, Number.isFinite(energy.low) ? energy.low : 0);
  const mid = Math.max(0, Number.isFinite(energy.mid) ? energy.mid : 0);
  const high = Math.max(0, Number.isFinite(energy.high) ? energy.high : 0);
  const total = low + mid + high;
  if (!Number.isFinite(total) || total <= EPS){
    return { chest: NaN, mask: NaN, head: NaN, total: 0, coverage: Number.isFinite(energy.coverage) ? energy.coverage : 0 };
  }
  const prior = total * RESONANCE_PRIOR_WEIGHT;
  const perBand = prior / 3;
  const denom = total + prior;
  return {
    chest: ((low + perBand) / denom),
    mask:  ((mid + perBand) / denom),
    head:  ((high + perBand) / denom),
    total,
    coverage: Number.isFinite(energy.coverage) ? energy.coverage : 0,
  };
}

function describeResonanceFromEnergy(energy){
  const pctFallback = { chest: 1 / 3, mask: 1 / 3, head: 1 / 3 };
  const normalized = normalizeResonanceBands(energy);
  const { chest, mask, head, total } = normalized;
  const hasAggregate = energy && (Number.isFinite(energy.coverage) || Number.isFinite(energy.validCount));
  const coverage = hasAggregate ? (Number.isFinite(energy.coverage) ? energy.coverage : 0) : NaN;
  const validCount = hasAggregate ? (Number.isFinite(energy.validCount) ? energy.validCount : 0) : 0;

  const insufficientEntry = () => {
    const insufficient = analysisText?.resonanceBalance?.insufficient;
    return {
      label: insufficient?.label || t("analysis.resonanceBalance.insufficient.label"),
      hint: insufficient?.hint || t("analysis.resonanceBalance.insufficient.hint"),
      pct: pctFallback,
      total: 0,
      coverage: hasAggregate ? coverage : NaN,
      display: insufficient?.label || t("analysis.resonanceBalance.insufficient.label"),
    };
  };

  if (hasAggregate){
    if (!Number.isFinite(coverage) || coverage < RESONANCE_MIN_COVERAGE || validCount < RESONANCE_MIN_SAMPLES){
      const base = insufficientEntry();
      const coverageNote = t("analysis.resonanceBalance.coverageLowHint", { value: Math.round((coverage || 0) * 100) });
      if (coverageNote) base.hint = `${base.hint} ${coverageNote}`.trim();
      if (Number.isFinite(coverage)){
        const suffix = t("analysis.resonanceBalance.coverageLowSuffix", { value: Math.round(coverage * 100) });
        base.display = suffix ? `${base.label}${suffix}` : base.label;
      }
      return base;
    }
  }

  if (!Number.isFinite(chest) || !Number.isFinite(mask) || !Number.isFinite(head)){
    return insufficientEntry();
  }

  const maxVal = Math.max(chest, mask, head);
  const minVal = Math.min(chest, mask, head);
  const span = maxVal - minVal;

  let key = "balanced";
  if (span >= 0.10){
    if (chest >= 0.45 && chest === maxVal) key = "chestHeavy";
    else if (mask >= 0.45 && mask === maxVal) key = "maskLead";
    else if (head >= 0.45 && head === maxVal) key = "headBright";
  }

  const entry = analysisText?.resonanceBalance?.[key];
  const label = entry?.label || t(`analysis.resonanceBalance.${key}.label`);
  let hint = entry?.hint || t(`analysis.resonanceBalance.${key}.hint`);
  let display = label;
  if (hasAggregate && Number.isFinite(coverage)){
    const coverageKey = coverage < RESONANCE_COVERAGE_GOOD ? "coverageLowHint" : "coverageHint";
    const hintNote = t(`analysis.resonanceBalance.${coverageKey}`, { value: Math.round(coverage * 100) });
    if (hintNote) hint = `${hint} ${hintNote}`.trim();
    let suffix = coverage < RESONANCE_COVERAGE_GOOD
      ? t("analysis.resonanceBalance.referenceSuffix", { value: Math.round(coverage * 100) })
      : t("analysis.resonanceBalance.coverageSuffix", { value: Math.round(coverage * 100) });
    if (!suffix && coverage < RESONANCE_COVERAGE_GOOD){
      suffix = t("analysis.resonanceBalance.referenceOnly");
    }
    if (suffix) display = `${label}${suffix}`;
  }
  return {
    label,
    display,
    hint,
    pct:{ chest, mask, head },
    total,
    coverage: hasAggregate ? coverage : NaN,
  };
}

function categorizeTilt(tilt){
  if (!Number.isFinite(tilt)) {
    const insufficient = analysisText?.tilt?.insufficient;
    return {
      label: insufficient?.label || t("analysis.tilt.insufficient.label"),
      hint: insufficient?.hint || t("analysis.tilt.insufficient.hint"),
    };
  }
  let key = "bright";
  if (tilt >= 7.5) key = "warm";
  else if (tilt >= 4.5) key = "gentleWarm";
  else if (tilt >= -1) key = "balanced";
  const entry = analysisText?.tilt?.[key];
  return {
    label: entry?.label || t(`analysis.tilt.${key}.label`),
    hint: entry?.hint || t(`analysis.tilt.${key}.hint`),
  };
}

function categorizeBreathiness(val, ctx = {}){
  if (!Number.isFinite(val)) {
    const insufficient = analysisText?.breathiness?.insufficient;
    return {
      label: insufficient?.label || t("analysis.breathiness.insufficient.label"),
      hint: insufficient?.hint || t("analysis.breathiness.insufficient.hint"),
    };
  }
  const snr = Number.isFinite(ctx.snr) ? ctx.snr : NaN;
  const brightnessKey = ctx.brightnessKey;
  const tilt = Number.isFinite(ctx.tilt) ? ctx.tilt : NaN;
  const styleEligible = Number.isFinite(snr) ? snr > 20 : false;
  const needsRelax = brightnessKey === "sharp" || (Number.isFinite(tilt) && tilt < BRIGHTNESS_TILT_SHARP);

  let key = "airy";
  if (val < 0.08) key = "dense";
  else if (val <= 0.18) key = "balanced";
  else if (val <= 0.28) key = "airy";
  else if (val <= 0.45) key = styleEligible ? "style" : "airy";
  else {
    if (needsRelax) key = "tooAiry";
    else key = styleEligible ? "style" : "airy";
  }

  const entry = analysisText?.breathiness?.[key];
  return {
    label: entry?.label || t(`analysis.breathiness.${key}.label`),
    hint: entry?.hint || t(`analysis.breathiness.${key}.hint`),
    key,
  };
}

function makeFormantHint(label, value, low, high){
  const rangeLabels = analysisText?.formant?.rangeLabels || {};
  const labelName = rangeLabels[label] || t(`analysis.formant.rangeLabels.${label}`);
  const labelWithName = `${label}（${labelName || label}）`;
  if (!Number.isFinite(value)) {
    return t("analysis.formant.insufficient", { label: labelWithName });
  }
  const lowHint = analysisText?.formant?.low?.[label] || t(`analysis.formant.low.${label}`);
  const highHint = analysisText?.formant?.high?.[label] || t(`analysis.formant.high.${label}`);
  if (value < low) {
    const msg = t("analysis.formant.lowMessage", { label: labelWithName, hint: lowHint });
    return msg || `${labelWithName} ${lowHint}`;
  }
  if (value > high) {
    const msg = t("analysis.formant.highMessage", { label: labelWithName, hint: highHint });
    return msg || `${labelWithName} ${highHint}`;
  }
  return t("analysis.formant.inRange", { label: labelWithName });
}

function summarizeFormantTrends(store, statsBundle, options = {}){
  let eligibleCount = Number.isFinite(options?.eligibleCount) ? options.eligibleCount : null;
  if ((eligibleCount == null || eligibleCount <= 0) && Array.isArray(store.voiced)){
    eligibleCount = store.voiced.reduce((acc, flag)=> acc + (flag ? 1 : 0), 0);
  }
  const hasAggregate = Number.isFinite(eligibleCount) && eligibleCount > 0;

  const makeEntry = (label, stats, values, low, high)=>{
    const sampleCount = values.length;
    const coverageRaw = hasAggregate ? (sampleCount / eligibleCount) : 0;
    const coverage = hasAggregate ? Math.max(0, Math.min(1, coverageRaw)) : NaN;
    const spread = stats ? (stats.p95 - stats.p05) : NaN;
    const reliable = hasAggregate
      && sampleCount >= FORMANT_MIN_SAMPLES
      && coverageRaw >= FORMANT_MIN_COVERAGE
      && Number.isFinite(spread)
      && spread >= FORMANT_MIN_SPREAD
      && Number.isFinite(stats?.med);

    const trendKey = reliable
      ? (stats.med < low ? "low" : (stats.med > high ? "high" : "inRange"))
      : "insufficient";

    const baseHint = reliable
      ? makeFormantHint(label, stats.med, low, high)
      : makeFormantHint(label, NaN, low, high);

    const extraHints = [];
    if (hasAggregate && sampleCount < FORMANT_MIN_SAMPLES){
      const msg = t("analysis.formant.moreSamplesHint");
      if (msg) extraHints.push(msg);
    }
    if (hasAggregate && coverageRaw < FORMANT_MIN_COVERAGE){
      const msg = t("analysis.formant.coverageLowHint", { value: Math.round(Math.max(0, Math.min(1, coverageRaw)) * 100) });
      if (msg) extraHints.push(msg);
    }

    const hint = [baseHint, ...extraHints].filter(Boolean).join(" ").trim();
    const display = buildFormantTrendDisplay(trendKey, coverage, hasAggregate);

    return {
      trend: trendKey,
      median: reliable ? stats.med : NaN,
      display,
      hint,
      coverage: hasAggregate ? coverage : NaN,
      samples: sampleCount,
    };
  };

  return {
    f1: makeEntry("F1", statsBundle.f1, statsBundle.f1Vals || [], 170, 420),
    f2: makeEntry("F2", statsBundle.f2, statsBundle.f2Vals || [], 1450, 2750),
    f3: makeEntry("F3", statsBundle.f3, statsBundle.f3Vals || [], 2400, 3400),
  };
}

function buildFormantTrendDisplay(trendKey, coverage, hasAggregate){
  const trendLabels = analysisText?.formant?.trendLabels;
  const baseRaw = (trendLabels && trendLabels[trendKey]) || t(`analysis.formant.trendLabels.${trendKey}`);
  const base = baseRaw || "—";
  if (!hasAggregate || !Number.isFinite(coverage) || coverage <= 0){
    return base;
  }
  const clamped = Math.max(0, Math.min(1, coverage));
  const key = clamped < FORMANT_GOOD_COVERAGE ? "coverageLowSuffix" : "coverageSuffix";
  const suffix = t(`analysis.formant.${key}`, { value: Math.round(clamped * 100) });
  if (!suffix) return base;
  return `${base}${suffix}`;
}

function analyzeVowelFocus(store, maskInfo){
  const formants = Array.isArray(store.formants) ? store.formants : [];
  let mask = null;
  if (Array.isArray(maskInfo)) mask = maskInfo;
  else if (maskInfo && Array.isArray(maskInfo.mask)) mask = maskInfo.mask;
  if (!mask || !mask.length){
    const built = buildEligibleFrameMask(store, {
      minConfidence: FORMANT_CONFIDENCE_THRESHOLD,
      maxGapFrames: FORMANT_MAX_GAP_FRAMES,
    });
    if (Array.isArray(built?.mask) && built.mask.length) mask = built.mask;
  }

  if (!mask || !mask.length){
    const insufficient = analysisText?.vowelFocus?.insufficient;
    return {
      ratio: NaN,
      label: insufficient?.label || t("analysis.vowelFocus.insufficient.label"),
      hint: insufficient?.hint || t("analysis.vowelFocus.insufficient.hint"),
    };
  }

  let voiced=0, focus=0;
  const limit = Math.min(formants.length, mask.length);
  for (let i=0;i<limit;i++){
    if (!mask[i]) continue;
    const form = formants[i];
    if (!form) continue;
    const f1=form[0], f2=form[1];
    if (!Number.isFinite(f1) || !Number.isFinite(f2)) continue;
    voiced++;
    if (f1 >= 170 && f1 <= 480 && f2 >= 1400 && f2 <= 3000) focus++;
  }
  const ratio = voiced ? focus/voiced : NaN;
  if (!Number.isFinite(ratio)) {
    const insufficient = analysisText?.vowelFocus?.insufficient;
    return {
      ratio: NaN,
      label: insufficient?.label || t("analysis.vowelFocus.insufficient.label"),
      hint: insufficient?.hint || t("analysis.vowelFocus.insufficient.hint"),
    };
  }
  let key = "weak";
  if (ratio >= 0.5) key = "strong";
  else if (ratio >= 0.3) key = "medium";
  const entry = analysisText?.vowelFocus?.[key];
  return {
    ratio,
    label: entry?.label || t(`analysis.vowelFocus.${key}.label`),
    hint: entry?.hint || t(`analysis.vowelFocus.${key}.hint`),
  };
}

function analyzeSpeechRate(store){
  const hopSec = store.frameSec || (PS_INTERVAL_MS/1000);
  const n = store.db.length;
  if (!n) {
    const insufficient = analysisText?.speechRate?.insufficient;
    return {
      syllPerSec: NaN,
      wordsPerMin: NaN,
      label: insufficient?.label || t("analysis.speechRate.insufficient.label"),
      hint: insufficient?.hint || t("analysis.speechRate.insufficient.hint"),
    };
  }
  const duration = hopSec * n;
  let peaks = 0;
  let lastPeak = -Infinity;
  for (let i=1;i<n-1;i++){
    if (!store.voiced[i]) continue;
    const prev = store.db[i-1] ?? store.db[i];
    const curr = store.db[i];
    const next = store.db[i+1] ?? store.db[i];
    if ((curr - prev) > 1.2 && curr >= next - 0.5){
      const t = i * hopSec;
      if (t - lastPeak >= 0.18){ peaks++; lastPeak = t; }
    }
  }
  if (!peaks){
    const voicedFrames = store.voiced.filter(Boolean).length;
    if (voicedFrames){ peaks = Math.max(1, Math.round((voicedFrames * hopSec) / 0.22)); }
  }
  const syllPerSec = peaks / Math.max(duration, EPS);
  const wordsPerMin = syllPerSec > 0 ? (syllPerSec / 1.5) * 60 : NaN;
  if (!Number.isFinite(syllPerSec) || syllPerSec <= 0) {
    const insufficient = analysisText?.speechRate?.insufficient;
    return {
      syllPerSec: NaN,
      wordsPerMin: NaN,
      label: insufficient?.label || t("analysis.speechRate.insufficient.label"),
      hint: insufficient?.hint || t("analysis.speechRate.insufficient.hint"),
    };
  }
  let key = "fast";
  if (syllPerSec < 2.2) key = "tooSlow";
  else if (syllPerSec <= 4.2) key = "balanced";
  const entry = analysisText?.speechRate?.[key];
  return {
    syllPerSec,
    wordsPerMin,
    label: entry?.label || t(`analysis.speechRate.${key}.label`),
    hint: entry?.hint || t(`analysis.speechRate.${key}.hint`),
  };
}

function analyzeConnectedSpeech(voicedArr, hopSec){
  if (!Array.isArray(voicedArr) || !voicedArr.length) {
    const insufficient = analysisText?.liaison?.insufficient;
    return {
      ratio: NaN,
      label: insufficient?.label || t("analysis.liaison.insufficient.label"),
      hint: insufficient?.hint || t("analysis.liaison.insufficient.hint"),
    };
  }
  let segments=0;
  let inVoiced=false;
  let gapDur=0;
  const gaps=[];
  for (let i=0;i<voicedArr.length;i++){
    if (voicedArr[i]){
      if (!inVoiced){
        segments++;
        if (gapDur>0){ gaps.push(gapDur); gapDur=0; }
      }
      inVoiced=true;
    } else {
      if (inVoiced){
        inVoiced=false;
        gapDur = hopSec;
      } else if (gapDur>0){
        gapDur += hopSec;
      } else {
        gapDur = hopSec;
      }
    }
  }
  const totalBreaks = Math.max(0, segments-1);
  const shortGaps = gaps.filter(g=>g <= 0.16).length;
  const ratio = totalBreaks ? shortGaps / totalBreaks : (segments>0 ? 1 : NaN);
  if (!Number.isFinite(ratio)) {
    const insufficient = analysisText?.liaison?.insufficient;
    return {
      ratio: NaN,
      label: insufficient?.label || t("analysis.liaison.insufficient.label"),
      hint: insufficient?.hint || t("analysis.liaison.insufficient.hint"),
    };
  }
  let key = "weak";
  if (ratio >= 0.7) key = "strong";
  else if (ratio >= 0.4) key = "medium";
  const entry = analysisText?.liaison?.[key];
  return {
    ratio,
    label: entry?.label || t(`analysis.liaison.${key}.label`),
    hint: entry?.hint || t(`analysis.liaison.${key}.hint`),
  };
}

function analyzeIntonation(data, hopSec){
  const metrics = computeIntonationMetrics(data, hopSec, {
    confidenceThreshold: CONFIDENCE_INCLUDE_THRESHOLD,
    voicedThreshold: CONFIDENCE_VOICED_THRESHOLD,
    eps: EPS,
  }) || {};
  const points = Array.isArray(metrics.points) ? metrics.points : [];
  const rawPoints = Array.isArray(metrics.rawPoints) ? metrics.rawPoints : [];
  const shadedRanges = Array.isArray(metrics.shadedRanges) ? metrics.shadedRanges : [];
  const slopeCount = Number(metrics.slopeSampleCount) || 0;

  if (slopeCount < 3){
    const insufficient = analysisText?.intonation?.insufficient;
    return {
      points: [],
      rawPoints,
      shadedRanges,
      slope: NaN,
      slopeLabel: insufficient?.slopeLabel || t("analysis.intonation.insufficient.slopeLabel"),
      hint: insufficient?.slopeHint || t("analysis.intonation.insufficient.slopeHint"),
      range: NaN,
      rangeHint: insufficient?.rangeHint || "",
      minHz: NaN,
      maxHz: NaN,
    };
  }

  const slope = Number.isFinite(metrics.slope) ? metrics.slope : NaN;
  const validRange = Number.isFinite(metrics.range) ? metrics.range : NaN;
  let slopeKey = "flat";
  if (slope > 12) slopeKey = "rising";
  else if (slope < -12) slopeKey = "falling";
  const slopeEntry = analysisText?.intonation?.slope?.[slopeKey];
  const slopeLabel = slopeEntry?.label || t(`analysis.intonation.slope.${slopeKey}.label`);
  const slopeHint = slopeEntry?.hint || t(`analysis.intonation.slope.${slopeKey}.hint`);

  let rangeKey = "narrow";
  if (validRange >= 90) rangeKey = "rich";
  else if (validRange >= 50) rangeKey = "medium";
  const rangeEntry = analysisText?.intonation?.range?.[rangeKey];
  const rangeHint = rangeEntry?.hint || t(`analysis.intonation.range.${rangeKey}.hint`);
  const rangeLabel = rangeEntry?.label || t(`analysis.intonation.range.${rangeKey}.label`);
  const hint = `${slopeHint} ${rangeHint}`.trim();

  return {
    points,
    rawPoints,
    shadedRanges,
    slope,
    slopeLabel,
    range: validRange,
    rangeLabel,
    hint,
    rangeHint,
    minHz: Number.isFinite(metrics.minHz) ? metrics.minHz : NaN,
    maxHz: Number.isFinite(metrics.maxHz) ? metrics.maxHz : NaN,
  };
}

function drawIntonationCurve(canvas, intonation){
  try{
    const pts = intonation?.points || [];
    const rawPts = intonation?.rawPoints || [];
    const shaded = intonation?.shadedRanges || [];
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.clientWidth || canvas.offsetWidth || canvas.width || 520;
    const height = canvas.clientHeight || canvas.offsetHeight || canvas.height || 140;
    const DPR = Math.max(1, window.devicePixelRatio||1);

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.max(1, Math.round(width * DPR));
    canvas.height = Math.max(1, Math.round(height * DPR));
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0,0,width,height);
    ctx.fillStyle = "#f8f8f8";
    ctx.fillRect(0,0,width,height);
    ctx.strokeStyle = "rgba(0,0,0,.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0,height-18);
    ctx.lineTo(width,height-18);
    ctx.stroke();
    if (!pts.length) return;
    const minT = pts[0].t;
    const maxT = pts[pts.length-1].t;
    const tRange = Math.max(maxT - minT, EPS);
    const minHz = Number.isFinite(intonation.minHz) ? intonation.minHz : Math.min(...pts.map(p=>p.hz));
    const maxHz = Number.isFinite(intonation.maxHz) ? intonation.maxHz : Math.max(...pts.map(p=>p.hz));
    const hzRange = Math.max(maxHz - minHz, 1);
    const projectX = (t)=> 10 + ((t - minT) / tRange) * (width - 20);
    const projectY = (hz)=> height - 20 - ((hz - minHz) / hzRange) * (height - 40);

    shaded.forEach(({ type, start, end })=>{
      const x0 = projectX(Math.max(minT, start));
      const x1 = projectX(Math.min(maxT, end));
      if (x1 <= x0) return;
      ctx.fillStyle = type === "mute" ? "rgba(110,110,110,0.24)" : "rgba(110,110,110,0.12)";
      ctx.fillRect(x0, 10, x1 - x0, height - 30);
    });

    if (showIntonationRawPoints && rawPts.length){
      ctx.fillStyle = "rgba(60,60,60,0.22)";
      rawPts.forEach(({ t, hz })=>{
        const x = projectX(t);
        const y = projectY(Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz)));
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI*2);
        ctx.fill();
      });
    }

    ctx.strokeStyle = "rgba(239,93,168,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let drawing=false;
    pts.forEach((p)=>{
      if (!Number.isFinite(p.hz)) { drawing=false; return; }
      const x = projectX(p.t);
      const y = projectY(Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, p.hz)));
      if (!drawing){ ctx.moveTo(x,y); drawing=true; }
      else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }catch(e){ console.error("[drawIntonationCurve]", e); }
}

function setupIntonationLegend(intonation){
  try{
    const legend = document.getElementById("intonationLegend");
    if (!legend) return;
    const hasRaw = Array.isArray(intonation?.rawPoints) && intonation.rawPoints.length > 0;
    legend.setAttribute("data-show-raw", showIntonationRawPoints ? "true" : "false");
    legend.setAttribute("data-has-raw", hasRaw ? "true" : "false");
    const toggle = document.getElementById("intonationRawToggle");
    if (toggle){
      toggle.disabled = !hasRaw;
      if (hasRaw){
        toggle.removeAttribute("aria-disabled");
      } else {
        toggle.setAttribute("aria-disabled", "true");
      }
      const updateToggleState = ()=>{
        const legendCopy = summaryText?.advanced?.intonationLegend || {};
        const labelShow = legendCopy.show || t("summary.advanced.intonationLegend.show");
        const labelHide = legendCopy.hide || t("summary.advanced.intonationLegend.hide");
        toggle.setAttribute("aria-pressed", showIntonationRawPoints ? "true" : "false");
        const stateEl = toggle.querySelector(".legend-toggle-state");
        if (stateEl){
          stateEl.textContent = showIntonationRawPoints ? labelHide : labelShow;
        }
        legend.setAttribute("data-show-raw", showIntonationRawPoints ? "true" : "false");
      };
      toggle.onclick = ()=>{
        if (!hasRaw) return;
        showIntonationRawPoints = !showIntonationRawPoints;
        saveIntonationRawPreference(showIntonationRawPoints);
        updateToggleState();
        const canvas = document.getElementById("intonationCanvas");
        if (canvas) drawIntonationCurve(canvas, intonation || {});
      };
      updateToggleState();
    }
  }catch(err){ console.error("[setupIntonationLegend]", err); }
}

function estimateSpectralFeatures(frame, sr){
  if (!frame || !frame.length) return null;
  let energy=0;
  for (let i=0;i<frame.length;i++){ const v=frame[i]; energy += v*v; }
  if (energy <= 1e-8) return null;

  const n = frame.length;
  const windowed = new Float32Array(n);
  for (let i=0;i<n;i++){
    const pre = frame[i] - 0.97*(i>0 ? frame[i-1] : 0);
    const win = 0.54 - 0.46*Math.cos((2*Math.PI*i)/Math.max(1,n-1));
    windowed[i] = pre * win;
  }
  const sizePow = Math.max(9, Math.ceil(Math.log2(n*2)));
  const N = 1 << sizePow;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(windowed);
  fftRadix2(re, im);

  const half = Math.floor(N/2);
  const mags = new Float32Array(half);
  for (let k=0;k<half;k++) mags[k] = Math.hypot(re[k], im[k]);

  const smooth = new Float32Array(half);
  const smoothWin = 4;
  for (let k=0;k<half;k++){
    let sum=0,count=0;
    for (let j=-smoothWin;j<=smoothWin;j++){
      const idx = k+j;
      if (idx>=0 && idx<half){ sum += mags[idx]; count++; }
    }
    smooth[k] = count ? (sum/count) : mags[k];
  }

  const freqStep = sr / N;
  const peaks=[];
  for (let k=3;k<half-3;k++){
    const freq = k * freqStep;
    if (freq < 90 || freq > 5000) continue;
    const val = smooth[k];
    if (val > smooth[k-1] && val >= smooth[k+1]) peaks.push({ freq, amp: val });
  }
  peaks.sort((a,b)=>a.freq-b.freq);
  const compact=[];
  for (const p of peaks){
    if (!compact.length){ compact.push(p); continue; }
    const last = compact[compact.length-1];
    if (Math.abs(last.freq - p.freq) < 80){
      if (p.amp > last.amp) compact[compact.length-1] = p;
    } else {
      compact.push(p);
    }
  }

  const f0 = detectPitchACF(frame, sr);
  const harmonicTolerance = f0 ? Math.max(40, f0 * 0.18) : 0;
  const isLikelyHarmonic = (freq)=>{
    if (!f0 || !Number.isFinite(f0) || f0 < 50) return false;
    const harmonic = Math.round(freq / f0) * f0;
    if (harmonic <= 0) return false;
    return Math.abs(freq - harmonic) <= harmonicTolerance;
  };

  function selectPeak(low, high, { allowHarmonic = false, minGap = 0, previousFreq = null } = {}){
    let best = null;
    for (const peak of compact){
      if (peak.freq < low || peak.freq > high) continue;
      if (!allowHarmonic && isLikelyHarmonic(peak.freq)) continue;
      if (previousFreq && peak.freq - previousFreq < minGap) continue;
      if (!best || peak.amp > best.amp) best = peak;
    }
    if (!best && !allowHarmonic){
      for (const peak of compact){
        if (peak.freq < low || peak.freq > high) continue;
        if (previousFreq && peak.freq - previousFreq < minGap) continue;
        if (!best || peak.amp > best.amp) best = peak;
      }
    }
    return best;
  }

  const f1Peak = selectPeak(180, 900, { allowHarmonic: true });
  const f1 = f1Peak?.freq ?? NaN;
  const f2Peak = selectPeak(900, 3000, { previousFreq: f1, minGap: 150 });
  const f2 = f2Peak?.freq ?? NaN;
  const f3Peak = selectPeak(1500, 4500, { previousFreq: f2 || f1, minGap: 150 });
  const f3 = f3Peak?.freq ?? NaN;

  let low=0, mid=0, high=0;
  for (let k=0;k<half;k++){
    const freq = k*freqStep;
    if (freq < 90 || freq > 5000) continue;
    const power = mags[k]*mags[k];
    if (freq < 1000) low += power;
    else if (freq < 3000) mid += power;
    else high += power;
  }
  const total = low + mid + high + EPS;
  const tilt = 10 * Math.log10((low + mid + EPS) / (high + EPS));
  const breathiness = Math.min(1, Math.max(0, high / total));
  const zcr = zeroCrossingRate(frame);

  return {
    f1: Number.isFinite(f1) ? f1 : NaN,
    f2: Number.isFinite(f2) ? f2 : NaN,
    f3: Number.isFinite(f3) ? f3 : NaN,
    energy: { low, mid, high, total },
    tilt,
    breathiness,
    zcr,
  };
}

function fftRadix2(re, im){
  const n = re.length;
  if (n <= 1) return;
  let j = 0;
  for (let i=1;i<n;i++){
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j){
      const tmpRe = re[i]; re[i] = re[j]; re[j] = tmpRe;
      const tmpIm = im[i]; im[i] = im[j]; im[j] = tmpIm;
    }
  }
  for (let len=2; len<=n; len<<=1){
    const ang = -2 * Math.PI / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i=0;i<n;i+=len){
      let wRe = 1, wIm = 0;
      for (let j=0;j<len/2;j++){
        const uRe = re[i+j], uIm = im[i+j];
        const vRe = re[i+j+len/2]*wRe - im[i+j+len/2]*wIm;
        const vIm = re[i+j+len/2]*wIm + im[i+j+len/2]*wRe;
        re[i+j] = uRe + vRe;
        im[i+j] = uIm + vIm;
        re[i+j+len/2] = uRe - vRe;
        im[i+j+len/2] = uIm - vIm;
        const nextRe = wRe*wLenRe - wIm*wLenIm;
        const nextIm = wRe*wLenIm + wIm*wLenRe;
        wRe = nextRe; wIm = nextIm;
      }
    }
  }
}

function zeroCrossingRate(arr){
  let count=0;
  for (let i=1;i<arr.length;i++){
    const prev = arr[i-1];
    const curr = arr[i];
    if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) count++;
  }
  return count / Math.max(1, arr.length-1);
}

function bandOf(medHz){
  if (!isFinite(medHz)) return "—";
  if (medHz < 85) return t("pitchBands.low");
  if (medHz < 165) return t("pitchBands.male");
  if (medHz < 180) return t("pitchBands.overlap");
  if (medHz < 310) return t("pitchBands.female");
  if (medHz < 450) return t("pitchBands.high");
  if (medHz <= PS_MAX_HZ) return t("pitchBands.falsetto");
  return t("pitchBands.outOfRange");
}
function isDivergent(medHz, pf, pm){
  if (!isFinite(medHz)) return false;
  // 165–180 的重疊帶不算分歧
  if (medHz >= 165 && medHz < 180) return false;
  // 音高偏高但模型偏男性；或音高偏低但模型偏女性
  if ((medHz >= 180 && pm >= 0.60) || (medHz <= 165 && pf >= 0.60)) return true;
  return false;
}
function escapeAttr(value){
  if (value == null) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
if (typeof window !== "undefined"){
  window.vpaDebugHooks = {
    drawIntonationCurve(data){
      const canvas = document.getElementById("intonationCanvas");
      if (canvas) drawIntonationCurve(canvas, data || {});
    },
    renderAdvancedSummary(summary){
      return renderAdvancedSummary(summary);
    },
  };
}
