// ===== Transformers pipeline =====
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

import { estimateAcousticPresentation } from "./js/acoustic-fast-path.js?v=1.4.20";
import { initI18n, t, getLocaleValue, onLocaleChange } from "./js/i18n.js?v=1.4.20";
import {
  analyzeWhole as sharedAnalyzeWhole,
  analyzeStreamed as sharedAnalyzeStreamed,
  runStreamedWithWindow as sharedRunStreamedWithWindow,
} from "./js/analysis-core.js";
import { createAnalysisEngineBridge } from "./js/analysis-engine-bridge.js?v=1.4.20";
import {
  createAnalysisFlowController,
  runDecodedAudioAnalyzers as sharedRunDecodedAudioAnalyzers,
} from "./js/analysis-flow.js";
import { createAnalysisSessionController } from "./js/analysis-session.js";
import { createAdvancedAdapters } from "./js/advanced-adapters.js";
import { createAdvancedRuntime } from "./js/advanced-runtime.js";
import {
  averageFinite as sharedAverageFinite,
  averageEnergy as sharedAverageEnergy,
  summarizeBreathiness as sharedSummarizeBreathiness,
  buildEligibleFrameMask as sharedBuildEligibleFrameMask,
  categorizeBrightness as sharedCategorizeBrightness,
  detectVoiceLeaning as sharedDetectVoiceLeaning,
  normalizeResonanceBands as sharedNormalizeResonanceBands,
  describeResonanceFromEnergy as sharedDescribeResonanceFromEnergy,
  categorizeTilt as sharedCategorizeTilt,
  categorizeBreathiness as sharedCategorizeBreathiness,
  makeFormantHint as sharedMakeFormantHint,
  summarizeFormantTrends as sharedSummarizeFormantTrends,
  buildFormantTrendDisplay as sharedBuildFormantTrendDisplay,
  analyzeVowelFocus as sharedAnalyzeVowelFocus,
  analyzeSpeechRate as sharedAnalyzeSpeechRate,
  analyzeConnectedSpeech as sharedAnalyzeConnectedSpeech,
  analyzeIntonation as sharedAnalyzeIntonation,
} from "./js/advanced-metrics.js";
import { computeAdvancedSummary as sharedComputeAdvancedSummary } from "./js/advanced-summary-core.js";
import {
  cloneOfflineFeatureStore as sharedCloneOfflineFeatureStore,
} from "./js/analysis-export.js";
import { createAnalysisTelemetryController } from "./js/analysis-telemetry.js";
import { wireAdvancedIntonation as sharedWireAdvancedIntonation } from "./js/advanced-intonation-wire.js";
import { mixChannelDataToMono } from "./js/audio-utils.js";
import { bootstrapAppRuntime } from "./js/app-bootstrap.js";
import { fillBuildMeta as sharedFillBuildMeta } from "./js/build-meta.js";
import {
  drawIntonationCurve as sharedDrawIntonationCurve,
  setupIntonationLegend as sharedSetupIntonationLegend,
} from "./js/intonation-visual.js";
import { decodeSmartToFloat32 as sharedDecodeSmartToFloat32 } from "./js/decode-smart.js";
import {
  offlineExtractStreamMetrics as sharedOfflineExtractStreamMetrics,
  resetOfflineFeatureStore as sharedResetOfflineFeatureStore,
} from "./js/offline-metrics.js";
import { ensurePipeline as sharedEnsurePipeline } from "./js/model-core.js";
import {
  trimYinBuffers as sharedTrimYinBuffers,
  detectPitchACF as sharedDetectPitchACF,
  detectPitchYinLite as sharedDetectPitchYinLite,
  estimateSpectralFeatures as sharedEstimateSpectralFeatures,
} from "./js/pitch-engine.js";
import {
  ensurePlayerUI as sharedEnsurePlayerUI,
  setupExportButton as sharedSetupExportButton,
  stopPlayback as sharedStopPlayback,
  pausePlayback as sharedPausePlayback,
  isPlaying as sharedIsPlaying,
  playLastRecording as sharedPlayLastRecording,
  setPlaybackSource as sharedSetPlaybackSource,
} from "./js/player-ui.js?v=1.4.20";
import { createPlayerSessionController } from "./js/player-session.js";
import { createPitchProfileController } from "./js/pitch-profile.js";
import { createPitchRuntimeCore } from "./js/pitch-runtime-core.js";
import { createPitchStrategyController } from "./js/pitch-strategy-core.js";
import { createRealtimePitchStreamController } from "./js/realtime-pitch-stream.js";
import {
  toMap as sharedToMap,
  renderScores as sharedRenderScores,
  startHeartbeat as sharedStartHeartbeat,
  stopHeartbeat as sharedStopHeartbeat,
  microYield as sharedMicroYield,
} from "./js/render-utils.js";
import {
  createAudioMediaRecorder as sharedCreateAudioMediaRecorder,
  getMicCaptureInfo as sharedGetMicCaptureInfo,
  pickSupportedMime as sharedPickSupportedMime,
  requestMicStream as sharedRequestMicStream,
} from "./js/recording-utils.js";
import { createRecordingFlowController } from "./js/recording-flow.js?v=1.4.20";
import {
  bandOf as sharedBandOf,
  isDivergent as sharedIsDivergent,
} from "./js/summary-helpers.js";
import { detectThreadCount as sharedDetectThreadCount } from "./js/thread-count.js";
import { installEmbeddedBrowserGuard } from "./js/embedded-browser.js?v=1.4.20";
import {
  mobileInferenceMaxSec,
  shouldUseEmbeddedAcousticFastPath,
  shouldUseMobileFastPath,
  selectRepresentativeSamples,
} from "./js/inference-sampling.js?v=1.4.20";
import { pickStreamStrategy as sharedPickStreamStrategy } from "./js/stream-strategy.js";
import { finishStreamStats as sharedFinishStreamStats } from "./js/stats-core.js";
import { createStatsOrchestration } from "./js/stats-orchestration.js";
import { maybeApplyAdaptiveVAD as sharedMaybeApplyAdaptiveVAD } from "./js/vad-adaptive.js";
import { bindMainUIEvents as sharedBindMainUIEvents } from "./js/ui-events.js";
import { createUIStateControls } from "./js/ui-state-controls.js?v=1.4.20";
import { evaluateAdvancedExperience } from "./experiments/advanced-evaluator.js?v=1.4.20";

import {
  recordBtn,
  dropZone,
  fileInput,
  uploadFab,
  statusEl,
  meter,
  femaleVal,
  maleVal,
  warmupCard,
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
  EPS,
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
  setStatusTimer,
  toggleStatusTimer,
  STATUS_TIMER_RESET,
} from "./js/ui.js";
import {
  PS_INTERVAL_MS,
  PS_MIN_HZ,
  PS_MAX_HZ,
  CONFIDENCE_INCLUDE_THRESHOLD,
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
  fmt1,
  logPostProcessingDiagnostics,
} from "./js/pitch-shared.js";
import { setupPracticeUI, openPracticeCategory } from "./js/practice.js";
import { initManualUI } from "./js/manual-ui.js";

// ===== Advanced UI state & gauges =====
const WARMUP_CARD_OPEN_KEY = "vpa::warmup.open";

/** 只用遠端（Hugging Face Hub），停用本機 /models 尋址 */
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
const THREAD_STORAGE_KEY = "vpa::onnxThreads";
const VOLUME_DISPLAY_MODE = "relative";
const MEDIA_RECORDER_DATA_TIMEOUT_MS = 5000;
const playerSessionController = createPlayerSessionController({
  sharedEnsurePlayerUI,
  sharedIsPlaying,
  sharedPausePlayback,
  sharedPlayLastRecording,
  sharedSetPlaybackSource,
  sharedSetupExportButton,
  sharedStopPlayback,
  t,
});

const runtimeBootstrap = await bootstrapAppRuntime({
  env,
  getLocaleValue,
  initI18n,
  onLocaleChange,
  onLocaleUpdated: () => playerSessionController.updatePlayerCopy(),
  sharedDetectThreadCount,
  threadStorageKey: THREAD_STORAGE_KEY,
});
const embeddedBrowserGuard = installEmbeddedBrowserGuard();
const embeddedBrowserContext = embeddedBrowserGuard.context;
window.vpaEmbeddedBrowser = embeddedBrowserContext;
const getSummaryText = () => runtimeBootstrap.getSummaryText();
const {
  advancedSectionController,
  advancedSummaryRenderer,
  applyDbCalibration,
  escapeAttr,
  focusHelpers,
  summaryString,
} = createAdvancedRuntime({
  fmt1,
  getSummaryText,
  onLocaleChange,
  t,
  volumeDisplayMode: VOLUME_DISPLAY_MODE,
});

// ===== 狀態 =====
const analysisSession = createAnalysisSessionController();

export function onInferenceDone(cb) {
  return analysisSession.onInferenceDone(cb);
}

const notifyInferenceListeners = (pf, pm, analysis = null, presentation = null) => {
  if (presentation?.ready && Number.isFinite(presentation.score)) {
    const feminine = Math.max(0, Math.min(1, presentation.score / 100));
    sharedRenderScores(feminine, 1 - feminine, { femaleVal, maleVal });
  }
  analysisSession.notifyInferenceListeners(pf, pm, analysis, presentation);
};

export const recorderCtl = {
  get isRecording() { return analysisSession.getIsRecording(); },
  get busy() { return analysisSession.getBusy(); },
  get hasLastRecording() { return playerSessionController.hasLastRecording(); },
  get isPlaying() { return playerSessionController.isPlaying(); },
  getAudioEl: () => playerSessionController.getAudioEl(),
  getLastRecordingUrl: () => playerSessionController.getLastRecordingUrl(),
  start: () => recordingFlowController.startRecording(),
  stop: () => recordingFlowController.stopRecording(),
  stopPlayback: () => playerSessionController.stopPlayback(),
  pausePlayback: () => playerSessionController.pausePlayback(),
  playLast: () => playerSessionController.playLastRecording(),
  handleFileOrBlob: (fileOrBlob, source = "upload") => analysisFlowController.handleFileOrBlob(fileOrBlob, source),
  getMicCaptureInfo: (stream) => sharedGetMicCaptureInfo(stream),
  togglePlayback: () => {
    if (playerSessionController.isPlaying()) {
      playerSessionController.pausePlayback();
      return false;
    }
    return playerSessionController.playLastRecording();
  },
};

const uiStateControls = createUIStateControls({
  fileInput,
  getBusy: () => analysisSession.getBusy(),
  getHasPlaybackSource: () => playerSessionController.hasPlaybackSource(),
  getIsRecording: () => analysisSession.getIsRecording(),
  getPlayBtn: () => playerSessionController.getPlayBtn(),
  recordBtn,
  setStatusTimer,
  statusTimerReset: STATUS_TIMER_RESET,
  toggleStatusTimer,
  uploadFab,
  warmupCard,
  warmupOpenKey: WARMUP_CARD_OPEN_KEY,
});

const analysisTelemetryController = createAnalysisTelemetryController();
const updatePlaybackAvailability = () => uiStateControls.updatePlaybackAvailability();

playerSessionController.ensurePlayerUI(updatePlaybackAvailability);
playerSessionController.setupExportButton({
  exportBtn,
  getLatestAnalysisExport: () => analysisTelemetryController.getLatestAnalysisExport(),
  setStatus,
  t,
});
try {
  await setupPracticeUI({ subscribeInference: onInferenceDone, recorder: recorderCtl });
} catch (err) {
  console.error("[practice] init failed", err);
}
uiStateControls.initWarmupCard();
initManualUI();

uiStateControls.refreshAvailability();

// Pitch Stream 狀態
const pitchRuntimeCore = createPitchRuntimeCore({
  clamp01,
  clampPitchRange,
  CONFIDENCE_INCLUDE_THRESHOLD,
  createPitchPostState,
  createPitchProfileController,
  createPitchStrategyController,
  createRealtimePitchStreamController,
  DEFAULT_PITCH_RANGE,
  fmt1,
  log,
  makeNoiseTracker,
  PITCH_PROFILE_DEFAULT,
  PITCH_RANGE_HARD,
  PS_INTERVAL_MS,
  PS_MAX_HZ,
  PS_MIN_HZ,
  resetPitchPostState,
  resetRealtimePanels,
  setRealtimePanelsActive,
  sharedAppendPitchSample,
  sharedDetectPitchACF,
  sharedDetectPitchYinLite,
  sharedOfflineExtractStreamMetrics,
  sharedResetOfflineFeatureStore,
  sharedTrimYinBuffers,
  t,
  VOICE_PRESETS,
});
const {
  acfBuffers,
  offlineFeatureStore,
  pitchPostState,
  psConfidence,
  psDb,
  psHz,
  psHzSmooth,
  psVoiced,
} = pitchRuntimeCore;

const advancedAdapters = createAdvancedAdapters({
  CONFIDENCE_INCLUDE_THRESHOLD,
  EPS,
  PS_MAX_HZ,
  PS_MIN_HZ,
  acfBuffers,
  averageEnergyFn: sharedAverageEnergy,
  averageFiniteFn: sharedAverageFinite,
  bandOfFn: sharedBandOf,
  buildEligibleFrameMaskFn: sharedBuildEligibleFrameMask,
  buildFormantTrendDisplayFn: sharedBuildFormantTrendDisplay,
  categorizeBreathinessFn: sharedCategorizeBreathiness,
  categorizeBrightnessFn: sharedCategorizeBrightness,
  categorizeTiltFn: sharedCategorizeTilt,
  describeResonanceFromEnergyFn: sharedDescribeResonanceFromEnergy,
  detectPitchACFFn: sharedDetectPitchACF,
  detectVoiceLeaningFn: sharedDetectVoiceLeaning,
  drawIntonationCurveFn: sharedDrawIntonationCurve,
  escapeAttrFn: escapeAttr,
  getPitchDetectorRange: () => pitchRuntimeCore.getPitchDetectorRange(),
  getShowIntonationRawPoints: () => pitchRuntimeCore.getShowIntonationRawPoints(),
  isDivergentFn: sharedIsDivergent,
  makeFormantHintFn: sharedMakeFormantHint,
  normalizeResonanceBandsFn: sharedNormalizeResonanceBands,
  saveIntonationRawPreference: (flag) => pitchRuntimeCore.saveIntonationRawPreference(flag),
  setShowIntonationRawPoints: (value) => pitchRuntimeCore.setShowIntonationRawPoints(value),
  setupIntonationLegendFn: sharedSetupIntonationLegend,
  summarizeBreathinessFn: sharedSummarizeBreathiness,
  summarizeFormantTrendsFn: sharedSummarizeFormantTrends,
  t,
  analyzeConnectedSpeechFn: sharedAnalyzeConnectedSpeech,
  analyzeIntonationFn: sharedAnalyzeIntonation,
  analyzeSpeechRateFn: sharedAnalyzeSpeechRate,
  analyzeVowelFocusFn: sharedAnalyzeVowelFocus,
  estimateSpectralFeaturesFn: sharedEstimateSpectralFeatures,
});

const {
  averageFinite,
  averageEnergy,
  summarizeBreathiness,
  buildEligibleFrameMask,
  categorizeBrightness,
  detectVoiceLeaning,
  normalizeResonanceBands,
  describeResonanceFromEnergy,
  categorizeTilt,
  categorizeBreathiness,
  makeFormantHint,
  summarizeFormantTrends,
  buildFormantTrendDisplay,
  analyzeVowelFocus,
  analyzeSpeechRate,
  analyzeConnectedSpeech,
  analyzeIntonation,
  drawIntonationCurve,
  setupIntonationLegend,
  estimateSpectralFeatures,
  bandOf,
  isDivergent,
  FORMANT_CONFIDENCE_THRESHOLD,
  FORMANT_MAX_GAP_FRAMES,
} = advancedAdapters;


function resetAnalysisOutputs() {
  resetMeter();
  resetRealtimePanels();
  analysisSession.resetLastScores();
  clearStreamStatsPanel();
  analysisTelemetryController.setLatestAnalysisExport(null);
}

function clearStreamStatsPanel() {
  try {
    const statsEl = document.getElementById("streamStats");
    if (statsEl) statsEl.innerHTML = "";
  } catch { }
}

const recordingFlowController = createRecordingFlowController({
  createMediaRecorder: (stream, mimeType) => sharedCreateAudioMediaRecorder(stream, mimeType),
  dismissOnboardTip,
  handleFileOrBlob: (fileOrBlob, source = "upload") => analysisFlowController.handleFileOrBlob(fileOrBlob, source),
  pickSupportedMime: () => sharedPickSupportedMime(),
  prepareAnalysis: () => (
    shouldUseMobileFastPath(embeddedBrowserContext)
      && !shouldUseEmbeddedAcousticFastPath(embeddedBrowserContext)
      ? analysisEngineBridge.preloadPipeline()
      : null
  ),
  refreshAvailability: () => uiStateControls.refreshAvailability(),
  requestMicStream: () => sharedRequestMicStream(),
  setBusy: (value) => analysisSession.setBusy(value),
  setIsRecording: (value) => analysisSession.setIsRecording(value),
  setStatus,
  startPitchStream,
  startRecordingTimer: () => uiStateControls.startRecordingTimer(),
  stopPitchStream,
  stopPlayback: () => playerSessionController.stopPlayback(),
  stopRecordingTimer: () => uiStateControls.stopRecordingTimer(),
  t,
  MEDIA_RECORDER_DATA_TIMEOUT_MS,
});

// ===== 版本資訊（build 與日期） =====
const buildMetaSource = (import.meta && import.meta.url) ? import.meta.url : "assets/app.js";
void sharedFillBuildMeta(buildMetaSource);

// ===== 事件 =====
sharedBindMainUIEvents({
  recordBtn,
  dropZone,
  fileInput,
  uploadFab,
  isBusy: () => analysisSession.getBusy(),
  isRecording: () => analysisSession.getIsRecording(),
  getMediaRecorder: () => recordingFlowController.getMediaRecorder(),
  resetMeter,
  startRecording: () => recordingFlowController.startRecording(),
  stopRecording: () => recordingFlowController.stopRecording(),
  setStatus,
  t,
  dismissOnboardTip,
  stopPlayback: () => playerSessionController.stopPlayback(),
  handleFileOrBlob: (fileOrBlob, source = "upload") => analysisFlowController.handleFileOrBlob(fileOrBlob, source),
});

const analysisEngineBridge = createAnalysisEngineBridge({
  MODEL_ID,
  clamp01,
  fmtSec,
  getClf: () => analysisSession.getClf(),
  getCurrentDevice: () => analysisSession.getCurrentDevice(),
  isAnalysisActive: (token) => analysisSession.isAnalysisActive(token),
  isOOMError,
  log,
  meter,
  microYield: sharedMicroYield,
  mixChannelDataToMono,
  pipeline,
  render,
  setClf: (value) => analysisSession.setClf(value),
  setCurrentDevice: (value) => analysisSession.setCurrentDevice(value),
  setStatus,
  sharedAnalyzeStreamed,
  sharedAnalyzeWhole,
  sharedDecodeSmartToFloat32,
  sharedEnsurePipeline,
  sharedPickStreamStrategy,
  sharedRunStreamedWithWindow,
  startHeartbeat,
  stopHeartbeat,
  t,
  toMap: sharedToMap,
});

const statsOrchestrationController = createStatsOrchestration({
  CONFIDENCE_INCLUDE_THRESHOLD,
  EPS,
  FORMANT_CONFIDENCE_THRESHOLD,
  FORMANT_MAX_GAP_FRAMES,
  PITCH_COUNTER_KEYS,
  PS_INTERVAL_MS,
  VOLUME_DISPLAY_MODE,
  analysisSession,
  analyzeConnectedSpeech,
  analyzeIntonation,
  analyzeSpeechRate,
  analyzeVowelFocus,
  averageEnergy,
  averageFinite,
  bandOf,
  buildEligibleFrameMask,
  buildFocusInsights: (context) => focusHelpers.buildFocusInsights(context),
  categorizeBreathiness,
  categorizeBrightness,
  categorizeTilt,
  cloneOfflineFeatureStore: () => sharedCloneOfflineFeatureStore(offlineFeatureStore),
  describeResonanceFromEnergy,
  detectVoiceLeaning,
  drawIntonationCurve,
  evaluatePresentation: evaluateAdvancedExperience,
  escapeAttr,
  filterPitchForStats,
  fmt1,
  getSummaryText,
  isDivergent,
  logPostProcessingDiagnostics,
  makeStats,
  notifyInferenceListeners,
  offlineFeatureStore,
  openPracticeCategory,
  percentileSorted,
  pitchPostState,
  psConfidence,
  psDb,
  psHz,
  psHzSmooth,
  psVoiced,
  renderAdvancedSummary,
  renderFocusBlock: (focus) => focusHelpers.renderFocusBlock(focus),
  resizeIntonationCanvas,
  setLatestAnalysisExport: (payload) => analysisTelemetryController.setLatestAnalysisExport(payload),
  setupAdvancedSection: (root) => advancedSectionController.setupAdvancedSection(root),
  setupIntonationLegend,
  sharedComputeAdvancedSummary,
  sharedFinishStreamStats,
  sharedWireAdvancedIntonation,
  summaryString,
  summarizeBreathiness,
  summarizeFormantTrends,
  t,
});

onLocaleChange(() => {
  void sharedFillBuildMeta(buildMetaSource);
  if (analysisSession.getLastPf() != null || pitchRuntimeCore.pitchPostState.count > 0) {
    try {
      statsOrchestrationController.finishStreamStats();
    } catch (e) {
      console.error("[app] failed to re-render stats on locale change", e);
    }
  }
});

const analysisFlowController = createAnalysisFlowController({
  analyzeStreamed: (float32, sr, durationSec, reason, token) =>
    analysisEngineBridge.analyzeStreamed(float32, sr, durationSec, reason, token),
  analyzeWhole: (float32, sr, durationSec, token) =>
    analysisEngineBridge.analyzeWhole(float32, sr, durationSec, token),
  analyzeWithoutModel: () => {
    if (!shouldUseEmbeddedAcousticFastPath(embeddedBrowserContext)) return false;
    const estimate = estimateAcousticPresentation(pitchRuntimeCore.offlineFeatureStore);
    analysisSession.setCurrentDevice(estimate.source);
    render(estimate.feminine, estimate.masculine);
    setStatus(t("status.embeddedFastDone"));
    return true;
  },
  decodeSmartToFloat32: (blobOrFile, targetSR) =>
    analysisEngineBridge.decodeSmartToFloat32(blobOrFile, targetSR),
  finishAnalysisRun: (token) => analysisSession.finishAnalysisRun(token, () => {
    uiStateControls.refreshAvailability();
  }),
  finishStreamStats: () => statsOrchestrationController.finishStreamStats(),
  fmtSec,
  isAnalysisActive: (token) => analysisSession.isAnalysisActive(token),
  MAX_WHOLE_SEC,
  maybeApplyAdaptiveVAD,
  microYield: sharedMicroYield,
  notifyInferenceListeners,
  offlineExtractStreamMetrics,
  prepareInferenceSamples: ({ samples, sampleRate }) => (
    shouldUseMobileFastPath(embeddedBrowserContext)
      ? selectRepresentativeSamples(samples, sampleRate, {
        maxDurationSec: mobileInferenceMaxSec(embeddedBrowserContext),
      })
      : null
  ),
  runDecodedAudioAnalyzers: sharedRunDecodedAudioAnalyzers,
  setAnalysisExtensions: (value) => pitchRuntimeCore.setAnalysisExtensions(value),
  setPlaybackSource: (blob) => playerSessionController.setPlaybackSource(blob, updatePlaybackAvailability),
  setStatus,
  startAnalysisRun: () => analysisSession.startAnalysisRun(() => {
    resetAnalysisOutputs();
    uiStateControls.refreshAvailability();
  }),
  t,
  TARGET_SR,
  updatePlaybackAvailability,
  WARN_LONG_SEC,
});

// ===== Render / Utils =====
function render(pf, pm) {
  const rendered = sharedRenderScores(pf, pm, { femaleVal, maleVal });
  analysisSession.setLastScores(rendered.female, rendered.male);
}
function startHeartbeat(fn) {
  analysisSession.setHeartbeatTimer(sharedStartHeartbeat(analysisSession.getHeartbeatTimer(), fn));
}
function stopHeartbeat() {
  analysisSession.setHeartbeatTimer(sharedStopHeartbeat(analysisSession.getHeartbeatTimer()));
}

// ===== VAD（只「選段」） =====
function maybeApplyAdaptiveVAD(float32, sr) {
  return sharedMaybeApplyAdaptiveVAD(float32, sr);
}

// ===== Pitch Stream（ACF 音高 + 畫布） =====
const realtimePitchStreamController = pitchRuntimeCore.createRealtimeController({
  applyDbCalibration,
  describeResonanceFromEnergy,
  dom: {
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
  },
  estimateSpectralFeatures,
  normalizeResonanceBands,
});

function startPitchStream(userMediaStream) {
  return realtimePitchStreamController.startPitchStream(userMediaStream);
}

function stopPitchStream() {
  return realtimePitchStreamController.stopPitchStream();
}

// ===== 離線抽樣（上傳檔用；錄音也會補） =====
function offlineExtractStreamMetrics(float32, sr, append = false) {
  return pitchRuntimeCore.offlineExtractStreamMetrics(float32, sr, append, {
    applyDbCalibration,
    estimateSpectralFeatures,
  });
}

// 讓曲線圖吃到容器實際寬度，避免在 details 關著時變 0
function resizeIntonationCanvas(canvas) {
  const pxRatio = Math.max(1, window.devicePixelRatio || 1);
  const box = canvas.parentElement || canvas;
  const cssWidth = Math.max(1, Math.floor(box.clientWidth || 600));
  const cssHeight = 160; // 你原本的高度，如果有常數就沿用
  canvas.style.width = cssWidth + "px";
  canvas.style.height = cssHeight + "px";
  canvas.width = Math.floor(cssWidth * pxRatio);
  canvas.height = Math.floor(cssHeight * pxRatio);
}


// ===== 統計卡（停止&分析完成後，含「簡評」與分歧提示） =====
function renderAdvancedSummary(summary, context = {}) {
  return advancedSummaryRenderer.renderAdvancedSummary(summary, context);
}
if (typeof window !== "undefined") {
  window.vpaDebugHooks = {
    drawIntonationCurve(data) {
      const canvas = document.getElementById("intonationCanvas");
      if (canvas) drawIntonationCurve(canvas, data || {});
    },
    renderAdvancedSummary(summary) {
      return renderAdvancedSummary(summary);
    },
  };
}
