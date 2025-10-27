// ===== Transformers pipeline =====
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

import { initI18n, t, getLocaleValue, onLocaleChange } from "./js/i18n.js";

import {
  recordBtn,
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
let psHz=[], psHzSmooth=[], psDb=[], psVoiced=[]; // 50ms/點
const PS_INTERVAL_MS = 50;
const PS_MIN_HZ = 50, PS_MAX_HZ = 450;
const PS_SMOOTH_BASE_ALPHA = 0.08; // 細微抖動採慢速平滑；統計與即時顯示都使用平滑後資料並過濾諧波異常值
const PS_SMOOTH_FAST_ALPHA = 0.45; // 真實音高跳動時加速追上
const PS_SMOOTH_FAST_THRESHOLD_SEMITONES = 1.5;
const PS_SMOOTH_MAX_STEP_SEMITONES = 2.4;
const PS_SMOOTH_MEDIAN_WINDOW = 7;
const PS_MIN_DB_FOR_PITCH = 42; // 約 -58 dBFS；低於此視為環境底噪
const PS_NOISE_BUFFER_MS = 10000;
const PS_NOISE_MIN_SAMPLES = 8;
const PS_NOISE_CAPTURE_RANGE_DB = 18;
const PS_MIN_DB_ABOVE_NOISE = 5;
const PS_NOISE_GATE_MAX_BOOST_DB = 14;
const psRealtimeNoiseTracker = makeNoiseTracker();
const psOfflineNoiseTracker = makeNoiseTracker();
const offlineFeatureStore = {
  frameSec: 0,
  pitch: [],
  db: [],
  voiced: [],
  formants: [],
  tilt: [],
  breathiness: [],
  energy: [],
  zcr: [],
};

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
  stopPlayback();
  fileInput?.click();
});

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
function makeNoiseTracker(){
  const buf = [];
  const maxSamples = Math.max(1, Math.round(PS_NOISE_BUFFER_MS / PS_INTERVAL_MS));
  return {
    reset(){ buf.length = 0; },
    capture(db){
      if (!Number.isFinite(db)) return;
      const capped = Math.min(db, PS_MIN_DB_FOR_PITCH + PS_NOISE_CAPTURE_RANGE_DB);
      buf.push(capped);
      if (buf.length > maxSamples) buf.shift();
    },
    getAmbient(){
      if (buf.length < PS_NOISE_MIN_SAMPLES) return null;
      const sorted = buf.slice().sort((a,b)=>a-b);
      const val = percentileSorted(sorted, 20);
      return Number.isFinite(val) ? val : null;
    },
    getThreshold(){
      let threshold = PS_MIN_DB_FOR_PITCH;
      const ambient = this.getAmbient();
      if (ambient != null){
        const dynamic = ambient + PS_MIN_DB_ABOVE_NOISE;
        const maxBoosted = PS_MIN_DB_FOR_PITCH + PS_NOISE_GATE_MAX_BOOST_DB;
        threshold = Math.max(threshold, Math.min(maxBoosted, dynamic));
      }
      return { threshold, ambient };
    },
    shouldDetect(db, wasVoiced){
      if (!Number.isFinite(db)) return { detect:false, threshold:PS_MIN_DB_FOR_PITCH, ambient:null };
      const { threshold, ambient } = this.getThreshold();
      if (db >= threshold) return { detect:true, threshold, ambient };
      if (wasVoiced && db >= PS_MIN_DB_FOR_PITCH){
        return { detect:true, threshold, ambient, hysteresis:true };
      }
      return { detect:false, threshold, ambient };
    },
  };
}

function appendPitchSample(rawHz){
  psHz.push(rawHz);
  if (!Number.isFinite(rawHz)){
    psHzSmooth.push(null);
    return;
  }
  const finiteSamples = [];
  for (let i = psHz.length - 1; i >= 0 && finiteSamples.length < PS_SMOOTH_MEDIAN_WINDOW; i--){
    const v = psHz[i];
    if (Number.isFinite(v)) finiteSamples.push(v);
  }
  finiteSamples.sort((a,b)=>a-b);
  const mid = finiteSamples.length ? finiteSamples[Math.floor(finiteSamples.length/2)] : rawHz;
  const clampedRaw = Math.min(PS_MAX_HZ, Math.max(PS_MIN_HZ, rawHz));
  let target = clampedRaw;
  if (finiteSamples.length >= 3){
    const safeMedian = Math.min(PS_MAX_HZ, Math.max(PS_MIN_HZ, mid));
    const diffFromMedian = Math.abs(Math.log2(clampedRaw / safeMedian)) * 12;
    if (Number.isFinite(diffFromMedian) && diffFromMedian <= 0.8){
      target = safeMedian;
    }
  }

  const prev = psHzSmooth.length ? psHzSmooth[psHzSmooth.length-1] : null;
  if (!Number.isFinite(prev)){
    psHzSmooth.push(target);
    return;
  }

  const safePrev = Math.min(PS_MAX_HZ, Math.max(PS_MIN_HZ, prev));
  const prevLog2 = Math.log2(safePrev);
  const safeTarget = Math.min(PS_MAX_HZ, Math.max(PS_MIN_HZ, target));
  let targetLog2 = Math.log2(safeTarget);
  let baseDiffSemitones = Math.abs(targetLog2 - prevLog2) * 12;

  if (Number.isFinite(baseDiffSemitones) && baseDiffSemitones > 3){
    let bestHz = safeTarget;
    let bestDiff = baseDiffSemitones;
    const multipliers = [2, 3, 4];
    for (const m of multipliers){
      const up = safeTarget * m;
      if (up <= PS_MAX_HZ){
        const diff = Math.abs(Math.log2(up) - prevLog2) * 12;
        if (diff < bestDiff){
          bestDiff = diff;
          bestHz = up;
        }
      }
      const down = safeTarget / m;
      if (down >= PS_MIN_HZ){
        const diff = Math.abs(Math.log2(down) - prevLog2) * 12;
        if (diff < bestDiff){
          bestDiff = diff;
          bestHz = down;
        }
      }
    }
    if (bestHz !== safeTarget && bestDiff <= baseDiffSemitones * 0.6){
      targetLog2 = Math.log2(bestHz);
      baseDiffSemitones = bestDiff;
    }
  }

  const deltaSemitones = Math.abs(targetLog2 - prevLog2) * 12;
  const alpha = deltaSemitones > PS_SMOOTH_FAST_THRESHOLD_SEMITONES
    ? PS_SMOOTH_FAST_ALPHA
    : PS_SMOOTH_BASE_ALPHA;
  let nextLog2 = prevLog2 + alpha * (targetLog2 - prevLog2);
  const maxStepLog2 = PS_SMOOTH_MAX_STEP_SEMITONES / 12;
  if (Math.abs(nextLog2 - prevLog2) > maxStepLog2){
    nextLog2 = prevLog2 + Math.sign(nextLog2 - prevLog2) * maxStepLog2;
  }
  const next = 2 ** nextLog2;
  psHzSmooth.push(next);
}

function startPitchStream(userMediaStream){
  try{
    if (!pitchWrap || !pitchCanvas) return;
    psHz.length=0; psHzSmooth.length=0; psDb.length=0; psVoiced.length=0;
    psRealtimeNoiseTracker.reset();

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
        const candHz = detectPitchACF(input, sampleRate);
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
        appendPitchSample(hz ?? null);
        const displayHz = psHzSmooth.length ? psHzSmooth[psHzSmooth.length-1] : (hz ?? null);
        psVoiced.push(hz!=null);
        const maxN = Math.round(15000 / PS_INTERVAL_MS); // 保留約 15 秒
        if (psDb.length>maxN){ psDb.shift(); psHz.shift(); psHzSmooth.shift(); psVoiced.shift(); }
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
  // 簡化自相關（ACF）+ 降採樣到 ~16k；限制 50–450 Hz
  const ds = Math.max(1, Math.floor(sr / 16000));
  const N  = Math.floor(input.length / ds);
  if (N < 128) return null;
  const x = new Float32Array(N);
  let mean=0; for (let i=0;i<N;i++){ mean += input[i*ds]; }
  mean /= N;
  let energy=0; for (let i=0;i<N;i++){ const v=input[i*ds]-mean; x[i]=v; energy += v*v; }
  if (energy <= 1e-8) return null;

  const srDS = sr / ds;
  const lagMin = Math.floor(srDS / PS_MAX_HZ);
  const lagMax = Math.floor(srDS / PS_MIN_HZ);

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
  return srDS / bestLag;
}
function bandLabel(hz){
  if (!hz) return "—";
  if (hz < 85) return t("pitchBands.bandLow");
  if (hz < 165) return t("pitchBands.bandBlue");
  if (hz < 180) return t("pitchBands.bandNeutral");
  if (hz < 310) return t("pitchBands.bandPink");
  if (hz <= 450) return t("pitchBands.bandHigh");
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
    const w=pitchCanvas.width, h=pitchCanvas.height;

    // 區帶：灰(50–85) / 藍(85–165) / 灰(165–180) / 粉(180–310) / 灰(310–450)
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(85),  w, h - yOf(85));
    ctx.fillStyle = cBlue; ctx.fillRect(0, yOf(165), w, yOf(85)-yOf(165));
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(180), w, yOf(165)-yOf(180));
    ctx.fillStyle = cPink; ctx.fillRect(0, yOf(310), w, yOf(180)-yOf(310));
    ctx.fillStyle = cGray; ctx.fillRect(0, 0,        w, yOf(310));

    // 網格線
    ctx.strokeStyle = "rgba(0,0,0,.08)"; ctx.lineWidth = 1*DPR;
    [50,85,165,180,310,450].forEach(f=>{ const y=yOf(f); ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); });
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
    const axisTicks = [450, 400, 350, 300, 250, 200, 150, 100, 50];
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
      psHz.length=0; psHzSmooth.length=0; psDb.length=0; psVoiced.length=0;
      resetOfflineFeatureStore();
      psOfflineNoiseTracker.reset();
    }
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
        const candHz = detectPitchACF(seg, sr);
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
      appendPitchSample(hz ?? null);
      psVoiced.push(hz!=null);
      offlineFeatureStore.pitch.push(hz ?? NaN);
      offlineFeatureStore.db.push(db);
      offlineFeatureStore.voiced.push(hz!=null);
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
  offlineFeatureStore.pitch.length = 0;
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
    const voicedHzRaw = psHzSmooth.filter(v => Number.isFinite(v));
    const vols     = psDb.slice();
    if (!voicedHzRaw.length && !vols.length){ statsEl.innerHTML=""; return; }

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
    let stabilityKey = "steady";
    if (isFinite(spread)){
      if (spread >= 80) stabilityKey = "wide";
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
    const voicedRatio = psVoiced.length ? (psVoiced.filter(Boolean).length / psVoiced.length) : NaN;
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
    const statsRows = [
      { key: "pitchAvg", value: `${fmt1(pitchStats.avg)}Hz` },
      { key: "pitchMed", value: `${fmt1(pitchStats.med)}Hz` },
      { key: "pitchHigh", value: `${fmt1(pitchStats.p95)}Hz` },
      { key: "pitchLow", value: `${fmt1(pitchStats.p05)}Hz` },
      { key: "volumeAvg", value: `${fmt1(volStats.avg)}dB (${fmt1(volStats.sd)}dB)` },
      { key: "volumeMed", value: `${fmt1(volStats.med)}dB (${fmt1(volStats.sd)}dB)` },
      { key: "volumeHigh", value: `${fmt1(volStats.p95)}dB` },
      { key: "volumeLow", value: `${fmt1(volStats.p05)}dB` },
    ];
    const statsRowsHtml = statsRows.map(({ key, value }) => {
      const label = statsLabels[key] || t(`summary.statsLabels.${key}`);
      return `<div class="kv"><div class="k">${label}</div><div class="v">${value}</div></div>`;
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
    const advancedHTML = renderAdvancedSummary(advSummary);

    statsEl.innerHTML = headerHTML + oneLiner + divergeNote + envNote + voicedNote + statsHTML + advancedHTML;

    // 標籤列
    const tags = statsEl.querySelector(".tags");
    if (tags){
      let tagHTML = `
        <span class="tag">${summaryString("tags.pitchBand", { band })}</span>
        <span class="tag">${summaryString("tags.noise", { noise: fmt1(envDb) })}</span>
      `;
      if (advSummary){
        const resonanceTag = advSummary.resonanceDisplay || advSummary.resonanceLabel;
        if (resonanceTag) tagHTML += `<span class="tag">${summaryString("tags.resonance", { label: resonanceTag })}</span>`;
        if (advSummary.speechRateLabel) tagHTML += `<span class="tag">${summaryString("tags.speechRate", { label: advSummary.speechRateLabel })}</span>`;
        if (advSummary.breathinessLabel) tagHTML += `<span class="tag">${summaryString("tags.breathiness", { label: advSummary.breathinessLabel })}</span>`;
      }
      tags.innerHTML = tagHTML;
    }

    if (advSummary?.intonation?.points?.length){
      const canvas = document.getElementById("intonationCanvas");
      if (canvas) drawIntonationCurve(canvas, advSummary.intonation);
    }
  }catch(e){ console.error("[finishStreamStats]", e); }
}
function computeAdvancedSummary(){
  const store = offlineFeatureStore;
  const n = store.pitch.length;
  const hopSec = store.frameSec || (PS_INTERVAL_MS/1000);
  const duration = hopSec * n;
  if (!n || duration < 0.5) return null;

  const f1Vals=[], f2Vals=[], f3Vals=[];
  for (const form of store.formants){
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
  });

  const energyAvg = averageEnergy(store.energy);
  const resonanceDesc = describeResonanceFromEnergy(energyAvg);
  const tiltAvg = averageFinite(store.tilt);
  const tiltInfo = categorizeTilt(tiltAvg);
  const breathAvg = averageFinite(store.breathiness);
  const breathInfo = categorizeBreathiness(breathAvg);
  const vowelInfo = analyzeVowelFocus(store);
  const speech = analyzeSpeechRate(store);
  const liaison = analyzeConnectedSpeech(store.voiced, hopSec);
  const intonation = analyzeIntonation(store.pitch, store.voiced, hopSec);

  return {
    formants: formantSummary,
    resonanceLabel: resonanceDesc.label,
    resonanceDisplay: resonanceDesc.display || resonanceDesc.label,
    resonanceHint: resonanceDesc.hint,
    energyPct: resonanceDesc.pct,
    tiltAvg,
    tiltLabel: tiltInfo.label,
    tiltHint: tiltInfo.hint,
    breathinessAvg: breathAvg,
    breathinessLabel: breathInfo.label,
    breathinessHint: breathInfo.hint,
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

  const liaisonValue = summary.liaisonLabel + (liaisonDisplay ? liaisonDisplay : "");
  const speechRateHint = speechWpmDisplay ? `${summary.speechRateHint} ${speechWpmDisplay}` : summary.speechRateHint;

  return `
    <div class="advanced-section">
      <h3 class="adv-title">${formantTitle}</h3>
      <div class="advanced-grid advanced-grid--four">
        <div class="adv-card"><div class="k">${formantCards.f1 || t("summary.advanced.formantCards.f1")}</div><div class="v">${f1Value}</div><div class="hint">${f1Hint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.f2 || t("summary.advanced.formantCards.f2")}</div><div class="v">${f2Value}</div><div class="hint">${f2Hint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.f3 || t("summary.advanced.formantCards.f3")}</div><div class="v">${f3Value}</div><div class="hint">${f3Hint}</div></div>
        <div class="adv-card"><div class="k">${formantCards.tilt || t("summary.advanced.formantCards.tilt")}</div><div class="v">${fmt1(summary.tiltAvg)} dB</div><div class="hint">${summary.tiltHint}</div></div>
      </div>
      <div class="resonance-summary">
        <div class="resonance-bar resonance-bar--summary">
          <div class="res-part chest" style="flex:${Math.max(summary.energyPct?.chest ?? 0.001, 0.001)}">${chestPct}%</div>
          <div class="res-part mask" style="flex:${Math.max(summary.energyPct?.mask ?? 0.001, 0.001)}">${maskPct}%</div>
          <div class="res-part head" style="flex:${Math.max(summary.energyPct?.head ?? 0.001, 0.001)}">${headPct}%</div>
        </div>
        <p class="subline">${summary.resonanceHint}</p>
      </div>
    </div>
    <div class="advanced-section">
      <h3 class="adv-title">${intonationTitle}</h3>
      <canvas id="intonationCanvas" width="520" height="140" aria-label="${canvasAria}"></canvas>
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

function averageFinite(arr){
  const vals = Array.isArray(arr) ? arr.filter(Number.isFinite) : [];
  if (!vals.length) return NaN;
  return vals.reduce((a,b)=>a+b,0) / vals.length;
}

function averageEnergy(arr){
  if (!Array.isArray(arr) || !arr.length) return { low:0, mid:0, high:0, total:0, coverage:0, validCount:0 };
  let low=0, mid=0, high=0, valid=0;
  for (const v of arr){
    if (!Array.isArray(v)) continue;
    const [l,m,h] = v;
    if (!Number.isFinite(l) && !Number.isFinite(m) && !Number.isFinite(h)) continue;
    low += Number.isFinite(l) ? l : 0;
    mid += Number.isFinite(m) ? m : 0;
    high += Number.isFinite(h) ? h : 0;
    valid++;
  }
  if (!valid) return { low:0, mid:0, high:0, total:0, coverage:0, validCount:0 };
  const avgLow = low / valid;
  const avgMid = mid / valid;
  const avgHigh = high / valid;
  return {
    low: avgLow,
    mid: avgMid,
    high: avgHigh,
    total: avgLow + avgMid + avgHigh,
    coverage: valid / arr.length,
    validCount: valid,
  };
}

const RESONANCE_PRIOR_WEIGHT = 6;
const RESONANCE_DOMINANCE_DELTA = 0.22;
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

  const sorted = [
    { key: "chest", value: chest },
    { key: "mask", value: mask },
    { key: "head", value: head },
  ].sort((a,b)=>b.value - a.value);
  const dominance = sorted[0].value - sorted[2].value;

  let key = "balanced";
  if (dominance >= RESONANCE_DOMINANCE_DELTA){
    if (sorted[0].key === "head" && sorted[0].value >= RESONANCE_HEAD_MIN){
      key = "headBright";
    } else if (sorted[0].key === "mask" && sorted[0].value >= RESONANCE_MASK_MIN){
      key = "maskLead";
    } else if (sorted[0].key === "chest" && sorted[0].value >= RESONANCE_CHEST_MIN){
      key = "chestHeavy";
    } else {
      key = "balanced";
    }
  }

  const entry = analysisText?.resonanceBalance?.[key];
  const label = entry?.label || t(`analysis.resonanceBalance.${key}.label`);
  let hint = entry?.hint || t(`analysis.resonanceBalance.${key}.hint`);
  let display = label;
  if (hasAggregate && Number.isFinite(coverage)){
    const coverageKey = coverage < RESONANCE_COVERAGE_GOOD ? "coverageLowHint" : "coverageHint";
    const hintNote = t(`analysis.resonanceBalance.${coverageKey}`, { value: Math.round(coverage * 100) });
    if (hintNote) hint = `${hint} ${hintNote}`.trim();
    const suffixKey = coverage < RESONANCE_COVERAGE_GOOD ? "coverageLowSuffix" : "coverageSuffix";
    const suffix = t(`analysis.resonanceBalance.${suffixKey}`, { value: Math.round(coverage * 100) });
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
  if (tilt >= 8) key = "warm";
  else if (tilt >= 3) key = "gentleWarm";
  else if (tilt >= -1) key = "balanced";
  const entry = analysisText?.tilt?.[key];
  return {
    label: entry?.label || t(`analysis.tilt.${key}.label`),
    hint: entry?.hint || t(`analysis.tilt.${key}.hint`),
  };
}

function categorizeBreathiness(val){
  if (!Number.isFinite(val)) {
    const insufficient = analysisText?.breathiness?.insufficient;
    return {
      label: insufficient?.label || t("analysis.breathiness.insufficient.label"),
      hint: insufficient?.hint || t("analysis.breathiness.insufficient.hint"),
    };
  }
  let key = "tooAiry";
  if (val < 0.08) key = "dense";
  else if (val <= 0.18) key = "balanced";
  else if (val <= 0.28) key = "airy";
  const entry = analysisText?.breathiness?.[key];
  return {
    label: entry?.label || t(`analysis.breathiness.${key}.label`),
    hint: entry?.hint || t(`analysis.breathiness.${key}.hint`),
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

function summarizeFormantTrends(store, statsBundle){
  const voicedTotal = Array.isArray(store.voiced)
    ? store.voiced.reduce((acc, flag)=> acc + (flag ? 1 : 0), 0)
    : 0;

  const makeEntry = (label, stats, values, low, high)=>{
    const sampleCount = values.length;
    const coverage = voicedTotal ? (sampleCount / voicedTotal) : 0;
    const hasAggregate = voicedTotal > 0;
    const spread = stats ? (stats.p95 - stats.p05) : NaN;
    const reliable = hasAggregate
      && sampleCount >= FORMANT_MIN_SAMPLES
      && coverage >= FORMANT_MIN_COVERAGE
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
    if (hasAggregate && coverage < FORMANT_MIN_COVERAGE){
      const msg = t("analysis.formant.coverageLowHint", { value: Math.round(coverage * 100) });
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
    f1: makeEntry("F1", statsBundle.f1, statsBundle.f1Vals || [], 180, 350),
    f2: makeEntry("F2", statsBundle.f2, statsBundle.f2Vals || [], 1600, 2500),
    f3: makeEntry("F3", statsBundle.f3, statsBundle.f3Vals || [], 2500, 3200),
  };
}

function buildFormantTrendDisplay(trendKey, coverage, hasAggregate){
  const trendLabels = analysisText?.formant?.trendLabels;
  const baseRaw = (trendLabels && trendLabels[trendKey]) || t(`analysis.formant.trendLabels.${trendKey}`);
  const base = baseRaw || "—";
  if (!hasAggregate || !Number.isFinite(coverage) || coverage <= 0){
    return base;
  }
  const key = coverage < FORMANT_GOOD_COVERAGE ? "coverageLowSuffix" : "coverageSuffix";
  const suffix = t(`analysis.formant.${key}`, { value: Math.round(coverage * 100) });
  if (!suffix) return base;
  return `${base}${suffix}`;
}

function analyzeVowelFocus(store){
  let voiced=0, focus=0;
  for (let i=0;i<store.formants.length;i++){
    if (!store.voiced[i]) continue;
    const form = store.formants[i];
    if (!form) continue;
    const f1=form[0], f2=form[1];
    if (!Number.isFinite(f1) || !Number.isFinite(f2)) continue;
    voiced++;
    if (f1 >= 180 && f1 <= 450 && f2 >= 1500 && f2 <= 2800) focus++;
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
  if (ratio >= 0.6) key = "strong";
  else if (ratio >= 0.4) key = "medium";
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

function analyzeIntonation(pitchArr, voicedArr, hopSec){
  const points=[];
  let minHz=Infinity, maxHz=-Infinity;
  for (let i=0;i<pitchArr.length;i++){
    const hz = pitchArr[i];
    if (!Number.isFinite(hz)) continue;
    const t = i * hopSec;
    points.push({ t, hz });
    if (hz < minHz) minHz = hz;
    if (hz > maxHz) maxHz = hz;
  }
  if (points.length < 3){
    const insufficient = analysisText?.intonation?.insufficient;
    return {
      points:[],
      slope:NaN,
      slopeLabel: insufficient?.slopeLabel || t("analysis.intonation.insufficient.slopeLabel"),
      hint: insufficient?.slopeHint || t("analysis.intonation.insufficient.slopeHint"),
      range:NaN,
      rangeHint: insufficient?.rangeHint || "",
      minHz:NaN,
      maxHz:NaN,
    };
  }
  const n = points.length;
  let sumT=0, sumH=0, sumTT=0, sumTH=0;
  for (const {t,hz} of points){
    sumT += t; sumH += hz; sumTT += t*t; sumTH += t*hz;
  }
  const slope = (n*sumTH - sumT*sumH) / Math.max((n*sumTT - sumT*sumT), EPS);
  const range = maxHz - minHz;
  let slopeKey = "flat";
  if (slope > 12) slopeKey = "rising";
  else if (slope < -12) slopeKey = "falling";
  const slopeEntry = analysisText?.intonation?.slope?.[slopeKey];
  const slopeLabel = slopeEntry?.label || t(`analysis.intonation.slope.${slopeKey}.label`);
  const slopeHint = slopeEntry?.hint || t(`analysis.intonation.slope.${slopeKey}.hint`);

  let rangeKey = "narrow";
  if (range >= 90) rangeKey = "rich";
  else if (range >= 50) rangeKey = "medium";
  const rangeEntry = analysisText?.intonation?.range?.[rangeKey];
  const rangeHint = rangeEntry?.hint || t(`analysis.intonation.range.${rangeKey}.hint`);

  const rangeLabel = rangeEntry?.label || t(`analysis.intonation.range.${rangeKey}.label`);
  const hint = `${slopeHint} ${rangeHint}`.trim();
  return { points, slope, slopeLabel, range, rangeLabel, hint, rangeHint, minHz, maxHz };
}

function drawIntonationCurve(canvas, intonation){
  try{
    const pts = intonation?.points || [];
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
    ctx.strokeStyle = "rgba(239,93,168,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, idx)=>{
      const x = 10 + ((p.t - minT) / tRange) * (width - 20);
      const y = height - 20 - ((p.hz - minHz) / hzRange) * (height - 40);
      if (idx===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  }catch(e){ console.error("[drawIntonationCurve]", e); }
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
    if (compact.length>=4) break;
  }
  const f1 = compact[0]?.freq ?? NaN;
  const f2 = compact[1]?.freq ?? NaN;
  const f3 = compact[2]?.freq ?? NaN;

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
  if (medHz <= 450) return t("pitchBands.high");
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
function fmt1(x){ return Number.isFinite(x) ? (Math.round(x*10)/10).toFixed(1) : "—"; }
function filterPitchForStats(samples){
  if (!Array.isArray(samples)) return [];
  const finite = samples.filter(Number.isFinite);
  if (finite.length < 3) return finite.slice();

  const logVals = finite.map((v)=>Math.log2(Math.max(v, PS_MIN_HZ)));
  const sortedLog = logVals.slice().sort((a,b)=>a-b);
  const medianLog = percentileSorted(sortedLog, 50);
  const diffSemitones = logVals.map((v)=>Math.abs(v - medianLog) * 12);
  const sortedDiffs = diffSemitones.slice().sort((a,b)=>a-b);
  const mad = percentileSorted(sortedDiffs, 50);
  const tolerance = Math.max(0.5, (mad || 0) * 3);
  const filtered = finite.filter((_, idx)=> diffSemitones[idx] <= tolerance);

  const minKeep = Math.max(3, Math.ceil(finite.length * 0.6));
  if (filtered.length < minKeep) return finite;
  return filtered;
}
function makeStats(arr){
  if (!arr.length) return { avg:NaN, med:NaN, p95:NaN, p05:NaN, sd:NaN };
  const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
  const sorted = arr.slice().sort((x,y)=>x-y);
  const med  = percentileSorted(sorted, 50);
  const p95  = percentileSorted(sorted, 95);
  const p05  = percentileSorted(sorted, 5);
  const sd   = Math.sqrt(arr.reduce((a,v)=> a + (v-mean)*(v-mean), 0) / arr.length);
  return { avg:mean, med, p95, p05, sd };
}
function percentileSorted(sorted, p){
  if (!sorted.length) return NaN;
  const i = (p/100) * (sorted.length - 1);
  const i0 = Math.floor(i), i1 = Math.min(sorted.length-1, i0+1), t = i - i0;
  return sorted[i0]*(1-t) + sorted[i1]*t;
}
