// assets/app.js — 準確優先 + 自動升級（全前端 / Transformers.js + ONNX WASM）
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// 沒 COOP/COEP 的靜態站 → 使用單執行緒 WASM
env.backends.onnx.wasm.numThreads = 1;

// ===== 參數 =====
const MODEL_ID       = (window.ONNX_MODEL_ID || "prithivMLmods/Common-Voice-Gender-Detection-ONNX");
const TARGET_SR      = 16000;
const DIRECT_MAX_SEC = 25;

// 自適應分段（平衡速度/準確）：固定 50% 重疊，用時間預算反推視窗大小
const OVERLAP_RATIO          = 0.5;     // 50% overlap → hop = window * (1 - 0.5) = window/2
const TIME_BUDGET_S          = 30;      // 目標 30 秒內
const EST_MS_PER_CHUNK_INIT  = 4500;    // 初始估計（會動態更新）
const MIN_WINDOW_S           = 6;
const MAX_WINDOW_S           = 12;
const MIN_HOP_S              = 0.5;

// 高準確升級（遇到不確定才觸發）
const HA_WINDOW_S = 20;     // 視窗 20 秒
const HA_HOP_S    = 5;      // 75% overlap
const HA_MAX_SEC  = 120;    // 升級最多跑到 ~2 分鐘（長檔會慢，但更接近整段一次性）

// 前處理
const TRIM_PAD_SEC = 0.25;
const TARGET_RMS   = 0.08;
const MAX_GAIN_DB  = 20;
const EPS          = 1e-9;

// ===== DOM =====
const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// ===== 狀態 =====
let mediaRecorder = null;
let chunks = [];
let clf = null;
let busy = false;
let avgChunkMs = 0;

log("[app] accuracy-first build ready.");

function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}
function log(...a){ try{ console.log(...a);}catch{} }
function fmtSec(s){ if (!isFinite(s)) return "—"; const m=Math.floor(s/60), ss=Math.round(s%60); return m? `${m}分${ss}秒`:`${ss}秒`; }
function clamp01(x){ return Math.min(1, Math.max(EPS, x)); }
function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }

// ===== 事件 =====
recordBtn?.addEventListener("click", async () => {
  if (busy) return;
  try {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      await startRecording();
    } else {
      await stopRecording();
    }
  } catch (err){ console.error("[recordBtn]", err); setStatus("錄音啟動失敗"); }
});

fileInput?.addEventListener("change", async (e) => {
  if (busy) return;
  try {
    const f = e.target.files?.[0];
    if (!f) return;
    await handleFileOrBlob(f);
    e.target.value = "";
  } catch (err){ console.error("[fileInput]", err); setStatus("上傳處理失敗"); }
});

// ===== 錄音 =====
function pickSupportedMime(){
  const cands = ["audio/webm;codecs=opus","audio/webm","audio/mp4","audio/ogg"];
  try{
    if (typeof MediaRecorder!=="undefined" && MediaRecorder.isTypeSupported) {
      for (const t of cands) if (MediaRecorder.isTypeSupported(t)) return t;
    }
  }catch{}
  return "";
}
async function startRecording(){
  if (typeof MediaRecorder === "undefined"){
    setStatus("此瀏覽器不支援錄音，請改用右下角上傳"); return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio:true });
  chunks = [];
  const mimeType = pickSupportedMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  mediaRecorder.ondataavailable = (ev)=>{ if (ev.data?.size) chunks.push(ev.data); };
  mediaRecorder.onstop = async () => {
    try { await handleFileOrBlob(new Blob(chunks, { type: mimeType || "audio/webm" })); }
    catch (e) { console.error("[onstop]", e); setStatus("錄音處理失敗"); }
    finally { stream.getTracks().forEach(t=>t.stop()); }
  };
  document.body.classList.add("recording");
  document.querySelector(".container")?.classList.add("recording");
  setStatus("錄音中… 再按一次停止");
  mediaRecorder.start();
}
async function stopRecording(){
  if (mediaRecorder && mediaRecorder.state!=="inactive"){
    setStatus("處理音訊…", true);
    mediaRecorder.stop();
  }
  document.body.classList.remove("recording");
  document.querySelector(".container")?.classList.remove("recording");
}

// ===== 主流程 =====
async function handleFileOrBlob(fileOrBlob){
  busy = true;
  try {
    setStatus("解析檔案…", true);
    const { float32, sr } = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);

    const trimmed = trimSilence(float32, sr, TRIM_PAD_SEC);
    const { y: normed, gain } = loudnessNormalize(trimmed, TARGET_RMS, MAX_GAIN_DB);
    log(`[pre] duration=${fmtSec(normed.length/sr)}, gain=${gain.toFixed(2)}x`);

    if (normed.length / sr <= DIRECT_MAX_SEC) {
      await analyzeWhole(normed, sr);
    } else {
      avgChunkMs = 0;
      const { pf, pm } = await analyzeChunkedAdaptive(normed, sr);  // 第一階段
      // 若不確定 → 升級準確度
      const margin = Math.abs(pf - 0.5);
      if (margin < 0.2) {
        setStatus("信心不足，升級為高準確模式…", true);
        await microYield();
        await analyzeHighAccuracy(normed, sr);
      } else {
        setStatus("完成");
      }
    }
  } catch (e) {
    console.error("[handleFileOrBlob]", e); setStatus("處理失敗");
  } finally {
    busy = false;
  }
}

// ===== 解碼：音訊先 WebAudio；影片或失敗 → FFmpeg =====
async function decodeSmartToFloat32(blobOrFile, targetSR){
  const type = (blobOrFile.type || "").toLowerCase();
  const name = (blobOrFile.name || "");
  const isVideoContainer =
    type.startsWith("video/") ||
    /(\.mp4|\.mov|\.m4v)$/i.test(name) ||
    type.includes("video/quicktime");

  if (!isVideoContainer) {
    try {
      setStatus("直接解碼（WebAudio）…", true);
      return await decodeViaWebAudio(blobOrFile, targetSR);
    } catch (e) {
      log("[decode] WebAudio failed, fallback to ffmpeg:", e?.message || e);
    }
  }
  setStatus("轉檔（ffmpeg）準備中…", true);
  const wavBlob = await transcodeToWav16kViaFFmpeg(blobOrFile);
  const { float32, sr } = wavToFloat32(await wavBlob.arrayBuffer());
  return { float32, sr, durationSec: float32.length / sr };
}
async function decodeViaWebAudio(blobOrFile, targetSR=16000){
  const arrayBuf = await blobOrFile.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

  const mono = new AudioBuffer({ length: audioBuf.length, numberOfChannels: 1, sampleRate: audioBuf.sampleRate });
  const ch0 = audioBuf.getChannelData(0);
  if (audioBuf.numberOfChannels > 1) {
    const ch1 = audioBuf.getChannelData(1);
    const out = mono.getChannelData(0);
    for (let i=0;i<ch0.length;i++) out[i] = (ch0[i] + ch1[i]) / 2;
  } else {
    mono.copyToChannel(ch0, 0);
  }

  let out;
  if (audioBuf.sampleRate === targetSR) {
    out = mono.getChannelData(0).slice(0);
  } else {
    const offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
    const src = offline.createBufferSource();
    src.buffer = mono; src.connect(offline.destination); src.start(0);
    const rendered = await offline.startRendering();
    out = rendered.getChannelData(0).slice(0);
  }
  return { float32: out, sr: targetSR, durationSec: out.length / targetSR };
}

// ---- FFmpeg 載入 ----
async function loadFFmpegModule(){
  try {
    const m = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/+esm");
    if (typeof m.createFFmpeg === "function" && typeof m.fetchFile === "function") {
      return { createFFmpeg: m.createFFmpeg, fetchFile: m.fetchFile, mode: "esm" };
    }
  } catch (e) { log("[ffmpeg] +esm import failed:", e?.message || e); }
  const g = await import("https://unpkg.com/@ffmpeg/ffmpeg@0.12.6/dist/ffmpeg.min.js");
  const FF = (window.FFmpeg || g.FFmpeg);
  if (!FF || typeof FF.createFFmpeg !== "function") throw new Error("FFmpeg loader unavailable");
  return { createFFmpeg: FF.createFFmpeg, fetchFile: FF.fetchFile, mode: "global" };
}
async function transcodeToWav16kViaFFmpeg(blobOrFile){
  const { createFFmpeg, fetchFile, mode } = await loadFFmpegModule();
  log(`[ffmpeg] loader mode = ${mode}`);
  const ffmpeg = createFFmpeg({
    corePath: "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/ffmpeg-core.js",
    log: false
  });
  if (!ffmpeg.isLoaded()) setStatus("下載 ffmpeg…", true);
  ffmpeg.setProgress(({ ratio }) => {
    if (ratio > 0 && ratio <= 1) setStatus(`轉檔（ffmpeg）… ${Math.round(ratio*100)}%`, true);
  });
  if (!ffmpeg.isLoaded()) await ffmpeg.load();

  const inName = "in.bin", outName = "out.wav";
  ffmpeg.FS("writeFile", inName, await fetchFile(blobOrFile));
  await ffmpeg.run("-i", inName, "-vn", "-ac", "1", "-ar", `${TARGET_SR}`, "-f", "wav", outName);
  const out = ffmpeg.FS("readFile", outName);
  try { ffmpeg.FS("unlink", inName); } catch {}
  try { ffmpeg.FS("unlink", outName); } catch {}
  return new Blob([out.buffer], { type: "audio/wav" });
}

// ===== 前處理 =====
function frameRMS(x, sr, frameSec=0.02, hopSec=0.01) {
  const f = Math.max(1, Math.floor(frameSec*sr));
  const h = Math.max(1, Math.floor(hopSec*sr));
  const out = [];
  for (let s=0; s + f <= x.length; s+=h) {
    let sum=0; for (let i=0;i<f;i++){ const v=x[s+i]; sum+=v*v; }
    out.push(Math.sqrt(sum/Math.max(1,f)));
  }
  return out;
}
function percentile(arr, p){
  if (!arr.length) return 0;
  const a = arr.slice().sort((x,y)=>x-y);
  const idx = Math.min(a.length-1, Math.max(0, Math.floor(p/100 * a.length)));
  return a[idx];
}
function trimSilence(x, sr, padSec=0.25){
  const rmsArr = frameRMS(x, sr, 0.02, 0.01);
  const med = percentile(rmsArr, 60);
  const thr = Math.max(0.0025, med * 0.3);
  let first = 0, last = x.length-1;
  for (let i=0;i<rmsArr.length;i++){ if (rmsArr[i] > thr){ first = Math.floor(i * 0.01 * sr); break; } }
  for (let i=rmsArr.length-1;i>=0;i--){ if (rmsArr[i] > thr){ last = Math.min(x.length-1, Math.floor((i*0.01+0.02)*sr)); break; } }
  const pad = Math.floor(padSec * sr);
  const s = Math.max(0, first - pad);
  const e = Math.min(x.length, last + pad);
  if (e <= s + Math.floor(0.5*sr)) return x;
  return x.slice(s, e);
}
function rms(x){ let s=0; for (let i=0;i<x.length;i++){ const v=x[i]; s+=v*v; } return Math.sqrt(s/Math.max(1,x.length)); }
function loudnessNormalize(x, target=0.08, maxGainDb=20){
  const cur = rms(x);
  let gain = target / Math.max(EPS, cur);
  const maxGain = Math.pow(10, maxGainDb/20);
  gain = Math.min(gain, maxGain);
  const y = new Float32Array(x.length);
  for (let i=0;i<x.length;i++){
    const v = x[i] * gain;
    y[i] = v < -1 ? -1 : v > 1 ? 1 : v;
  }
  return { y, gain };
}

// ===== 模型 =====
async function ensurePipeline(){
  if (clf) return clf;
  setStatus("下載模型中…（首次會久一點）", true);
  const progress_callback = (p)=>{
    if (p?.status==="progress" && typeof p.progress==="number") {
      setStatus(`下載模型 ${Math.round(p.progress*100)}% …`, true);
    } else if (p?.status) {
      setStatus(`${p.status}…`, true);
    }
  };
  clf = await pipeline("audio-classification", MODEL_ID, { progress_callback });
  setStatus("模型就緒");
  return clf;
}

// ===== 推論（整段） =====
async function analyzeWhole(float32, sr){
  const model = await ensurePipeline();
  setStatus("分析中（整段）…", true);
  meter?.classList.remove("hidden");
  const res = await model(float32, { sampling_rate: sr, topk: 2 });
  const map = toMap(res);
  render(map.female||0, map.male||0);
  setStatus("完成");
}

// ===== 推論（自適應，50% 重疊，時長加權） =====
async function analyzeChunkedAdaptive(float32, sr){
  const model = await ensurePipeline();
  meter?.classList.remove("hidden");

  const totalSec = float32.length / sr;
  // 目標片段數（依時間上限）
  const targetChunks = Math.max(3, Math.floor((TIME_BUDGET_S * 1000) / Math.max(1, (avgChunkMs || EST_MS_PER_CHUNK_INIT))));
  // window ≈ 2T/N（因為 hop = window/2）
  let windowSec = clamp((2 * totalSec) / targetChunks, MIN_WINDOW_S, MAX_WINDOW_S);
  let hopSec    = Math.max(MIN_HOP_S, windowSec * (1 - OVERLAP_RATIO)); // = window/2

  let chunks = buildChunks(float32.length, sr, windowSec, hopSec);
  setStatus(`分析中（自適應 50% 重疊）… 0%｜片段 0/${chunks.length}`, true);

  let logitSum = 0, weightSum = 0;
  let processed = 0;
  const started = performance.now();
  const votes = []; // 每片段 pf 記錄，用來判斷是否升級

  for (let i=0;i<chunks.length;i++){
    const [s0,s1] = chunks[i];
    const seg = float32.subarray(s0, s1);
    const dur = (s1 - s0) / sr;

    const t0 = performance.now();
    const out = await model(seg, { sampling_rate: sr, topk: 2 });
    const dt = performance.now() - t0;
    avgChunkMs = avgChunkMs === 0 ? dt : (avgChunkMs*0.6 + dt*0.4);

    const map = toMap(out);
    const pf = clamp01(map.female || EPS);
    const pm = clamp01(map.male   || EPS);
    votes.push(pf);

    // 對數勝算 +「時長加權」（不再用能量，避免伴奏拉偏）
    const logit = Math.log(pf) - Math.log(pm);     // 等於 log(pf/(1-pf))
    const w = dur;                                  // 權重 = 片段時長
    logitSum += logit * w;
    weightSum += w;
    processed = s1;

    const logitAvg = logitSum / Math.max(weightSum, EPS);
    const pf_now = 1 / (1 + Math.exp(-logitAvg));
    const pm_now = 1 - pf_now;
    render(pf_now, pm_now);

    const pct = Math.round(((i+1)/chunks.length)*100);
    const remain = (chunks.length - i - 1);
    const etaSec = remain * (avgChunkMs/1000);
    const elapsed = (performance.now() - started)/1000;
    setStatus(`分析中 ${pct}%｜片段 ${i+1}/${chunks.length}｜已處理 ${fmtSec(processed/sr)} / ${fmtSec(totalSec)}｜預估剩餘 ~ ${fmtSec(etaSec)}｜已用 ${fmtSec(elapsed)}`, true);

    // 前 2 段後重新計算 window（更準確的 avgChunkMs）
    if (i === 1) {
      const tgt = Math.max(3, Math.floor((TIME_BUDGET_S * 1000) / Math.max(1, avgChunkMs)));
      windowSec = clamp((2 * totalSec) / tgt, MIN_WINDOW_S, MAX_WINDOW_S);
      hopSec    = Math.max(MIN_HOP_S, windowSec * (1 - OVERLAP_RATIO));
      const nextStart = s1;
      if (nextStart < float32.length) {
        const rest = buildChunks(float32.length - nextStart, sr, windowSec, hopSec, nextStart);
        chunks = chunks.slice(0, i+1).concat(rest);
        setStatus(`重新規劃：window=${windowSec.toFixed(1)}s / hop=${hopSec.toFixed(1)}s｜剩餘 ${rest.length} 片段`, true);
      }
    }

    await microYield();
  }

  const logitAvg = logitSum / Math.max(weightSum, EPS);
  const pf = 1 / (1 + Math.exp(-logitAvg));
  const pm = 1 - pf;

  // 回傳給上層判斷是否需要升級
  return { pf, pm, votes };
}

// ===== 高準確升級（20s 視窗 / 5s hop，75% 重疊，時長加權） =====
async function analyzeHighAccuracy(float32, sr){
  const model = await ensurePipeline();
  const totalSec = float32.length / sr;
  let windowSec = Math.min(HA_WINDOW_S, totalSec);
  let hopSec    = Math.min(HA_HOP_S, Math.max(0.5, windowSec * 0.25)); // 75% overlap

  const chunks = buildChunks(float32.length, sr, windowSec, hopSec);
  setStatus(`高準確模式… 0%｜片段 0/${chunks.length}`, true);
  meter?.classList.remove("hidden");

  let logitSum = 0, weightSum = 0;
  let processed = 0;
  const started = performance.now();

  for (let i=0;i<chunks.length;i++){
    const [s0,s1] = chunks[i];
    const seg = float32.subarray(s0, s1);
    const dur = (s1 - s0) / sr;

    const t0 = performance.now();
    const out = await model(seg, { sampling_rate: sr, topk: 2 });
    const dt = performance.now() - t0;
    avgChunkMs = avgChunkMs === 0 ? dt : (avgChunkMs*0.6 + dt*0.4);

    const map = toMap(out);
    const pf = clamp01(map.female || EPS);
    const pm = clamp01(map.male   || EPS);

    const logit = Math.log(pf) - Math.log(pm);
    const w = dur; // 時長加權
    logitSum += logit * w;
    weightSum += w;
    processed = s1;

    const logitAvg = logitSum / Math.max(weightSum, EPS);
    const pf_now = 1 / (1 + Math.exp(-logitAvg));
    const pm_now = 1 - pf_now;
    render(pf_now, pm_now);

    const pct = Math.round(((i+1)/chunks.length)*100);
    const remain = (chunks.length - i - 1);
    const etaSec = remain * (avgChunkMs/1000);
    const elapsed = (performance.now() - started)/1000;
    setStatus(`高準確模式 ${pct}%｜片段 ${i+1}/${chunks.length}｜已處理 ${fmtSec(processed/sr)} / ${fmtSec(totalSec)}｜預估剩餘 ~ ${fmtSec(etaSec)}｜已用 ${fmtSec(elapsed)}`, true);

    // 上限保護（避免極長檔跑太久）
    if (elapsed > HA_MAX_SEC) {
      setStatus("高準確模式超時，已提前結束。");
      break;
    }
    await microYield();
  }

  const logitAvg = logitSum / Math.max(weightSum, EPS);
  const pf = 1 / (1 + Math.exp(-logitAvg));
  const pm = 1 - pf;
  render(pf, pm);
  setStatus("完成（高準確）");
}

// ===== chunk 工具 =====
function buildChunks(totalSamples, sr, windowSec, hopSec, startSample=0){
  const win = Math.max(1, Math.floor(windowSec * sr));
  const hop = Math.max(1, Math.floor(hopSec   * sr));
  const end = startSample + totalSamples;
  const chunks = [];
  for (let s = startSample; s < end; s += hop) {
    const e = Math.min(s + win, end);
    if (e - s <= Math.floor(0.5 * sr)) break;
    chunks.push([s, e]);
    if (e === end) break;
  }
  if (!chunks.length) chunks.push([startSample, Math.min(startSample + win, end)]);
  return chunks;
}

// ===== 其它工具 =====
function toMap(arr){
  const m = { female: 0, male: 0 };
  if (Array.isArray(arr)) for (const r of arr) {
    if (r && typeof r.label === "string") m[r.label] = typeof r.score === "number" ? r.score : 0;
  }
  return m;
}
function render(pf, pm){
  const barF = document.querySelector(".bar.female");
  const barM = document.querySelector(".bar.male");
  if (barF) barF.style.setProperty("--p", pf ?? 0);
  if (barM) barM.style.setProperty("--p", pm ?? 0);
  if (femaleVal) femaleVal.textContent = `${((pf ?? 0) * 100).toFixed(1)}%`;
  if (maleVal)   maleVal.textContent   = `${((pm ?? 0) * 100).toFixed(1)}%`;
}
function microYield(){ return new Promise(r=>setTimeout(r,0)); }

// 解析 16-bit PCM WAV → Float32
function wavToFloat32(arrayBuffer){
  const view = new DataView(arrayBuffer);
  const riff = str(view,0,4), wave = str(view,8,4);
  if (riff!=="RIFF"||wave!=="WAVE") throw new Error("Not a WAV");
  let pos=12, fmt={}, dataOffset=-1, dataSize=0;
  while (pos < view.byteLength) {
    const id = str(view,pos,4); pos+=4;
    const size = view.getUint32(pos,true); pos+=4;
    if (id==="fmt ") {
      fmt.audioFormat   = view.getUint16(pos+0,true);
      fmt.numChannels   = view.getUint16(pos+2,true);
      fmt.sampleRate    = view.getUint32(pos+4,true);
      fmt.bitsPerSample = view.getUint16(pos+14,true);
    } else if (id==="data") { dataOffset=pos; dataSize=size; break; }
    pos += size;
  }
  if (dataOffset<0) throw new Error("WAV data not found");
  if (fmt.audioFormat!==1 || fmt.bitsPerSample!==16) throw new Error("Expect 16-bit PCM");
  const bytes = fmt.bitsPerSample/8;
  const frames = dataSize / (bytes * fmt.numChannels);
  const out = new Float32Array(frames);
  for (let i=0;i<frames;i++){
    const off = dataOffset + i*bytes*fmt.numChannels;
    const s = view.getInt16(off,true);
    out[i] = s/32768;
  }
  return { float32: out, sr: fmt.sampleRate };
}
function str(v,s,l){ let x=""; for(let i=0;i<l;i++) x+=String.fromCharCode(v.getUint8(s+i)); return x; }
