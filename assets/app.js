// assets/app.js — 高準確 + 高體感版（全前端 / Transformers.js + ONNX WASM）
// 特色：短檔整段、長檔重疊切窗 + 對數勝算聚合；自動去靜音與音量校正；全程即時進度/ETA。
// 依賴：@xenova/transformers（自動抓 ONNX），@ffmpeg/ffmpeg（僅 mp4 或解碼失敗時載入）.

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// 讓 GH Pages 這種無 COOP/COEP 也能跑：WASM 單執行緒
env.backends.onnx.wasm.numThreads = 1;

// ===== 模型與全域參數 =====
const MODEL_ID       = (window.ONNX_MODEL_ID || "prithivMLmods/Common-Voice-Gender-Detection-ONNX");
const TARGET_SR      = 16000; // 16k
const DIRECT_MAX_SEC = 25;    // ≤ 這秒數「整段一次跑」
const WINDOW_SEC     = 8;     // 長檔分段：窗長（秒）
const HOP_SEC        = 1;     // 長檔分段：位移（秒，8/1 = 87.5% overlap）
const ENERGY_GAMMA   = 2;     // 權重 = duration * (RMS^gamma)
const SILENCE_FLOOR  = 1e-4;  // 最小權重下限（避免純靜音把分數拉亂）
const TRIM_PAD_SEC   = 0.25;  // 去頭尾靜音時保留的邊界
const TARGET_RMS     = 0.08;  // 音量校正目標 RMS（約 -22 dBFS）
const MAX_GAIN_DB    = 20;    // 校正最大增益（避免過度放大噪音）
const EPS            = 1e-9;

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
let clf = null;        // transformers.js pipeline
let busy = false;      // 避免重入
let t0_chunk = 0;      // for ETA

log("[app] accuracy+UX build ready (browser-only).");

function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}
function log(...a){ try{ console.log(...a);}catch{} }
function fmtSec(s){ if (!isFinite(s)) return "—"; const m=Math.floor(s/60), ss=Math.round(s%60); return m? `${m}分${ss}秒`:`${ss}秒`; }
function clamp01(x){ return Math.min(1, Math.max(EPS, x)); }

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
    try {
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      await handleFileOrBlob(blob);
    } catch (e) {
      console.error("[onstop]", e); setStatus("錄音處理失敗");
    } finally {
      stream.getTracks().forEach(t=>t.stop());
    }
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

    // 去頭尾靜音（保留邊界）
    const trimmed = trimSilence(float32, sr, TRIM_PAD_SEC);
    // 音量校正（限 20 dB）
    const { y: normed, gain } = loudnessNormalize(trimmed, TARGET_RMS, MAX_GAIN_DB);

    log(`[pre] duration=${fmtSec(normed.length/sr)}, gain=${gain.toFixed(2)}x`);

    // 短檔整段、長檔切窗
    if (normed.length / sr <= DIRECT_MAX_SEC) {
      await analyzeWhole(normed, sr);
    } else {
      await analyzeChunked(normed, sr);
    }
  } catch (e) {
    console.error("[handleFileOrBlob]", e); setStatus("處理失敗");
  } finally {
    busy = false;
  }
}

// ===== 智慧解碼：WebAudio 優先；mp4 或失敗 → FFmpeg.wasm =====
async function decodeSmartToFloat32(blobOrFile, targetSR){
  const type = (blobOrFile.type || "").toLowerCase();
  const name = (blobOrFile.name || "");
  const isMP4 = type.includes("video/mp4") || type.includes("audio/mp4") || /\.mp4$/i.test(name);

  if (!isMP4) {
    try {
      return await decodeViaWebAudio(blobOrFile, targetSR);
    } catch (e) {
      log("[decode] WebAudio failed, fallback to ffmpeg:", e?.message || e);
    }
  }
  const wavBlob = await transcodeToWav16kViaFFmpeg(blobOrFile);
  const { float32, sr } = wavToFloat32(await wavBlob.arrayBuffer()); // 已是 16k/mono
  return { float32, sr, durationSec: float32.length / sr };
}

async function decodeViaWebAudio(blobOrFile, targetSR=16000){
  const arrayBuf = await blobOrFile.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

  // 單聲道
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

async function transcodeToWav16kViaFFmpeg(blobOrFile){
  setStatus("轉檔（ffmpeg）準備中…", true);
  const { createFFmpeg, fetchFile } = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/+esm");
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

// ===== 前處理：去頭尾靜音、音量校正 =====
function frameRMS(x, sr, frameSec=0.02, hopSec=0.01) {
  const f = Math.max(1, Math.floor(frameSec*sr));
  const h = Math.max(1, Math.floor(hopSec*sr));
  const out = [];
  for (let s=0; s + f <= x.length; s+=h) {
    let sum=0;
    for (let i=0;i<f;i++){ const v=x[s+i]; sum+=v*v; }
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

// 去頭尾靜音（保留 padSec 邊界）
function trimSilence(x, sr, padSec=0.25){
  const rmsArr = frameRMS(x, sr, 0.02, 0.01);
  const med = percentile(rmsArr, 60);
  const thr = Math.max(0.0025, med * 0.3);
  let first = 0, last = x.length-1;

  // 找第一個超過門檻的 frame
  for (let i=0;i<rmsArr.length;i++){ if (rmsArr[i] > thr){ first = Math.floor(i * 0.01 * sr); break; } }
  // 找最後一個超過門檻的 frame
  for (let i=rmsArr.length-1;i>=0;i--){ if (rmsArr[i] > thr){ last = Math.min(x.length-1, Math.floor((i*0.01+0.02)*sr)); break; } }

  const pad = Math.floor(padSec * sr);
  const s = Math.max(0, first - pad);
  const e = Math.min(x.length, last + pad);
  if (e <= s + Math.floor(0.5*sr)) return x; // 太短就不裁
  return x.slice(s, e);
}

// 音量校正（限最大增益）
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

// ===== 推論：短檔整段 =====
async function analyzeWhole(float32, sr){
  const model = await ensurePipeline();
  setStatus("分析中（整段）…", true);
  meter?.classList.remove("hidden");
  const res = await model(float32, { sampling_rate: sr, topk: 2 });
  const map = toMap(res);
  render(map.female||0, map.male||0);
  setStatus("完成");
}

// ===== 推論：長檔重疊切窗 + 對數勝算聚合 =====
async function analyzeChunked(float32, sr){
  const model = await ensurePipeline();
  const totalSec = float32.length / sr;
  meter?.classList.remove("hidden");

  const win = Math.floor(WINDOW_SEC * sr);
  const hop = Math.floor(HOP_SEC * sr);

  // 建 chunk 清單
  const chunks = [];
  for (let s=0; s<float32.length; s+=hop){
    const e = Math.min(s + win, float32.length);
    if (e - s <= Math.floor(0.5*sr)) break; // 最後不足 0.5s 就跳過
    chunks.push([s,e]);
    if (e === float32.length) break; // 到尾了
  }
  if (!chunks.length) chunks.push([0, Math.min(win, float32.length)]);

  let logitSum = 0, weightSum = 0;
  let processedSec = 0;
  let avgChunkMs = 0;

  for (let i=0;i<chunks.length;i++){
    const [s0,s1] = chunks[i];
    const seg = float32.subarray(s0, s1);
    const dur = (s1 - s0) / sr;

    // 權重：時長 × (能量^gamma)，但至少有 SILENCE_FLOOR × 時長
    const eSeg = rms(seg);
    const w = Math.max(dur * Math.pow(eSeg + EPS, ENERGY_GAMMA), dur * SILENCE_FLOOR);

    // 推論
    t0_chunk = performance.now();
    const out = await model(seg, { sampling_rate: sr, topk: 2 });
    const dt = performance.now() - t0_chunk;
    avgChunkMs = avgChunkMs === 0 ? dt : (avgChunkMs*0.6 + dt*0.4);

    const map = toMap(out);
    const pf = clamp01(map.female || EPS);
    const pm = clamp01(map.male   || EPS);
    const logit = Math.log(pf) - Math.log(pm);

    logitSum += logit * w;
    weightSum += w;
    processedSec = Math.min(totalSec, (s1 / sr));

    // 即時聚合 → 機率
    const logitAvg = logitSum / Math.max(weightSum, EPS);
    const pf_now = 1 / (1 + Math.exp(-logitAvg));
    const pm_now = 1 - pf_now;
    render(pf_now, pm_now);

    // ETA
    const remainChunks = (chunks.length - i - 1);
    const etaSec = (remainChunks * (avgChunkMs/1000));
    const pct = Math.round(((i+1)/chunks.length)*100);
    setStatus(`分析中 ${pct}%｜片段 ${i+1}/${chunks.length}｜已處理 ${fmtSec(processedSec)} / ${fmtSec(totalSec)}｜預估剩餘 ~ ${fmtSec(etaSec)}`, true);

    await microYield();
  }

  const logitAvg = logitSum / Math.max(weightSum, EPS);
  const pf = 1 / (1 + Math.exp(-logitAvg));
  const pm = 1 - pf;
  render(pf, pm);
  setStatus("完成");
}

// ===== 工具 =====
function toMap(arr){
  const m = { female: 0, male: 0 };
  if (Array.isArray(arr)) for (const r of arr) {
    if (r && typeof r.label === "string") m[r.label] = typeof r.score === "number" ? r.score : 0;
  }
  return m;
}

// ★★★ 這個就是你漏掉的函式：把分數畫到兩條進度條上 ★★★
function render(pf, pm){
  const barF = document.querySelector(".bar.female");
  const barM = document.querySelector(".bar.male");
  if (barF) barF.style.setProperty("--p", pf ?? 0);
  if (barM) barM.style.setProperty("--p", pm ?? 0);
  if (femaleVal) femaleVal.textContent = `${((pf ?? 0) * 100).toFixed(1)}%`;
  if (maleVal)   maleVal.textContent   = `${((pm ?? 0) * 100).toFixed(1)}%`;
}

function microYield(){ return new Promise(r=>setTimeout(r,0)); }

// 解析 16-bit PCM WAV → Float32（單聲道）
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
