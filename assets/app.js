// assets/app.js — 整段一次、不分段、不改音量/不去靜音（純前端 / Transformers.js + ONNX）
// 注意：為符合模型輸入，僅做「必要」處理：單聲道混合 + 重採樣至 16kHz。
// 其餘（靜音、音量、內容）皆不改，整段送入模型。

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// GH Pages 沒有 COOP/COEP → ONNX WASM 設成單執行緒，WebGPU 可用時會自動走 GPU
env.backends.onnx.wasm.numThreads = 1;

// ===== 參數（你可視需要微調） =====
const MODEL_ID  = (window.ONNX_MODEL_ID || "prithivMLmods/Common-Voice-Gender-Detection-ONNX");
const TARGET_SR = 16000;   // 模型需求：16 kHz
const WARN_LONG_SEC = 180; // 超過 3 分鐘提醒（仍會照跑，但瀏覽器可能較吃力）
const EPS = 1e-9;

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
let busy = false;
let heartbeatTimer = null;

log("[app] single-pass (no-chunk, no-trim, no-norm) ready.");

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

// ===== 主流程（整段一次） =====
async function handleFileOrBlob(fileOrBlob){
  busy = true;
  try {
    setStatus("解析檔案…", true);
    const { float32, sr, durationSec } = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);

    if (durationSec > WARN_LONG_SEC) {
      setStatus(`提示：長度 ${fmtSec(durationSec)}，直接整段分析可能較久。開始推論…`, true);
      await microYield();
    }

    await analyzeWhole(float32, sr, durationSec);
  } catch (e) {
    console.error("[handleFileOrBlob]", e);
    setStatus("處理失敗");
  } finally {
    busy = false;
  }
}

// ===== 解碼：音訊先 WebAudio；影片或失敗 → FFmpeg.wasm（仍「不分段、不修音」，僅重採樣到 16k & 單聲道） =====
async function decodeSmartToFloat32(blobOrFile, targetSR){
  const type = (blobOrFile.type || "").toLowerCase();
  const name = (blobOrFile.name || "");
  const maybeVideo =
    type.startsWith("video/") ||
    /(\.mp4|\.mov|\.m4v)$/i.test(name) ||
    type.includes("video/quicktime");

  if (!maybeVideo) {
    try {
      setStatus("直接解碼（WebAudio）…", true);
      return await decodeViaWebAudio(blobOrFile, targetSR);
    } catch (e) {
      log("[decode] WebAudio failed, fallback to ffmpeg:", e?.message || e);
    }
  }

  setStatus("轉檔（ffmpeg）準備中…", true);
  const wavBlob = await transcodeToWav16kViaFFmpeg(blobOrFile);
  const { float32, sr } = wavToFloat32(await wavBlob.arrayBuffer()); // 已是 16k/mono
  return { float32, sr, durationSec: float32.length / sr };
}

async function decodeViaWebAudio(blobOrFile, targetSR=16000){
  const arrayBuf = await blobOrFile.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

  // 單聲道（必要）：模型輸入是一維向量
  const mono = new AudioBuffer({ length: audioBuf.length, numberOfChannels: 1, sampleRate: audioBuf.sampleRate });
  const ch0 = audioBuf.getChannelData(0);
  if (audioBuf.numberOfChannels > 1) {
    const ch1 = audioBuf.getChannelData(1);
    const out = mono.getChannelData(0);
    for (let i=0;i<ch0.length;i++) out[i] = (ch0[i] + ch1[i]) / 2;
  } else {
    mono.copyToChannel(ch0, 0);
  }

  // 僅為符合模型而重採樣到 16k（內容不裁、不調音量）
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

// ---- FFmpeg 載入（影片容器才會用到） ----
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

// ===== 推論（整段一次，不分段） =====
async function analyzeWhole(float32, sr, durationSec){
  const model = await ensurePipeline();
  meter?.classList.remove("hidden");

  // 心跳式進度：每秒更新已用時，避免以為卡住
  const started = performance.now();
  startHeartbeat(() => {
    const elapsed = (performance.now() - started)/1000;
    setStatus(`分析中（整段、不分段）｜音檔 ${fmtSec(durationSec)}｜已用 ${fmtSec(elapsed)}`, true);
  });

  try {
    const res = await model(float32, { sampling_rate: sr, topk: 2 });
    const map = toMap(res);
    render(map.female||0, map.male||0);
    setStatus("完成");
  } finally {
    stopHeartbeat();
  }
}

// ===== UI 工具 =====
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
function startHeartbeat(tickFn){
  stopHeartbeat();
  heartbeatTimer = setInterval(() => { try{ tickFn(); }catch{} }, 1000);
}
function stopHeartbeat(){
  if (heartbeatTimer){ clearInterval(heartbeatTimer); heartbeatTimer = null; }
}
function microYield(){ return new Promise(r=>setTimeout(r,0)); }

// ===== 解析 16-bit PCM WAV → Float32（單聲道），僅供 FFmpeg 轉出的 WAV 使用 =====
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
