// assets/app.js — 整段一次、不分段、不改音量/不去靜音（純前端 / Transformers.js + ONNX）
// 美觀 + 播放鍵 + 記憶體防爆版
// - 先 WebAudio 解碼；失敗才 ffmpeg.wasm（jsDelivr）→ 轉完立刻 exit() 釋放
// - 只保留「最後一次」音檔：舊 URL revoke、不囤 blob
// - 關閉 AudioContext、清 chunks、丟大型陣列參考，避免長玩變胖
// - 心跳進度文字，避免以為卡住

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// GH Pages 無 COOP/COEP → ONNX WASM 單執行緒（若可用 WebGPU 會自動走 GPU）
env.backends.onnx.wasm.numThreads = 1;

// ===== 參數 =====
const MODEL_ID  = (window.ONNX_MODEL_ID || "prithivMLmods/Common-Voice-Gender-Detection-ONNX");
const TARGET_SR = 16000;         // 模型需求：16 kHz
const WARN_LONG_SEC = 180;       // >3 分鐘提醒（仍會照跑）
const EPS = 1e-9;

// ===== DOM =====
const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// ===== 播放器 UI（動態建立，無需改 HTML/CSS） =====
let playBtn = null;
let audioEl = null;
let lastAudioUrl = null;

ensurePlayerUI();

// ===== 狀態 =====
let mediaRecorder = null;
let chunks = [];
let clf = null;        // transformers.js pipeline
let busy = false;
let heartbeatTimer = null;

log("[app] single-pass (no-chunk, no-trim, no-norm) + player + GC-safe ready.");

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
      chunks.length = 0; // ★ 立刻丟掉暫存
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
  let decoded = null;
  try {
    // 先把原始檔綁到播放器（只保留最新一個）
    setPlaybackSource(fileOrBlob);

    setStatus("解析檔案…", true);
    decoded = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
    const { float32, sr, durationSec } = decoded;

    if (durationSec > WARN_LONG_SEC) {
      setStatus(`提示：長度 ${fmtSec(durationSec)}，整段分析可能較久。開始推論…`, true);
      await microYield();
    }

    await analyzeWhole(float32, sr, durationSec);
  } catch (e) {
    console.error("[handleFileOrBlob]", e);
    setStatus("處理失敗");
  } finally {
    // ★ 放掉大陣列參考（讓 GC 收）
    if (decoded) decoded.float32 = null;
    decoded = null;
    busy = false;
  }
}

// ===== 解碼策略 =====
// 1) 一律先嘗試 WebAudio 直接解碼 → 保留原聲（僅做單聲道 + 16k 重採樣）
// 2) 失敗才用 ffmpeg.wasm 轉 16k/mono WAV（轉完 exit() 釋放記憶體）
async function decodeSmartToFloat32(blobOrFile, targetSR){
  try {
    setStatus("直接解碼（WebAudio）…", true);
    return await decodeViaWebAudio(blobOrFile, targetSR);
  } catch (e) {
    log("[decode] WebAudio failed, fallback to ffmpeg:", e?.message || e);
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
  let offline = null;
  try {
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
      offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
      const src = offline.createBufferSource();
      src.buffer = mono; src.connect(offline.destination); src.start(0);
      const rendered = await offline.startRendering();
      out = rendered.getChannelData(0).slice(0);
    }
    return { float32: out, sr: targetSR, durationSec: out.length / targetSR };
  } finally {
    try { await ctx.close(); } catch {}
    offline = null; // 讓 GC 收
  }
}

// ---- FFmpeg 載入（只有當 WebAudio 失敗時才會用到） ----
// 優先 ESM（jsDelivr +esm）；若失敗，動態插入 <script> UMD（同樣 jsDelivr）
async function loadFFmpegModule(){
  try {
    const m = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/+esm");
    if (typeof m.createFFmpeg === "function" && typeof m.fetchFile === "function") {
      return { createFFmpeg: m.createFFmpeg, fetchFile: m.fetchFile, mode: "esm" };
    }
  } catch (e) {
    log("[ffmpeg] +esm import failed:", e?.message || e);
  }
  await loadScriptTag("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/dist/ffmpeg.min.js");
  const FF = window.FFmpeg;
  if (!FF || typeof FF.createFFmpeg !== "function") {
    throw new Error("FFmpeg UMD load failed");
  }
  return { createFFmpeg: FF.createFFmpeg, fetchFile: FF.fetchFile, mode: "umd" };
}

function loadScriptTag(src){
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.crossOrigin = "anonymous";
    s.referrerPolicy = "no-referrer";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("script load error: " + src));
    document.head.appendChild(s);
  });
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
  // -vn 去視訊；-ac 1 單聲道；-ar 16000 取樣率；輸出 PCM WAV
  await ffmpeg.run("-i", inName, "-vn", "-ac", "1", "-ar", `${TARGET_SR}`, "-f", "wav", outName);
  const out = ffmpeg.FS("readFile", outName);
  try { ffmpeg.FS("unlink", inName); } catch {}
  try { ffmpeg.FS("unlink", outName); } catch {}

  // ★ 關鍵：退出釋放 WASM 記憶體
  try { await ffmpeg.exit(); } catch {}

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

// ===== 播放器：用「剛才送去分析的原檔」 =====
function ensurePlayerUI(){
  const container = document.querySelector(".container");
  if (!container) return;

  // 包一個小卡片，美美的
  const wrap = document.createElement("div");
  wrap.className = "player";
  wrap.style.cssText = `
    margin-top: 18px; padding: 14px; border-radius: 14px;
    background: linear-gradient(180deg, rgba(21,25,34,0.55), rgba(21,25,34,0.35));
    border: 1px solid #1f2937; display: flex; align-items: center; gap: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,.25);
  `;

  const btn = document.createElement("button");
  btn.id = "playBtn";
  btn.type = "button";
  btn.disabled = true;
  btn.textContent = "▶︎ 播放剛才的聲音";
  btn.ariaLabel = "播放剛才的聲音";
  btn.style.cssText = `
    padding: 10px 14px; font-size: 14px; border-radius: 999px;
    background: linear-gradient(135deg, color-mix(in oklab, var(--accent) 85%, #fff 8%), #1f2937);
    color: #0b0c10; font-weight: 700; letter-spacing: .2px;
    border: none; cursor: pointer; box-shadow: 0 6px 18px rgba(110,231,183,.25);
    transition: transform .06s ease, filter .2s ease, opacity .2s ease;
    opacity: .92;
  `;
  btn.onmouseenter = () => { btn.style.transform = "translateY(-1px)"; btn.style.filter = "brightness(1.05)"; };
  btn.onmouseleave = () => { btn.style.transform = "translateY(0)"; btn.style.filter = "none"; };

  const hint = document.createElement("div");
  hint.textContent = "想再聽一次剛才那段嗎？點這裡。";
  hint.style.cssText = "color: var(--muted); font-size: 13px;";

  const audio = document.createElement("audio");
  audio.id = "playback";
  audio.preload = "metadata";
  audio.style.display = "none";

  wrap.appendChild(btn);
  wrap.appendChild(hint);
  wrap.appendChild(audio);
  container.appendChild(wrap);

  playBtn = btn;
  audioEl = audio;

  playBtn.onclick = async () => {
    if (!audioEl.src) return;
    try {
      if (audioEl.paused) {
        await audioEl.play();
        playBtn.textContent = "⏸ 暫停播放";
      } else {
        audioEl.pause();
        playBtn.textContent = "▶︎ 播放剛才的聲音";
      }
    } catch (e) {
      console.error("[audio play]", e);
    }
  };
  audioEl.onended = () => { playBtn.textContent = "▶︎ 播放剛才的聲音"; };
}

function setPlaybackSource(blob){
  try {
    if (!audioEl || !playBtn) return;
    // 清理舊 URL，避免記憶體洩漏
    if (lastAudioUrl) { try { URL.revokeObjectURL(lastAudioUrl); } catch {} lastAudioUrl = null; }
    lastAudioUrl = URL.createObjectURL(blob);
    audioEl.src = lastAudioUrl;
    audioEl.load();
    playBtn.disabled = false;
    playBtn.textContent = "▶︎ 播放剛才的聲音";
  } catch (e) {
    console.error("[setPlaybackSource]", e);
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

// ===== 離站清理：離開頁面時釋放最後 URL =====
window.addEventListener("beforeunload", () => {
  if (lastAudioUrl) { try { URL.revokeObjectURL(lastAudioUrl); } catch {} }
});
