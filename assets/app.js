// assets/app.js — 全前端（GitHub Pages 可用），不走任何自家 API。
// 功能：錄音 / 上傳（含 .mp3/.m4a/.wav/.mp4），完整解碼 → 16k 單聲道 → 分段（非取樣）跑整段推論 → 輸出 female/male。
// 依賴：Transformers.js（ONNX WASM），FFmpeg.wasm（僅在需要解 .mp4 或瀏覽器解碼失敗時才載入）。

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// 確保在沒有 COOP/COEP 的靜態頁也能跑（關閉多執行緒）
env.backends.onnx.wasm.numThreads = 1;

// 模型（需提供 ONNX 版）
const MODEL_ID = "prithivMLmods/Common-Voice-Gender-Detection-ONNX";

// 介面元素
const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// 狀態
let mediaRecorder = null;
let chunks = [];
let clf = null; // transformers.js pipeline（lazy）
let isRunning = false;

// 分段推論參數（處理「整段」而非取樣中間）
const WINDOW_SEC = 10;     // 每片段長度（秒）
const HOP_SEC    = 10;     // 片段位移（秒），=WINDOW_SEC 表示無重疊
const TARGET_SR  = 16000;  // 目標取樣率

log("[app] browser-only mode ready");

function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}
function log(...a){ try{ console.log(...a); }catch{} }

// 綁定事件
if (recordBtn) {
  recordBtn.addEventListener("click", async () => {
    if (isRunning) return; // 分析中避免重入
    try {
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        await startRecording();
      } else {
        await stopRecording();
      }
    } catch (err) {
      console.error("[recordBtn]", err);
      setStatus("錄音啟動失敗");
    }
  });
}
if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    if (isRunning) return;
    try {
      if (!e.target.files || e.target.files.length === 0) return;
      const file = e.target.files[0];
      await handleFileOrBlob(file);
      e.target.value = "";
    } catch (err) {
      console.error("[fileInput]", err);
      setStatus("上傳處理失敗");
    }
  });
}

// 錄音
function pickSupportedMime() {
  const cands = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg"
  ];
  try {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported) {
      for (const t of cands) if (MediaRecorder.isTypeSupported(t)) return t;
    }
  } catch {}
  return "";
}
async function startRecording() {
  if (typeof MediaRecorder === "undefined") {
    setStatus("此瀏覽器不支援錄音，請用右下角上傳");
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  const mimeType = pickSupportedMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

  mediaRecorder.ondataavailable = (ev) => { if (ev.data && ev.data.size > 0) chunks.push(ev.data); };
  mediaRecorder.onstop = async () => {
    try {
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      await handleFileOrBlob(blob);
    } catch (e) {
      console.error("[onstop]", e);
      setStatus("錄音處理失敗");
    } finally {
      stream.getTracks().forEach((t) => t.stop());
    }
  };

  document.body.classList.add("recording");
  const box = document.querySelector(".container");
  if (box) box.classList.add("recording");
  setStatus("錄音中… 再按一次停止");
  mediaRecorder.start();
}
async function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    setStatus("處理音訊…", true);
    mediaRecorder.stop();
  }
  document.body.classList.remove("recording");
  const box = document.querySelector(".container");
  if (box) box.classList.remove("recording");
}

// 主流程：檔案/Blob → Float32(16k, mono) → 分段推論
async function handleFileOrBlob(fileOrBlob) {
  try {
    isRunning = true;
    setStatus("解析檔案…", true);
    const { float32, sr } = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
    await analyzeByChunks(float32, sr);
  } catch (e) {
    console.error("[handleFileOrBlob]", e);
    setStatus("處理失敗");
  } finally {
    isRunning = false;
  }
}

// 智慧解碼：優先用 WebAudio；失敗或遇到 .mp4（含影片容器）則用 FFmpeg.wasm 轉成 16k 單聲道 WAV 再讀
async function decodeSmartToFloat32(blobOrFile, targetSR) {
  const type = (blobOrFile.type || "").toLowerCase();
  const name = (blobOrFile.name || "");
  const isLikelyMP4 = type.includes("video/mp4") || type.includes("audio/mp4") || /\.mp4$/i.test(name);

  // 1) 嘗試用 WebAudio 直接解
  if (!isLikelyMP4) {
    try {
      return await decodeViaWebAudio(blobOrFile, targetSR);
    } catch (e) {
      log("[decode] WebAudio failed, fall back to ffmpeg", e?.message || e);
    }
  }

  // 2) Fallback: FFmpeg.wasm 轉 16k 單聲道 WAV，再解析
  const wavBlob = await transcodeToWav16kViaFFmpeg(blobOrFile);
  // 直接轉成 Float32（避免再 decodeAudioData 一次）
  const { float32 } = wavToFloat32(await wavBlob.arrayBuffer());
  return { float32, sr: targetSR };
}

// WebAudio 解碼 + OfflineAudioContext 重取樣
async function decodeViaWebAudio(blobOrFile, targetSR=16000) {
  const arrayBuf = await blobOrFile.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

  // 先單聲道
  const mono = new AudioBuffer({ length: audioBuf.length, numberOfChannels: 1, sampleRate: audioBuf.sampleRate });
  const ch0 = audioBuf.getChannelData(0);
  if (audioBuf.numberOfChannels > 1) {
    const ch1 = audioBuf.getChannelData(1);
    const out = mono.getChannelData(0);
    for (let i = 0; i < ch0.length; i++) out[i] = (ch0[i] + ch1[i]) / 2;
  } else {
    mono.copyToChannel(ch0, 0);
  }

  if (audioBuf.sampleRate === targetSR) {
    // 不需重取樣
    return { float32: mono.getChannelData(0).slice(0), sr: targetSR };
  }

  // 重取樣
  const offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
  const src = offline.createBufferSource();
  src.buffer = mono;
  src.connect(offline.destination);
  src.start(0);
  const rendered = await offline.startRendering();
  return { float32: rendered.getChannelData(0).slice(0), sr: targetSR };
}

// FFmpeg.wasm：任意（含 mp4）→ 16k 單聲道 WAV（wav Blob）
async function transcodeToWav16kViaFFmpeg(blobOrFile) {
  setStatus("轉檔（ffmpeg）…", true);
  const { createFFmpeg, fetchFile } = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/+esm");
  const ffmpeg = createFFmpeg({
    corePath: "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/ffmpeg-core.js",
    log: false
  });
  if (!ffmpeg.isLoaded()) await ffmpeg.load();

  const inName = "in.bin";
  const outName = "out.wav";
  ffmpeg.FS("writeFile", inName, await fetchFile(blobOrFile));

  // -vn 去掉影像、-ac 1 單聲道、-ar 16000 取樣率
  await ffmpeg.run("-i", inName, "-vn", "-ac", "1", "-ar", `${TARGET_SR}`, "-f", "wav", outName);
  const out = ffmpeg.FS("readFile", outName);

  // 清理
  try { ffmpeg.FS("unlink", inName); } catch {}
  try { ffmpeg.FS("unlink", outName); } catch {}

  return new Blob([out.buffer], { type: "audio/wav" });
}

// 將 16-bit PCM WAV ArrayBuffer 轉 Float32（單聲道）
// ※ 假設 wav 已是 16k/mono/16-bit
function wavToFloat32(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  // 讀取格式
  const riff = getString(view, 0, 4);           // 'RIFF'
  const wave = getString(view, 8, 4);           // 'WAVE'
  if (riff !== "RIFF" || wave !== "WAVE") throw new Error("Not a WAV");

  // 找到 'fmt ' 與 'data' 區塊
  let pos = 12;
  let audioFormat = 1, numChannels = 1, sampleRate = TARGET_SR, bitsPerSample = 16;
  let dataOffset = -1, dataSize = 0;

  while (pos < view.byteLength) {
    const id = getString(view, pos, 4); pos += 4;
    const size = view.getUint32(pos, true); pos += 4;
    if (id === "fmt ") {
      audioFormat   = view.getUint16(pos + 0, true);
      numChannels   = view.getUint16(pos + 2, true);
      sampleRate    = view.getUint32(pos + 4, true);
      bitsPerSample = view.getUint16(pos + 14, true);
    } else if (id === "data") {
      dataOffset = pos;
      dataSize = size;
      break;
    }
    pos += size;
  }
  if (dataOffset < 0) throw new Error("WAV data not found");
  if (audioFormat !== 1 || bitsPerSample !== 16) throw new Error("Expect 16-bit PCM WAV");
  // 讀 PCM（只取第一聲道）
  const bytesPerSample = bitsPerSample / 8;
  const frameCount = dataSize / (bytesPerSample * numChannels);
  const out = new Float32Array(frameCount);
  let idx = 0;
  for (let i = 0; i < frameCount; i++) {
    const offset = dataOffset + i * bytesPerSample * numChannels;
    const sample = view.getInt16(offset, true);
    out[idx++] = sample / 32768;
  }
  return { float32: out, sr: sampleRate };
}
function getString(view, start, len) {
  let s = ""; for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(start + i)); return s;
}

// 建立 / 取用模型
async function ensurePipeline() {
  if (clf) return clf;
  setStatus("下載模型中…（首次會久一點）", true);
  const progress_callback = (p) => {
    if (p?.status === "progress" && typeof p.progress === "number") {
      setStatus(`下載模型 ${Math.round(p.progress * 100)}% …`, true);
    } else if (p?.status) {
      setStatus(`${p.status}…`, true);
    }
  };
  clf = await pipeline("audio-classification", MODEL_ID, { progress_callback });
  setStatus("模型就緒");
  return clf;
}

// 分段（整段處理，非只取樣一小段）：將整個音檔切成多個 WINDOW_SEC 片段依序推論、加權平均
async function analyzeByChunks(float32, sr) {
  const model = await ensurePipeline();
  const totalLen = float32.length;
  const win = Math.floor(WINDOW_SEC * sr);
  const hop = Math.floor(HOP_SEC * sr);
  const chunks = [];
  for (let start = 0; start < totalLen; start += hop) {
    const end = Math.min(start + win, totalLen);
    if (end - start <= sr * 0.5) break; // 尾端太短就不跑
    chunks.push([start, end]);
  }
  if (chunks.length === 0) chunks.push([0, Math.min(win, totalLen)]);

  setStatus(`分析中… (0/${chunks.length})`, true);
  if (meter) meter.classList.remove("hidden");

  let femaleSum = 0, maleSum = 0, weightSum = 0;

  for (let i = 0; i < chunks.length; i++) {
    const [s0, s1] = chunks[i];
    const seg = float32.subarray(s0, s1);
    // 注意：Transformers.js 支援直接給 Float32Array + sampling_rate
    const res = await model(seg, { sampling_rate: sr, topk: 2 });
    // 聚合：用片段時長做權重
    const w = (s1 - s0) / sr;
    const map = { female: 0, male: 0 };
    if (Array.isArray(res)) for (const r of res) if (r && typeof r.label === "string") map[r.label] = r.score || 0;
    femaleSum += (map.female || 0) * w;
    maleSum   += (map.male   || 0) * w;
    weightSum += w;

    // 即時更新（跑到哪算到哪）
    const fNow = femaleSum / weightSum;
    const mNow = maleSum / weightSum;
    renderBars(fNow, mNow);
    setStatus(`分析中… (${i+1}/${chunks.length})`, true);

    // 讓 UI 有機會刷新
    await microYield();
  }

  const female = femaleSum / weightSum;
  const male   = maleSum   / weightSum;
  renderBars(female, male);
  setStatus("完成");
}

function renderBars(f, m) {
  const barF = document.querySelector(".bar.female");
  const barM = document.querySelector(".bar.male");
  if (barF) barF.style.setProperty("--p", typeof f === "number" ? f : 0);
  if (barM) barM.style.setProperty("--p", typeof m === "number" ? m : 0);
  if (femaleVal) femaleVal.textContent = `${((f||0) * 100).toFixed(1)}%`;
  if (maleVal)   maleVal.textContent   = `${((m||0) * 100).toFixed(1)}%`;
}

function microYield() {
  return new Promise((r) => setTimeout(r, 0));
}
