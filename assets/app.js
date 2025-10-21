// assets/app.js  —  SAFE v3 (ESM). 需要在 index.html 以 <script type="module" src="assets/app.js?v=..."></script> 載入。

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0/dist/transformers.min.js";

// ONNXRuntime Web WASM 路徑（給 ONNX 後端用）
env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/";

// ---- DOM ----
const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// ---- 狀態 ----
let mediaRecorder = null;
let chunks = [];
let pipe = null; // transformers.js pipeline（lazy）
const device = (navigator.gpu ? "webgpu" : "wasm");

// 預設參數（可在 index.html 事先設 window.* 蓋掉）
if (!("INFERENCE_MODE" in window)) window.INFERENCE_MODE = "browser"; // 或 "server"
if (!("API_BASE_URL" in window))   window.API_BASE_URL   = "/api/classify";
if (!("ONNX_MODEL_ID" in window))  window.ONNX_MODEL_ID  = "prithivMLmods/Common-Voice-Gender-Detection-ONNX";

console.log("[app] loaded, device =", device, "| mode =", window.INFERENCE_MODE);

function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}

// ---- 事件 ----
if (recordBtn) {
  recordBtn.addEventListener("click", async () => {
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
    try {
      if (!e.target.files || e.target.files.length === 0) return;
      const file = e.target.files[0];
      await handleBlob(file);
      e.target.value = "";
    } catch (err) {
      console.error("[fileInput]", err);
      setStatus("上傳處理失敗");
    }
  });
}

// ---- 錄音 ----
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
    setStatus("此瀏覽器不支援錄音，請改用右下角上傳", false);
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  const mimeType = pickSupportedMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

  mediaRecorder.ondataavailable = (ev) => {
    if (ev.data && ev.data.size > 0) chunks.push(ev.data);
  };

  mediaRecorder.onstop = async () => {
    try {
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      await handleBlob(blob);
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

// ---- 音訊處理 ----
async function handleBlob(blob) {
  setStatus("解析與重取樣…", true);
  const { data } = await decodeAndResample(blob, 16000);
  if (window.INFERENCE_MODE === "browser") {
    const ok = await runInBrowser(data, 16000);
    if (!ok) {
      console.warn("[fallback] browser inference failed → switching to server");
      window.INFERENCE_MODE = "server";
      const wav = floatToWavBlob(data, 16000);
      await runViaServer(wav);
    }
  } else {
    const wav = floatToWavBlob(data, 16000);
    await runViaServer(wav);
  }
}

// 解析 + 重取樣到 targetSR，回傳 Float32Array（單聲道）
async function decodeAndResample(blob, targetSR = 16000) {
  const arrayBuf = await blob.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

  // 建離線重取樣器
  const offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
  const src = offline.createBufferSource();

  // 先做單聲道
  const mono = new AudioBuffer({ length: audioBuf.length, numberOfChannels: 1, sampleRate: audioBuf.sampleRate });
  const ch0 = audioBuf.getChannelData(0);
  if (audioBuf.numberOfChannels > 1) {
    const ch1 = audioBuf.getChannelData(1);
    const out = mono.getChannelData(0);
    for (let i = 0; i < ch0.length; i++) out[i] = (ch0[i] + ch1[i]) / 2;
  } else {
    mono.copyToChannel(ch0, 0);
  }

  src.buffer = mono;
  src.connect(offline.destination);
  src.start(0);

  const rendered = await offline.startRendering();
  const out = rendered.getChannelData(0);
  return { data: new Float32Array(out), sr: targetSR };
}

// ---- 推論（瀏覽器端） ----
async function runInBrowser(float32PCM, samplingRate) {
  try {
    setStatus("載入模型（首次需較久）…", true);
    if (meter) meter.classList.remove("hidden");
    if (!pipe) {
      // 若 ONNX 模型不存在或載不動會 throw；外層會 fallback 到 server
      pipe = await pipeline("audio-classification", window.ONNX_MODEL_ID, { device });
    }
    const results = await pipe(float32PCM, { sampling_rate: samplingRate, topk: 2 });
    renderResults(results);
    setStatus("完成");
    return true;
  } catch (e) {
    console.error("[browser inference error]", e);
    setStatus("瀏覽器端推論失敗，嘗試伺服器分析…", true);
    return false;
  }
}

// ---- 推論（伺服器端 API）----
async function runViaServer(wavBlob) {
  try {
    setStatus("上傳到伺服器分析…", true);
    if (meter) meter.classList.remove("hidden");
    const res = await fetch(window.API_BASE_URL, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: await wavBlob.arrayBuffer()
    });
    if (!res.ok) throw new Error(`API ${res.status}`);
    const json = await res.json();
    renderResults(json.results || json);
    setStatus("完成");
  } catch (e) {
    console.error("[server inference error]", e);
    setStatus("伺服器分析失敗");
  }
}

// ---- 畫面更新 ----
function renderResults(arr) {
  // 期望格式：[{label:"female", score:0.98}, {label:"male", score:0.02}]
  const map = { female: 0, male: 0 };
  if (Array.isArray(arr)) {
    for (const r of arr) {
      if (r && typeof r.label === "string" && typeof r.score === "number") {
        map[r.label] = r.score;
      }
    }
  }
  const f = map.female || 0;
  const m = map.male || 0;
  const barF = document.querySelector(".bar.female");
  const barM = document.querySelector(".bar.male");
  if (barF) barF.style.setProperty("--p", f);
  if (barM) barM.style.setProperty("--p", m);
  if (femaleVal) femaleVal.textContent = `${(f * 100).toFixed(1)}%`;
  if (maleVal)   maleVal.textContent   = `${(m * 100).toFixed(1)}%`;
}

// ---- 工具：Float32 → 16-bit PCM WAV Blob ----
function floatToWavBlob(float32, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32.length * 2);
  const view = new DataView(buffer);

  // Helpers
  function writeStr(off, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }

  // RIFF header
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + float32.length * 2, true);
  writeStr(8, "WAVE");

  // fmt chunk
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);     // chunk size
  view.setUint16(20, 1, true);      // PCM
  view.setUint16(22, 1, true);      // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono 16-bit)
  view.setUint16(32, 2, true);      // block align
  view.setUint16(34, 16, true);     // bits per sample

  // data chunk
  writeStr(36, "data");
  view.setUint32(40, float32.length * 2, true);

  let offset = 44;
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}
