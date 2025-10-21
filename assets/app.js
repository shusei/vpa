// assets/app.js — 全前端、無伺服器、在瀏覽器用 Transformers.js + ONNXRuntime(WASM) 跑推論
// 重點：不呼叫任何外部 API（除了從 Hugging Face 下載模型檔），錄音/上傳 → 16k 重取樣 → 直接在瀏覽器做 audio-classification

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// ---- ONNXRuntime Web 設定（避免 crossOriginIsolation 需求，強制單執行緒）----
env.backends.onnx.wasm.numThreads = 1;          // 無多執行緒 → 不需 COOP/COEP
// 不指定 wasmPaths，讓 transformers.js 自己配對對應版本的 .wasm（避免你遇到的 1.19 檔名差異）
// env.allowRemoteModels 預設為 true（會從 Hugging Face Hub 載模型）

// ---- 模型ID（Hugging Face）----
const MODEL_ID = "prithivMLmods/Common-Voice-Gender-Detection-ONNX";

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
let clf = null; // transformers.js pipeline（lazy）

function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}

console.log("[app] loaded | mode=browser-only");

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
    setStatus("此瀏覽器不支援錄音，請改用右下角上傳");
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

// ---- 音訊處理 → 16k 單聲道 → 推論 ----
async function handleBlob(blob) {
  try {
    setStatus("解析與重取樣…", true);
    const { data } = await decodeAndResample(blob, 16000);
    await runInBrowser(data, 16000);
  } catch (e) {
    console.error("[handleBlob]", e);
    setStatus("處理失敗");
  }
}

// 解碼＋重取樣 → Float32Array (mono, targetSR)
async function decodeAndResample(blob, targetSR = 16000) {
  const arrayBuf = await blob.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

  // 使用 OfflineAudioContext 重取樣
  const offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
  const src = offline.createBufferSource();

  // 先轉單聲道
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

// ---- 瀏覽器端推論（Transformers.js + ONNXRuntime WASM）----
async function runInBrowser(float32PCM, samplingRate) {
  try {
    if (!clf) {
      setStatus("下載模型中…（首次會久一點）", true);
      // 進度顯示（可選）
      const progress_callback = (p) => {
        if (p?.status === "progress" && typeof p.progress === "number") {
          setStatus(`下載模型 ${Math.round(p.progress * 100)}% …`, true);
        } else if (p?.status) {
          setStatus(`${p.status}…`, true);
        }
      };
      clf = await pipeline("audio-classification", MODEL_ID, { progress_callback });
      setStatus("模型就緒");
    }

    setStatus("分析中…", true);
    if (meter) meter.classList.remove("hidden");
    const results = await clf(float32PCM, { sampling_rate: samplingRate, topk: 2 });
    renderResults(results);
    setStatus("完成");
  } catch (e) {
    console.error("[browser inference error]", e);
    setStatus("瀏覽器推論失敗");
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
