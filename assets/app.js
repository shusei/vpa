// assets/app.js — GH Pages 版（不跑 ONNX；直接呼叫 Hugging Face Inference API）
// 建議用 <script src="assets/app.js?v=20251021hf"></script> 以避免快取。
// 介面：大錄音鍵＋右下角上傳；結果顯示 female / male 百分比。

// ========== DOM ==========
const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// ========== 設定 ==========
const HF_MODEL_ID = "prithivMLmods/Common-Voice-Gender-Detection";
const HF_API_URL  = `https://api-inference.huggingface.co/models/${HF_MODEL_ID}`;
// 若你有私用 Token，可填入（公開網站不建議）
// const HF_TOKEN = "hf_XXXXXXXX"; // 留空就匿名呼叫（有限流量）

// ========== 狀態 ==========
let mediaRecorder = null;
let chunks = [];

function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}

// ========== 事件 ==========
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

// ========== 錄音 ==========
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

// ========== 音訊處理 ==========
async function handleBlob(blob) {
  setStatus("解析與重取樣…", true);
  const { data } = await decodeAndResample(blob, 16000);
  const wav = floatToWavBlob(data, 16000);
  await runViaHF(wav);
}

// 解碼＋重取樣 → Float32Array (mono, 16 kHz)
async function decodeAndResample(blob, targetSR = 16000) {
  const arrayBuf = await blob.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);

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

// ========== 推論（呼叫 Hugging Face Inference API）==========
async function runViaHF(wavBlob) {
  try {
    setStatus("送到模型分析…", true);
    if (meter) meter.classList.remove("hidden");

    const bytes = await wavBlob.arrayBuffer();

    // 最多重試 5 次（503 代表模型正在暖機）
    const maxRetry = 5;
    let attempt = 0;
    let lastErr = null;

    while (attempt < maxRetry) {
      attempt++;
      try {
        const res = await fetch(HF_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "audio/wav",
            // ...(HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {})
          },
          body: bytes
        });

        if (res.status === 503) {
          // 模型在加載，等待一下再試
          const info = await res.json().catch(() => ({}));
          const waitSec = Math.min(3 + attempt, (info.estimated_time || 3));
          setStatus(`模型啟動中… ${waitSec}s 後再試（第 ${attempt}/${maxRetry} 次）`, true);
          await sleep(waitSec * 1000);
          continue;
        }

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`HF ${res.status}: ${text}`);
        }

        const out = await res.json();
        const arr = Array.isArray(out) ? out : (out.results || out[0] || []);
        renderResults(arr);
        setStatus("完成");
        return;
      } catch (e) {
        lastErr = e;
        // 非 503 錯誤也稍等一下再試
        await sleep(800);
      }
    }

    throw lastErr || new Error("HF API failed");

  } catch (e) {
    console.error("[HF inference error]", e);
    setStatus("分析失敗，稍後再試");
  }
}

// ========== 畫面更新 ==========
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

// ========== 小工具 ==========
function floatToWavBlob(float32, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32.length * 2);
  const view = new DataView(buffer);

  function writeStr(off, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }

  // RIFF
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + float32.length * 2, true);
  writeStr(8, "WAVE");

  // fmt
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);      // PCM
  view.setUint16(22, 1, true);      // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);      // block align
  view.setUint16(34, 16, true);     // bits per sample

  // data
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

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
