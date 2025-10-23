// ===== Transformers pipeline =====
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

/** 只用遠端（Hugging Face Hub），停用本機 /models 尋址 */
env.allowLocalModels = false;
env.allowRemoteModels = true;

/** 視需要可調整：WASM 執行緒數 */
env.backends.onnx.wasm.numThreads = 1;

// ===== Theme（三切 + 每派記憶 + 一次性提示） =====
const MODE_KEY = "vpa.themeMode";          // 'auto' | 'light' | 'dark'
const LAST_LIGHT_KEY = "vpa.lastLight";    // ex: 'day'
const LAST_DARK_KEY  = "vpa.lastDark";     // ex: 'night'
const LEGACY_KEY     = "vpa.theme";        // 舊版單一主題鍵（會自動遷移）
const TIP_KEY        = "vpa.themeTipDone"; // 一次性齒輪提示

// 26 主題（名稱需與 index.html data-theme 對齊）
const THEMES = [
  "warm","lavender","peach","ink","day","night","contrast","slate","graphite","sand","latte","clay",
  "rose","blush","coral","amber","gold","cocoa","olive","emerald","teal","aqua","cyan","sky","azure",
  "cobalt","indigo","violet","grape","plum","magenta","fuchsia"
];

// 兩派：背景白（light）與背景深（dark）
// 注意：依據樣式檔的 --bg-start / --bg-end 設計而分
const THEME_FACTION = {
  day:"light", sand:"light", latte:"light", blush:"light", amber:"light",
  aqua:"light", sky:"light", azure:"light", fuchsia:"light",

  warm:"dark", lavender:"dark", peach:"dark", ink:"dark", night:"dark", contrast:"dark",
  slate:"dark", graphite:"dark", clay:"dark", rose:"dark", coral:"dark", gold:"dark", cocoa:"dark",
  olive:"dark", emerald:"dark", teal:"dark", cyan:"dark", cobalt:"dark", indigo:"dark",
  violet:"dark", grape:"dark", plum:"dark", magenta:"dark"
};

function getSystemFaction(){
  try {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch { return "light"; }
}
function getSavedMode(){
  try{ return (localStorage.getItem(MODE_KEY) || "auto"); }catch{ return "auto"; }
}
function setSavedMode(m){ try{ localStorage.setItem(MODE_KEY, m); }catch{} }
function getSavedLast(faction){
  const key = faction === "light" ? LAST_LIGHT_KEY : LAST_DARK_KEY;
  let val = null; try{ val = localStorage.getItem(key); }catch{}
  if (!val || !THEMES.includes(val) || THEME_FACTION[val] !== faction){
    // 預設各派
    val = faction === "light" ? "day" : "night";
  }
  return val;
}
function setSavedLast(theme){
  const faction = THEME_FACTION[theme] || "dark";
  const key = faction === "light" ? LAST_LIGHT_KEY : LAST_DARK_KEY;
  try{ localStorage.setItem(key, theme); }catch{}
}

function migrateLegacyTheme(){
  try{
    const old = localStorage.getItem(LEGACY_KEY);
    if (!old) return;
    const f = THEME_FACTION[old] || getSystemFaction();
    setSavedLast(old);
    localStorage.removeItem(LEGACY_KEY);
  }catch{}
}

// 目前模式與派系（以 data- 屬性呈現）
function applyMode(mode){
  const m = (mode === "light" || mode === "dark" || mode === "auto") ? mode : "auto";
  setSavedMode(m);
  document.documentElement.setAttribute("data-mode", m);

  const faction = (m === "auto") ? getSystemFaction() : m;
  document.documentElement.setAttribute("data-faction", faction);

  // 如果已套的主題不在該派，就切到該派最後一次主題
  const now = document.documentElement.getAttribute("data-theme");
  if (!now || THEME_FACTION[now] !== faction){
    applyTheme(getSavedLast(faction), false);
  }
  refreshThemeChecks();
  refreshTabUI();
}
function applyTheme(t, persist=true){
  if (!THEMES.includes(t)) t = getSavedLast(getSystemFaction());
  const faction = THEME_FACTION[t] || getSystemFaction();
  document.documentElement.setAttribute("data-theme", t);
  document.documentElement.setAttribute("data-faction", faction);
  if (persist) setSavedLast(t);
  refreshThemeChecks();
}
function refreshThemeChecks(){
  document.querySelectorAll(".theme-item").forEach(btn=>{
    const checked = btn.dataset.theme === document.documentElement.getAttribute("data-theme");
    btn.setAttribute("aria-checked", checked ? "true" : "false");
  });
}
function refreshTabUI(){
  const mode = document.documentElement.getAttribute("data-mode") || "auto";
  const map = { auto: "#tabAuto", light: "#tabLight", dark: "#tabDark" };
  for (const [k, sel] of Object.entries(map)){
    const el = document.querySelector(sel);
    if (!el) continue;
    el.setAttribute("aria-selected", k === mode ? "true" : "false");
    el.classList.toggle("active", k === mode);
  }
}

function initThemeUI(){
  migrateLegacyTheme();

  // 初始 Mode + 主題
  applyMode(getSavedMode());

  const gear = document.getElementById("settingsBtn");
  const menu = document.getElementById("themeMenu");
  const tip  = document.getElementById("themeTip");

  if (gear && menu){
    gear.addEventListener("click", (e)=>{
      e.stopPropagation();
      const open = !menu.hasAttribute("hidden");
      if (open){ menu.setAttribute("hidden",""); gear.setAttribute("aria-expanded","false"); }
      else {
        menu.removeAttribute("hidden");
        gear.setAttribute("aria-expanded","true");
        // 第一次打開齒輪時，關閉一次性提示
        if (tip && !tip.hasAttribute("hidden")){
          tip.setAttribute("hidden","");
          try{ localStorage.setItem(TIP_KEY, "1"); }catch{}
        }
      }
    });
    document.addEventListener("click", (e)=>{
      if (!menu.contains(e.target) && e.target !== gear && !gear.contains(e.target)){
        if (!menu.hasAttribute("hidden")){ menu.setAttribute("hidden",""); gear.setAttribute("aria-expanded","false"); }
      }
      if (tip && !tip.hasAttribute("hidden")){ tip.setAttribute("hidden",""); try{ localStorage.setItem(TIP_KEY, "1"); }catch{} }
    });

    // 主題項
    menu.querySelectorAll(".theme-item").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        applyTheme(btn.dataset.theme, true);
        menu.setAttribute("hidden",""); gear.setAttribute("aria-expanded","false");
      });
    });

    // 三切 Tabs
    const modeBtns = [
      ["auto",  document.getElementById("tabAuto")],
      ["light", document.getElementById("tabLight")],
      ["dark",  document.getElementById("tabDark")],
    ];
    for (const [name, el] of modeBtns){
      if (!el) continue;
      el.addEventListener("click", ()=>{
        applyMode(name);
      });
    }

    // Auto 監聽系統深淺變化
    try{
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      mq.addEventListener?.("change", ()=>{
        if ((localStorage.getItem(MODE_KEY)||"auto") === "auto"){
          applyMode("auto");
        }
      });
    }catch{}
  }

  // 一次性提示（齒輪）
  try{
    const done = localStorage.getItem(TIP_KEY) === "1";
    if (!done && tip){
      tip.removeAttribute("hidden");
      // 自動收起
      setTimeout(()=>{ try{ tip.setAttribute("hidden",""); localStorage.setItem(TIP_KEY,"1"); }catch{} }, 4000);
    }
  }catch{}
}
initThemeUI();

// ===== 常量 =====
/** 防呆：剝掉外層花括號或空白（若有） */
const RAW_MODEL_ID = (window.ONNX_MODEL_ID || "prithivMLmods/Common-Voice-Gender-Detection-ONNX");
const MODEL_ID     = String(RAW_MODEL_ID).trim().replace(/^\{+|\}+$/g, "");

const TARGET_SR       = 16000;
const MAX_WHOLE_SEC   = 150;
const WARN_LONG_SEC   = 180;
const STREAM_WIN_CAND = [12, 8, 6, 4];
const STREAM_HOP_S    = 3;
const EPS             = 1e-9;

// VAD
const VAD_MIN_APPLY_SEC   = 20;
const VAD_FRAME_MS        = 30;
const VAD_HOP_MS          = 10;
const VAD_PAD_MS          = 60;
const VAD_MIN_SEG_MS      = 200;
const VAD_MIN_VOICED_SEC  = 2;
const VAD_SILENCE_RATIO_TO_APPLY = 0.15;

// Safari 檢測
const IS_SAFARI = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// ===== DOM =====
const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// 播放器（動態建立）
let playBtn = null;
let audioEl = null;
let lastAudioUrl = null;
ensurePlayerUI();

// ===== 狀態 =====
let mediaRecorder = null;
let chunks = [];
let clf = null;
let busy = false;
let heartbeatTimer = null;

log("[app] ready: theming + player + pipeline");

// ===== 版本資訊 =====
(async function fillBuildMeta(){
  try {
    const verEl = document.getElementById('ver');
    const updEl = document.getElementById('updatedAt');
    if (!verEl && !updEl) return;
    const selfUrl = (import.meta && import.meta.url) ? import.meta.url : 'assets/app.js';
    const res = await fetch(selfUrl, { method: 'HEAD', cache: 'no-store' });
    let d = null;
    if (res.ok) {
      const lm = res.headers.get('last-modified');
      if (lm) d = new Date(lm);
    }
    if (!d || isNaN(d.getTime())) d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth()+1).padStart(2,'0');
    const day = String(d.getDate()).padStart(2,'0');
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    if (updEl) updEl.textContent = `${y}-${m}-${day}`;
    if (verEl) verEl.textContent = `build-${y}${m}${day}-${hh}${mm}`;
  } catch {}
})();

// ===== UI 小工具 =====
function setStatus(text, spin=false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}
function log(...a){ try{ console.log(...a);}catch{} }
function fmtSec(s){ if (!isFinite(s)) return "—"; const m=Math.floor(s/60), ss=Math.round(s%60); return m? `${m}分${ss}秒`:`${ss}秒`; }
function clamp01(x){ return Math.min(1, Math.max(EPS, x)); }

function resetMeter() {
  try {
    meter?.classList.remove("hidden");
    const barF = document.querySelector(".bar.female");
    const barM = document.querySelector(".bar.male");
    if (barF) { barF.style.setProperty("--p", 0); barF.setAttribute("aria-valuenow","0"); }
    if (barM) { barM.style.setProperty("--p", 0); barM.setAttribute("aria-valuenow","0"); }
    if (femaleVal) femaleVal.textContent = "0.0%";
    if (maleVal)   maleVal.textContent   = "0.0%";
  } catch {}
}

// 檢查 OOM
function isOOMError(err){
  const msg = String(err?.message || err || "");
  return /OrtRun|bad_alloc|out of memory|memory|alloc/i.test(msg);
}

// ===== 事件 =====
recordBtn?.addEventListener("click", async () => {
  if (busy) return;
  try {
    // 使用者按「開始」就先把結果條歸零
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      resetMeter();
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
    // 使用者選定檔案當下就歸零
    resetMeter();
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
      chunks.length = 0;
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
  let decoded = null;
  try {
    // 先給播放器用原檔；只保留最新一個
    setPlaybackSource(fileOrBlob);

    setStatus("解析檔案…", true);
    decoded = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
    let { float32, sr, durationSec } = decoded;

    if (durationSec > WARN_LONG_SEC) {
      setStatus(`提示：長度 ${fmtSec(durationSec)}，分析可能較久。準備推論…`, true);
      await microYield();
    }

    // 自適應 VAD（只選段、不動原音）
    const vad = maybeApplyAdaptiveVAD(float32, sr);
    if (vad && vad.used) {
      const reducedRatio = 1 - (vad.keptSec / durationSec);
      float32    = vad.arr;
      durationSec = vad.keptSec;
      setStatus(`已去除靜音（約 ${(reducedRatio*100).toFixed(0)}%）→ 有效時長 ${fmtSec(durationSec)}，開始推論…`, true);
      await microYield();
    }

    if (durationSec <= MAX_WHOLE_SEC) {
      await analyzeWhole(float32, sr, durationSec);
    } else {
      await analyzeStreamed(float32, sr, durationSec, `長度超過 ${MAX_WHOLE_SEC} 秒，自動切換串流分段`);
    }
  } catch (e) {
    console.error("[handleFileOrBlob]", e);
    setStatus("處理失敗：" + (e?.message || "無法解碼或分析此音檔"));
  } finally {
    if (decoded) decoded.float32 = null;
    decoded = null;
    busy = false;
  }
}

// ===== 解碼策略 =====
async function decodeSmartToFloat32(blobOrFile, targetSR){
  const name = (blobOrFile.name || "").toLowerCase();
  const type = (blobOrFile.type || "").toLowerCase();
  const looksLikeM4A = /\.m4a$/i.test(name) || type.includes("audio/mp4") || type.includes("audio/x-m4a");

  if (IS_SAFARI && looksLikeM4A) {
    setStatus("轉檔（ffmpeg，Safari m4a）準備中…", true);
    const wavBlob = await transcodeToWav16kViaFFmpeg(blobOrFile);
    const { float32, sr } = wavToFloat32(await wavBlob.arrayBuffer());
    return { float32, sr, durationSec: float32.length / sr };
  }

  try {
    setStatus("直接解碼（WebAudio）…", true);
    return await decodeViaWebAudio(blobOrFile, targetSR);
  } catch (e) {
    log("[decode] WebAudio failed, fallback to ffmpeg:", e?.message || e);
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
  let offline = null;
  try {
    const audioBuf = await ctx.decodeAudioData(arrayBuf);
    const mono = ctx.createBuffer(1, audioBuf.length, audioBuf.sampleRate);
    const outCh = mono.getChannelData(0);
    const ch0 = audioBuf.getChannelData(0);

    if (audioBuf.numberOfChannels > 1) {
      const ch1 = audioBuf.getChannelData(1);
      for (let i=0;i<ch0.length;i++) outCh[i] = (ch0[i] + ch1[i]) / 2;
    } else { outCh.set(ch0); }

    let out;
    if (audioBuf.sampleRate === targetSR) {
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
    try { await ctx.close(); } catch {}
    offline = null;
  }
}

// ---- FFmpeg ----
async function loadFFmpegModule(){
  try {
    const m = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/+esm");
    if (typeof m.createFFmpeg === "function" && typeof m.fetchFile === "function") {
      return { createFFmpeg: m.createFFmpeg, fetchFile: m.fetchFile, mode: "esm" };
    }
  } catch (e) { log("[ffmpeg] +esm import failed:", e?.message || e); }
  await loadScriptTag("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/dist/ffmpeg.min.js");
  const FF = window.FFmpeg;
  if (!FF || typeof FF.createFFmpeg !== "function") throw new Error("FFmpeg UMD load failed");
  return { createFFmpeg: FF.createFFmpeg, fetchFile: FF.fetchFile, mode: "umd" };
}
function loadScriptTag(src){
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src; s.async = true; s.crossOrigin = "anonymous"; s.referrerPolicy = "no-referrer";
    s.onload = () => resolve(); s.onerror = () => reject(new Error("script load error: " + src));
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
    const r = Math.min(1, Math.max(0, Number.isFinite(ratio) ? ratio : 0));
    setStatus(`轉檔（ffmpeg）… ${Math.round(r*100)}%`, true);
  });
  if (!ffmpeg.isLoaded()) await ffmpeg.load();

  const inName = "in.bin", outName = "out.wav";
  ffmpeg.FS("writeFile", inName, await fetchFile(blobOrFile));
  await ffmpeg.run("-i", inName, "-vn", "-ac", "1", "-ar", `${TARGET_SR}`, "-f", "wav", outName);
  const out = ffmpeg.FS("readFile", outName);
  try { ffmpeg.FS("unlink", inName); } catch {}
  try { ffmpeg.FS("unlink", outName); } catch {}
  try { await ffmpeg.exit(); } catch {}

  return new Blob([out.buffer], { type: "audio/wav" });
}

// ===== 模型 =====
async function ensurePipeline(){
  if (clf) return clf;
  setStatus("下載模型中…（首次會久一點）", true);

  const progress_callback = (p)=>{
    if (!p) return;
    let pct = null;
    if (typeof p.loadedBytes === 'number' && typeof p.totalBytes === 'number' && p.totalBytes > 0) {
      pct = p.loadedBytes / p.totalBytes;
    } else if (typeof p.progress === 'number' && isFinite(p.progress)) {
      pct = p.progress;
    }
    const label = p.status || "下載模型";
    if (pct == null) setStatus(`${label}…`, true);
    else setStatus(`${label} ${Math.min(99, Math.max(0, Math.floor(pct*100)))}% …`, true);
  };

  const device = (typeof navigator !== 'undefined' && navigator.gpu) ? 'webgpu' : 'wasm';
  clf = await pipeline("audio-classification", MODEL_ID, { progress_callback, device });
  setStatus(`模型就緒（device: ${device}）`);
  return clf;
}

// ===== 分析（整段） =====
async function analyzeWhole(float32, sr, durationSec){
  const model = await ensurePipeline();
  meter?.classList.remove("hidden");

  const started = performance.now();
  startHeartbeat(() => {
    const elapsed = (performance.now() - started)/1000;
    setStatus(`分析中（整段）｜音檔 ${fmtSec(durationSec)}｜已用 ${fmtSec(elapsed)}`, true);
  });

  try {
    const res = await model(float32, { sampling_rate: sr, topk: 2 });
    const map = toMap(res);
    render(map.female||0, map.male||0);
    setStatus("完成（整段）");
  } catch (err) {
    if (isOOMError(err)) {
      console.warn("[analyzeWhole] OOM → switch to streamed mode…");
      await analyzeStreamed(float32, sr, durationSec, "偵測到記憶體不足，自動改串流分段");
      return;
    }
    console.error("[analyzeWhole]", err);
    setStatus("分析失敗（整段）");
  } finally { stopHeartbeat(); }
}

// ===== 分析（串流分段） =====
async function analyzeStreamed(float32, sr, durationSec, reason="串流分段"){
  const model = await ensurePipeline();
  meter?.classList.remove("hidden");

  let lastErr = null;
  for (const winSec of STREAM_WIN_CAND) {
    try {
      await runStreamedWithWindow(model, float32, sr, durationSec, winSec, STREAM_HOP_S, reason);
      return;
    } catch (e) {
      lastErr = e;
      if (isOOMError(e)) { console.warn(`[streamed] OOM at win=${winSec}s → downshift`); continue; }
      else { console.error(`[streamed] error at win=${winSec}s`, e); break; }
    }
  }
  console.error("[analyzeStreamed] failed", lastErr);
  setStatus("分析失敗（串流分段）");
}

async function runStreamedWithWindow(model, float32, sr, durationSec, WIN_S, HOP_S, reason){
  const win = Math.max(1, Math.floor(WIN_S * sr));
  const hop = Math.max(1, Math.floor(HOP_S * sr));

  const chunks = [];
  for (let s=0; s<float32.length; s+=hop){
    const e = Math.min(s + win, float32.length);
    if (e - s < Math.floor(0.5 * sr)) break;
    chunks.push([s,e]);
    if (e === float32.length) break;
  }
  if (!chunks.length) chunks.push([0, Math.min(win, float32.length)]);

  let avgMs = 0, processedSec = 0;
  let logitSum = 0, wSum = 0;

  const started = performance.now();
  startHeartbeat(() => {
    const elapsed = (performance.now() - started)/1000;
    const pct = processedSec > 0 ? Math.min(99, Math.round((processedSec/durationSec)*100)) : 0;
    setStatus(`分析中（串流；win=${WIN_S}s/step=${HOP_S}s）｜${reason}｜${pct}%｜已用 ${fmtSec(elapsed)}`, true);
  });

  try {
    for (let i=0;i<chunks.length;i++){
      const [s0,s1] = chunks[i];
      const seg = float32.subarray(s0, s1);
      const dur = (s1 - s0) / sr;

      const t0 = performance.now();
      const out = await model(seg, { sampling_rate: sr, topk: 2 });
      const dt = performance.now() - t0;
      avgMs = avgMs === 0 ? dt : (avgMs*0.65 + dt*0.35);

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
      setStatus(`分析中（串流；win=${WIN_S}s）｜片段 ${i+1}/${chunks.length}｜${pct}%｜已處理 ${fmtSec(processedSec)} / ${fmtSec(durationSec)}｜預估剩餘 ~ ${fmtSec(etaSec)}`, true);

      await microYield();
    }

    const logitAvg = logitSum / Math.max(wSum, EPS);
    const pf = 1 / (1 + Math.exp(-logitAvg));
    const pm = 1 - pf;
    render(pf, pm);
    setStatus("完成（串流分段）");
  } finally { stopHeartbeat(); }
}

// ===== 播放器（CSS 驅動） =====
function ensurePlayerUI(){
  const container = document.querySelector(".container");
  if (!container) return;
  if (document.getElementById("playBtn")) return;

  const wrap = document.createElement("div");
  wrap.className = "player";

  const btn = document.createElement("button");
  btn.id = "playBtn"; btn.type = "button"; btn.disabled = true;
  btn.textContent = "▶︎ 播放剛才的聲音";
  btn.setAttribute("aria-label", "播放剛才的聲音");

  const hint = document.createElement("div");
  hint.className = "hint";
  hint.innerHTML = `想再聽一次剛才那段嗎？<a href="#" id="replayLink">點這裡</a>。`;

  const audio = document.createElement("audio");
  audio.id = "playback"; audio.preload = "metadata"; audio.style.display = "none";

  wrap.appendChild(btn); wrap.appendChild(hint); wrap.appendChild(audio);

  const tipEl = container.querySelector(".callout");
  if (tipEl) container.insertBefore(wrap, tipEl); else container.appendChild(wrap);

  playBtn = btn; audioEl = audio;

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
    } catch (e) { console.error("[audio play]", e); }
  };
  audioEl.onended = () => { playBtn.textContent = "▶︎ 播放剛才的聲音"; };

  wrap.querySelector("#replayLink")?.addEventListener("click", (e)=>{
    e.preventDefault();
    playBtn.click();
  });
}

function setPlaybackSource(blob){
  try {
    if (!audioEl || !playBtn) return;
    if (lastAudioUrl) { try { URL.revokeObjectURL(lastAudioUrl); } catch {} }
    lastAudioUrl = URL.createObjectURL(blob);
    audioEl.src = lastAudioUrl;
    audioEl.load();
    playBtn.disabled = false;
    playBtn.textContent = "▶︎ 播放剛才的聲音";
  } catch (e) { console.error("[setPlaybackSource]", e); }
}

// ===== Render / Utils =====
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
  if (barF) { barF.style.setProperty("--p", pf ?? 0); barF.setAttribute("aria-valuenow", Math.round(((pf ?? 0) * 100))); }
  if (barM) { barM.style.setProperty("--p", pm ?? 0); barM.setAttribute("aria-valuenow", Math.round(((pm ?? 0) * 100))); }
  if (femaleVal) femaleVal.textContent = `${((pf ?? 0) * 100).toFixed(1)}%`;
  if (maleVal)   maleVal.textContent   = `${((pm ?? 0) * 100).toFixed(1)}%`;
}

function startHeartbeat(tickFn){ stopHeartbeat(); heartbeatTimer = setInterval(() => { try{ tickFn(); }catch{} }, 1000); }
function stopHeartbeat(){ if (heartbeatTimer){ clearInterval(heartbeatTimer); heartbeatTimer = null; } }
function microYield(){ return new Promise(r=>setTimeout(r,0)); }

// ===== WAV 解析 =====
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
    let j = i;
    while (j < voicedMask.length && voicedMask[j]) j++;
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
  return { used: true, arr: out, keptSec, segs };
}
function percentile(arr, p){
  const a = arr.slice().sort((x,y)=>x-y);
  const idx = Math.min(a.length-1, Math.max(0, Math.round((p/100)* (a.length-1))));
  return a[idx];
}
function smoothMask(mask, k=3){
  // 把長度 < k 的 0 洞補成 1
  let count=0;
  for (let i=0;i<=mask.length;i++){
    if (i<mask.length && !mask[i]) count++;
    else {
      if (count>0 && count<k){ for (let j=i-count; j<i; j++) mask[j]=true; }
      count=0;
    }
  }
  // 把長度 < k 的 1 島補成 0
  count=0;
  for (let i=0;i<=mask.length;i++){
    if (i<mask.length && mask[i]) count++;
    else {
      if (count>0 && count<k){ for (let j=i-count; j<i; j++) mask[j]=false; }
      count=0;
    }
  }
}

// ===== 離站清理 =====
window.addEventListener("beforeunload", () => {
  if (lastAudioUrl) { try { URL.revokeObjectURL(lastAudioUrl); } catch {} }
});
