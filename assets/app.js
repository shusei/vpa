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

// 主題清單（需與 index.html data-theme 對齊）
const THEMES = [
  "warm","lavender","peach","ink","day","night","contrast","slate","graphite","sand","latte","clay",
  "rose","blush","coral","amber","gold","cocoa","olive","emerald","teal","aqua","cyan","sky","azure",
  "cobalt","indigo","violet","grape","plum","magenta","fuchsia"
];

// 兩派：背景白（light）與背景深（dark）
const THEME_FACTION = {
  day:"light", sand:"light", latte:"light", blush:"light", amber:"light",
  aqua:"light", sky:"light", azure:"light", fuchsia:"light",

  warm:"dark", lavender:"dark", peach:"dark", ink:"dark", night:"dark", contrast:"dark",
  slate:"dark", graphite:"dark", clay:"dark", rose:"dark", coral:"dark", gold:"dark", cocoa:"dark",
  olive:"dark", emerald:"dark", teal:"dark", cyan:"dark", cobalt:"dark", indigo:"dark",
  violet:"dark", grape:"dark", plum:"dark", magenta:"dark"
};

function getSystemFaction(){
  try { return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; }
  catch { return "light"; }
}
function getSavedMode(){ try{ return (localStorage.getItem(MODE_KEY) || "auto"); }catch{ return "auto"; } }
function setSavedMode(m){ try{ localStorage.setItem(MODE_KEY, m);}catch{} }
function getSavedLast(faction){
  const key = faction === "light" ? LAST_LIGHT_KEY : LAST_DARK_KEY;
  let v = null; try{ v = localStorage.getItem(key);}catch{}
  if (!v || !THEMES.includes(v) || THEME_FACTION[v] !== faction) v = faction==="light" ? "day" : "night";
  return v;
}
function setSavedLast(theme){
  const faction = THEME_FACTION[theme] || "dark";
  const key = faction === "light" ? LAST_LIGHT_KEY : LAST_DARK_KEY;
  try{ localStorage.setItem(key, theme);}catch{}
}
function migrateLegacyTheme(){
  try{
    const old = localStorage.getItem(LEGACY_KEY);
    if (!old) return;
    setSavedLast(old);
    localStorage.removeItem(LEGACY_KEY);
  }catch{}
}

function applyMode(mode){
  const m = (mode==="light"||mode==="dark"||mode==="auto") ? mode : "auto";
  setSavedMode(m);
  document.documentElement.setAttribute("data-mode", m);

  const faction = (m==="auto") ? getSystemFaction() : m;
  document.documentElement.setAttribute("data-faction", faction);

  // 若現行主題不屬於該派，切回該派上次使用的主題
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
  const map = { auto:"#tabAuto", light:"#tabLight", dark:"#tabDark" };
  for (const [k, sel] of Object.entries(map)){
    const el = document.querySelector(sel); if (!el) continue;
    el.setAttribute("aria-selected", k===mode ? "true" : "false");
    el.classList.toggle("active", k===mode);
  }
}
function initThemeUI(){
  migrateLegacyTheme();
  applyMode(getSavedMode());

  const gear = document.getElementById("settingsBtn");
  const menu = document.getElementById("themeMenu");
  const tip  = document.getElementById("themeTip");

  if (gear && menu){
    gear.addEventListener("click",(e)=>{
      e.stopPropagation();
      const open = !menu.hasAttribute("hidden");
      if (open){ menu.setAttribute("hidden",""); gear.setAttribute("aria-expanded","false"); }
      else {
        menu.removeAttribute("hidden"); gear.setAttribute("aria-expanded","true");
        if (tip && !tip.hasAttribute("hidden")){
          tip.setAttribute("hidden",""); try{ localStorage.setItem(TIP_KEY,"1"); }catch{}
        }
      }
    });
    document.addEventListener("click",(e)=>{
      if (!menu.contains(e.target) && e.target!==gear && !gear.contains(e.target)){
        if (!menu.hasAttribute("hidden")){ menu.setAttribute("hidden",""); gear.setAttribute("aria-expanded","false"); }
      }
      if (tip && !tip.hasAttribute("hidden")){
        tip.setAttribute("hidden",""); try{ localStorage.setItem(TIP_KEY,"1"); }catch{}
      }
    });
    menu.querySelectorAll(".theme-item").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        applyTheme(btn.dataset.theme, true);
        menu.setAttribute("hidden",""); gear.setAttribute("aria-expanded","false");
      });
    });
    [["auto","#tabAuto"],["light","#tabLight"],["dark","#tabDark"]].forEach(([name, sel])=>{
      const el = document.querySelector(sel); if (!el) return;
      el.addEventListener("click", ()=> applyMode(name));
    });

    try{
      const mq = matchMedia("(prefers-color-scheme: dark)");
      mq.addEventListener?.("change", ()=>{
        if ((localStorage.getItem(MODE_KEY)||"auto")==="auto") applyMode("auto");
      });
    }catch{}
  }

  // 一次性提示（齒輪）
  try{
    const done = localStorage.getItem(TIP_KEY)==="1";
    if (!done && tip){
      tip.removeAttribute("hidden");
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
const STREAM_WIN_CAND = [12,8,6,4];
const STREAM_HOP_S    = 3;
const EPS             = 1e-9;

// VAD（自適應，只「選段」）
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
const uploadFab = document.getElementById("uploadFab");
const statusEl  = document.getElementById("status");
const meter     = document.getElementById("meter");
const femaleVal = document.getElementById("femaleVal");
const maleVal   = document.getElementById("maleVal");

// Pitch Stream DOM
const pitchWrap   = document.getElementById("pitchWrap");
const pitchCanvas = document.getElementById("pitchCanvas");
const pitchNowEl  = document.getElementById("pitchNow");
const bandNowEl   = document.getElementById("bandNow");
const volNowEl    = document.getElementById("volNow");
const formantWrap = document.getElementById("formantWrap");
const f1NowEl     = document.getElementById("f1Now");
const f2NowEl     = document.getElementById("f2Now");
const f3NowEl     = document.getElementById("f3Now");
const breathNowEl = document.getElementById("breathNow");
const resonanceNowEl = document.getElementById("resonanceNow");
const tiltNowEl   = document.getElementById("tiltNow");
const resBarChest = document.getElementById("resChest");
const resBarMask  = document.getElementById("resMask");
const resBarHead  = document.getElementById("resHead");
const resValChest = document.getElementById("resChestVal");
const resValMask  = document.getElementById("resMaskVal");
const resValHead  = document.getElementById("resHeadVal");

// 播放器（動態建立；並在其下方插入統計卡容器）
let playBtn = null, audioEl = null, lastAudioUrl = null;
ensurePlayerUI();

// ===== 狀態 =====
let mediaRecorder = null, chunks = [];
let clf = null, busy = false, heartbeatTimer = null;

// Pitch Stream 狀態
let psCtx=null, psSrc=null, psProc=null;
let psRAF=null, psRunning=false;
let psHz=[], psDb=[], psVoiced=[]; // 50ms/點
const PS_INTERVAL_MS = 50;
const PS_MIN_HZ = 50, PS_MAX_HZ = 450;
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

// ===== UI 工具 =====
function setStatus(text, spin=false){
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}
function log(...a){ try{ console.log(...a);}catch{} }
function fmtSec(s){ if(!isFinite(s)) return "—"; const m=Math.floor(s/60), ss=Math.round(s%60); return m? `${m}分${ss}秒`:`${ss}秒`; }
function clamp01(x){ return Math.min(1, Math.max(EPS, x)); }

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

// ===== Meter 工具 =====
function resetMeter(){
  try{
    meter?.classList.remove("hidden");
    const barF = document.querySelector(".bar.female");
    const barM = document.querySelector(".bar.male");
    if (barF){ barF.style.setProperty("--p", 0); barF.setAttribute("aria-valuenow","0"); }
    if (barM){ barM.style.setProperty("--p", 0); barM.setAttribute("aria-valuenow","0"); }
    if (femaleVal) femaleVal.textContent = "0.0%";
    if (maleVal)   maleVal.textContent   = "0.0%";
  }catch{}
}

// OOM 檢查
function isOOMError(err){ const msg=String(err?.message||err||""); return /OrtRun|bad_alloc|out of memory|memory|alloc/i.test(msg); }

// ===== 事件 =====
recordBtn?.addEventListener("click", async ()=>{
  if (busy) return;
  try{
    if (!mediaRecorder || mediaRecorder.state==="inactive"){
      resetMeter();
      await startRecording();
    } else {
      await stopRecording();
    }
  }catch(err){ console.error("[recordBtn]", err); setStatus("錄音啟動失敗"); }
});
fileInput?.addEventListener("change", async (e)=>{
  if (busy) return;
  try{
    const f = e.target.files?.[0]; if(!f) return;
    resetMeter();
    await handleFileOrBlob(f);
    e.target.value = "";
  }catch(err){ console.error("[fileInput]", err); setStatus("上傳處理失敗"); }
});

uploadFab?.addEventListener("click", ()=>{
  fileInput?.click();
});

// ===== 錄音 =====
function pickSupportedMime(){
  const cands = ["audio/webm;codecs=opus","audio/webm","audio/mp4","audio/ogg"];
  try{ if(typeof MediaRecorder!=="undefined" && MediaRecorder.isTypeSupported){ for(const t of cands) if(MediaRecorder.isTypeSupported(t)) return t; } }catch{}
  return "";
}
async function startRecording(){
  if (typeof MediaRecorder === "undefined"){ setStatus("此瀏覽器不支援錄音，請改用右側上傳", false); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio:true });
  chunks = [];
  const mimeType = pickSupportedMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  mediaRecorder.ondataavailable = (ev)=>{ if(ev.data?.size) chunks.push(ev.data); };
  mediaRecorder.onstop = async ()=>{
    try{
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      chunks.length = 0;
      stopPitchStream();                 // 停止即時圖，但保留資料做統計
      await handleFileOrBlob(blob);      // 分析完成後會呼叫 finishStreamStats()
    }catch(e){ console.error("[onstop]", e); setStatus("錄音處理失敗"); }
    finally{ stream.getTracks().forEach(t=>t.stop()); }
  };

  document.body.classList.add("recording");
  document.querySelector(".container")?.classList.add("recording");
  setStatus("錄音中… 再按一次停止");
  mediaRecorder.start();

  // 啟動 Pitch Stream
  startPitchStream(stream);
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
  try{
    setPlaybackSource(fileOrBlob);

    setStatus("解析檔案…", true);
    decoded = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
    let { float32, sr, durationSec } = decoded;

    // 離線抽樣（供 Statistics / 簡評）。先對原始音檔做一次。
    offlineExtractStreamMetrics(float32, sr, /*append*/false);

    if (durationSec > WARN_LONG_SEC){
      setStatus(`提示：長度 ${fmtSec(durationSec)}，分析可能較久。準備推論…`, true);
      await microYield();
    }

    // VAD（只選段）
    const vad = maybeApplyAdaptiveVAD(float32, sr);
    if (vad && vad.used){
      const reducedRatio = 1 - (vad.keptSec / durationSec);
      float32 = vad.arr; durationSec = vad.keptSec;
      setStatus(`已去除靜音（約 ${(reducedRatio*100).toFixed(0)}%）→ 有效時長 ${fmtSec(durationSec)}，開始推論…`, true);
      // 針對「有效語音」再抽樣一次，提升代表性
      offlineExtractStreamMetrics(float32, sr, /*append*/true);
      await microYield();
    }

    if (durationSec <= MAX_WHOLE_SEC){
      await analyzeWhole(float32, sr, durationSec);
    } else {
      await analyzeStreamed(float32, sr, durationSec, `長度超過 ${MAX_WHOLE_SEC} 秒，自動切換串流分段`);
    }

    // 顯示統計（錄音/上傳皆會有）
    finishStreamStats();
  }catch(e){
    console.error("[handleFileOrBlob]", e);
    setStatus("處理失敗：" + (e?.message || "無法解碼或分析此音檔"));
  }finally{
    if (decoded) decoded.float32 = null;
    decoded = null; busy = false;
  }
}

// ===== 解碼策略（WebAudio 優先，失敗再 ffmpeg.wasm） =====
async function decodeSmartToFloat32(blobOrFile, targetSR){
  const name = (blobOrFile.name || "").toLowerCase();
  const type = (blobOrFile.type || "").toLowerCase();
  const looksLikeM4A = /\.m4a$/i.test(name) || type.includes("audio/mp4") || type.includes("audio/x-m4a");

  if (IS_SAFARI && looksLikeM4A){
    setStatus("轉檔（ffmpeg，Safari m4a）準備中…", true);
    const wavBlob = await transcodeToWav16kViaFFmpeg(blobOrFile);
    const { float32, sr } = wavToFloat32(await wavBlob.arrayBuffer());
    return { float32, sr, durationSec: float32.length / sr };
  }

  try{
    setStatus("直接解碼（WebAudio）…", true);
    return await decodeViaWebAudio(blobOrFile, targetSR);
  }catch(e){
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
  try{
    const audioBuf = await ctx.decodeAudioData(arrayBuf);
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

// ===== FFmpeg（ESM 優先，失敗用 UMD） =====
async function loadFFmpegModule(){
  try{
    const m = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/+esm");
    if (typeof m.createFFmpeg==="function" && typeof m.fetchFile==="function"){
      return { createFFmpeg: m.createFFmpeg, fetchFile: m.fetchFile, mode:"esm" };
    }
  }catch(e){ log("[ffmpeg] +esm import failed:", e?.message || e); }
  await loadScriptTag("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.6/dist/ffmpeg.min.js");
  const FF = window.FFmpeg;
  if (!FF || typeof FF.createFFmpeg!=="function") throw new Error("FFmpeg UMD load failed");
  return { createFFmpeg: FF.createFFmpeg, fetchFile: FF.fetchFile, mode:"umd" };
}
function loadScriptTag(src){
  return new Promise((resolve, reject)=>{
    const s = document.createElement("script");
    s.src = src; s.async = true; s.crossOrigin = "anonymous"; s.referrerPolicy = "no-referrer";
    s.onload = ()=>resolve(); s.onerror = ()=>reject(new Error("script load error: " + src));
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
  ffmpeg.setProgress(({ ratio })=>{
    const r = Math.min(1, Math.max(0, Number.isFinite(ratio)?ratio:0));
    setStatus(`轉檔（ffmpeg）… ${Math.round(r*100)}%`, true);
  });
  if (!ffmpeg.isLoaded()) await ffmpeg.load();

  const inName="in.bin", outName="out.wav";
  ffmpeg.FS("writeFile", inName, await fetchFile(blobOrFile));
  await ffmpeg.run("-i", inName, "-vn", "-ac", "1", "-ar", `${TARGET_SR}`, "-f", "wav", outName);
  const out = ffmpeg.FS("readFile", outName);
  try{ ffmpeg.FS("unlink", inName); }catch{}
  try{ ffmpeg.FS("unlink", outName); }catch{}
  try{ await ffmpeg.exit(); }catch{}
  return new Blob([out.buffer], { type:"audio/wav" });
}

// ===== 模型 =====
async function ensurePipeline(){
  if (clf) return clf;
  setStatus("下載模型中…（首次會久一點）", true);
  const progress_callback = (p)=>{
    if (!p) return;
    let pct=null;
    if (typeof p.loadedBytes==='number' && typeof p.totalBytes==='number' && p.totalBytes>0) pct=p.loadedBytes/p.totalBytes;
    else if (typeof p.progress==='number' && isFinite(p.progress)) pct=p.progress;
    const label = p.status || "下載模型";
    if (pct==null) setStatus(`${label}…`, true);
    else setStatus(`${label} ${Math.min(99, Math.max(0, Math.floor(pct*100)))}% …`, true);
  };
  const device = (typeof navigator!=='undefined' && navigator.gpu) ? 'webgpu' : 'wasm';
  clf = await pipeline("audio-classification", MODEL_ID, { progress_callback, device });
  setStatus(`模型就緒（device: ${device}）`);
  return clf;
}

// ===== 分析（整段） =====
async function analyzeWhole(float32, sr, durationSec){
  const model = await ensurePipeline();
  meter?.classList.remove("hidden");

  const started = performance.now();
  startHeartbeat(()=>{
    const elapsed=(performance.now()-started)/1000;
    setStatus(`分析中（整段）｜音檔 ${fmtSec(durationSec)}｜已用 ${fmtSec(elapsed)}`, true);
  });

  try{
    const res = await model(float32, { sampling_rate: sr, topk: 2 });
    const map = toMap(res);
    render(map.female||0, map.male||0);
    setStatus("完成（整段）");
  }catch(err){
    if (isOOMError(err)){
      console.warn("[analyzeWhole] OOM → switch to streamed mode…");
      await analyzeStreamed(float32, sr, durationSec, "偵測到記憶體不足，自動改串流分段");
      return;
    }
    console.error("[analyzeWhole]", err);
    setStatus("分析失敗（整段）");
  }finally{ stopHeartbeat(); }
}

// ===== 分析（串流分段） =====
async function analyzeStreamed(float32, sr, durationSec, reason="串流分段"){
  const model = await ensurePipeline();
  meter?.classList.remove("hidden");

  let lastErr=null;
  for (const winSec of STREAM_WIN_CAND){
    try{
      await runStreamedWithWindow(model, float32, sr, durationSec, winSec, STREAM_HOP_S, reason);
      return;
    }catch(e){
      lastErr=e;
      if (isOOMError(e)){ console.warn(`[streamed] OOM at win=${winSec}s → downshift`); continue; }
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
    const elapsed=(performance.now()-started)/1000;
    const pct = processedSec>0 ? Math.min(99, Math.round((processedSec/durationSec)*100)) : 0;
    setStatus(`分析中（串流；win=${WIN_S}s/step=${HOP_S}s）｜${reason}｜${pct}%｜已用 ${fmtSec(elapsed)}`, true);
  });

  try{
    for (let i=0;i<chunks.length;i++){
      const [s0,s1] = chunks[i];
      const seg = float32.subarray(s0, s1);
      const dur = (s1 - s0) / sr;

      const t0 = performance.now();
      const out = await model(seg, { sampling_rate: sr, topk: 2 });
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

// ===== 播放器與統計卡容器 =====
function ensurePlayerUI(){
  const container = document.querySelector("main.container");
  if (!container) return;
  if (document.getElementById("playBtn")) return;

  const wrap = document.createElement("div");
  wrap.className = "player";

  const btn = document.createElement("button");
  btn.id="playBtn"; btn.type="button"; btn.disabled=true;
  btn.textContent="▶︎ 播放剛才的聲音"; btn.setAttribute("aria-label","播放剛才的聲音");

  const hint = document.createElement("div");
  hint.className="hint";
  hint.innerHTML=`想再聽一次剛才那段嗎？<a href="#" id="replayLink">點這裡</a>。`;

  const audio = document.createElement("audio");
  audio.id="playback"; audio.preload="metadata"; audio.style.display="none";

  wrap.appendChild(btn); wrap.appendChild(hint); wrap.appendChild(audio);

  const tipEl = container.querySelector(".callout");
  if (tipEl) container.insertBefore(wrap, tipEl); else container.appendChild(wrap);

  playBtn = btn; audioEl = audio;

  playBtn.onclick = async ()=>{
    if (!audioEl.src) return;
    try{
      if (audioEl.paused){ await audioEl.play(); playBtn.textContent="⏸ 暫停播放"; }
      else { audioEl.pause(); playBtn.textContent="▶︎ 播放剛才的聲音"; }
    }catch(e){ console.error("[audio play]", e); }
  };
  audioEl.onended = ()=>{ playBtn.textContent = "▶︎ 播放剛才的聲音"; };

  wrap.querySelector("#replayLink")?.addEventListener("click",(e)=>{ e.preventDefault(); playBtn.click(); });

  // 統計卡容器（插在播放器區塊後）
  if (!document.getElementById("streamStats")){
    const stats = document.createElement("div");
    stats.id = "streamStats";
    stats.className = "insight";
    stats.innerHTML = "";
    wrap.insertAdjacentElement("afterend", stats);
  }
}
function setPlaybackSource(blob){
  try{
    if(!audioEl || !playBtn) return;
    if(lastAudioUrl){ try{ URL.revokeObjectURL(lastAudioUrl);}catch{} }
    lastAudioUrl = URL.createObjectURL(blob);
    audioEl.src = lastAudioUrl; audioEl.load();
    playBtn.disabled = false; playBtn.textContent = "▶︎ 播放剛才的聲音";
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

// ===== WAV 解析（16-bit PCM / float32 支援） =====
function wavToFloat32(arrayBuffer){
  const dv = new DataView(arrayBuffer);
  function str(o,n){ let s=""; for(let i=0;i<n;i++) s+=String.fromCharCode(dv.getUint8(o+i)); return s; }
  if(str(0,4)!=="RIFF" || str(8,4)!=="WAVE") throw new Error("Not WAV");
  let off=12, fmt=null, dataOff=0, dataLen=0;
  while(off<dv.byteLength){
    const id=str(off,4); const sz=dv.getUint32(off+4,true); const body=off+8;
    if(id==="fmt "){
      fmt={ tag:dv.getUint16(body,true), ch:dv.getUint16(body+2,true), sr:dv.getUint32(body+4,true), bps:dv.getUint16(body+14,true) };
    } else if(id==="data"){ dataOff=body; dataLen=sz; break; }
    off += 8+sz;
  }
  if(!fmt||!dataOff) throw new Error("Invalid WAV");
  const totalSamples = dataLen / (fmt.bps/8);
  const out = new Float32Array(totalSamples / fmt.ch);
  if(fmt.tag===1 && fmt.bps===16){ // PCM 16
    let j=0;
    for(let i=0;i<totalSamples;i+=fmt.ch){ const v=dv.getInt16(dataOff + (i*2), true)/32768; out[j++]=v; }
  }else if(fmt.tag===3 && fmt.bps===32){ // IEEE float32
    let j=0;
    for(let i=0;i<totalSamples;i+=fmt.ch){ const v=dv.getFloat32(dataOff + (i*4), true); out[j++]=v; }
  }else{
    throw new Error("Unsupported WAV encoding");
  }
  return { float32: out, sr: fmt.sr };
}

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
function startPitchStream(userMediaStream){
  try{
    if (!pitchWrap || !pitchCanvas) return;
    psHz.length=0; psDb.length=0; psVoiced.length=0;

    const Ctx = window.AudioContext || window.webkitAudioContext;
    psCtx = new Ctx();
    psSrc = psCtx.createMediaStreamSource(userMediaStream);
    psProc = psCtx.createScriptProcessor(2048, 1, 1);

    let lastTick = 0;
    psProc.onaudioprocess = (ev)=>{
      const input = ev.inputBuffer.getChannelData(0);
      const rms = Math.sqrt(input.reduce((a,v)=>a+v*v,0) / Math.max(1,input.length));
      const db  = 20*Math.log10(Math.max(rms, 1e-6)) + 100; // 相對 dB
      const hz  = detectPitchACF(input, psCtx.sampleRate);  // null 表 unvoiced
      const now = performance.now();
      if (now - lastTick >= PS_INTERVAL_MS){
        const spectral = estimateSpectralFeatures(input, psCtx.sampleRate);
        psDb.push(db);
        psHz.push(hz ?? null);
        psVoiced.push(hz!=null);
        const maxN = Math.round(15000 / PS_INTERVAL_MS); // 保留約 15 秒
        if (psDb.length>maxN){ psDb.shift(); psHz.shift(); psVoiced.shift(); }
        lastTick = now;

        if (pitchNowEl) pitchNowEl.textContent = hz ? `${hz.toFixed(1)}Hz` : "— Hz";
        if (volNowEl)   volNowEl.textContent   = `${db.toFixed(1)} dB`;
        if (bandNowEl)  bandNowEl.textContent  = bandLabel(hz);
        updateRealtimeMonitor(spectral);
      }
    };

    psSrc.connect(psProc); psProc.connect(psCtx.destination);
    psRunning = true; pitchWrap.hidden = false;
    startDrawLoop();
  }catch(e){ console.error("[startPitchStream]", e); }
}

function updateRealtimeMonitor(features){
  try{
    if (!formantWrap) return;
    if (!features){
      return;
    }
    formantWrap.hidden = false;
    const { f1, f2, f3, breathiness, tilt, energy } = features;
    if (f1NowEl) f1NowEl.textContent = Number.isFinite(f1) ? `${Math.round(f1)} Hz` : "— Hz";
    if (f2NowEl) f2NowEl.textContent = Number.isFinite(f2) ? `${Math.round(f2)} Hz` : "— Hz";
    if (f3NowEl) f3NowEl.textContent = Number.isFinite(f3) ? `${Math.round(f3)} Hz` : "— Hz";
    if (breathNowEl) breathNowEl.textContent = Number.isFinite(breathiness)
      ? `${Math.round(breathiness*100)}%`
      : "—";

    const desc = describeResonanceFromEnergy(energy);
    if (resonanceNowEl) resonanceNowEl.textContent = desc.label || "—";
    if (tiltNowEl) tiltNowEl.textContent = Number.isFinite(tilt) ? `Tilt ${fmt1(tilt)} dB` : "Tilt —";

    const total = Math.max(energy?.total || 0, EPS);
    const chest = Math.max(0, Math.min(1, (energy?.low || 0) / total));
    const mask  = Math.max(0, Math.min(1, (energy?.mid || 0) / total));
    const head  = Math.max(0, Math.min(1, (energy?.high || 0) / total));
    const sum   = Math.max(chest + mask + head, EPS);
    const chestPct = chest / sum;
    const maskPct  = mask / sum;
    const headPct  = head / sum;

    if (resBarChest){ resBarChest.style.flexGrow = Math.max(chestPct, 0.001); resBarChest.style.flexBasis = `${(chestPct*100).toFixed(1)}%`; }
    if (resBarMask){ resBarMask.style.flexGrow = Math.max(maskPct, 0.001); resBarMask.style.flexBasis = `${(maskPct*100).toFixed(1)}%`; }
    if (resBarHead){ resBarHead.style.flexGrow = Math.max(headPct, 0.001); resBarHead.style.flexBasis = `${(headPct*100).toFixed(1)}%`; }
    if (resValChest) resValChest.textContent = `胸 ${Math.round(chestPct*100)}%`;
    if (resValMask)  resValMask.textContent  = `面罩 ${Math.round(maskPct*100)}%`;
    if (resValHead)  resValHead.textContent  = `頭 ${Math.round(headPct*100)}%`;
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
  if (hz < 85) return "低（<85）";
  if (hz < 165) return "藍 85–165";
  if (hz < 180) return "中性 165–180";
  if (hz < 310) return "粉 180–310";
  if (hz <= 450) return "高 310–450";
  return "超出範圍";
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
    if (!psRunning && psHz.length===0){ psRAF = requestAnimationFrame(draw); return; }
    const w=pitchCanvas.width, h=pitchCanvas.height;
    ctx.clearRect(0,0,w,h);
    drawBands();

    const styles = getComputedStyle(document.documentElement);
    ctx.lineWidth = 2*DPR;
    ctx.strokeStyle = styles.getPropertyValue("--stream-ink") || "#222";

    // 往右跑：最右是最新
    const stepX = 3*DPR;
    const maxN  = Math.floor(w/stepX)-2;
    const n = Math.min(psHz.length, maxN);
    ctx.beginPath();
    for (let i=0;i<n;i++){
      const hz = psHz[psHz.length-n+i];
      const x = w - (n-i)*stepX;
      if (hz==null) continue;
      const y = yOf(hz);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();

    psRAF = requestAnimationFrame(draw);
  }
  draw();
}

// ===== 離線抽樣（上傳檔用；錄音也會補） =====
function offlineExtractStreamMetrics(float32, sr, append=false){
  try{
    if(!append){
      psHz.length=0; psDb.length=0; psVoiced.length=0;
      resetOfflineFeatureStore();
    }
    const step = Math.max(1, Math.floor((PS_INTERVAL_MS/1000)*sr));
    const frame = Math.min(Math.floor(0.08*sr), 8192); // ~80ms ACF 視窗
    offlineFeatureStore.frameSec = step / sr;
    for(let i=0;i+frame<=float32.length; i+=step){
      const seg = float32.subarray(i, i+frame);
      const db = 20*Math.log10(Math.max(rms(seg,0,seg.length), 1e-6)) + 100;
      const hz = detectPitchACF(seg, sr);
      const spectral = estimateSpectralFeatures(seg, sr);
      psDb.push(db);
      psHz.push(hz ?? null);
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
        <span class="badge">Statistics</span>
        <div class="tags"></div>
      </div>
    `;

    // 僅對有聲點統計；若沒有資料就清空
    const voicedHz = psHz.filter(v => Number.isFinite(v));
    const vols     = psDb.slice();
    if (!voicedHz.length && !vols.length){ statsEl.innerHTML=""; return; }

    const pitchStats = makeStats(voicedHz);
    const volStats   = makeStats(vols);
    const volsSorted = vols.slice().sort((a,b)=>a-b);
    const envDb      = percentileSorted(volsSorted, 10); // 10th 近似環境底噪
    const snr        = Number.isFinite(volStats.med) && Number.isFinite(envDb) ? (volStats.med - envDb) : NaN;

    // ====== 簡評（可一眼看懂）======
    const band = bandOf(pitchStats.med);                 // 常見音高區（依 Median）
    const spread = (pitchStats.p95 - pitchStats.p05);    // 變化幅度
    const stability = (isFinite(spread) ? (spread < 40 ? "穩定" : spread <= 80 ? "中等變化" : "變化較大") : "—");
    const snrTag = (isFinite(snr) ? (snr >= 20 ? "安靜／很清楚" : snr >= 12 ? "可用／一般" : "偏吵／建議換環境") : "—");
    const volSigmaTag = (isFinite(volStats.sd) ? (volStats.sd < 6 ? "穩定" : volStats.sd <= 12 ? "中等" : "波動大") : "—");

    // 指標分歧（模型傾向 vs 音高常見區）
    const diverge = isDivergent(pitchStats.med, lastPf, lastPm);
    const divergeBadge = diverge
      ? `<span class="chip">指標分歧</span>`
      : "";

    // 取樣覆蓋率（錄音期間有聲點比例）
    const voicedRatio = psVoiced.length ? (psVoiced.filter(Boolean).length / psVoiced.length) : NaN;
    const voicedHint = (!isFinite(voicedRatio) || voicedRatio < 0.25)
      ? "取樣偏少，建議 5–10 秒連續語句。"
      : (voicedRatio < 0.5 ? "取樣略少，可再多說一點讓統計更穩。" : "");

    const oneLiner = `
      <div class="summary-line">
        <strong>簡評：</strong>
        模型傾向（多特徵） F ${(lastPf*100).toFixed(1)}% / M ${(lastPm*100).toFixed(1)}% ｜
        音高（單一特徵）Median ${fmt1(pitchStats.med)} Hz（常見音高區：${band}；${stability}）｜
        SNR：${isFinite(snr)? fmt1(snr) + " dB" : "—"}（${snrTag}） ${divergeBadge}
      </div>
    `;

    const divergeNote = diverge
      ? `<p class="subline" style="margin:6px 0 0">
          <b>指標分歧</b>：音高落在 <b>${band}</b>，但模型仍偏向 <b>${lastPf>=lastPm?"女性化":"男性化"}</b>。
          兩者量測不同面向（模型含共鳴、音色、發音模式等；音高僅看 Hz），屬正常情形。
        </p>`
      : "";

    const envNote = isFinite(snr) && snr < 12
      ? `<p class="subline" style="margin:4px 0 0">環境偏吵（SNR 低於 12 dB），建議更安靜場景或拉近麥克風。</p>`
      : "";

    const voicedNote = voicedHint
      ? `<p class="subline" style="margin:4px 0 0">${voicedHint}</p>`
      : "";

    const statsHTML = `
      <div class="stats-grid">
        <div class="kv"><div class="k">Pitch · Average</div><div class="v">${fmt1(pitchStats.avg)}Hz</div></div>
        <div class="kv"><div class="k">Pitch · Median</div><div class="v">${fmt1(pitchStats.med)}Hz</div></div>
        <div class="kv"><div class="k">Pitch · High (95th)</div><div class="v">${fmt1(pitchStats.p95)}Hz</div></div>
        <div class="kv"><div class="k">Pitch · Low (5th)</div><div class="v">${fmt1(pitchStats.p05)}Hz</div></div>

        <div class="kv"><div class="k">Volume · Average (σ)</div><div class="v">${fmt1(volStats.avg)}dB (${fmt1(volStats.sd)}dB)</div></div>
        <div class="kv"><div class="k">Volume · Median (σ)</div><div class="v">${fmt1(volStats.med)}dB (${fmt1(volStats.sd)}dB)</div></div>
        <div class="kv"><div class="k">Volume · High (95th)</div><div class="v">${fmt1(volStats.p95)}dB</div></div>
        <div class="kv"><div class="k">Volume · Low (5th)</div><div class="v">${fmt1(volStats.p05)}dB</div></div>
      </div>
      <div class="kv" style="margin-top:10px"><div class="k">Environment</div><div class="v">${fmt1(envDb)}dB</div></div>
      <p class="subline" style="margin:8px 0 0">音量波動（σ）：<b>${volSigmaTag}</b>；這些指標是練習回饋，<u>不是性別認定</u>。</p>
    `;
    const advSummary = computeAdvancedSummary();
    const advancedHTML = renderAdvancedSummary(advSummary);

    statsEl.innerHTML = headerHTML + oneLiner + divergeNote + envNote + voicedNote + statsHTML + advancedHTML;

    // 標籤列
    const tags = statsEl.querySelector(".tags");
    if (tags){
      let tagHTML = `
        <span class="tag">Pitch band：${band}</span>
        <span class="tag">環境底噪：約 ${fmt1(envDb)} dB</span>
      `;
      if (advSummary){
        if (advSummary.resonanceLabel) tagHTML += `<span class="tag">共鳴：${advSummary.resonanceLabel}</span>`;
        if (advSummary.speechRateLabel) tagHTML += `<span class="tag">語速：${advSummary.speechRateLabel}</span>`;
        if (advSummary.breathinessLabel) tagHTML += `<span class="tag">氣聲：${advSummary.breathinessLabel}</span>`;
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
    f1Med: f1Stats?.med ?? NaN,
    f2Med: f2Stats?.med ?? NaN,
    f3Med: f3Stats?.med ?? NaN,
    f1Hint: makeFormantHint("F1", f1Stats?.med, 180, 350),
    f2Hint: makeFormantHint("F2", f2Stats?.med, 1600, 2500),
    f3Hint: makeFormantHint("F3", f3Stats?.med, 2500, 3200),
    resonanceLabel: resonanceDesc.label,
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
        <div class="note">語音時長不足，暫無 formant、共鳴與語速分析。請錄製 5–10 秒連續語句。</div>
      </div>
    `;
  }
  const chestPct = Math.round((summary.energyPct?.chest ?? 0.33)*100);
  const maskPct  = Math.round((summary.energyPct?.mask ?? 0.33)*100);
  const headPct  = Math.round((summary.energyPct?.head ?? 0.34)*100);
  const speechRateDisplay = Number.isFinite(summary.speechRate?.syllPerSec)
    ? `${fmt1(summary.speechRate.syllPerSec)} 音節/秒`
    : "—";
  const speechWpm = Number.isFinite(summary.speechRate?.wordsPerMin)
    ? `${Math.round(summary.speechRate.wordsPerMin)} wpm`
    : "—";
  const liaisonDisplay = Number.isFinite(summary.liaisonRatio)
    ? `${Math.round(summary.liaisonRatio*100)}%`
    : "—";
  const vowelDisplay = summary.vowelLabel + (Number.isFinite(summary.vowelFocusRatio) ? ` · ${Math.round(summary.vowelFocusRatio*100)}%` : "");
  const breathDisplay = summary.breathinessLabel + (Number.isFinite(summary.breathinessAvg) ? ` · ${Math.round(summary.breathinessAvg*100)}%` : "");
  const rangeDisplay = Number.isFinite(summary.intonation?.range) ? `${fmt1(summary.intonation.range)} Hz` : "—";
  const intonationLabel = summary.intonation?.slopeLabel || "—";
  const intonationHint = summary.intonation?.hint || "語調資訊不足，建議錄製更長語句。";
  const rangeHint = summary.intonation?.rangeHint || "";

  return `
    <div class="advanced-section">
      <h3 class="adv-title">Formant 與共鳴</h3>
      <div class="advanced-grid advanced-grid--four">
        <div class="adv-card"><div class="k">F1 · Median</div><div class="v">${fmt1(summary.f1Med)}Hz</div><div class="hint">${summary.f1Hint}</div></div>
        <div class="adv-card"><div class="k">F2 · Median</div><div class="v">${fmt1(summary.f2Med)}Hz</div><div class="hint">${summary.f2Hint}</div></div>
        <div class="adv-card"><div class="k">F3 · Median</div><div class="v">${fmt1(summary.f3Med)}Hz</div><div class="hint">${summary.f3Hint}</div></div>
        <div class="adv-card"><div class="k">Spectral Tilt</div><div class="v">${fmt1(summary.tiltAvg)} dB</div><div class="hint">${summary.tiltHint}</div></div>
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
      <h3 class="adv-title">語調與語速</h3>
      <canvas id="intonationCanvas" width="520" height="140" aria-label="Intonation curve"></canvas>
      <div class="advanced-grid advanced-grid--four">
        <div class="adv-card"><div class="k">語調趨勢</div><div class="v">${intonationLabel}</div><div class="hint">${intonationHint}</div></div>
        <div class="adv-card"><div class="k">音高動態</div><div class="v">${rangeDisplay}</div><div class="hint">${rangeHint}</div></div>
        <div class="adv-card"><div class="k">語速估計</div><div class="v">${speechRateDisplay}</div><div class="hint">${summary.speechRateHint}（約 ${speechWpm}）</div></div>
        <div class="adv-card"><div class="k">連音比例</div><div class="v">${summary.liaisonLabel} · ${liaisonDisplay}</div><div class="hint">${summary.liaisonHint}</div></div>
      </div>
    </div>
    <div class="advanced-section">
      <h3 class="adv-title">元音聚焦與氣聲</h3>
      <div class="advanced-grid advanced-grid--three">
        <div class="adv-card"><div class="k">元音聚焦</div><div class="v">${vowelDisplay}</div><div class="hint">${summary.vowelHint}</div></div>
        <div class="adv-card"><div class="k">氣聲比例</div><div class="v">${breathDisplay}</div><div class="hint">${summary.breathinessHint}</div></div>
        <div class="adv-card"><div class="k">共鳴傾向</div><div class="v">${summary.tiltLabel}</div><div class="hint">${summary.tiltHint}</div></div>
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
  if (!Array.isArray(arr) || !arr.length) return { low:0, mid:0, high:0, total:0 };
  let low=0, mid=0, high=0, count=0;
  for (const v of arr){
    if (!Array.isArray(v)) continue;
    const [l,m,h] = v;
    if (!Number.isFinite(l) && !Number.isFinite(m) && !Number.isFinite(h)) continue;
    low += Number.isFinite(l) ? l : 0;
    mid += Number.isFinite(m) ? m : 0;
    high += Number.isFinite(h) ? h : 0;
    count++;
  }
  if (!count) return { low:0, mid:0, high:0, total:0 };
  return { low:low/count, mid:mid/count, high:high/count, total:(low+mid+high)/count };
}

function describeResonanceFromEnergy(energy){
  if (!energy || (!Number.isFinite(energy.low) && !Number.isFinite(energy.mid) && !Number.isFinite(energy.high))){
    return { label:"資料不足", hint:"需要更多穩定語音才能估算共鳴分布。", pct:{ chest:1/3, mask:1/3, head:1/3 }, total:0 };
  }
  const total = Math.max((energy.low||0) + (energy.mid||0) + (energy.high||0), EPS);
  const chestPct = Math.max(0, (energy.low||0) / total);
  const maskPct  = Math.max(0, (energy.mid||0) / total);
  const headPct  = Math.max(0, (energy.high||0) / total);
  let label="平衡", hint="胸 / 面罩 / 頭腔能量分布均衡，可依需求微調亮度。";
  if (headPct >= 0.32){
    label = "頭腔亮度強";
    hint = "高頻占比較高，保持放鬆的軟顎與氣息，避免聲音過尖。";
  } else if (chestPct >= 0.45){
    label = "胸腔偏重";
    hint = "胸腔能量較多，可抬高喉頭、增加口腔共鳴來提亮聲音。";
  } else if (maskPct >= chestPct && maskPct >= headPct){
    label = "面罩共鳴主導";
    hint = "面罩區域占比最高，維持鼻腔開放同時注意放鬆喉部。";
  }
  return { label, hint, pct:{ chest:chestPct, mask:maskPct, head:headPct }, total };
}

function categorizeTilt(tilt){
  if (!Number.isFinite(tilt)) return { label:"資料不足", hint:"錄製更多穩定語音片段以估算共鳴傾向。" };
  if (tilt >= 8) return { label:"濃厚暖色", hint:"低頻占比較高，可加入頭腔共鳴提升明亮度。" };
  if (tilt >= 3) return { label:"平衡偏暖", hint:"低頻略多，維持支撐的同時讓口腔更前放。" };
  if (tilt >= -1) return { label:"平衡", hint:"頻譜傾斜度均衡，可依語意微調亮度。" };
  return { label:"亮度強", hint:"高頻偏多，放鬆喉頭並加強氣息支撐來柔化聲音。" };
}

function categorizeBreathiness(val){
  if (!Number.isFinite(val)) return { label:"資料不足", hint:"需要更多有聲樣本才能判斷氣聲比例。" };
  if (val < 0.08) return { label:"偏實聲", hint:"氣聲較少，可加入些許氣息讓聲音更柔和。" };
  if (val <= 0.18) return { label:"平衡", hint:"氣聲落在建議的 8%–18% 區間，維持目前的氣息控制。" };
  if (val <= 0.28) return { label:"偏氣聲", hint:"氣聲略多，收緊聲帶或增加呼吸支撐可提升聚焦。" };
  return { label:"氣聲過多", hint:"氣流外洩明顯，建議練習連續母音與收聲帶閉合。" };
}

function makeFormantHint(label, value, low, high){
  if (!Number.isFinite(value)) return `${label} 無法估計，錄音過短或噪音過高。`;
  if (value < low) return `${label} 偏低，可抬高舌位、縮小口腔體積讓共鳴往前。`;
  if (value > high) return `${label} 偏高，試著放鬆舌根或增加口腔開度保持厚度。`;
  return `${label} 落在建議範圍，維持目前的發聲位置。`;
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
  if (!Number.isFinite(ratio)) return { ratio:NaN, label:"資料不足", hint:"需要更多穩定母音才能評估聚焦程度。" };
  if (ratio >= 0.6) return { ratio, label:"聚焦良好", hint:"超過 60% 母音落在女性常見區，維持目前的舌位與口形。" };
  if (ratio >= 0.4) return { ratio, label:"可再集中", hint:"約一半母音達標，可練習延長母音、保持舌尖前放。" };
  return { ratio, label:"需加強", hint:"聚焦比例低，建議練習 /i/、/e/ 等前母音建立高 F2。" };
}

function analyzeSpeechRate(store){
  const hopSec = store.frameSec || (PS_INTERVAL_MS/1000);
  const n = store.db.length;
  if (!n) return { syllPerSec:NaN, wordsPerMin:NaN, label:"資料不足", hint:"語速資訊不足，建議錄 5–10 秒語句。" };
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
  if (!Number.isFinite(syllPerSec) || syllPerSec <= 0) return { syllPerSec:NaN, wordsPerMin:NaN, label:"資料不足", hint:"語速資訊不足，建議再錄一次。" };
  if (syllPerSec < 2.2) return { syllPerSec, wordsPerMin, label:"偏慢", hint:"語速偏慢（建議 2.5–4 音節/秒），可縮短停頓、保持語尾上揚。" };
  if (syllPerSec <= 4.2) return { syllPerSec, wordsPerMin, label:"適中", hint:"語速落在常見練習區間，維持吐字清晰與節奏。" };
  return { syllPerSec, wordsPerMin, label:"偏快", hint:"語速偏快，可放慢語尾母音並確認發音完整。" };
}

function analyzeConnectedSpeech(voicedArr, hopSec){
  if (!Array.isArray(voicedArr) || !voicedArr.length) return { ratio:NaN, label:"資料不足", hint:"語句過短，無法評估連音。" };
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
  if (!Number.isFinite(ratio)) return { ratio:NaN, label:"資料不足", hint:"語音片段不足以計算連音比例。" };
  if (ratio >= 0.7) return { ratio, label:"連音良好", hint:"大部分停頓低於 160ms，保持順滑連接。" };
  if (ratio >= 0.4) return { ratio, label:"中等", hint:"部分字詞仍有停頓，可練習連讀或弱化輔音。" };
  return { ratio, label:"偏斷裂", hint:"短停頓比例低，試著延長母音或加強連音練習。" };
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
    return { points:[], slope:NaN, slopeLabel:"資料不足", hint:"語調點太少，建議錄製更長語句。", range:NaN, rangeHint:"", minHz:NaN, maxHz:NaN };
  }
  const n = points.length;
  let sumT=0, sumH=0, sumTT=0, sumTH=0;
  for (const {t,hz} of points){
    sumT += t; sumH += hz; sumTT += t*t; sumTH += t*hz;
  }
  const slope = (n*sumTH - sumT*sumH) / Math.max((n*sumTT - sumT*sumT), EPS);
  const range = maxHz - minHz;
  let slopeLabel="平穩", slopeHint="整體趨勢平穩，可在句尾加些上揚提升表情。";
  if (slope > 12){ slopeLabel="上揚"; slopeHint="語尾持續上揚，有助營造更明亮的語調。"; }
  else if (slope < -12){ slopeLabel="下降"; slopeHint="語尾明顯下降，可在結尾加入上揚以維持女性語感。"; }
  let rangeLabel="偏平", rangeHint="音高變化較小，可練習階梯式上揚。";
  if (range >= 90){ rangeLabel="變化豐富"; rangeHint="音高跨距大，注意控制不要失去穩定。"; }
  else if (range >= 50){ rangeLabel="適中"; rangeHint="音高動態適中，可依語意微調。"; }
  const hint = `${slopeHint} ${rangeHint}`.trim();
  return { points, slope, slopeLabel, range, rangeLabel, hint, rangeHint, minHz, maxHz };
}

function drawIntonationCurve(canvas, intonation){
  try{
    const pts = intonation?.points || [];
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.clientWidth || canvas.width || 520;
    const height = canvas.clientHeight || canvas.height || 140;
    const DPR = Math.max(1, window.devicePixelRatio||1);
    canvas.width = width * DPR;
    canvas.height = height * DPR;
    ctx.scale(DPR, DPR);
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
  if (medHz < 85) return "低域（<85Hz）";
  if (medHz < 165) return "男性常見區（85–165Hz）";
  if (medHz < 180) return "重疊帶（中性 165–180Hz）";
  if (medHz < 310) return "女性常見區（180–310Hz）";
  if (medHz <= 450) return "高域（>310Hz）";
  return "超出範圍";
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
