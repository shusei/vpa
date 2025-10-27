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
        psDb.push(db);
        psHz.push(hz ?? null);
        psVoiced.push(hz!=null);
        const maxN = Math.round(15000 / PS_INTERVAL_MS); // 保留約 15 秒
        if (psDb.length>maxN){ psDb.shift(); psHz.shift(); psVoiced.shift(); }
        lastTick = now;

        if (pitchNowEl) pitchNowEl.textContent = hz ? `${hz.toFixed(1)}Hz` : "— Hz";
        if (volNowEl)   volNowEl.textContent   = `${db.toFixed(1)} dB`;
        if (bandNowEl)  bandNowEl.textContent  = bandLabel(hz);
      }
    };

    psSrc.connect(psProc); psProc.connect(psCtx.destination);
    psRunning = true; pitchWrap.hidden = false;
    startDrawLoop();
  }catch(e){ console.error("[startPitchStream]", e); }
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
    if(!append){ psHz.length=0; psDb.length=0; psVoiced.length=0; }
    const step = Math.max(1, Math.floor((PS_INTERVAL_MS/1000)*sr));
    const frame = Math.min(Math.floor(0.08*sr), 8192); // ~80ms ACF 視窗
    for(let i=0;i+frame<=float32.length; i+=step){
      const seg = float32.subarray(i, i+frame);
      const db = 20*Math.log10(Math.max(rms(seg,0,seg.length), 1e-6)) + 100;
      const hz = detectPitchACF(seg, sr);
      psDb.push(db);
      psHz.push(hz ?? null);
      psVoiced.push(hz!=null);
    }
  }catch(e){ console.error("[offlineExtractStreamMetrics]", e); }
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

    statsEl.innerHTML = headerHTML + oneLiner + divergeNote + envNote + voicedNote + statsHTML;

    // 標籤列
    const tags = statsEl.querySelector(".tags");
    if (tags){
      tags.innerHTML = `
        <span class="tag">Pitch band：${band}</span>
        <span class="tag">環境底噪：約 ${fmt1(envDb)} dB</span>
      `;
    }
  }catch(e){ console.error("[finishStreamStats]", e); }
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
