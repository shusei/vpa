import { t, onLocaleChange } from "./i18n.js";
import { fmtSec, clamp01, escapeHtml, escapeAttr, fmt1 } from "./utils.js";
import { getAdvancedMode, setAdvancedMode, getDetailsOpen, setDetailsOpen, WARMUP_CARD_OPEN_KEY, ADVANCED_MODE_KEY, ADV_DETAILS_KEY_PREFIX } from "./ui/ui-state.js";
import { psHz, psHzSmooth, psRunning } from "./service.js";
import { pitchCanvas } from "./dom.js";
import { EPS } from "./config.js";
import { makeStats, percentileSorted, computeIntonationMetrics, PS_MIN_HZ, PS_MAX_HZ } from "./pitch-shared.js";

let psRAF = null;

export function setStatus(msg, showLoading) {
  const el = document.getElementById("status");
  if (el) {
    el.innerText = msg;
    if (showLoading) el.classList.add("loading");
    else el.classList.remove("loading");
  }
}

// Store last analysis results for statistics
let lastPf = 0;
let lastPm = 0;

export function render(pf, pm) {
  // Store values for stats card
  lastPf = pf;
  lastPm = pm;

  const f = document.getElementById("femaleVal");
  const m = document.getElementById("maleVal");
  if (f) {
    f.innerText = (pf * 100).toFixed(1) + "%";
    const bar = f.closest(".bar");
    if (bar) bar.style.setProperty("--p", pf);
  }
  if (m) {
    m.innerText = (pm * 100).toFixed(1) + "%";
    const bar = m.closest(".bar");
    if (bar) bar.style.setProperty("--p", pm);
  }
}

export function startDrawLoop() {
  const ctx = pitchCanvas.getContext("2d");
  const DPR = Math.max(1, window.devicePixelRatio || 1);
  function resize() {
    const r = pitchCanvas.getBoundingClientRect();
    pitchCanvas.width = Math.max(600, Math.round(r.width * DPR));
    pitchCanvas.height = Math.round(r.height * DPR);
  }
  resize(); addEventListener("resize", resize);

  function yOf(hz) {
    const h = pitchCanvas.height;
    const clamped = Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz));
    return h - ((clamped - PS_MIN_HZ) / (PS_MAX_HZ - PS_MIN_HZ)) * h;
  }
  function drawBands() {
    const styles = getComputedStyle(document.documentElement);
    const cGray = styles.getPropertyValue("--band-gray") || "#ddd";
    const cBlue = styles.getPropertyValue("--band-blue") || "#bfe7ff";
    const cPink = styles.getPropertyValue("--band-pink") || "#ffd1dc";
    const cLilac = styles.getPropertyValue("--band-lilac") || "#e2d5ff";
    const w = pitchCanvas.width, h = pitchCanvas.height;

    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(85), w, h - yOf(85));
    ctx.fillStyle = cBlue; ctx.fillRect(0, yOf(165), w, yOf(85) - yOf(165));
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(180), w, yOf(165) - yOf(180));
    ctx.fillStyle = cPink; ctx.fillRect(0, yOf(310), w, yOf(180) - yOf(310));
    ctx.fillStyle = cGray; ctx.fillRect(0, yOf(450), w, yOf(310) - yOf(450));
    ctx.fillStyle = cLilac; ctx.fillRect(0, 0, w, yOf(450));

    ctx.strokeStyle = "rgba(0,0,0,.08)"; ctx.lineWidth = 1 * DPR;
    [50, 85, 165, 180, 310, 450, PS_MAX_HZ].forEach(f => { const y = yOf(f); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); });
  }

  function draw() {
    if (!psRunning && psHzSmooth.length === 0) { psRAF = requestAnimationFrame(draw); return; }
    const w = pitchCanvas.width, h = pitchCanvas.height;
    ctx.clearRect(0, 0, w, h);
    drawBands();

    const styles = getComputedStyle(document.documentElement);
    ctx.lineWidth = 2 * DPR;
    ctx.strokeStyle = styles.getPropertyValue("--stream-ink") || "#222";

    const stepX = 3 * DPR;
    const maxN = Math.floor(w / stepX) - 2;
    const n = Math.min(psHzSmooth.length, maxN);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const hz = psHzSmooth[psHzSmooth.length - n + i] ?? psHz[psHz.length - n + i];
      const x = w - (n - i) * stepX;
      if (hz == null) continue;
      const y = yOf(hz);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const axisColor = styles.getPropertyValue("--stream-axis") || styles.getPropertyValue("--muted") || "rgba(0,0,0,.5)";
    const axisFont = (styles.getPropertyValue("--font-ui") || "sans-serif").trim() || "sans-serif";
    const axisFontSize = 11 * DPR;
    const axisTicks = [PS_MAX_HZ, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50];
    const tickLen = 6 * DPR;
    const leftX = 8 * DPR;
    const rightX = w - 8 * DPR;
    const labelHalf = axisFontSize * 0.6;

    ctx.save();
    ctx.fillStyle = axisColor;
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1 * DPR;
    ctx.font = `${axisFontSize}px ${axisFont}`;
    ctx.textBaseline = "middle";

    axisTicks.forEach((hz) => {
      const y = yOf(hz);
      const textY = Math.min(Math.max(y, labelHalf), h - labelHalf);

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(tickLen, y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(w, y);
      ctx.lineTo(w - tickLen, y);
      ctx.stroke();

      ctx.textAlign = "left";
      ctx.fillText(`${hz} Hz`, leftX, textY);
      ctx.textAlign = "right";
      ctx.fillText(`${hz} Hz`, rightX, textY);
    });

    ctx.restore();

    psRAF = requestAnimationFrame(draw);
  }
  draw();
}



export function renderBeginnerHighlights(summary, context = {}) {
  return "";
}



export function finishStreamStats() {
  // Dynamic import to avoid circular dependency
  import("./ui/stats-card.js").then(({ renderStatsCard }) => {
    import("./service.js").then(({ psHzSmooth, psConfidence, psDb, psVoiced, offlineFeatureStore }) => {
      const dataBundle = {
        psHzSmooth,
        psConfidence,
        psDb,
        psVoiced,
        lastPf,
        lastPm,
        offlineFeatureStore,
        advSummary: offlineFeatureStore?.advSummary, // Pass it if already computed/stored
      };
      renderStatsCard(dataBundle);
    });
  });
}

export function updateRealtimeMonitor({ hz, db, band, spectral }) {
  const pitchEl = document.getElementById("pitchNow");
  const bandEl = document.getElementById("bandNow");
  const volEl = document.getElementById("volNow");

  if (pitchEl) pitchEl.innerText = Number.isFinite(hz) ? Math.round(hz) : "—";
  if (bandEl) bandEl.innerText = band || "—";
  if (volEl) volEl.innerText = Number.isFinite(db) ? Math.round(db) + " dB" : "—";
}



// ===== Advanced UI state & gauges =====
// Moved to ./ui/ui-state.js

export function setupAdvancedSection(root) {
  if (!root) return;

  const toggleBtn = root.querySelector("[data-adv-toggle]");

  const labelFor = (mode) =>
    mode === "advanced"
      ? (t("ui.advancedMode.beginner") || "Switch to Beginner (collapse all)")
      : (t("ui.advancedMode.advanced") || "Switch to Advanced (expand all)");

  // force: null=用記憶, "expand"=全部展開, "collapse"=全部收起
  // persist: 是否把這次結果寫回每塊的記憶
  function applyMode(force, persist) {
    const mode = getAdvancedMode(); // "beginner" | "advanced"
    root.setAttribute("data-mode", mode);
    if (toggleBtn) {
      toggleBtn.setAttribute("aria-pressed", mode === "advanced" ? "true" : "false");
      toggleBtn.textContent = labelFor(mode);
    }

    const blocks = root.querySelectorAll("details[data-adv], details.adv-details, details.adv");
    blocks.forEach(d => {
      const key = d.getAttribute("data-adv") || "";
      let open;
      if (force === "expand") open = true;
      else if (force === "collapse") open = false;
      else open = getDetailsOpen(key, mode === "advanced"); // 用記憶，沒有就依模式預設

      d.open = open;
      d.setAttribute("aria-expanded", open ? "true" : "false");
      if (persist === true) setDetailsOpen(key, open);
    });
  }

  // 初次套用：尊重既有記憶（不強制、不覆蓋）
  applyMode(null, false);

  // 一鍵切換：預設不覆蓋使用者記憶；按住 Shift/Alt 可「順便寫入記憶」
  if (toggleBtn) {
    toggleBtn.addEventListener("click", (ev) => {
      const next = getAdvancedMode() === "advanced" ? "beginner" : "advanced";
      setAdvancedMode(next);

      const force = next === "advanced" ? "expand" : "collapse";
      const persist = ev.shiftKey || ev.altKey; // Shift/Alt 點擊 = 覆蓋記憶
      applyMode(force, persist);
    });
  }

  // 使用者手動展開/收起某一塊時，更新該塊的記憶
  root.querySelectorAll("details[data-adv], details.adv-details, details.adv").forEach(d => {
    const key = d.getAttribute("data-adv") || "";
    d.addEventListener("toggle", () => setDetailsOpen(key, d.open));
  });

  // 語系切換時同步更新按鈕文案
  if (typeof onLocaleChange === "function") {
    onLocaleChange(() => {
      if (toggleBtn) toggleBtn.textContent = labelFor(getAdvancedMode());
    });
  }
}

export function initWarmupCard() {
  const warmupCard = document.getElementById("warmupCard");
  if (!warmupCard) return;
  let defaultOpen = true;
  try {
    const raw = localStorage.getItem(WARMUP_CARD_OPEN_KEY);
    if (raw === "1") defaultOpen = true;
    else if (raw === "0") defaultOpen = false;
  } catch { }
  warmupCard.open = defaultOpen;
  warmupCard.setAttribute("aria-expanded", warmupCard.open ? "true" : "false");
  warmupCard.addEventListener("toggle", () => {
    warmupCard.setAttribute("aria-expanded", warmupCard.open ? "true" : "false");
    try {
      localStorage.setItem(WARMUP_CARD_OPEN_KEY, warmupCard.open ? "1" : "0");
    } catch { }
  });
}

export function initUI() {
  initWarmupCard();
  setupAdvancedSection(document.body);
}
