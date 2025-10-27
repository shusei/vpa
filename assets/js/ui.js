import {
  statusEl,
  pitchWrap,
  formantWrap,
  pitchNowEl,
  bandNowEl,
  volNowEl,
  f1NowEl,
  f2NowEl,
  f3NowEl,
  breathNowEl,
  resonanceNowEl,
  tiltNowEl,
  resValChest,
  resValMask,
  resValHead,
  resBarChest,
  resBarMask,
  resBarHead,
  meter,
  femaleVal,
  maleVal,
} from "./dom.js";
import { EPS } from "./constants.js";

export function setStatus(text, spin = false) {
  if (!statusEl) return;
  statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}

export function log(...args) {
  try {
    console.log(...args);
  } catch {
    // ignore logging failures (Safari private mode, etc.)
  }
}

export function fmtSec(seconds) {
  if (!isFinite(seconds)) return "—";
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return minutes ? `${minutes}分${secs}秒` : `${secs}秒`;
}

export function clamp01(value) {
  return Math.min(1, Math.max(EPS, value));
}

export function resetRealtimePanels() {
  try {
    if (pitchNowEl) pitchNowEl.textContent = "— Hz";
    if (bandNowEl) bandNowEl.textContent = "—";
    if (volNowEl) volNowEl.textContent = "— dB";
    if (f1NowEl) f1NowEl.textContent = "— Hz";
    if (f2NowEl) f2NowEl.textContent = "— Hz";
    if (f3NowEl) f3NowEl.textContent = "— Hz";
    if (breathNowEl) breathNowEl.textContent = "—";
    if (resonanceNowEl) resonanceNowEl.textContent = "—";
    if (tiltNowEl) tiltNowEl.textContent = "Tilt —";
    if (resValChest) resValChest.textContent = "胸 0%";
    if (resValMask) resValMask.textContent = "面罩 0%";
    if (resValHead) resValHead.textContent = "頭 0%";
    if (resBarChest) {
      resBarChest.style.flexGrow = 1;
      resBarChest.style.flexBasis = "33%";
    }
    if (resBarMask) {
      resBarMask.style.flexGrow = 1;
      resBarMask.style.flexBasis = "34%";
    }
    if (resBarHead) {
      resBarHead.style.flexGrow = 1;
      resBarHead.style.flexBasis = "33%";
    }
  } catch {
    // ignore DOM update errors
  }
}

export function setRealtimePanelsActive(active) {
  try {
    if (pitchWrap) {
      if (active) pitchWrap.removeAttribute("hidden");
      else pitchWrap.setAttribute("hidden", "");
    }
    if (formantWrap) {
      if (active) formantWrap.removeAttribute("hidden");
      else formantWrap.setAttribute("hidden", "");
    }
    resetRealtimePanels();
  } catch {
    // ignore DOM update errors
  }
}

export function resetMeter() {
  try {
    meter?.classList.remove("hidden");
    const barF = document.querySelector(".bar.female");
    const barM = document.querySelector(".bar.male");
    if (barF) {
      barF.style.setProperty("--p", 0);
      barF.setAttribute("aria-valuenow", "0");
    }
    if (barM) {
      barM.style.setProperty("--p", 0);
      barM.setAttribute("aria-valuenow", "0");
    }
    if (femaleVal) femaleVal.textContent = "0.0%";
    if (maleVal) maleVal.textContent = "0.0%";
  } catch {
    // ignore DOM update errors
  }
}

export function isOOMError(err) {
  const message = String(err?.message || err || "");
  return /OrtRun|bad_alloc|out of memory|memory|alloc/i.test(message);
}
