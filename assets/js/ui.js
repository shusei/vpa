import { t } from "./i18n.js?v=1.4.26";
import {
  statusEl,
  statusLabel,
  statusTimer,
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

const doc = typeof document !== "undefined" ? document : null;
const spinnerTemplate = doc ? doc.createElement("span") : null;
if (spinnerTemplate) {
  spinnerTemplate.className = "spinner";
}

export const STATUS_TIMER_RESET = "00:00";

export function setStatus(text, spin = false) {
  if (!statusEl) return;
  const message = text == null ? "" : String(text);
  const labelTarget = statusLabel || statusEl;
  if (spin) {
    const nodes = [];
    if (spinnerTemplate) {
      nodes.push(spinnerTemplate.cloneNode(false));
    }
    if (message) {
      nodes.push(
        doc ? doc.createTextNode(nodes.length ? ` ${message}` : message) : message
      );
    }
    if (labelTarget && typeof labelTarget.replaceChildren === "function") {
      if (nodes.length) {
        labelTarget.replaceChildren(...nodes);
      } else {
        labelTarget.replaceChildren();
      }
    } else if (labelTarget) {
      labelTarget.textContent = message;
    }
    statusEl.dataset.busy = "true";
    statusEl.setAttribute("aria-busy", "true");
  } else {
    if (labelTarget && typeof labelTarget.replaceChildren === "function") {
      const textNode = doc ? doc.createTextNode(message) : message;
      labelTarget.replaceChildren(textNode);
    } else if (labelTarget) {
      labelTarget.textContent = message;
    }
    delete statusEl.dataset.busy;
    statusEl.removeAttribute("aria-busy");
  }
  statusEl.classList.toggle("is-busy", spin);
}

export function setStatusTimer(value) {
  if (!statusTimer) return;
  const text = value == null ? STATUS_TIMER_RESET : String(value);
  statusTimer.textContent = text;
}

export function toggleStatusTimer(active) {
  if (statusEl) {
    statusEl.classList.toggle("show-timer", !!active);
  }
  if (!active) {
    setStatusTimer(STATUS_TIMER_RESET);
  }
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
  if (minutes) return t("time.minutesSeconds", { minutes, seconds: secs });
  return t("time.seconds", { seconds: secs });
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
    if (tiltNowEl) tiltNowEl.textContent = t("realtime.resonance.tiltPlaceholder");
    if (resValChest) resValChest.textContent = t("analysis.resonanceBar.default.chest");
    if (resValMask) resValMask.textContent = t("analysis.resonanceBar.default.mask");
    if (resValHead) resValHead.textContent = t("analysis.resonanceBar.default.head");
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
