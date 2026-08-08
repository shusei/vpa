import {
  helpBtn,
  helpOverlay,
  localeBtn,
  localeMenu,
  onboardTip,
  onboardDismissBtn,
  recordBtn,
  uploadFab,
} from "./dom.js";
import { setLocale, getCurrentLocale, onLocaleChange } from "./i18n.js?v=1.4.12";

const MODE_KEY = "vpa::theme.mode";          // 'auto' | 'light' | 'dark'
const LAST_LIGHT_KEY = "vpa::theme.light";  // ex: 'day'
const LAST_DARK_KEY = "vpa::theme.dark";   // ex: 'night'
const LEGACY_KEY = "vpa.theme";        // 舊版單一主題鍵（會自動遷移）
const TIP_KEY = "vpa.themeTipDone"; // 一次性齒輪提示
const HELP_KEY = "vpa.helpSeen";
const ONBOARD_KEY = "vpa.onboardTipDone";

const THEMES = [
  "warm", "lavender", "peach", "ink", "day", "night", "contrast", "slate", "graphite", "sand", "latte", "clay",
  "rose", "blush", "coral", "amber", "gold", "luxegold", "pony2026", "cocoa", "olive", "emerald", "teal", "aqua", "cyan", "sky", "azure",
  "cobalt", "indigo", "violet", "grape", "plum", "magenta", "fuchsia"
];

const THEME_FACTION = {
  day: "light", sand: "light", latte: "light", blush: "light", amber: "light",
  aqua: "light", sky: "light", azure: "light", fuchsia: "light", pony2026: "light",

  warm: "dark", lavender: "dark", peach: "dark", ink: "dark", night: "dark", contrast: "dark",
  slate: "dark", graphite: "dark", clay: "dark", rose: "dark", coral: "dark", gold: "dark", cocoa: "dark", luxegold: "dark",
  olive: "dark", emerald: "dark", teal: "dark", cyan: "dark", cobalt: "dark", indigo: "dark",
  violet: "dark", grape: "dark", plum: "dark", magenta: "dark",
};

function getSystemFaction() {
  try { return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; }
  catch { return "light"; }
}
function getSavedMode() { try { return (localStorage.getItem(MODE_KEY) || "auto"); } catch { return "auto"; } }
function setSavedMode(m) { try { localStorage.setItem(MODE_KEY, m); } catch { } }
function getSavedLast(faction) {
  const key = faction === "light" ? LAST_LIGHT_KEY : LAST_DARK_KEY;
  let v = null; try { v = localStorage.getItem(key); } catch { }
  if (!v || !THEMES.includes(v) || THEME_FACTION[v] !== faction) v = faction === "light" ? "day" : "night";
  return v;
}
function setSavedLast(theme) {
  const faction = THEME_FACTION[theme] || "dark";
  const key = faction === "light" ? LAST_LIGHT_KEY : LAST_DARK_KEY;
  try { localStorage.setItem(key, theme); } catch { }
}
function migrateLegacyTheme() {
  try {
    const old = localStorage.getItem(LEGACY_KEY);
    if (!old) return;
    setSavedLast(old);
    localStorage.removeItem(LEGACY_KEY);
  } catch { }
}

export function applyMode(mode) {
  const m = (mode === "light" || mode === "dark" || mode === "auto") ? mode : "auto";
  setSavedMode(m);
  document.documentElement.setAttribute("data-mode", m);

  const faction = (m === "auto") ? getSystemFaction() : m;
  document.documentElement.setAttribute("data-faction", faction);

  const now = document.documentElement.getAttribute("data-theme");
  if (!now || THEME_FACTION[now] !== faction) {
    applyTheme(getSavedLast(faction), false);
  }
  refreshThemeChecks();
  refreshTabUI();
}
export function applyTheme(t, persist = true) {
  if (!THEMES.includes(t)) t = getSavedLast(getSystemFaction());
  const faction = THEME_FACTION[t] || getSystemFaction();

  // Sync mode if needed
  if (persist) {
    const currentMode = getSavedMode();
    // If explicit mode mismatch, force update
    if (currentMode === "light" && faction === "dark") applyMode("dark");
    else if (currentMode === "dark" && faction === "light") applyMode("light");
    // If auto, and we picked a theme opposite to system, we must force manual mode
    else if (currentMode === "auto" && faction !== getSystemFaction()) applyMode(faction);
  }

  document.documentElement.setAttribute("data-theme", t);
  document.documentElement.setAttribute("data-faction", faction);
  if (persist) setSavedLast(t);
  refreshThemeChecks();
}

function refreshThemeChecks() {
  document.querySelectorAll(".theme-item").forEach(btn => {
    const checked = btn.dataset.theme === document.documentElement.getAttribute("data-theme");
    btn.setAttribute("aria-checked", checked ? "true" : "false");
  });
}
function refreshTabUI() {
  const mode = document.documentElement.getAttribute("data-mode") || "auto";
  const map = { auto: "#tabAuto", light: "#tabLight", dark: "#tabDark" };
  for (const [k, sel] of Object.entries(map)) {
    const el = document.querySelector(sel); if (!el) continue;
    el.setAttribute("aria-selected", k === mode ? "true" : "false");
    el.classList.toggle("active", k === mode);
  }
}
function initThemeUI() {
  migrateLegacyTheme();
  applyMode(getSavedMode());

  const gear = document.getElementById("settingsBtn");
  const menu = document.getElementById("themeMenu");
  const tip = document.getElementById("themeTip");

  if (gear && menu) {
    gear.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = !menu.hasAttribute("hidden");
      if (open) { menu.setAttribute("hidden", ""); gear.setAttribute("aria-expanded", "false"); }
      else {
        menu.removeAttribute("hidden"); gear.setAttribute("aria-expanded", "true");
        if (tip && !tip.hasAttribute("hidden")) {
          tip.setAttribute("hidden", ""); try { localStorage.setItem(TIP_KEY, "1"); } catch { }
        }
      }
    });
    document.addEventListener("click", (e) => {
      if (!menu.contains(e.target) && e.target !== gear && !gear.contains(e.target)) {
        if (!menu.hasAttribute("hidden")) { menu.setAttribute("hidden", ""); gear.setAttribute("aria-expanded", "false"); }
      }
      if (tip && !tip.hasAttribute("hidden")) {
        tip.setAttribute("hidden", ""); try { localStorage.setItem(TIP_KEY, "1"); } catch { }
      }
    });
    menu.querySelectorAll(".theme-item").forEach(btn => {
      btn.addEventListener("click", () => {
        applyTheme(btn.dataset.theme, true);
        menu.setAttribute("hidden", ""); gear.setAttribute("aria-expanded", "false");
      });
    });
    [["auto", "#tabAuto"], ["light", "#tabLight"], ["dark", "#tabDark"]].forEach(([name, sel]) => {
      const el = document.querySelector(sel); if (!el) return;
      el.addEventListener("click", () => applyMode(name));
    });

    try {
      const mq = matchMedia("(prefers-color-scheme: dark)");
      mq.addEventListener?.("change", () => {
        if ((localStorage.getItem(MODE_KEY) || "auto") === "auto") applyMode("auto");
      });
    } catch { }
  }

  try {
    const done = localStorage.getItem(TIP_KEY) === "1";
    if (!done && tip) {
      tip.removeAttribute("hidden");
      setTimeout(() => { try { tip.setAttribute("hidden", ""); localStorage.setItem(TIP_KEY, "1"); } catch { } }, 4000);
    }
  } catch { }
}

export function dismissOnboardTip(permanent = false) {
  try {
    if (onboardTip && !onboardTip.hasAttribute("hidden")) {
      onboardTip.setAttribute("hidden", "");
    }
    if (permanent) { localStorage.setItem(ONBOARD_KEY, "1"); }
  } catch {
    if (onboardTip) { onboardTip.setAttribute("hidden", ""); }
  }
}

function initOnboardingTip() {
  if (!onboardTip) return;
  let done = false;
  try { done = localStorage.getItem(ONBOARD_KEY) === "1"; } catch { }
  if (!done) {
    setTimeout(() => {
      if (!onboardTip) return;
      onboardTip.removeAttribute("hidden");
    }, 600);
  }
  onboardDismissBtn?.addEventListener("click", () => {
    dismissOnboardTip(true);
  });
  recordBtn?.addEventListener("click", () => dismissOnboardTip(true));
  uploadFab?.addEventListener("click", () => dismissOnboardTip(true));
}

function initHelpOverlay() {
  if (!helpBtn || !helpOverlay) return;
  let returnFocus = helpBtn;
  const focusClose = () => {
    const closeBtn = helpOverlay.querySelector(".help-close");
    closeBtn?.focus?.();
  };
  const closeButtons = new WeakSet();
  const topButtons = new WeakSet();
  const close = () => {
    helpOverlay.setAttribute("hidden", "");
    helpBtn.setAttribute("aria-expanded", "false");
    document.body.classList.remove("help-open");
    returnFocus?.focus?.();
  };
  const scrollTop = () => {
    const dialog = helpOverlay.querySelector(".help-dialog");
    if (!dialog) return;
    if (typeof dialog.scrollTo === "function") {
      dialog.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      dialog.scrollTop = 0;
    }
  };
  const scrollAuthorIntoView = () => {
    const author = helpOverlay.querySelector(".help-author");
    if (!author) return;
    author.scrollIntoView({ block: "center" });
  };
  const ensureCloseButtons = () => {
    helpOverlay.querySelectorAll(".help-close").forEach((btn) => {
      if (closeButtons.has(btn)) return;
      btn.addEventListener("click", (event) => {
        event.preventDefault();
        close();
      });
      closeButtons.add(btn);
    });
  };
  const ensureTopButtons = () => {
    helpOverlay.querySelectorAll(".help-top").forEach((btn) => {
      if (topButtons.has(btn)) return;
      btn.addEventListener("click", (event) => {
        event.preventDefault();
        scrollTop();
      });
      topButtons.add(btn);
    });
  };
  const open = (trigger = helpBtn, target = null) => {
    returnFocus = trigger;
    try { localStorage.setItem(HELP_KEY, "1"); } catch { }
    dismissOnboardTip(true);
    ensureCloseButtons();
    ensureTopButtons();
    helpOverlay.removeAttribute("hidden");
    helpBtn.setAttribute("aria-expanded", "true");
    document.body.classList.add("help-open");
    requestAnimationFrame(() => {
      focusClose();
      if (target === "author") requestAnimationFrame(scrollAuthorIntoView);
    });
  };

  helpBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = !helpOverlay.hasAttribute("hidden");
    if (isOpen) close(); else open(helpBtn);
  });
  document.addEventListener("click", (event) => {
    const trigger = event.target.closest?.("[data-author-shortcut]");
    if (!trigger) return;
    event.preventDefault();
    event.stopPropagation();
    open(trigger, "author");
  });
  helpOverlay.addEventListener("click", (e) => {
    if (e.target === helpOverlay) {
      close();
      return;
    }
    const closeTrigger = e.target.closest?.(".help-close");
    if (closeTrigger) {
      e.preventDefault();
      close();
    }
    const topTrigger = e.target.closest?.(".help-top");
    if (topTrigger) {
      e.preventDefault();
      scrollTop();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !helpOverlay.hasAttribute("hidden")) {
      close();
    }
  });
  const dialog = helpOverlay.querySelector(".help-dialog");
  if (dialog) {
    const observer = new MutationObserver(() => {
      ensureCloseButtons();
      ensureTopButtons();
    });
    observer.observe(dialog, { childList: true });
  }
  onLocaleChange(() => {
    ensureCloseButtons();
    ensureTopButtons();
  });
  ensureCloseButtons();
  ensureTopButtons();
}

function initLocaleMenu() {
  if (!localeBtn || !localeMenu) return;
  const items = () => Array.from(localeMenu.querySelectorAll("[data-locale]"));
  const setActive = (locale) => {
    items().forEach((item) => {
      const isActive = item.dataset.locale === locale;
      item.setAttribute("aria-checked", isActive ? "true" : "false");
      item.classList.toggle("active", isActive);
    });
  };
  const open = () => {
    localeMenu.removeAttribute("hidden");
    localeBtn.setAttribute("aria-expanded", "true");
    const current = localeMenu.querySelector(`[data-locale="${getCurrentLocale()}"]`);
    requestAnimationFrame(() => current?.focus?.());
  };
  const close = () => {
    if (localeMenu.hasAttribute("hidden")) return;
    localeMenu.setAttribute("hidden", "");
    localeBtn.setAttribute("aria-expanded", "false");
  };

  localeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = !localeMenu.hasAttribute("hidden");
    if (isOpen) close(); else open();
  });
  document.addEventListener("click", (e) => {
    if (!localeMenu.contains(e.target) && e.target !== localeBtn && !localeBtn.contains(e.target)) {
      close();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !localeMenu.hasAttribute("hidden")) {
      close();
      localeBtn.focus?.();
    }
  });
  localeMenu.addEventListener("click", async (e) => {
    const item = e.target.closest?.("[data-locale]");
    if (!item) return;
    e.preventDefault();
    const target = item.dataset.locale;
    if (!target) return;
    setActive(target);
    await setLocale(target);
    close();
    localeBtn.focus?.();
  });

  setActive(getCurrentLocale());
  onLocaleChange((locale) => setActive(locale));
}

function initializeTheme() {
  initThemeUI();
  initLocaleMenu();
  initHelpOverlay();
  initOnboardingTip();
}

initializeTheme();
