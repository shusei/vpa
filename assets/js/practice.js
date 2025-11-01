import { t, getCurrentLocale, onLocaleChange } from "./i18n.js";
import { loadPracticeData } from "./practice-data.js";

const LS_SETTINGS = "vpa.practice.v1.settings";
const LS_HISTORY = "vpa.practice.v1.history";
const REF_VOICE_MODE_KEY = "practice:refVoiceMode";
const REF_VOICE_LOCALE_KEY = "practice:refVoiceLocale";
const DEFAULT_REF_LOCALE = "zh-Hant";

const VOICE_PACK_BASE = {
  "zh-Hant": {
    oneesan: "assets/audio/zh-Hant/oneesan/",
    loli: "assets/audio/zh-Hant/loli/",
  },
  "zh-Hans": {
    oneesan: "assets/audio/zh-Hans/oneesan/",
    loli: "assets/audio/zh-Hans/loli/",
  },
  en: {
    oneesan: "assets/audio/en/oneesan/",
    loli: "assets/audio/en/loli/",
  },
};

const SYSTEM_TTS_PRESETS = {
  oneesan: { rate: 0.95, pitch: 0.95 },
  loli: { rate: 1.08, pitch: 1.22 },
  system: { rate: 1.0, pitch: 1.0 },
};

const bridge = {
  subscribeInference: null,
  recorder: null,
};

const state = {
  data: { categories: [], phrases: [] },
  selectedCategory: null,
  shadowMode: true,
  autoAdvance: true,
  history: new Map(),
  activeId: null,
  unsub: null,
  runToken: null,
};

const ttsState = {
  bound: false,
  pending: null,
};

function ensureTtsListeners() {
  try {
    if (ttsState.bound) return;
    const synth = window?.speechSynthesis;
    if (!synth) return;
    const handler = () => {
      if (!ttsState.pending) return;
      if (state.runToken || getRecorder()?.isRecording) {
        ttsState.pending = null;
        return;
      }
      const pending = ttsState.pending;
      ttsState.pending = null;
      requestSystemTts(pending.text, pending.locale, pending.style, { allowRetry: false });
    };
    synth.addEventListener("voiceschanged", handler);
    ttsState.bound = true;
  } catch (err) {
    console.warn("[practice] bind voiceschanged failed", err);
  }
}

function qs(selector, root = document) {
  return root ? root.querySelector(selector) : null;
}

function qsa(selector, root = document) {
  return root ? Array.from(root.querySelectorAll(selector) || []) : [];
}

function createEl(tag, attrs = {}, content = null) {
  const el = document.createElement(tag);
  if (attrs && typeof attrs === "object") {
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null) continue;
      if (key === "class") {
        el.className = value;
      } else if (key === "dataset" && value && typeof value === "object") {
        for (const [dKey, dValue] of Object.entries(value)) {
          if (dValue == null) continue;
          el.dataset[dKey] = dValue;
        }
      } else if (key in el) {
        try {
          el[key] = value;
        } catch {
          el.setAttribute(key, value);
        }
      } else {
        el.setAttribute(key, value);
      }
    }
  }
  if (content != null) {
    if (Array.isArray(content)) {
      for (const child of content) {
        if (child == null) continue;
        el.append(child);
      }
    } else if (typeof content === "string") {
      el.innerHTML = content;
    } else {
      el.append(content);
    }
  }
  return el;
}

function setSelectValue(select, value) {
  if (!select) return null;
  const options = Array.from(select.options || []);
  if (!options.length) {
    select.value = value;
    return value;
  }
  const has = options.some((opt) => opt.value === value);
  const finalValue = has ? value : options[0].value;
  select.value = finalValue;
  return finalValue;
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(LS_SETTINGS);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      if (parsed.shadowMode != null) state.shadowMode = Boolean(parsed.shadowMode);
      if (parsed.autoAdvance != null) state.autoAdvance = Boolean(parsed.autoAdvance);
    }
  } catch (err) {
    console.warn("[practice] read settings failed", err);
  }
}

function saveSettings() {
  try {
    const payload = {
      shadowMode: state.shadowMode,
      autoAdvance: state.autoAdvance,
    };
    localStorage.setItem(LS_SETTINGS, JSON.stringify(payload));
  } catch (err) {
    console.warn("[practice] save settings failed", err);
  }
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(LS_HISTORY);
    if (!raw) {
      state.history = new Map();
      return;
    }
    const arr = JSON.parse(raw);
    if (Array.isArray(arr)) {
      state.history = new Map(arr);
    } else {
      state.history = new Map();
    }
  } catch (err) {
    console.warn("[practice] read history failed", err);
    state.history = new Map();
  }
}

function saveHistory() {
  try {
    const serialized = Array.from(state.history.entries());
    localStorage.setItem(LS_HISTORY, JSON.stringify(serialized));
  } catch (err) {
    console.warn("[practice] save history failed", err);
  }
}

function byCategory(data) {
  const map = new Map();
  for (const cat of data.categories || []) {
    map.set(cat.id, { ...cat, items: [] });
  }
  for (const phrase of data.phrases || []) {
    const target = map.get(phrase.cat) || map.get(phrase.category);
    if (target) {
      target.items.push(phrase);
    } else {
      map.set(phrase.cat, { id: phrase.cat, title: phrase.cat, items: [phrase] });
    }
  }
  return Array.from(map.values());
}

async function tryPlayPackAudio(phraseId, mode = "oneesan", locale = DEFAULT_REF_LOCALE) {
  try {
    const base = VOICE_PACK_BASE[locale]?.[mode];
    if (!base) return false;
    const url = `${base}${phraseId}.mp3`;
    const res = await fetch(url, { method: "HEAD", cache: "no-store" });
    if (!res.ok) return false;
    const audio = new Audio(url);
    await audio.play();
    return true;
  } catch (err) {
    return false;
  }
}

async function speakSystemTTS(text, locale = "zh-TW", style = "system") {
  if (!("speechSynthesis" in window)) return false;
  const synth = window.speechSynthesis;
  const voices = synth.getVoices();

  const prefer = (voice) => {
    const name = (voice.name || "").toLowerCase();
    const lang = (voice.lang || "").toLowerCase();
    return (
      (lang.includes(locale.toLowerCase()) ? 10 : 0)
      + (/(natural|neural|aria|sara|xiao|hsiao|han|female)/.test(name) ? 3 : 0)
      + (/female|女|小姐|sara/.test(name) ? 1 : 0)
    );
  };

  const sorted = [...voices].sort((a, b) => prefer(b) - prefer(a));
  const best = sorted[0];
  const utterance = new SpeechSynthesisUtterance(text);
  if (best) utterance.voice = best;

  const preset = SYSTEM_TTS_PRESETS[style] || SYSTEM_TTS_PRESETS.system;
  utterance.rate = preset.rate;
  utterance.pitch = preset.pitch;

  return new Promise((resolve) => {
    utterance.onend = () => resolve(true);
    utterance.onerror = () => resolve(false);
    try {
      synth.cancel();
      synth.speak(utterance);
    } catch (err) {
      console.warn("[practice] system TTS failed", err);
      resolve(false);
    }
  });
}

async function requestSystemTts(text, locale = "zh-TW", style = "system", { allowRetry = true } = {}) {
  try {
    const synth = window.speechSynthesis;
    if (!synth) return false;
    ensureTtsListeners();
    const ok = await speakSystemTTS(text, locale, style);
    if (!ok && allowRetry) {
      ttsState.pending = { text, locale, style };
    } else if (ok) {
      ttsState.pending = null;
    }
    return ok;
  } catch (err) {
    console.warn("[practice] request system TTS failed", err);
    return false;
  }
}

function resolvePreferredRefLocale(locale) {
  if (locale && VOICE_PACK_BASE[locale]) {
    return locale;
  }
  return DEFAULT_REF_LOCALE;
}

export async function playReferenceForPhrase(phrase) {
  if (!phrase || !phrase.text) return false;
  let mode = "system";
  try {
    const storedMode = localStorage.getItem(REF_VOICE_MODE_KEY);
    if (storedMode && ["system", "oneesan", "loli"].includes(storedMode)) {
      mode = storedMode;
    }
  } catch (err) {
    console.warn("[practice] read ref voice mode failed", err);
  }
  const uiLocale = getCurrentLocale();
  let storedLocale = null;
  try {
    storedLocale = localStorage.getItem(REF_VOICE_LOCALE_KEY);
  } catch (err) {
    console.warn("[practice] read ref voice locale failed", err);
  }
  const locale = storedLocale && VOICE_PACK_BASE[storedLocale]
    ? storedLocale
    : resolvePreferredRefLocale(uiLocale);

  if (mode === "oneesan" || mode === "loli") {
    const ok = await tryPlayPackAudio(phrase.id, mode, locale);
    if (ok) {
      return true;
    }
  }

  const ttsLocale = locale === "zh-Hant"
    ? "zh-TW"
    : locale === "zh-Hans"
      ? "zh-CN"
      : "en-US";
  return requestSystemTts(phrase.text, ttsLocale, mode);
}

function showCountdown(card, seconds = 3) {
  const el = card ? card.querySelector(".countdown") : null;
  if (!el) return Promise.resolve();
  if (el.dataset.timerId) {
    clearInterval(Number(el.dataset.timerId));
    delete el.dataset.timerId;
  }
  el.hidden = false;
  el.textContent = String(seconds);
  el.classList.remove("is-counting");
  // force reflow to restart animation when reused
  void el.offsetWidth;
  el.classList.add("is-counting");
  let remaining = seconds;
  return new Promise((resolve) => {
    const timer = window.setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(timer);
        delete el.dataset.timerId;
        el.hidden = true;
        el.classList.remove("is-counting");
        resolve();
        return;
      }
      el.textContent = String(remaining);
      el.classList.remove("is-counting");
      void el.offsetWidth;
      el.classList.add("is-counting");
    }, 1000);
    el.dataset.timerId = String(timer);
  });
}

function hydrateLastBadge(card, id) {
  if (!card) return;
  const badge = card.querySelector(".badge.last");
  const history = state.history.get(id);
  if (!badge) return;
  if (Array.isArray(history) && history.length) {
    const last = history[history.length - 1];
    const pct = Number.isFinite(last?.pf) ? Math.round(last.pf * 100) : 0;
    badge.textContent = `${t("practice.lastScore") || "上次"} ${pct}%`;
    badge.classList.remove("muted");
  } else {
    badge.textContent = t("practice.lastNone") || "尚無紀錄";
    badge.classList.add("muted");
  }
}

function writeResult(card, pf, pm) {
  if (!card) return;
  const femEl = card.querySelector(".practice-result .fem");
  const mascEl = card.querySelector(".practice-result .masc");
  const femVal = Number.isFinite(pf) ? pf : 0;
  const mascVal = Number.isFinite(pm) ? pm : 0;
  const femPct = `${(femVal * 100).toFixed(1)}%`;
  const mascPct = `${(mascVal * 100).toFixed(1)}%`;
  if (femEl) {
    femEl.textContent = femPct;
    femEl.setAttribute("aria-label", `${t("practice.feminine") || "Feminine"} ${femPct}`);
  }
  if (mascEl) {
    mascEl.textContent = mascPct;
    mascEl.setAttribute("aria-label", `${t("practice.masculine") || "Masculine"} ${mascPct}`);
  }
}

function persistHistory(id, pf, pm) {
  const list = state.history.get(id) || [];
  list.push({ ts: Date.now(), pf, pm });
  while (list.length > 20) list.shift();
  state.history.set(id, list);
  saveHistory();
}

function setBusyCard(card, busy) {
  if (!card) return;
  const recordBtn = card.querySelector('[data-act="record"]');
  const stopBtn = card.querySelector('[data-act="stop"]');
  if (recordBtn) recordBtn.disabled = Boolean(busy);
  if (stopBtn) stopBtn.disabled = !busy;
}

function focusNextCard(card) {
  const cards = Array.from(document.querySelectorAll(".practice-card"));
  if (!cards.length) return;
  const index = cards.indexOf(card);
  const next = cards[(index + 1) % cards.length];
  if (next) {
    next.scrollIntoView({ behavior: "smooth", block: "center" });
    const focusable = next.querySelector('[data-act="play"]');
    focusable?.focus();
  }
}

function subscribeInference(cb) {
  if (typeof bridge.subscribeInference === "function") {
    return bridge.subscribeInference(cb);
  }
  return () => {};
}

function getRecorder() {
  return bridge.recorder || null;
}

function cancelActiveRun(card) {
  state.unsub?.();
  state.unsub = null;
  state.activeId = null;
  state.runToken = null;
  setBusyCard(card, false);
  const countdown = card?.querySelector?.(".countdown");
  if (countdown) {
    if (countdown.dataset.timerId) {
      clearInterval(Number(countdown.dataset.timerId));
      delete countdown.dataset.timerId;
    }
    countdown.hidden = true;
    countdown.classList.remove("is-counting");
  }
}

function bindCardEvents(card, phrase) {
  card.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.getAttribute("data-act");
    if (!action) return;
    const recorder = getRecorder();
    if (!recorder) return;

    if (recorder.busy || (recorder.isRecording && action === "record")) {
      return;
    }

    if (action === "play") {
      await playReferenceForPhrase(phrase);
      return;
    }

    if (action === "record") {
      const runToken = Symbol("practiceRun");
      state.activeId = phrase.id;
      state.runToken = runToken;
      setBusyCard(card, true);

      if (state.shadowMode) {
        await showCountdown(card, 3);
        if (state.runToken !== runToken) {
          return;
        }
      }

      state.unsub?.();
      state.unsub = subscribeInference(({ pf, pm }) => {
        if (state.runToken !== runToken) {
          return;
        }
        cancelActiveRun(card);
        writeResult(card, pf, pm);
        persistHistory(phrase.id, pf, pm);
        hydrateLastBadge(card, phrase.id);
        if (state.autoAdvance) {
          focusNextCard(card);
        }
      });

      try {
        await Promise.resolve(recorder.start());
      } catch (err) {
        console.error("[practice] start recording failed", err);
        cancelActiveRun(card);
      }
      return;
    }

    if (action === "stop") {
      if (!recorder.isRecording) {
        cancelActiveRun(card);
        return;
      }
      try {
        await Promise.resolve(recorder.stop());
      } catch (err) {
        console.error("[practice] stop recording failed", err);
        cancelActiveRun(card);
      }
    }
  });
}

function renderCategoryNav(cats = byCategory(state.data)) {
  const nav = qs("#practiceNav");
  if (!nav) return;
  const usable = cats.filter((cat) => Array.isArray(cat.items) && cat.items.length);
  const label = t("practice.filterLabel") || "Browse by category";
  nav.setAttribute("aria-label", label);
  nav.setAttribute("role", "radiogroup");
  nav.innerHTML = "";

  if (!usable.length) {
    nav.hidden = true;
    return;
  }

  nav.hidden = usable.length <= 1;

  const activeId = state.selectedCategory || "";

  const makeButton = (title, catId) => {
    const button = createEl("button", {
      type: "button",
      class: "practice-nav__item",
      dataset: { cat: catId ?? "" },
      title,
    }, document.createTextNode(title));
    button.setAttribute("role", "radio");
    const id = catId ?? "";
    const isActive = id === activeId;
    button.setAttribute("aria-checked", String(isActive));
    if (isActive) {
      button.classList.add("is-active");
    }
    return button;
  };

  const allTitle = t("practice.allCategories") || "All";
  nav.appendChild(makeButton(allTitle, null));

  for (const cat of usable) {
    const title = cat.title || cat.name || cat.id || "";
    nav.appendChild(makeButton(title, cat.id));
  }
}

function renderList(cats = byCategory(state.data)) {
  const list = qs("#practiceList");
  if (!list) return;
  list.setAttribute("aria-busy", "true");
  list.innerHTML = "";
  const filtered = state.selectedCategory
    ? cats.filter((cat) => cat.id === state.selectedCategory)
    : cats;
  for (const cat of filtered) {
    if (!cat.items || !cat.items.length) continue;
    const heading = createEl("h4", { class: "practice-cat" }, document.createTextNode(cat.title || cat.id || ""));
    list.appendChild(heading);
    for (const phrase of cat.items) {
      const card = createCard(phrase);
      list.appendChild(card);
    }
  }
  list.removeAttribute("aria-busy");
}

function createCard(phrase) {
  const card = createEl("article", { class: "practice-card", dataset: { id: phrase.id } });
  const text = createEl("div", { class: "practice-text" }, document.createTextNode(phrase.text || ""));
  const tip = createEl("div", { class: "practice-tip" }, document.createTextNode(phrase.tip || ""));
  const controls = createEl("div", { class: "practice-controls" });
  const playBtn = createEl("button", { class: "btn sm", dataset: { act: "play" } }, document.createTextNode(t("practice.playRef") || "播放參考"));
  const recordBtn = createEl("button", { class: "btn sm primary", dataset: { act: "record" } }, document.createTextNode(t("practice.record") || "錄音"));
  const stopBtn = createEl("button", { class: "btn sm danger", dataset: { act: "stop" }, disabled: true }, document.createTextNode(t("practice.stop") || "停止"));
  controls.append(playBtn, recordBtn, stopBtn);

  const result = createEl("div", { class: "practice-result", "aria-live": "polite" });
  result.innerHTML = `
    <div><b class="fem" aria-label="${t("practice.feminine") || "Feminine"} --%">--%</b><span>${t("practice.feminine") || "Feminine"}</span></div>
    <div><b class="masc" aria-label="${t("practice.masculine") || "Masculine"} --%">--%</b><span>${t("practice.masculine") || "Masculine"}</span></div>
  `;

  const badges = createEl("div", { class: "practice-badges" });
  const lastBadge = createEl("span", { class: "badge muted last" }, document.createTextNode(t("practice.lastNone") || "尚無紀錄"));
  badges.appendChild(lastBadge);

  const countdown = createEl("div", { class: "countdown", hidden: true });

  card.append(text, tip, controls, result, badges, countdown);
  bindCardEvents(card, phrase);
  hydrateLastBadge(card, phrase.id);
  return card;
}

function focusRelative(current, delta) {
  const cards = Array.from(document.querySelectorAll(".practice-card"));
  if (!cards.length) return;
  let index = cards.indexOf(current);
  if (index === -1) index = 0;
  index = (index + delta + cards.length) % cards.length;
  const target = cards[index];
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    const play = target.querySelector('[data-act="play"]');
    play?.focus();
  }
}

async function refreshData(locale) {
  state.data = await loadPracticeData(locale);
  const cats = byCategory(state.data);
  if (state.selectedCategory && !cats.some((cat) => cat.id === state.selectedCategory && cat.items && cat.items.length)) {
    state.selectedCategory = null;
  }
  renderCategoryNav(cats);
  renderList(cats);
}

export async function setupPracticeUI({ subscribeInference, recorder } = {}) {
  bridge.subscribeInference = subscribeInference || bridge.subscribeInference;
  bridge.recorder = recorder || bridge.recorder;

  const toggle = qs("#practiceToggle");
  const panel = qs("#practicePanel");
  const list = qs("#practiceList");
  const nav = qs("#practiceNav");
  const shadow = qs("#practiceShadowMode");
  const advance = qs("#practiceAutoAdvance");
  const randomBtn = qs("#practiceRandomBtn");
  const refMode = qs("#refVoiceMode");
  const refLocale = qs("#refVoiceLocale");

  if (!toggle || !panel || !list || !nav || !shadow || !advance || !randomBtn) {
    return;
  }

  loadSettings();
  loadHistory();

  await refreshData(getCurrentLocale());

  if (refMode) {
    let storedMode = "system";
    try {
      const read = localStorage.getItem(REF_VOICE_MODE_KEY);
      if (read) storedMode = read;
    } catch (err) {
      console.warn("[practice] read ref voice mode failed", err);
    }
    const appliedMode = setSelectValue(refMode, storedMode);
    if (!appliedMode) {
      setSelectValue(refMode, "system");
    }
    refMode.addEventListener("change", () => {
      try {
        localStorage.setItem(REF_VOICE_MODE_KEY, refMode.value || "system");
      } catch (err) {
        console.warn("[practice] save ref voice mode failed", err);
      }
    });
  }

  let hasExplicitRefLocale = false;
  if (refLocale) {
    let storedLocale = null;
    try {
      storedLocale = localStorage.getItem(REF_VOICE_LOCALE_KEY);
    } catch (err) {
      console.warn("[practice] read ref voice locale failed", err);
    }
    const defaultLocale = resolvePreferredRefLocale(getCurrentLocale());
    const initialLocale = storedLocale && VOICE_PACK_BASE[storedLocale]
      ? storedLocale
      : defaultLocale;
    setSelectValue(refLocale, initialLocale);
    hasExplicitRefLocale = Boolean(storedLocale && VOICE_PACK_BASE[storedLocale]);
    refLocale.addEventListener("change", () => {
      const next = refLocale.value;
      hasExplicitRefLocale = Boolean(next);
      try {
        if (next) {
          localStorage.setItem(REF_VOICE_LOCALE_KEY, next);
        } else {
          localStorage.removeItem(REF_VOICE_LOCALE_KEY);
        }
      } catch (err) {
        console.warn("[practice] save ref voice locale failed", err);
      }
    });
  }

  nav.addEventListener("click", (event) => {
    const target = event.target instanceof HTMLElement
      ? event.target.closest("button[data-cat]")
      : null;
    if (!target) return;
    const catId = target.dataset.cat || "";
    const normalized = catId ? catId : null;
    if (state.selectedCategory === normalized) return;
    state.selectedCategory = normalized;
    const cats = byCategory(state.data);
    renderCategoryNav(cats);
    renderList(cats);
    const safeCat = typeof CSS !== "undefined" && typeof CSS.escape === "function"
      ? CSS.escape(catId)
      : catId.replace(/["\\]/g, "\\$&");
    const selector = `[data-cat="${safeCat}"]`;
    requestAnimationFrame(() => {
      const active = nav.querySelector(selector);
      if (active instanceof HTMLElement) {
        active.focus();
      }
    });
  });

  nav.addEventListener("keydown", (event) => {
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;
    const buttons = Array.from(nav.querySelectorAll('[role="radio"]'));
    if (!buttons.length) return;
    event.preventDefault();
    const focused = document.activeElement instanceof HTMLElement
      ? document.activeElement.closest('[role="radio"]')
      : null;
    let index = buttons.indexOf(focused);
    if (index === -1) {
      index = buttons.findIndex((btn) => btn.classList.contains("is-active"));
    }
    if (index === -1) index = 0;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      index = (index + 1) % buttons.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      index = (index - 1 + buttons.length) % buttons.length;
    } else if (event.key === "Home") {
      index = 0;
    } else if (event.key === "End") {
      index = buttons.length - 1;
    }
    const next = buttons[index];
    if (next instanceof HTMLElement) {
      next.focus();
      next.click();
    }
  });

  toggle.addEventListener("click", () => {
    const isHidden = panel.hasAttribute("hidden");
    if (isHidden) {
      panel.removeAttribute("hidden");
    } else {
      panel.setAttribute("hidden", "");
    }
    toggle.setAttribute("aria-expanded", String(isHidden));
  });

  shadow.checked = state.shadowMode;
  advance.checked = state.autoAdvance;

  shadow.addEventListener("change", () => {
    state.shadowMode = Boolean(shadow.checked);
    saveSettings();
  });

  advance.addEventListener("change", () => {
    state.autoAdvance = Boolean(advance.checked);
    saveSettings();
  });

  randomBtn.addEventListener("click", () => {
    const cards = Array.from(document.querySelectorAll(".practice-card"));
    if (!cards.length) return;
    const index = Math.floor(Math.random() * cards.length);
    const target = cards[index];
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.querySelector('[data-act="play"]')?.focus();
  });

  onLocaleChange(async (locale) => {
    await refreshData(locale);
    if (refLocale && !hasExplicitRefLocale) {
      const nextLocale = resolvePreferredRefLocale(locale);
      setSelectValue(refLocale, nextLocale);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (panel.hasAttribute("hidden")) return;
    const active = document.activeElement instanceof HTMLElement
      ? document.activeElement.closest(".practice-card")
      : null;
    if (event.code === "Space") {
      const recordButton = (active?.querySelector('[data-act="record"]')
        || document.querySelector(".practice-card [data-act=\"record\"]"));
      if (recordButton) {
        event.preventDefault();
        recordButton.click();
      }
    } else if (event.key === "j" || event.key === "J") {
      event.preventDefault();
      focusRelative(active, 1);
    } else if (event.key === "k" || event.key === "K") {
      event.preventDefault();
      focusRelative(active, -1);
    }
  });
}
