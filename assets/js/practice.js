import { t, getCurrentLocale, onLocaleChange } from "./i18n.js";
import { loadPracticeData } from "./practice-data.js";

const LS_HISTORY = "vpa.practice.v1.history";
const bridge = {
  subscribeInference: null,
  recorder: null,
};

const state = {
  data: { categories: [], phrases: [] },
  selectedCategory: null,
  history: new Map(),
  activeId: null,
  unsub: null,
  runToken: null,
  lastPlayableId: null,
  playingId: null,
  player: null,
  pendingCategory: null,
};

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

function setCardRecording(card, recording) {
  if (!card) return;
  const button = card.querySelector('[data-act="toggle"]');
  if (!button) return;
  const startLabel = t("practice.recordStart") || t("practice.record") || "開始錄音";
  const stopLabel = t("practice.recordStop") || t("practice.stop") || "停止錄音";
  button.textContent = recording ? stopLabel : startLabel;
  button.setAttribute("aria-pressed", recording ? "true" : "false");
  button.classList.toggle("danger", recording);
  button.classList.toggle("primary", !recording);
}

function setCardBusy(card, busy) {
  if (!card) return;
  const button = card.querySelector('[data-act="toggle"]');
  if (button) {
    button.disabled = Boolean(busy);
    button.classList.toggle("is-busy", Boolean(busy));
  }
  const playBtn = card.querySelector('[data-act="play"]');
  if (playBtn) {
    if (busy) {
      playBtn.disabled = true;
      playBtn.setAttribute("aria-disabled", "true");
    } else {
      const recorder = getRecorder();
      const canPlay = Boolean(recorder?.hasLastRecording) && state.lastPlayableId === card.dataset.id;
      setCardPlayable(card, canPlay);
    }
  }
}

function setCardPlayable(card, playable) {
  if (!card) return;
  const playBtn = card.querySelector('[data-act="play"]');
  if (!playBtn) return;
  playBtn.disabled = !playable;
  playBtn.setAttribute("aria-disabled", playable ? "false" : "true");
}

function setCardPlaying(card, playing) {
  if (!card) return;
  const playBtn = card.querySelector('[data-act="play"]');
  if (!playBtn) return;
  const playLabel = t("practice.playLast") || "播放上一段";
  const stopLabel = t("practice.playStop") || "停止播放";
  playBtn.textContent = playing ? stopLabel : playLabel;
  playBtn.setAttribute("aria-pressed", playing ? "true" : "false");
}

function stripCardDismissals(root) {
  if (!root) return;
  const extras = root.querySelectorAll('[data-act="dismiss"], .practice-card__dismiss, .practice-dismiss');
  if (!extras.length) return;
  for (const control of extras) {
    const card = control.closest(".practice-card");
    control.remove();
    if (card) {
      card.removeAttribute("data-dismissable");
      card.classList.remove("has-dismiss", "practice-card--dismissable", "practice-card_has-dismiss");
    }
  }
}

function getCardById(id) {
  if (!id) return null;
  const safe = typeof CSS !== "undefined" && typeof CSS.escape === "function"
    ? CSS.escape(id)
    : String(id).replace(/["\\]/g, "\\$&");
  return qs(`.practice-card[data-id="${safe}"]`);
}

function resetPlayingState() {
  if (!state.playingId) return;
  const active = getCardById(state.playingId);
  setCardPlaying(active, false);
  state.playingId = null;
}

function ensurePlayer() {
  if (state.player) {
    return state.player;
  }
  if (typeof Audio !== "function") {
    return null;
  }
  const audio = new Audio();
  audio.preload = "auto";
  audio.addEventListener("ended", () => {
    resetPlayingState();
  });
  audio.addEventListener("pause", () => {
    if (audio.ended) return;
    resetPlayingState();
  });
  audio.addEventListener("error", () => {
    resetPlayingState();
  });
  state.player = audio;
  return audio;
}

function stopPracticePlayback() {
  if (!state.player) return;
  try {
    state.player.pause();
    state.player.currentTime = 0;
  } catch (err) {
    console.warn("[practice] stop playback failed", err);
  }
  resetPlayingState();
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
    const femLabel = t("practice.feminineLabel") || t("practice.feminine") || "女性傾向";
    femEl.textContent = `${femLabel} ${femPct}`;
    femEl.setAttribute("aria-label", `${femLabel} ${femPct}`);
  }
  if (mascEl) {
    const mascLabel = t("practice.masculineLabel") || t("practice.masculine") || "男性傾向";
    mascEl.textContent = `${mascLabel} ${mascPct}`;
    mascEl.setAttribute("aria-label", `${mascLabel} ${mascPct}`);
  }
}

function persistHistory(id, pf, pm) {
  const list = state.history.get(id) || [];
  list.push({ ts: Date.now(), pf, pm });
  while (list.length > 20) list.shift();
  state.history.set(id, list);
  saveHistory();
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
  setCardRecording(card, false);
  setCardBusy(card, false);
}

function bindCardEvents(card, phrase) {
  card.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const action = target.getAttribute("data-act");
    if (!action) return;
    const recorder = getRecorder();
    if (!recorder) return;

    if (recorder.busy && action !== "toggle") {
      return;
    }

    if (action === "toggle") {
      const isActiveCard = state.activeId === phrase.id && recorder.isRecording;
      if (isActiveCard) {
        setCardBusy(card, true);
        try {
          await Promise.resolve(recorder.stop());
        } catch (err) {
          console.error("[practice] stop recording failed", err);
          cancelActiveRun(card);
        }
        return;
      }
      if (recorder.isRecording) {
        return;
      }

      const previousPlayableId = state.lastPlayableId;
      const runToken = Symbol("practiceRun");
      state.activeId = phrase.id;
      state.runToken = runToken;
      setCardBusy(card, true);
      if (state.lastPlayableId && state.lastPlayableId !== phrase.id) {
        const previous = getCardById(state.lastPlayableId);
        setCardPlayable(previous, false);
        setCardPlaying(previous, false);
      }
      state.lastPlayableId = null;
      stopPracticePlayback();

      state.unsub?.();
      state.unsub = subscribeInference(({ pf, pm }) => {
        if (state.runToken !== runToken) {
          return;
        }
        cancelActiveRun(card);
        writeResult(card, pf, pm);
        persistHistory(phrase.id, pf, pm);
        hydrateLastBadge(card, phrase.id);
        const recorderCtl = getRecorder();
        const prevPlayable = state.lastPlayableId;
        if (recorderCtl?.hasLastRecording) {
          state.lastPlayableId = phrase.id;
          setCardPlayable(card, true);
          setCardPlaying(card, false);
          if (prevPlayable && prevPlayable !== phrase.id) {
            const previous = getCardById(prevPlayable);
            setCardPlayable(previous, false);
            setCardPlaying(previous, false);
          }
        } else {
          state.lastPlayableId = null;
        }
      });

      try {
        await Promise.resolve(recorder.start());
        setCardRecording(card, true);
        setCardBusy(card, false);
      } catch (err) {
        console.error("[practice] start recording failed", err);
        cancelActiveRun(card);
        if (previousPlayableId) {
          state.lastPlayableId = previousPlayableId;
          const prevCard = getCardById(previousPlayableId);
          if (prevCard) {
            setCardPlayable(prevCard, Boolean(getRecorder()?.hasLastRecording));
          }
        }
      }
      return;
    }
    if (action === "play") {
      const recorderCtl = getRecorder();
      if (!recorderCtl?.hasLastRecording) {
        return;
      }
      const player = ensurePlayer();
      if (!player) {
        return;
      }
      const currentId = state.playingId;
      if (currentId === phrase.id) {
        stopPracticePlayback();
        return;
      }
      stopPracticePlayback();
      const url = typeof recorderCtl.getLastRecordingUrl === "function"
        ? recorderCtl.getLastRecordingUrl()
        : null;
      if (!url) {
        return;
      }
      try {
        player.src = url;
        state.playingId = phrase.id;
        setCardPlaying(card, true);
        const maybePromise = player.play();
        if (maybePromise && typeof maybePromise.then === "function") {
          await maybePromise;
        }
      } catch (err) {
        console.error("[practice] play failed", err);
        resetPlayingState();
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
  stripCardDismissals(list);
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
  stripCardDismissals(list);
  list.removeAttribute("aria-busy");
}

function selectPracticeCategory(catId, { focusNav = false } = {}) {
  const nav = qs("#practiceNav");
  const normalized = catId ? String(catId) : null;
  const cats = byCategory(state.data);
  const hasTarget = !normalized || cats.some((cat) => cat.id === normalized && Array.isArray(cat.items) && cat.items.length);
  const nextCat = hasTarget ? normalized : null;
  state.selectedCategory = nextCat;
  renderCategoryNav(cats);
  renderList(cats);
  if (focusNav && nav) {
    const safe = normalized && typeof CSS !== "undefined" && typeof CSS.escape === "function"
      ? CSS.escape(normalized)
      : normalized
        ? normalized.replace(/["\\]/g, "\\$&")
        : "";
    const selector = normalized ? `[data-cat="${safe}"]` : '[data-cat=""]';
    requestAnimationFrame(() => {
      const btn = nav.querySelector(selector);
      if (btn instanceof HTMLElement) {
        btn.focus();
      }
    });
  }
}

export function openPracticeCategory(catId) {
  const panel = qs("#practicePanel");
  const toggle = qs("#practiceToggle");
  const nav = qs("#practiceNav");
  const normalized = catId ? String(catId) : null;
  if (!panel || !nav) {
    state.pendingCategory = normalized;
    return false;
  }
  if (panel.hasAttribute("hidden")) {
    panel.removeAttribute("hidden");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "true");
    }
  }
  state.pendingCategory = null;
  selectPracticeCategory(normalized, { focusNav: true });
  if (typeof panel.scrollIntoView === "function") {
    panel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  return true;
}

function createCard(phrase) {
  const card = createEl("article", { class: "practice-card", dataset: { id: phrase.id } });
  const text = createEl("div", { class: "practice-text" }, document.createTextNode(phrase.text || ""));
  const tip = createEl("div", { class: "practice-tip" }, document.createTextNode(phrase.tip || ""));
  const controls = createEl("div", { class: "practice-controls" });
  const recordBtn = createEl("button", { class: "btn sm primary", dataset: { act: "toggle" }, "aria-pressed": "false" }, document.createTextNode(t("practice.recordStart") || t("practice.record") || "開始錄音"));
  const playBtn = createEl("button", {
    class: "btn sm secondary",
    dataset: { act: "play" },
    "aria-pressed": "false",
    type: "button",
    disabled: true,
  }, document.createTextNode(t("practice.playLast") || "播放上一段"));
  controls.append(recordBtn);
  controls.append(playBtn);

  const result = createEl("div", { class: "practice-result", "aria-live": "polite" });
  result.innerHTML = `
    <div><b class="fem" aria-label="${t("practice.feminineLabel") || t("practice.feminine") || "女性傾向"} --%">${t("practice.feminineLabel") || t("practice.feminine") || "女性傾向"} --%</b></div>
    <div><b class="masc" aria-label="${t("practice.masculineLabel") || t("practice.masculine") || "男性傾向"} --%">${t("practice.masculineLabel") || t("practice.masculine") || "男性傾向"} --%</b></div>
  `;

  const badges = createEl("div", { class: "practice-badges" });
  const lastBadge = createEl("span", { class: "badge muted last" }, document.createTextNode(t("practice.lastNone") || "尚無紀錄"));
  badges.appendChild(lastBadge);

  card.append(text, tip, controls, result, badges);
  bindCardEvents(card, phrase);
  hydrateLastBadge(card, phrase.id);
  setCardRecording(card, false);
  setCardBusy(card, false);
  const canPlay = state.lastPlayableId === phrase.id && getRecorder()?.hasLastRecording;
  setCardPlayable(card, Boolean(canPlay));
  setCardPlaying(card, state.playingId === phrase.id);
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
    const focusable = target.querySelector('[data-act="toggle"]')
      || target.querySelector('[data-act="play"]');
    focusable?.focus();
  }
}

async function refreshData(locale) {
  state.data = await loadPracticeData(locale);
  const cats = byCategory(state.data);
  if (state.selectedCategory && !cats.some((cat) => cat.id === state.selectedCategory && cat.items && cat.items.length)) {
    state.selectedCategory = null;
  }
  selectPracticeCategory(state.selectedCategory, { focusNav: false });
  if (state.pendingCategory != null) {
    selectPracticeCategory(state.pendingCategory, { focusNav: false });
  }
}

export async function setupPracticeUI({ subscribeInference, recorder } = {}) {
  bridge.subscribeInference = subscribeInference || bridge.subscribeInference;
  bridge.recorder = recorder || bridge.recorder;

  const toggle = qs("#practiceToggle");
  const panel = qs("#practicePanel");
  const list = qs("#practiceList");
  const nav = qs("#practiceNav");
  if (!toggle || !panel || !list || !nav) {
    return;
  }

  loadHistory();
  ensurePlayer();

  await refreshData(getCurrentLocale());
  if (state.pendingCategory != null) {
    const pending = state.pendingCategory;
    state.pendingCategory = null;
    openPracticeCategory(pending);
  }
  nav.addEventListener("click", (event) => {
    const target = event.target instanceof HTMLElement
      ? event.target.closest("button[data-cat]")
      : null;
    if (!target) return;
    const catId = target.dataset.cat || "";
    const normalized = catId ? catId : null;
    selectPracticeCategory(normalized, { focusNav: true });
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

  async function hidePracticePanel({ focusToggle = false } = {}) {
    stopPracticePlayback();
    const recorder = getRecorder();
    if (state.activeId) {
      const activeCard = getCardById(state.activeId);
      if (recorder?.isRecording) {
        try {
          await Promise.resolve(recorder.stop());
        } catch (err) {
          console.warn("[practice] stop recording on panel close failed", err);
        }
      }
      cancelActiveRun(activeCard);
    }
    panel.setAttribute("hidden", "");
    toggle.setAttribute("aria-expanded", "false");
    if (focusToggle && toggle instanceof HTMLElement) {
      requestAnimationFrame(() => {
        toggle.focus();
      });
    }
  }

  toggle.addEventListener("click", async () => {
    if (panel.hasAttribute("hidden")) {
      panel.removeAttribute("hidden");
      toggle.setAttribute("aria-expanded", "true");
    } else {
      await hidePracticePanel();
    }
  });

  onLocaleChange(async (locale) => {
    await refreshData(locale);
    if (state.pendingCategory != null) {
      const pending = state.pendingCategory;
      state.pendingCategory = null;
      openPracticeCategory(pending);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (panel.hasAttribute("hidden")) return;
    const active = document.activeElement instanceof HTMLElement
      ? document.activeElement.closest(".practice-card")
      : null;
    if (event.code === "Space") {
      const recordButton = (active?.querySelector('[data-act="toggle"]')
        || document.querySelector(".practice-card [data-act=\"toggle\"]"));
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
