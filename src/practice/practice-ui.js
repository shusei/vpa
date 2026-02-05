import { t, getCurrentLocale, onLocaleChange } from "../i18n.js";
import { state, bridge, subscribeInference, getRecorder, loadHistory, persistHistory, byCategory, refreshData, ensurePlayer, stopPracticePlayback, setOnPlayerStop } from "./practice-core.js";

// --- DOM Helpers ---

function qs(selector, root = document) {
    return root ? root.querySelector(selector) : null;
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
                try { el[key] = value; } catch { el.setAttribute(key, value); }
            } else {
                el.setAttribute(key, value);
            }
        }
    }
    if (content != null) {
        if (Array.isArray(content)) {
            for (const child of content) { if (child != null) el.append(child); }
        } else if (typeof content === "string") {
            el.innerHTML = content;
        } else {
            el.append(content);
        }
    }
    return el;
}

// --- Card UI Manipulations ---

function getCardById(id) {
    if (!id) return null;
    const safe = typeof CSS !== "undefined" && typeof CSS.escape === "function" ? CSS.escape(id) : String(id).replace(/["\\]/g, "\\$&");
    return qs(`.practice-card[data-id="${safe}"]`);
}

function setCardRecording(card, recording) {
    if (!card) return;
    const button = card.querySelector('[data-act="toggle"]');
    if (!button) return;
    const startLabel = t("practice.recordStart") || "開始錄音";
    const stopLabel = t("practice.recordStop") || "停止錄音";
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

function resetPlayingState() {
    // Called by Core when player stops
    // We need to find which card was playing (stored in state.playingId BEFORE it was cleared by core)
    // Wait, Core clears `state.playingId`. 
    // We should iterate all cards or store a local ref? 
    // Or Core's callback tells us "stop UI for this ID".
    // Core's `onPlayerStop` sets `playingId = null`.
    // So we need to act BEFORE or just scan DOM.
    const cards = document.querySelectorAll('.practice-card button[data-act="play"][aria-pressed="true"]');
    cards.forEach(btn => {
        const card = btn.closest(".practice-card");
        setCardPlaying(card, false);
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
        badge.textContent = `${t("practice.lastScore")} ${pct}%`;
        badge.classList.remove("muted");
    } else {
        badge.textContent = t("practice.lastNone");
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
        const femLabel = t("practice.feminineLabel");
        femEl.textContent = `${femLabel} ${femPct}`;
        femEl.setAttribute("aria-label", `${femLabel} ${femPct}`);
    }
    if (mascEl) {
        const mascLabel = t("practice.masculineLabel");
        mascEl.textContent = `${mascLabel} ${mascPct}`;
        mascEl.setAttribute("aria-label", `${mascLabel} ${mascPct}`);
    }
}

// --- Event Binding ---

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

        if (recorder.busy && action !== "toggle") return;

        if (action === "toggle") {
            const isActiveCard = state.activeId === phrase.id && recorder.isRecording;
            if (isActiveCard) {
                setCardBusy(card, true);
                try { await Promise.resolve(recorder.stop()); }
                catch (err) {
                    console.error("[practice] stop recording failed", err);
                    cancelActiveRun(card);
                }
                return;
            }
            if (recorder.isRecording) return;

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
                if (state.runToken !== runToken) return;
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
                    setCardPlayable(prevCard, Boolean(getRecorder()?.hasLastRecording));
                }
            }
            return;
        }

        if (action === "play") {
            const recorderCtl = getRecorder();
            if (!recorderCtl?.hasLastRecording) return;
            const player = ensurePlayer();
            if (!player) return; // Should not happen

            if (state.playingId === phrase.id) {
                stopPracticePlayback();
                return;
            }
            stopPracticePlayback();
            const url = typeof recorderCtl.getLastRecordingUrl === "function" ? recorderCtl.getLastRecordingUrl() : null;
            if (!url) return;

            try {
                player.src = url;
                state.playingId = phrase.id;
                setCardPlaying(card, true);
                await player.play();
            } catch (err) {
                console.error("[practice] play failed", err);
                stopPracticePlayback(); // reset
            }
        }
    });
}

// --- Renderers ---

export function createCard(phrase) {
    const card = createEl("article", { class: "practice-card", dataset: { id: phrase.id } });
    const text = createEl("div", { class: "practice-text" }, document.createTextNode(phrase.text || ""));
    const tip = createEl("div", { class: "practice-tip" }, document.createTextNode(phrase.tip || ""));
    const controls = createEl("div", { class: "practice-controls" });
    const recordBtn = createEl("button", { class: "btn sm primary", dataset: { act: "toggle" }, "aria-pressed": "false" }, document.createTextNode(t("practice.recordStart") || "開始錄音"));
    const playBtn = createEl("button", { class: "btn sm secondary", dataset: { act: "play" }, "aria-pressed": "false", type: "button", disabled: true }, document.createTextNode(t("practice.playLast") || "播放上一段"));
    controls.append(recordBtn, playBtn);

    const result = createEl("div", { class: "practice-result", "aria-live": "polite" });
    const femLabel = t("practice.feminineLabel") || "女性傾向";
    const mascLabel = t("practice.masculineLabel") || "男性傾向";
    result.innerHTML = `<div><b class="fem" aria-label="${femLabel} --%">${femLabel} --%</b></div><div><b class="masc" aria-label="${mascLabel} --%">${mascLabel} --%</b></div>`;

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

function renderCategoryNav(cats = byCategory(state.data)) {
    const nav = qs("#practiceNav");
    if (!nav) return;
    const usable = cats.filter((cat) => Array.isArray(cat.items) && cat.items.length);
    const label = t("practice.filterLabel");
    nav.setAttribute("aria-label", label);
    nav.setAttribute("role", "radiogroup");
    nav.innerHTML = "";

    if (!usable.length) { nav.hidden = true; return; }
    nav.hidden = usable.length <= 1;

    const activeId = state.selectedCategory || "";
    const makeButton = (title, catId) => {
        const button = createEl("button", { type: "button", class: "practice-nav__item", dataset: { cat: catId ?? "" }, title }, document.createTextNode(title));
        button.setAttribute("role", "radio");
        const isActive = (catId ?? "") === activeId;
        button.setAttribute("aria-checked", String(isActive));
        if (isActive) button.classList.add("is-active");
        return button;
    };

    nav.appendChild(makeButton(t("practice.allCategories"), null));
    for (const cat of usable) {
        nav.appendChild(makeButton(cat.title || cat.name || cat.id || "", cat.id));
    }
}

function renderList(cats = byCategory(state.data)) {
    const list = qs("#practiceList");
    if (!list) return;
    list.setAttribute("aria-busy", "true");
    list.innerHTML = "";
    const filtered = state.selectedCategory ? cats.filter((cat) => cat.id === state.selectedCategory) : cats;
    for (const cat of filtered) {
        if (!cat.items || !cat.items.length) continue;
        const heading = createEl("h4", { class: "practice-cat" }, document.createTextNode(cat.title || cat.id || ""));
        list.appendChild(heading);
        for (const phrase of cat.items) {
            list.appendChild(createCard(phrase));
        }
    }
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
        const selector = normalized ? `[data-cat="${CSS.escape(normalized)}"]` : '[data-cat=""]';
        requestAnimationFrame(() => nav.querySelector(selector)?.focus());
    }
}

// --- Main Setup ---

export function openPracticeCategory(catId) {
    const panel = qs("#practicePanel");
    const toggle = qs("#practiceToggle");
    if (!panel) { state.pendingCategory = catId ? String(catId) : null; return false; }
    if (panel.hasAttribute("hidden")) {
        panel.removeAttribute("hidden");
        toggle?.setAttribute("aria-expanded", "true");
    }
    state.pendingCategory = null;
    selectPracticeCategory(catId, { focusNav: true });
    panel.scrollIntoView({ behavior: "smooth", block: "center" });
    return true;
}

export async function setupPracticeUI({ subscribeInference: sub, recorder: rec } = {}) {
    bridge.subscribeInference = sub || bridge.subscribeInference;
    bridge.recorder = rec || bridge.recorder;

    const toggle = qs("#practiceToggle");
    const panel = qs("#practicePanel");
    const list = qs("#practiceList");
    const nav = qs("#practiceNav");

    if (!toggle || !panel || !list || !nav) {
        console.error(`[practice] Missing elements`);
        return;
    }

    loadHistory();
    ensurePlayer();
    setOnPlayerStop(resetPlayingState); // Hook UI update

    await refreshData(getCurrentLocale());
    if (state.pendingCategory != null) {
        openPracticeCategory(state.pendingCategory);
    }

    // Toggle and Nav events
    toggle.addEventListener("click", () => {
        if (panel.hasAttribute("hidden")) {
            panel.removeAttribute("hidden");
            toggle.setAttribute("aria-expanded", "true");
        } else {
            stopPracticePlayback();
            panel.setAttribute("hidden", "");
            toggle.setAttribute("aria-expanded", "false");
        }
    });

    nav.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-cat]");
        if (btn) selectPracticeCategory(btn.dataset.cat || null, { focusNav: true });
    });

    // Locale change
    onLocaleChange(async (locale) => {
        await refreshData(locale);
    });

    // Keyboard (Space to record)
    document.addEventListener("keydown", (event) => {
        if (panel.hasAttribute("hidden")) return;
        if (event.code === "Space") {
            const active = document.activeElement.closest(".practice-card");
            if (active) {
                event.preventDefault();
                active.querySelector('[data-act="toggle"]')?.click();
            }
        }
    });
}
