import { t } from "../i18n.js";
import { loadPracticeData } from "../practice-data.js";

const LS_HISTORY = "vpa.practice.v1.history";

export const state = {
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

// Bridge for external dependencies to avoid circular imports if possible,
// or just to mock/control.
export const bridge = {
    subscribeInference: null,
    recorder: null,
};

export function subscribeInference(cb) {
    if (typeof bridge.subscribeInference === "function") {
        return bridge.subscribeInference(cb);
    }
    return () => { };
}

export function getRecorder() {
    return bridge.recorder || null;
}

// --- History ---

export function loadHistory() {
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

export function saveHistory() {
    try {
        const serialized = Array.from(state.history.entries());
        localStorage.setItem(LS_HISTORY, JSON.stringify(serialized));
    } catch (err) {
        console.warn("[practice] save history failed", err);
    }
}

export function persistHistory(id, pf, pm) {
    const list = state.history.get(id) || [];
    list.push({ ts: Date.now(), pf, pm });
    while (list.length > 20) list.shift();
    state.history.set(id, list);
    saveHistory();
}

// --- Data ---
export function byCategory(data) {
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

export async function refreshData(locale) {
    state.data = await loadPracticeData(locale);
    const cats = byCategory(state.data);
    if (state.selectedCategory && !cats.some((cat) => cat.id === state.selectedCategory && cat.items && cat.items.length)) {
        state.selectedCategory = null;
    }
    return cats;
}

// --- Player ---

export function ensurePlayer() {
    if (state.player) {
        return state.player;
    }
    if (typeof Audio !== "function") {
        return null;
    }
    const audio = new Audio();
    audio.preload = "auto";
    // UI logic will need to listen to these events?
    // We can dispatch custom events or use callbacks if strictly separated.
    // For now, let's attach basic listeners that reset state,
    // and rely on UI to poll or use event bus if improved.
    // BUT! resetPlayingState is a UI function (updates DOM).
    // We need to decouple.
    audio.addEventListener("ended", () => {
        if (state.playingId) {
            // Dispatch event to UI?
            // Or just update state and let UI react?
            // Since we don't have a reactive framework, we need to call UI.
            // We can use a callback.
            onPlayerStop();
        }
    });
    audio.addEventListener("pause", () => {
        if (audio.ended) return;
        onPlayerStop();
    });
    audio.addEventListener("error", () => {
        onPlayerStop();
    });
    state.player = audio;
    return audio;
}

let onPlayerStopCallback = () => { };
export function setOnPlayerStop(cb) {
    onPlayerStopCallback = cb;
}
function onPlayerStop() {
    state.playingId = null;
    onPlayerStopCallback();
}

export function stopPracticePlayback() {
    if (!state.player) return;
    try {
        state.player.pause();
        state.player.currentTime = 0;
    } catch (err) {
        console.warn("[practice] stop playback failed", err);
    }
    onPlayerStop();
}
