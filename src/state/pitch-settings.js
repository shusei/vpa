import {
    VOICE_PRESETS,
    DEFAULT_PITCH_RANGE,
    PITCH_PROFILE_DEFAULT,
    clampPitchRange
} from "../pitch-shared.js";
import { AUTO_WIDE_RANGE } from "../config.js";

export const PITCH_RANGE_KEY = "vpa:pitchRangeHz";
export const PITCH_PROFILE_KEY = "vpa:pitchProfile";

let pitchProfileSetting = PITCH_PROFILE_DEFAULT;
let pitchRangeSetting = { ...DEFAULT_PITCH_RANGE };

const listeners = new Set();

export function onSettingsChange(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
}

function notify() {
    listeners.forEach(fn => {
        try { fn(); } catch (e) { console.error(e); }
    });
}

export function loadSettings() {
    pitchProfileSetting = loadPitchProfile();
    pitchRangeSetting = loadPitchRange();
    notify();
}

function loadPitchProfile() {
    try {
        const raw = localStorage.getItem(PITCH_PROFILE_KEY);
        if (!raw) return PITCH_PROFILE_DEFAULT;
        if (raw === "custom") return "custom";
        if (raw in VOICE_PRESETS) return raw;
        return PITCH_PROFILE_DEFAULT;
    } catch {
        return PITCH_PROFILE_DEFAULT;
    }
}

function loadPitchRange() {
    try {
        const raw = localStorage.getItem(PITCH_RANGE_KEY);
        if (!raw) return { ...DEFAULT_PITCH_RANGE };
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return { ...DEFAULT_PITCH_RANGE };
        return clampPitchRange(parsed);
    } catch {
        return { ...DEFAULT_PITCH_RANGE };
    }
}

export function savePitchProfile(profile) {
    pitchProfileSetting = profile;
    try { localStorage.setItem(PITCH_PROFILE_KEY, profile); } catch { }
    notify();
}

export function savePitchRange(range) {
    pitchRangeSetting = clampPitchRange(range || DEFAULT_PITCH_RANGE);
    if (pitchProfileSetting !== "auto") {
        try { localStorage.setItem(PITCH_RANGE_KEY, JSON.stringify(pitchRangeSetting)); } catch { }
    }
    notify();
    return pitchRangeSetting;
}

export function getPitchProfile() {
    return pitchProfileSetting;
}

export function getPitchRange() {
    return pitchRangeSetting;
}

export function getPitchProfileDisplayRange(autoRangeState) {
    if (pitchProfileSetting === "auto") {
        return clampPitchRange(autoRangeState?.currentRange || VOICE_PRESETS.neutral);
    }
    if (pitchProfileSetting && pitchProfileSetting in VOICE_PRESETS) {
        const preset = VOICE_PRESETS[pitchProfileSetting];
        if (preset) return clampPitchRange(preset);
    }
    return clampPitchRange(pitchRangeSetting || DEFAULT_PITCH_RANGE);
}

export function getPitchDetectorRange(autoRangeState) {
    if (pitchProfileSetting === "auto") {
        if (autoRangeState?.stage === "bootstrap") {
            return clampPitchRange(AUTO_WIDE_RANGE);
        }
        return clampPitchRange(autoRangeState?.currentRange || VOICE_PRESETS.neutral);
    }
    if (pitchProfileSetting && pitchProfileSetting in VOICE_PRESETS) {
        const preset = VOICE_PRESETS[pitchProfileSetting];
        if (preset) return clampPitchRange(preset);
    }
    return clampPitchRange(pitchRangeSetting || DEFAULT_PITCH_RANGE);
}
