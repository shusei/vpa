/**
 * Speech Rate Analysis
 * Extracted from legacy app.js (lines 4453-4506)
 */

import { t } from "../i18n.js";
import { PS_INTERVAL_MS } from "../pitch-shared.js";
import { EPS } from "./helpers.js";

const analysisText = null;

/**
 * Analyze speech rate (syllables per second, words per minute)
 */
export function analyzeSpeechRate(store) {
    const hopSec = store.frameSec || (PS_INTERVAL_MS / 1000);
    const n = store.db.length;
    if (!n) {
        const insufficient = analysisText?.speechRate?.insufficient;
        return {
            syllPerSec: NaN,
            wordsPerMin: NaN,
            label: insufficient?.label || t("analysis.speechRate.insufficient.label"),
            hint: insufficient?.hint || t("analysis.speechRate.insufficient.hint"),
            key: "insufficient",
        };
    }
    const duration = hopSec * n;
    let peaks = 0;
    let lastPeak = -Infinity;
    for (let i = 1; i < n - 1; i++) {
        if (!store.voiced[i]) continue;
        const prev = store.db[i - 1] ?? store.db[i];
        const curr = store.db[i];
        const next = store.db[i + 1] ?? store.db[i];
        if ((curr - prev) > 1.2 && curr >= next - 0.5) {
            const t = i * hopSec;
            if (t - lastPeak >= 0.18) { peaks++; lastPeak = t; }
        }
    }
    if (!peaks) {
        const voicedFrames = store.voiced.filter(Boolean).length;
        if (voicedFrames) { peaks = Math.max(1, Math.round((voicedFrames * hopSec) / 0.22)); }
    }
    const syllPerSec = peaks / Math.max(duration, EPS);
    const wordsPerMin = syllPerSec > 0 ? (syllPerSec / 1.5) * 60 : NaN;
    if (!Number.isFinite(syllPerSec) || syllPerSec <= 0) {
        const insufficient = analysisText?.speechRate?.insufficient;
        return {
            syllPerSec: NaN,
            wordsPerMin: NaN,
            label: insufficient?.label || t("analysis.speechRate.insufficient.label"),
            hint: insufficient?.hint || t("analysis.speechRate.insufficient.hint"),
            key: "insufficient",
        };
    }
    let key = "fast";
    if (syllPerSec < 2.2) key = "tooSlow";
    else if (syllPerSec <= 4.2) key = "balanced";
    const entry = analysisText?.speechRate?.[key];
    return {
        syllPerSec,
        wordsPerMin,
        label: entry?.label || t(`analysis.speechRate.${key}.label`),
        hint: entry?.hint || t(`analysis.speechRate.${key}.hint`),
        key,
    };
}
