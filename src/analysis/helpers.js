/**
 * Analysis Helper Functions
 * Extracted from legacy app.js for modular statistics analysis
 */

import { PS_INTERVAL_MS, CONFIDENCE_INCLUDE_THRESHOLD } from "../pitch-shared.js";

// ===== Constants =====

export const RESONANCE_PRIOR_WEIGHT = 0.02;
export const RESONANCE_PRIOR_MIN = 0.6;
export const RESONANCE_DOMINANCE_DELTA = 0.12;
export const RESONANCE_HEAD_MIN = 0.50;
export const RESONANCE_MASK_MIN = 0.46;
export const RESONANCE_CHEST_MIN = 0.54;
export const RESONANCE_MIN_SAMPLES = 18;
export const RESONANCE_MIN_COVERAGE = 0.2;
export const RESONANCE_COVERAGE_GOOD = 0.35;

export const FORMANT_MIN_SAMPLES = 24;
export const FORMANT_MIN_COVERAGE = 0.18;
export const FORMANT_GOOD_COVERAGE = 0.32;
export const FORMANT_MIN_SPREAD = 70;
export const FORMANT_CONFIDENCE_THRESHOLD = CONFIDENCE_INCLUDE_THRESHOLD;
export const FORMANT_MAX_GAP_FRAMES = 8;

export const BREATHINESS_EMA_TAU_SEC = 0.2;
export const BRIGHTNESS_F3_LOW = 2400;
export const BRIGHTNESS_F3_HIGH = 3400;
export const BRIGHTNESS_WARM_Z = -0.7;
export const BRIGHTNESS_SPARKLE_Z = 0.4;
export const BRIGHTNESS_SWEET_Z = 1.0;
export const BRIGHTNESS_TILT_SHARP = -1.5;
export const BRIGHTNESS_BREATH_THRESHOLD = 0.45;

export const EPS = 1e-9;

// ===== Helper Functions =====

/**
 * Average finite values from array with optional mask
 * @param {number[]} arr - Array of values
 * @param {boolean[]|{mask:boolean[]}} mask - Optional mask
 * @returns {number} Average or NaN
 */
export function averageFinite(arr, mask) {
    const values = Array.isArray(arr) ? arr : [];
    const maskArray = Array.isArray(mask?.mask) ? mask.mask : (Array.isArray(mask) ? mask : null);
    const limit = maskArray ? Math.min(values.length, maskArray.length) : values.length;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < limit; i++) {
        if (maskArray && !maskArray[i]) continue;
        const val = values[i];
        if (Number.isFinite(val)) {
            sum += val;
            count++;
        }
    }
    if (!count) return NaN;
    return sum / count;
}

/**
 * Average energy bands (low/mid/high) with coverage tracking
 * @param {Array} arr - Array of [low, mid, high] tuples
 * @param {Object} info - {mask, eligibleCount}
 * @returns {Object} {low, mid, high, total, coverage, validCount}
 */
export function averageEnergy(arr, info = {}) {
    if (!Array.isArray(arr) || !arr.length) return { low: 0, mid: 0, high: 0, total: 0, coverage: 0, validCount: 0 };
    const mask = Array.isArray(info.mask) ? info.mask : (Array.isArray(info.mask?.mask) ? info.mask.mask : null);
    const eligible = Number.isFinite(info.eligibleCount) ? info.eligibleCount : (mask ? mask.reduce((acc, flag) => acc + (flag ? 1 : 0), 0) : arr.length);
    const limit = mask ? Math.min(arr.length, mask.length) : arr.length;
    let low = 0, mid = 0, high = 0, valid = 0, considered = 0;
    for (let i = 0; i < limit; i++) {
        if (mask && !mask[i]) continue;
        considered++;
        const v = arr[i];
        if (!Array.isArray(v)) continue;
        const [l, m, h] = v;
        if (!Number.isFinite(l) && !Number.isFinite(m) && !Number.isFinite(h)) continue;
        low += Number.isFinite(l) ? l : 0;
        mid += Number.isFinite(m) ? m : 0;
        high += Number.isFinite(h) ? h : 0;
        valid++;
    }
    if (!valid) {
        const baseCoverage = eligible > 0 ? (considered / eligible) : 0;
        return { low: 0, mid: 0, high: 0, total: 0, coverage: Math.max(0, Math.min(1, baseCoverage)), validCount: 0 };
    }
    const avgLow = low / valid;
    const avgMid = mid / valid;
    const avgHigh = high / valid;
    const baseCoverage = eligible > 0 ? (valid / eligible) : 0;
    return {
        low: avgLow,
        mid: avgMid,
        high: avgHigh,
        total: avgLow + avgMid + avgHigh,
        coverage: Math.max(0, Math.min(1, baseCoverage)),
        validCount: valid,
    };
}

/**
 * Summarize breathiness with EMA smoothing
 * @param {number[]} arr - Breathiness values
 * @param {Object} info - {mask}
 * @param {number} hopSec - Frame duration in seconds
 * @returns {Object} {avg, count}
 */
export function summarizeBreathiness(arr, info = {}, hopSec) {
    if (!Array.isArray(arr) || !arr.length) return { avg: NaN, count: 0 };
    const mask = Array.isArray(info.mask) ? info.mask : (Array.isArray(info.mask?.mask) ? info.mask.mask : null);
    const limit = mask ? Math.min(arr.length, mask.length) : arr.length;
    const step = Number.isFinite(hopSec) && hopSec > 0 ? hopSec : (PS_INTERVAL_MS / 1000);
    const tau = Math.max(0.08, BREATHINESS_EMA_TAU_SEC);
    const alpha = 1 - Math.exp(-step / tau);
    let ema = null;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < limit; i++) {
        if (mask && !mask[i]) continue;
        let val = arr[i];
        if (!Number.isFinite(val)) continue;
        val = Math.max(0, Math.min(1, val));
        if (ema == null) ema = val;
        else ema = ema + alpha * (val - ema);
        sum += ema;
        count++;
    }
    if (!count) return { avg: NaN, count: 0 };
    const avg = Math.max(0, Math.min(1, sum / count));
    return { avg, count };
}

/**
 * Build eligible frame mask from voiced/confidence data
 * @param {Object} store - {voiced, pitchConfidence}
 * @param {Object} options - {minConfidence, maxGapFrames}
 * @returns {Object} {mask, count, minConfidence, maxGapFrames}
 */
export function buildEligibleFrameMask(store, { minConfidence = FORMANT_CONFIDENCE_THRESHOLD, maxGapFrames = FORMANT_MAX_GAP_FRAMES } = {}) {
    const voiced = Array.isArray(store.voiced) ? store.voiced : [];
    const confidence = Array.isArray(store.pitchConfidence) ? store.pitchConfidence : [];
    const n = Math.min(voiced.length, confidence.length);
    let mask = new Array(n).fill(false);
    for (let i = 0; i < n; i++) {
        const conf = confidence[i] ?? 0;
        mask[i] = Boolean(voiced[i]) && conf >= minConfidence;
    }
    if (maxGapFrames > 0 && mask.length) {
        let gapStart = -1;
        for (let i = 0; i <= n; i++) {
            const flag = i < n ? mask[i] : true;
            if (!flag) {
                if (gapStart < 0) gapStart = i;
            } else if (gapStart >= 0) {
                const gapLen = i - gapStart;
                const prev = gapStart > 0 ? mask[gapStart - 1] : false;
                const next = i < n ? mask[i] : false;
                if (prev && next && gapLen <= maxGapFrames) {
                    for (let j = gapStart; j < i; j++) mask[j] = true;
                }
                gapStart = -1;
            }
        }
        const dilation = Math.max(1, Math.floor(maxGapFrames / 2));
        if (dilation > 0) {
            const expanded = mask.slice();
            for (let i = 0; i < n; i++) {
                if (!mask[i]) continue;
                for (let d = 1; d <= dilation; d++) {
                    if (i - d >= 0) expanded[i - d] = true;
                    if (i + d < n) expanded[i + d] = true;
                }
            }
            mask = expanded;
        }
    }
    const count = mask.reduce((acc, flag) => acc + (flag ? 1 : 0), 0);
    return { mask, count, minConfidence, maxGapFrames };
}
