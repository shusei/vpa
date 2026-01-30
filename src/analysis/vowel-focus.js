/**
 * Vowel Focus Analysis
 * Extracted from legacy app.js (lines 4400-4451)
 */

import { t } from "../i18n.js";
import { buildEligibleFrameMask, FORMANT_CONFIDENCE_THRESHOLD, FORMANT_MAX_GAP_FRAMES } from "./helpers.js";

const analysisText = null;

/**
 * Analyze vowel focus based on F1/F2 formant space
 */
export function analyzeVowelFocus(store, maskInfo) {
    const formants = Array.isArray(store.formants) ? store.formants : [];
    let mask = null;
    if (Array.isArray(maskInfo)) mask = maskInfo;
    else if (maskInfo && Array.isArray(maskInfo.mask)) mask = maskInfo.mask;
    if (!mask || !mask.length) {
        const built = buildEligibleFrameMask(store, {
            minConfidence: FORMANT_CONFIDENCE_THRESHOLD,
            maxGapFrames: FORMANT_MAX_GAP_FRAMES,
        });
        if (Array.isArray(built?.mask) && built.mask.length) mask = built.mask;
    }

    if (!mask || !mask.length) {
        const insufficient = analysisText?.vowelFocus?.insufficient;
        return {
            ratio: NaN,
            label: insufficient?.label || t("analysis.vowelFocus.insufficient.label"),
            hint: insufficient?.hint || t("analysis.vowelFocus.insufficient.hint"),
        };
    }

    let voiced = 0, focus = 0;
    const limit = Math.min(formants.length, mask.length);
    for (let i = 0; i < limit; i++) {
        if (!mask[i]) continue;
        const form = formants[i];
        if (!form) continue;
        const f1 = form[0], f2 = form[1];
        if (!Number.isFinite(f1) || !Number.isFinite(f2)) continue;
        voiced++;
        if (f1 >= 170 && f1 <= 480 && f2 >= 1400 && f2 <= 3000) focus++;
    }
    const ratio = voiced ? focus / voiced : NaN;
    if (!Number.isFinite(ratio)) {
        const insufficient = analysisText?.vowelFocus?.insufficient;
        return {
            ratio: NaN,
            label: insufficient?.label || t("analysis.vowelFocus.insufficient.label"),
            hint: insufficient?.hint || t("analysis.vowelFocus.insufficient.hint"),
        };
    }
    let key = "weak";
    if (ratio >= 0.5) key = "strong";
    else if (ratio >= 0.3) key = "medium";
    const entry = analysisText?.vowelFocus?.[key];
    return {
        ratio,
        label: entry?.label || t(`analysis.vowelFocus.${key}.label`),
        hint: entry?.hint || t(`analysis.vowelFocus.${key}.hint`),
    };
}
