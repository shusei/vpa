/**
 * Connected Speech (Liaison) Analysis
 * Extracted from legacy app.js (lines 4508-4558)
 */

import { t } from "../i18n.js";

const analysisText = null;

/**
 * Analyze connected speech / liaison ratio
 */
export function analyzeConnectedSpeech(voicedArr, hopSec) {
    if (!Array.isArray(voicedArr) || !voicedArr.length) {
        const insufficient = analysisText?.liaison?.insufficient;
        return {
            ratio: NaN,
            label: insufficient?.label || t("analysis.liaison.insufficient.label"),
            hint: insufficient?.hint || t("analysis.liaison.insufficient.hint"),
        };
    }
    let segments = 0;
    let inVoiced = false;
    let gapDur = 0;
    const gaps = [];
    for (let i = 0; i < voicedArr.length; i++) {
        if (voicedArr[i]) {
            if (!inVoiced) {
                segments++;
                if (gapDur > 0) { gaps.push(gapDur); gapDur = 0; }
            }
            inVoiced = true;
        } else {
            if (inVoiced) {
                inVoiced = false;
                gapDur = hopSec;
            } else if (gapDur > 0) {
                gapDur += hopSec;
            } else {
                gapDur = hopSec;
            }
        }
    }
    const totalBreaks = Math.max(0, segments - 1);
    const shortGaps = gaps.filter(g => g <= 0.16).length;
    const ratio = totalBreaks ? shortGaps / totalBreaks : (segments > 0 ? 1 : NaN);
    if (!Number.isFinite(ratio)) {
        const insufficient = analysisText?.liaison?.insufficient;
        return {
            ratio: NaN,
            label: insufficient?.label || t("analysis.liaison.insufficient.label"),
            hint: insufficient?.hint || t("analysis.liaison.insufficient.hint"),
        };
    }
    let key = "weak";
    if (ratio >= 0.7) key = "strong";
    else if (ratio >= 0.4) key = "medium";
    const entry = analysisText?.liaison?.[key];
    return {
        ratio,
        label: entry?.label || t(`analysis.liaison.${key}.label`),
        hint: entry?.hint || t(`analysis.liaison.${key}.hint`),
    };
}
