/**
 * Intonation Analysis
 * Extracted from legacy app.js (lines 4561-4617)
 */

import { t } from "../i18n.js";
import { computeIntonationMetrics, CONFIDENCE_INCLUDE_THRESHOLD } from "../pitch-shared.js";
import { EPS } from "./helpers.js";

// CONFIDENCE_VOICED_THRESHOLD from pitch-shared.js
const CONFIDENCE_VOICED_THRESHOLD = 0.5;

const analysisText = null;

/**
 * Analyze intonation curve (slope, range)
 */
export function analyzeIntonation(data, hopSec) {
    const metrics = computeIntonationMetrics(data, hopSec, {
        confidenceThreshold: CONFIDENCE_INCLUDE_THRESHOLD,
        voicedThreshold: CONFIDENCE_VOICED_THRESHOLD,
        eps: EPS,
    }) || {};
    const points = Array.isArray(metrics.points) ? metrics.points : [];
    const rawPoints = Array.isArray(metrics.rawPoints) ? metrics.rawPoints : [];
    const shadedRanges = Array.isArray(metrics.shadedRanges) ? metrics.shadedRanges : [];
    const slopeCount = Number(metrics.slopeSampleCount) || 0;

    if (slopeCount < 3) {
        const insufficient = analysisText?.intonation?.insufficient;
        return {
            points: [],
            rawPoints,
            shadedRanges,
            slope: NaN,
            slopeLabel: insufficient?.slopeLabel || t("analysis.intonation.insufficient.slopeLabel"),
            hint: insufficient?.slopeHint || t("analysis.intonation.insufficient.slopeHint"),
            range: NaN,
            rangeHint: insufficient?.rangeHint || "",
            minHz: NaN,
            maxHz: NaN,
        };
    }

    const slope = Number.isFinite(metrics.slope) ? metrics.slope : NaN;
    const validRange = Number.isFinite(metrics.range) ? metrics.range : NaN;
    let slopeKey = "flat";
    if (slope > 12) slopeKey = "rising";
    else if (slope < -12) slopeKey = "falling";
    const slopeEntry = analysisText?.intonation?.slope?.[slopeKey];
    const slopeLabel = slopeEntry?.label || t(`analysis.intonation.slope.${slopeKey}.label`);
    const slopeHint = slopeEntry?.hint || t(`analysis.intonation.slope.${slopeKey}.hint`);

    let rangeKey = "narrow";
    if (validRange >= 90) rangeKey = "rich";
    else if (validRange >= 50) rangeKey = "medium";
    const rangeEntry = analysisText?.intonation?.range?.[rangeKey];
    const rangeHint = rangeEntry?.hint || t(`analysis.intonation.range.${rangeKey}.hint`);
    const rangeLabel = rangeEntry?.label || t(`analysis.intonation.range.${rangeKey}.label`);
    const hint = `${slopeHint} ${rangeHint}`.trim();

    return {
        points,
        rawPoints,
        shadedRanges,
        slope,
        slopeLabel,
        range: validRange,
        rangeLabel,
        hint,
        rangeHint,
        minHz: Number.isFinite(metrics.minHz) ? metrics.minHz : NaN,
        maxHz: Number.isFinite(metrics.maxHz) ? metrics.maxHz : NaN,
    };
}
