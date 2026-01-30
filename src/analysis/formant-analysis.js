/**
 * Formant Analysis Functions
 * Extracted from legacy app.js (lines 4309-4398)
 */

import { t } from "../i18n.js";
import { makeStats } from "../pitch-shared.js";
import {
    FORMANT_MIN_SAMPLES,
    FORMANT_MIN_COVERAGE,
    FORMANT_GOOD_COVERAGE,
    FORMANT_MIN_SPREAD,
} from "./helpers.js";

const analysisText = null;

/**
 * Make formant hint based on value and range
 */
export function makeFormantHint(label, value, low, high) {
    const rangeLabels = analysisText?.formant?.rangeLabels || {};
    const labelName = rangeLabels[label] || t(`analysis.formant.rangeLabels.${label}`);
    const labelWithName = `${label}（${labelName || label}）`;
    if (!Number.isFinite(value)) {
        return t("analysis.formant.insufficient", { label: labelWithName });
    }
    const lowHint = analysisText?.formant?.low?.[label] || t(`analysis.formant.low.${label}`);
    const highHint = analysisText?.formant?.high?.[label] || t(`analysis.formant.high.${label}`);
    if (value < low) {
        const msg = t("analysis.formant.lowMessage", { label: labelWithName, hint: lowHint });
        return msg || `${labelWithName} ${lowHint}`;
    }
    if (value > high) {
        const msg = t("analysis.formant.highMessage", { label: labelWithName, hint: highHint });
        return msg || `${labelWithName} ${highHint}`;
    }
    return t("analysis.formant.inRange", { label: labelWithName });
}

/**
 * Build formant trend display label with coverage suffix
 */
export function buildFormantTrendDisplay(trendKey, coverage, hasAggregate) {
    const trendLabels = analysisText?.formant?.trendLabels;
    const baseRaw = (trendLabels && trendLabels[trendKey]) || t(`analysis.formant.trendLabels.${trendKey}`);
    const base = baseRaw || "—";
    if (!hasAggregate || !Number.isFinite(coverage) || coverage <= 0) {
        return base;
    }
    const clamped = Math.max(0, Math.min(1, coverage));
    const key = clamped < FORMANT_GOOD_COVERAGE ? "coverageLowSuffix" : "coverageSuffix";
    const suffix = t(`analysis.formant.${key}`, { value: Math.round(clamped * 100) });
    if (!suffix) return base;
    return `${base}${suffix}`;
}

/**
 * Summarize formant trends (F1, F2, F3)
 */
export function summarizeFormantTrends(store, statsBundle, options = {}) {
    let eligibleCount = Number.isFinite(options?.eligibleCount) ? options.eligibleCount : null;
    if ((eligibleCount == null || eligibleCount <= 0) && Array.isArray(store.voiced)) {
        eligibleCount = store.voiced.reduce((acc, flag) => acc + (flag ? 1 : 0), 0);
    }
    const hasAggregate = Number.isFinite(eligibleCount) && eligibleCount > 0;

    const makeEntry = (label, stats, values, low, high) => {
        const sampleCount = values.length;
        const coverageRaw = hasAggregate ? (sampleCount / eligibleCount) : 0;
        const coverage = hasAggregate ? Math.max(0, Math.min(1, coverageRaw)) : NaN;
        const spread = stats ? (stats.p95 - stats.p05) : NaN;
        const reliable = hasAggregate
            && sampleCount >= FORMANT_MIN_SAMPLES
            && coverageRaw >= FORMANT_MIN_COVERAGE
            && Number.isFinite(spread)
            && spread >= FORMANT_MIN_SPREAD
            && Number.isFinite(stats?.med);

        const trendKey = reliable
            ? (stats.med < low ? "low" : (stats.med > high ? "high" : "inRange"))
            : "insufficient";

        const baseHint = reliable
            ? makeFormantHint(label, stats.med, low, high)
            : makeFormantHint(label, NaN, low, high);

        const extraHints = [];
        if (hasAggregate && sampleCount < FORMANT_MIN_SAMPLES) {
            const msg = t("analysis.formant.moreSamplesHint");
            if (msg) extraHints.push(msg);
        }
        if (hasAggregate && coverageRaw < FORMANT_MIN_COVERAGE) {
            const msg = t("analysis.formant.coverageLowHint", { value: Math.round(Math.max(0, Math.min(1, coverageRaw)) * 100) });
            if (msg) extraHints.push(msg);
        }

        const hint = [baseHint, ...extraHints].filter(Boolean).join(" ").trim();
        const display = buildFormantTrendDisplay(trendKey, coverage, hasAggregate);

        return {
            trend: trendKey,
            median: reliable ? stats.med : NaN,
            display,
            hint,
            coverage: hasAggregate ? coverage : NaN,
            samples: sampleCount,
        };
    };

    return {
        f1: makeEntry("F1", statsBundle.f1, statsBundle.f1Vals || [], 170, 420),
        f2: makeEntry("F2", statsBundle.f2, statsBundle.f2Vals || [], 1450, 2750),
        f3: makeEntry("F3", statsBundle.f3, statsBundle.f3Vals || [], 2400, 3400),
    };
}
