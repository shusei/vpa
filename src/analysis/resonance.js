/**
 * Resonance Analysis Functions
 * Extracted from legacy app.js (lines 4162-4256)
 */

import { t } from "../i18n.js";
import {
    EPS,
    RESONANCE_PRIOR_WEIGHT,
    RESONANCE_PRIOR_MIN,
    RESONANCE_DOMINANCE_DELTA,
    RESONANCE_HEAD_MIN,
    RESONANCE_MASK_MIN,
    RESONANCE_CHEST_MIN,
    RESONANCE_MIN_SAMPLES,
    RESONANCE_MIN_COVERAGE,
    RESONANCE_COVERAGE_GOOD,
} from "./helpers.js";

// Note: analysisText is from i18n, but we use t() for compatibility
const analysisText = null; // Legacy global, now using t() directly

/**
 * Normalize resonance energy bands to chest/mask/head percentages
 */
export function normalizeResonanceBands(energy) {
    if (!energy) return { chest: NaN, mask: NaN, head: NaN, total: 0, coverage: 0 };
    const low = Math.max(0, Number.isFinite(energy.low) ? energy.low : 0);
    const mid = Math.max(0, Number.isFinite(energy.mid) ? energy.mid : 0);
    const high = Math.max(0, Number.isFinite(energy.high) ? energy.high : 0);
    const total = low + mid + high;
    if (!Number.isFinite(total) || total <= EPS) {
        return { chest: NaN, mask: NaN, head: NaN, total: 0, coverage: Number.isFinite(energy.coverage) ? energy.coverage : 0 };
    }
    const prior = Math.max(RESONANCE_PRIOR_MIN, total * RESONANCE_PRIOR_WEIGHT);
    const perBand = prior / 3;
    const denom = total + prior;
    return {
        chest: ((low + perBand) / denom),
        mask: ((mid + perBand) / denom),
        head: ((high + perBand) / denom),
        total,
        coverage: Number.isFinite(energy.coverage) ? energy.coverage : 0,
    };
}

/**
 * Describe resonance balance from energy data
 */
export function describeResonanceFromEnergy(energy) {
    const pctFallback = { chest: 1 / 3, mask: 1 / 3, head: 1 / 3 };
    const normalized = normalizeResonanceBands(energy);
    const { chest, mask, head, total } = normalized;
    const hasAggregate = energy && (Number.isFinite(energy.coverage) || Number.isFinite(energy.validCount));
    const coverage = hasAggregate ? (Number.isFinite(energy.coverage) ? energy.coverage : 0) : NaN;
    const validCount = hasAggregate ? (Number.isFinite(energy.validCount) ? energy.validCount : 0) : 0;

    const insufficientEntry = () => {
        const insufficient = analysisText?.resonanceBalance?.insufficient;
        return {
            label: insufficient?.label || t("analysis.resonanceBalance.insufficient.label"),
            hint: insufficient?.hint || t("analysis.resonanceBalance.insufficient.hint"),
            pct: pctFallback,
            total: 0,
            coverage: hasAggregate ? coverage : NaN,
            display: insufficient?.label || t("analysis.resonanceBalance.insufficient.label"),
        };
    };

    if (hasAggregate) {
        if (!Number.isFinite(coverage) || coverage < RESONANCE_MIN_COVERAGE || validCount < RESONANCE_MIN_SAMPLES) {
            const base = insufficientEntry();
            const coverageNote = t("analysis.resonanceBalance.coverageLowHint", { value: Math.round((coverage || 0) * 100) });
            if (coverageNote) base.hint = `${base.hint} ${coverageNote}`.trim();
            if (Number.isFinite(coverage)) {
                const suffix = t("analysis.resonanceBalance.coverageLowSuffix", { value: Math.round(coverage * 100) });
                base.display = suffix ? `${base.label}${suffix}` : base.label;
            }
            return base;
        }
    }

    if (!Number.isFinite(chest) || !Number.isFinite(mask) || !Number.isFinite(head)) {
        return insufficientEntry();
    }

    const chestLead = chest - Math.max(mask, head);
    const maskLead = mask - Math.max(chest, head);
    const headLead = head - Math.max(chest, mask);
    let key = "balanced";
    if (chestLead >= RESONANCE_DOMINANCE_DELTA && chest >= RESONANCE_CHEST_MIN) {
        key = "chestHeavy";
    } else if (maskLead >= RESONANCE_DOMINANCE_DELTA && mask >= RESONANCE_MASK_MIN) {
        key = "maskLead";
    } else if (headLead >= RESONANCE_DOMINANCE_DELTA && head >= RESONANCE_HEAD_MIN) {
        key = "headBright";
    }

    const entry = analysisText?.resonanceBalance?.[key];
    const label = entry?.label || t(`analysis.resonanceBalance.${key}.label`);
    let hint = entry?.hint || t(`analysis.resonanceBalance.${key}.hint`);
    let display = label;
    if (hasAggregate && Number.isFinite(coverage)) {
        const coverageKey = coverage < RESONANCE_COVERAGE_GOOD ? "coverageLowHint" : "coverageHint";
        const hintNote = t(`analysis.resonanceBalance.${coverageKey}`, { value: Math.round(coverage * 100) });
        if (hintNote) hint = `${hint} ${hintNote}`.trim();
        let suffix = coverage < RESONANCE_COVERAGE_GOOD
            ? t("analysis.resonanceBalance.referenceSuffix", { value: Math.round(coverage * 100) })
            : t("analysis.resonanceBalance.coverageSuffix", { value: Math.round(coverage * 100) });
        if (!suffix && coverage < RESONANCE_COVERAGE_GOOD) {
            suffix = t("analysis.resonanceBalance.referenceOnly");
        }
        if (suffix) display = `${label}${suffix}`;
    }
    return {
        label,
        display,
        hint,
        pct: { chest, mask, head },
        total,
        coverage: hasAggregate ? coverage : NaN,
    };
}

/**
 * Detect voice leaning from model probabilities
 */
export function detectVoiceLeaning(pf, pm) {
    const pfVal = Number.isFinite(pf) ? pf : 0;
    const pmVal = Number.isFinite(pm) ? pm : 0;
    const diff = Math.abs(pfVal - pmVal);
    if (diff < 0.08) return "neutral";
    return pfVal > pmVal ? "feminine" : "masculine";
}
