/**
 * Brightness and Tilt Analysis
 * Extracted from legacy app.js (lines 4112-4275)
 */

import { t } from "../i18n.js";
import {
    BRIGHTNESS_F3_LOW,
    BRIGHTNESS_F3_HIGH,
    BRIGHTNESS_WARM_Z,
    BRIGHTNESS_SPARKLE_Z,
    BRIGHTNESS_SWEET_Z,
    BRIGHTNESS_TILT_SHARP,
    BRIGHTNESS_BREATH_THRESHOLD,
} from "./helpers.js";

const analysisText = null; // Legacy global

/**
 * Categorize brightness based on F3 frequency
 */
export function categorizeBrightness({ f3Stats, tilt, breath, leaning } = {}) {
    const brightnessText = analysisText?.brightness;
    if (!Number.isFinite(f3Stats?.med)) {
        const insufficient = brightnessText?.insufficient;
        return {
            label: insufficient?.label || t("analysis.brightness.insufficient.label"),
            hint: insufficient?.hint || t("analysis.brightness.insufficient.hint"),
            key: "insufficient",
        };
    }
    const med = f3Stats.med;
    const center = (BRIGHTNESS_F3_LOW + BRIGHTNESS_F3_HIGH) / 2;
    const span = Math.max(1, (BRIGHTNESS_F3_HIGH - BRIGHTNESS_F3_LOW) / 2);
    const z = (med - center) / span;
    const tiltVal = Number.isFinite(tilt) ? tilt : NaN;
    const breathVal = Number.isFinite(breath) ? breath : NaN;
    let key = "balanced";
    if (z <= BRIGHTNESS_WARM_Z) key = "warm";
    else if (z >= BRIGHTNESS_SWEET_Z) {
        const needsRelax = (Number.isFinite(breathVal) && breathVal > BRIGHTNESS_BREATH_THRESHOLD)
            || (Number.isFinite(tiltVal) && tiltVal < BRIGHTNESS_TILT_SHARP);
        key = needsRelax ? "sharp" : "sweet";
    } else if (z >= BRIGHTNESS_SPARKLE_Z) {
        key = "sparkle";
    }
    // Masculine-leaning：避免「甜」「閃」等性別暗示字眼
    let lookupKey = key;
    if (leaning === "masculine") {
        if (key === "sweet") lookupKey = "sweetMasculine";
        else if (key === "sparkle") lookupKey = "sparkleMasculine";
    }
    let entry = brightnessText?.[lookupKey];
    if (!entry && lookupKey !== key) entry = brightnessText?.[key];
    const label = entry?.label
        || t(`analysis.brightness.${lookupKey}.label`)
        || (lookupKey !== key ? t(`analysis.brightness.${key}.label`) : "");
    const hint = entry?.hint
        || t(`analysis.brightness.${lookupKey}.hint`)
        || (lookupKey !== key ? t(`analysis.brightness.${key}.hint`) : "");
    return { label, hint, key, zScore: z };
}

/**
 * Categorize spectral tilt
 */
export function categorizeTilt(tilt) {
    if (!Number.isFinite(tilt)) {
        const insufficient = analysisText?.tilt?.insufficient;
        return {
            label: insufficient?.label || t("analysis.tilt.insufficient.label"),
            hint: insufficient?.hint || t("analysis.tilt.insufficient.hint"),
        };
    }
    let key = "bright";
    if (tilt >= 7.5) key = "warm";
    else if (tilt >= 4.5) key = "gentleWarm";
    else if (tilt >= -1) key = "balanced";
    const entry = analysisText?.tilt?.[key];
    return {
        label: entry?.label || t(`analysis.tilt.${key}.label`),
        hint: entry?.hint || t(`analysis.tilt.${key}.hint`),
    };
}

/**
 * Categorize breathiness
 */
export function categorizeBreathiness(val, ctx = {}) {
    if (!Number.isFinite(val)) {
        const insufficient = analysisText?.breathiness?.insufficient;
        return {
            label: insufficient?.label || t("analysis.breathiness.insufficient.label"),
            hint: insufficient?.hint || t("analysis.breathiness.insufficient.hint"),
        };
    }
    const snr = Number.isFinite(ctx.snr) ? ctx.snr : NaN;
    const brightnessKey = ctx.brightnessKey;
    const tilt = Number.isFinite(ctx.tilt) ? ctx.tilt : NaN;
    const styleEligible = Number.isFinite(snr) ? snr > 20 : false;
    const needsRelax = brightnessKey === "sharp" || (Number.isFinite(tilt) && tilt < BRIGHTNESS_TILT_SHARP);

    let key = "airy";
    if (val < 0.08) key = "dense";
    else if (val <= 0.18) key = "balanced";
    else if (val <= 0.28) key = "airy";
    else if (val <= 0.45) key = styleEligible ? "style" : "airy";
    else {
        if (needsRelax) key = "tooAiry";
        else key = styleEligible ? "style" : "airy";
    }

    const entry = analysisText?.breathiness?.[key];
    return {
        label: entry?.label || t(`analysis.breathiness.${key}.label`),
        hint: entry?.hint || t(`analysis.breathiness.${key}.hint`),
        key,
    };
}
