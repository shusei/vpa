import { t } from "../i18n.js";

/**
 * Format number to 1 decimal place
 */
export function fmt1(n) {
    return Number.isFinite(n) ? n.toFixed(1) : "—";
}

/**
 * Escape HTML attribute value
 */
export function escapeAttr(value) {
    if (value == null) return "";
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

/**
 * Get summary string from i18n with parameter substitution
 */
export function summaryString(key, params = {}) {
    const template = t(`summary.${key}`, params);
    if (!template) return "";

    let result = template;
    Object.entries(params).forEach(([k, v]) => {
        result = result.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
    });
    return result;
}

/**
 * Determine pitch band label from median Hz
 */
export function bandOf(medHz) {
    if (!isFinite(medHz)) return "—";
    if (medHz < 85) return t("pitchBands.low") || "低於男聲";
    if (medHz < 165) return t("pitchBands.male") || "男聲範圍";
    if (medHz < 180) return t("pitchBands.overlap") || "重疊帶";
    if (medHz < 310) return t("pitchBands.female") || "女聲範圍";
    if (medHz < 450) return t("pitchBands.high") || "高音";
    if (medHz <= 600) return t("pitchBands.falsetto") || "假聲";
    return t("pitchBands.outOfRange") || "超出範圍";
}

/**
 * Check if pitch band conflicts with model prediction
 */
export function isDivergent(medHz, pf, pm) {
    if (!isFinite(medHz)) return false;
    // 165–180 overlap zone is not divergent
    if (medHz >= 165 && medHz < 180) return false;
    // High pitch but masculine prediction, or low pitch but feminine prediction
    if ((medHz >= 180 && pm >= 0.60) || (medHz <= 165 && pf >= 0.60)) return true;
    return false;
}

// Baseline ranges for gauges
export const BASELINES = {
    f1: { min: 170, max: 420, unit: "Hz" },
    f2: { min: 1450, max: 2750, unit: "Hz" },
    f3: { min: 2400, max: 3400, unit: "Hz" },
    tilt: { min: -1, max: 8, unit: "dB", visualMin: -8, visualMax: 10 },
    breath: { min: 8, max: 18, unit: "%", visualMin: 0, visualMax: 60 },
    syll: { min: 3.2, max: 5.2, unit: "syll/s" },
    wpm: { min: 120, max: 180, unit: "wpm" },
    liaison: { min: 40, max: 75, unit: "%", visualMin: 0, visualMax: 100 }
};

export function fmt0(x) {
    return Number.isFinite(x) ? Math.round(x) : 0;
}

export function formatBaselineRange(baseline) {
    if (!baseline) return "—";
    const { min, max, unit = "" } = baseline;
    const safeMin = Number.isFinite(min) ? fmt0(min) : "—";
    const safeMax = Number.isFinite(max) ? fmt0(max) : "—";
    return `${safeMin}–${safeMax}${unit}`;
}

export function renderGauge(value, baseline, label) {
    const { min, max, unit } = baseline;
    if (!Number.isFinite(value)) {
        return `<span class="gauge is-na" role="meter" aria-valuemin="${min}" aria-valuemax="${max}" aria-valuenow="0" aria-label="${escapeAttr(label)}">—</span>`;
    }
    const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
    const hasVisualRange = Number.isFinite(baseline.visualMin)
        && Number.isFinite(baseline.visualMax)
        && baseline.visualMax > baseline.visualMin;
    const visualMin = hasVisualRange ? baseline.visualMin : min;
    const visualMax = hasVisualRange ? baseline.visualMax : max;
    const pct = clamp((value - visualMin) / Math.max(1e-6, (visualMax - visualMin)) * 100, 0, 100);
    const cls = (value >= min && value <= max) ? "is-ok" : (value < min ? "is-low" : "is-high");
    const valueDisplay = (unit === "%") ? fmt0(value) : fmt1(value); // NOTE: input value for % MUST be 0-100
    const title = `${label}: ${valueDisplay}${unit || ""} · target ${fmt0(min)}–${fmt0(max)}${unit || ""}`;
    return `
    <span class="gauge ${cls}" title="${escapeAttr(title)}" role="meter" aria-valuemin="${min}" aria-valuemax="${max}" aria-valuenow="${fmt0(value)}" aria-label="${escapeAttr(label)}">
      <span class="gauge__track">
        <span class="gauge__range" style="left:${pct}%;"></span>
      </span>
      <span class="gauge__val">${valueDisplay}${unit || ""}</span>
    </span>`;
}

