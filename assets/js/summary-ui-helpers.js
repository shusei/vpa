export function createSummaryUIHelpers(deps) {
  const {
    escapeAttr,
    fmt1,
  } = deps;

  function fmt0(x) {
    return Number.isFinite(x) ? Math.round(x) : 0;
  }

  function renderGauge(value, baseline, label) {
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
    const valueDisplay = (unit === "%") ? fmt0(value) : fmt1(value);
    const title = `${label}: ${valueDisplay}${unit || ""} · target ${fmt0(min)}–${fmt0(max)}${unit || ""}`;
    return `
    <span class="gauge ${cls}" title="${escapeAttr(title)}" role="meter" aria-valuemin="${min}" aria-valuemax="${max}" aria-valuenow="${fmt0(value)}" aria-label="${escapeAttr(label)}">
      <span class="gauge__track">
        <span class="gauge__range" style="left:${pct}%;"></span>
      </span>
      <span class="gauge__val">${valueDisplay}${unit || ""}</span>
    </span>`;
  }

  function escapeHtml(input) {
    if (input == null) return "";
    return String(input)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatBaselineRange(baseline) {
    if (!baseline) return "—";
    const { min, max, unit = "" } = baseline;
    const safeMin = Number.isFinite(min) ? fmt0(min) : "—";
    const safeMax = Number.isFinite(max) ? fmt0(max) : "—";
    return `${safeMin}–${safeMax}${unit}`;
  }

  return {
    escapeHtml,
    formatBaselineRange,
    renderGauge,
  };
}
