export function bandOf(medHz, deps = {}) {
  const { t, PS_MAX_HZ } = deps;
  if (!isFinite(medHz)) return "—";
  if (medHz < 85) return t("pitchBands.low");
  if (medHz < 165) return t("pitchBands.male");
  if (medHz < 180) return t("pitchBands.overlap");
  if (medHz < 310) return t("pitchBands.female");
  if (medHz < 450) return t("pitchBands.high");
  if (medHz <= PS_MAX_HZ) return t("pitchBands.falsetto");
  return t("pitchBands.outOfRange");
}

export function isDivergent(medHz, pf, pm) {
  if (!isFinite(medHz)) return false;
  if (medHz >= 165 && medHz < 180) return false;
  if ((medHz >= 180 && pm >= 0.60) || (medHz <= 165 && pf >= 0.60)) return true;
  return false;
}

export function escapeAttr(value) {
  if (value == null) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
