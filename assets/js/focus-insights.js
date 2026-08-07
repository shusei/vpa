const PRACTICE_CATEGORY_BRIDGE = Object.freeze({
  pitch: "greet",
  resonance: "work",
  breath: "life",
  pace: "assist",
  clarity: "cs",
});

export function createFocusHelpers(deps) {
  const {
    fmt1,
    getSummaryText,
    summaryString,
    t,
  } = deps;

  function buildFocusInsights(context = {}) {
    const summaryText = getSummaryText?.() || {};
    const heading = summaryText?.focus?.heading || t("summary.focus.heading");
    const empty = summaryText?.focus?.empty || t("summary.focus.empty");
    const severityLabels = summaryText?.focus?.severity || {};
    const ctaLabels = summaryText?.focus?.cta || {};
    const severityRank = { high: 3, medium: 2, low: 1 };
    const seenKeys = new Set();
    const entries = [];

    const severityText = (level) => severityLabels[level] || t(`summary.focus.severity.${level}`);
    const practiceLabel = (key) => t(`summary.focus.cta.${key}`) || ctaLabels[key] || "Practice";
    const practiceCategoryFor = (key) => PRACTICE_CATEGORY_BRIDGE[key] || null;

    const push = (key, severity, score, params = {}, practiceKey = null) => {
      if (!key || seenKeys.has(key)) return;
      const raw = summaryString(`focus.items.${key}`, params);
      if (!raw || raw === `summary.focus.items.${key}`) return;
      const severityLabel = severityText(severity);
      const practiceCategory = practiceKey ? practiceCategoryFor(practiceKey) : null;
      const ctaLabel = practiceCategory ? practiceLabel(practiceKey) : null;
      entries.push({
        key,
        title: raw,
        severity,
        severityLabel,
        practiceCategory,
        ctaLabel,
        score,
      });
      seenKeys.add(key);
    };

    const {
      band,
      stabilityKey,
      stabilityLabel,
      spread,
      snr,
      snrDisplay,
      snrKey,
      snrLabel,
      diverge,
      trendLabel,
      advSummary,
      voicedHintKey,
      voicedHintLabel,
    } = context;

    if (diverge) {
      push("divergence", "high", 95, { band, trend: trendLabel }, "resonance");
    }

    if (snrKey === "noisy") {
      const veryLow = Number.isFinite(snr) && snr < 8;
      push("noisy", veryLow ? "high" : "medium", veryLow ? 90 : 75, { snrLabel, snrDisplay }, "clarity");
    }

    if (Number.isFinite(spread) && spread >= 40 && stabilityLabel && stabilityLabel !== "—") {
      const spreadDisplay = fmt1(spread);
      if (stabilityKey === "wide") {
        push("pitchWide", "high", 85, { spread: spreadDisplay, stability: stabilityLabel }, "pitch");
      } else {
        push("pitchModerate", "medium", 70, { spread: spreadDisplay, stability: stabilityLabel }, "pitch");
      }
    }

    const breathKey = advSummary?.breathinessKey;
    const breathLabel = breathKey ? t(`analysis.breathiness.${breathKey}.label`) : (advSummary?.breathinessLabel || "");
    if (breathLabel && breathKey) {
      if (breathKey === "tooAiry" || breathKey === "airy") {
        const severity = breathKey === "tooAiry" ? "high" : "medium";
        const score = breathKey === "tooAiry" ? 82 : 68;
        push("breathinessAiry", severity, score, { label: breathLabel }, "breath");
      } else if (breathKey === "dense") {
        push("breathinessDense", "medium", 66, { label: breathLabel }, "breath");
      }
    }

    const vowelRatio = Number(advSummary?.vowelFocusRatio);
    const vowelKey = Number.isFinite(vowelRatio) ? (vowelRatio >= 0.5 ? "strong" : vowelRatio >= 0.3 ? "medium" : "weak") : null;
    const vowelLabel = vowelKey ? t(`analysis.vowelFocus.${vowelKey}.label`) : (advSummary?.vowelLabel || null);
    if (Number.isFinite(vowelRatio) && vowelLabel && vowelRatio < 0.32) {
      const severity = vowelRatio < 0.2 ? "high" : "medium";
      const score = vowelRatio < 0.2 ? 80 : 69;
      push("vowelWeak", severity, score, { label: vowelLabel }, "resonance");
    }

    const speechRate = advSummary?.speechRate;
    const speechRateLabel = speechRate?.key && speechRate.key !== "insufficient"
      ? t(`analysis.speechRate.${speechRate.key}.label`)
      : (speechRate?.label || null);
    if (speechRateLabel && speechRate?.key) {
      if (speechRate.key === "fast") {
        push("speechFast", "medium", 64, { label: speechRateLabel }, "pace");
      } else if (speechRate.key === "tooSlow") {
        push("speechSlow", "medium", 62, { label: speechRateLabel }, "pace");
      }
    }

    if (voicedHintKey && voicedHintLabel) {
      const severity = voicedHintKey === "low" ? "high" : "medium";
      const score = voicedHintKey === "low" ? 72 : 60;
      push("voicedLow", severity, score, { label: voicedHintLabel }, "clarity");
    }

    const brightnessKey = advSummary?.brightnessKey;
    const brightnessLabel = advSummary?.brightnessLabel;
    if (brightnessKey === "sharp" && brightnessLabel) {
      push("brightnessSharp", "medium", 63, { label: brightnessLabel }, "resonance");
    }

    entries.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return (severityRank[b.severity] || 0) - (severityRank[a.severity] || 0);
    });

    const items = entries.slice(0, 3).map(({ score, ...rest }) => rest);
    return { heading, empty, items };
  }

  function renderFocusBlock(focus) {
    const heading = focus?.heading || t("summary.focus.heading");
    const empty = focus?.empty || t("summary.focus.empty");
    const items = Array.isArray(focus?.items) ? focus.items : [];
    if (!items.length) {
      return `
      <div class="focus-block">
        <h3 class="focus-heading">${heading}</h3>
        <div class="focus-empty">${empty}</div>
      </div>
    `;
    }
    const listHtml = items.map((item) => {
      const ctaHtml = "";
      return `
      <li class="focus-item focus-item--${item.severity}">
        <div class="focus-chip">
          <span class="focus-severity">${item.severityLabel}</span>
          <span class="focus-title">${item.title}</span>
        </div>
        ${ctaHtml}
      </li>
    `;
    }).join("");

    return `
    <div class="focus-block">
      <h3 class="focus-heading">${heading}</h3>
      <ul class="focus-list">
        ${listHtml}
      </ul>
    </div>
  `;
  }

  return {
    buildFocusInsights,
    renderFocusBlock,
  };
}
