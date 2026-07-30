import { createAdvancedSectionController } from "./advanced-section.js";
import { createAdvancedSummaryRenderer } from "./advanced-summary-render.js";
import { createFocusHelpers } from "./focus-insights.js";
import {
  escapeAttr as sharedEscapeAttr,
} from "./summary-helpers.js";
import { createSummaryUIHelpers } from "./summary-ui-helpers.js";

const ADVANCED_MODE_KEY = "ui:advancedMode";
const ADV_DETAILS_KEY_PREFIX = "ui:advOpen:";

// Baseline ranges（保守預設，可按你的語料微調）
const BASELINES = {
  f1: { min: 170, max: 420, unit: "Hz" },
  f2: { min: 1450, max: 2750, unit: "Hz" },
  f3: { min: 2400, max: 3400, unit: "Hz" },
  tilt: { min: -1, max: 8, unit: "dB", visualMin: -8, visualMax: 10 },
  breath: { min: 8, max: 18, unit: "%", visualMin: 0, visualMax: 60 },
  syll: { min: 3.2, max: 5.2, unit: "syll/s" },
  wpm: { min: 120, max: 180, unit: "wpm" },
  liaison: { min: 40, max: 75, unit: "%", visualMin: 0, visualMax: 100 },
};

export function createAdvancedRuntime(deps) {
  const {
    fmt1,
    getSummaryText,
    onLocaleChange,
    t,
    volumeDisplayMode,
  } = deps;

  const advancedSectionController = createAdvancedSectionController({
    detailsKeyPrefix: ADV_DETAILS_KEY_PREFIX,
    modeKey: ADVANCED_MODE_KEY,
    onLocaleChange,
    t,
  });

  function summaryString(path, params) {
    return t(`summary.${path}`, params);
  }

  function escapeAttr(value) {
    return sharedEscapeAttr(value);
  }

  const summaryUIHelpers = createSummaryUIHelpers({
    escapeAttr,
    fmt1,
  });

  const {
    escapeHtml,
    formatBaselineRange,
    renderGauge,
  } = summaryUIHelpers;

  const advancedSummaryRenderer = createAdvancedSummaryRenderer({
    BASELINES,
    escapeAttr,
    escapeHtml,
    fmt1,
    formatBaselineRange,
    getAdvancedMode: () => advancedSectionController.getAdvancedMode(),
    getDetailsOpen: (id, fallbackOpen) => advancedSectionController.getDetailsOpen(id, fallbackOpen),
    getSummaryText,
    renderGauge,
    summaryString,
    t,
  });

  const focusHelpers = createFocusHelpers({
    fmt1,
    getSummaryText,
    summaryString,
    t,
  });

  function applyDbCalibration(rawDb) {
    return { value: rawDb, mode: volumeDisplayMode };
  }

  return {
    advancedSectionController,
    advancedSummaryRenderer,
    applyDbCalibration,
    escapeAttr,
    focusHelpers,
    summaryString,
  };
}
