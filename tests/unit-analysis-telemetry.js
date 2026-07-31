import assert from "node:assert/strict";
import { createAnalysisTelemetryController } from "../assets/js/analysis-telemetry.js";

const calls = [];
const previousWindow = globalThis.window;
globalThis.window = {
  gtag: (...args) => calls.push(args),
};

try {
  const controller = createAnalysisTelemetryController();
  controller.setLatestAnalysisExport({
    meta: {
      source: "recording",
    },
    pitch: {
      spreadHz: 48.37,
      stats: {
        med: 231.42,
      },
    },
    summary: {
      voicedRatio: 0.8734,
    },
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], "event");
  assert.equal(calls[0][1], "analysis_completed");
  assert.deepEqual(calls[0][2], {
    event_category: "analysis",
    event_label: "recording",
    source: "recording",
  });
  assert.equal(window.vpaLatestAnalysis.pitch.stats.med, 231.42);
  assert.deepEqual(window.__vpaLastGAEvent.payload, calls[0][2]);

  controller.setLatestAnalysisExport(null);
  assert.equal(window.vpaLatestAnalysis, null);
} finally {
  if (previousWindow === undefined) {
    delete globalThis.window;
  } else {
    globalThis.window = previousWindow;
  }
}

console.log("analysis telemetry privacy checks passed");
