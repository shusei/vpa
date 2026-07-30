import { prepareAnalysisExport as sharedPrepareAnalysisExport } from "./analysis-export.js";

export function createAnalysisTelemetryController() {
  let latestAnalysisExport = null;

  function setLatestAnalysisExport(payload) {
    try {
      if (payload == null) {
        latestAnalysisExport = null;
        if (typeof window !== "undefined") window.vpaLatestAnalysis = null;
        return;
      }
      const sanitized = sharedPrepareAnalysisExport(payload);
      latestAnalysisExport = sanitized;
      if (typeof window !== "undefined") {
        window.vpaLatestAnalysis = sanitized;
        if (typeof window.gtag === "function") {
          try {
            const pitchMed = Number(sanitized?.pitch?.stats?.med);
            const pitchSpread = Number(sanitized?.pitch?.spreadHz);
            const voicedRatio = Number(sanitized?.summary?.voicedRatio);
            const source = sanitized?.source || sanitized?.meta?.source || "unknown";
            const eventPayload = {
              event_category: "analysis",
              event_label: String(source),
              source: String(source),
            };
            if (Number.isFinite(pitchMed)) eventPayload.pitch_median_hz = Number(pitchMed.toFixed(2));
            if (Number.isFinite(pitchSpread)) eventPayload.pitch_spread_hz = Number(pitchSpread.toFixed(2));
            if (Number.isFinite(voicedRatio)) eventPayload.voiced_ratio = Number(voicedRatio.toFixed(4));
            window.gtag("event", "analysis_completed", eventPayload);
            window.__vpaLastGAEvent = {
              name: "analysis_completed",
              payload: eventPayload,
              at: Date.now(),
            };
          } catch (eventErr) {
            console.warn("[ga4] analysis_completed event failed", eventErr);
          }
        }
      }
    } catch (err) {
      console.error("[export] capture failed", err);
      latestAnalysisExport = null;
    }
  }

  function getLatestAnalysisExport() {
    return latestAnalysisExport;
  }

  return {
    getLatestAnalysisExport,
    setLatestAnalysisExport,
  };
}
