import {
  MAX_WHOLE_SEC,
  STREAM_HOP_S,
  STREAM_WIN_CAND,
} from "./constants.js";

const STREAM_STRATEGY_DEFAULT = Object.freeze({
  hop: STREAM_HOP_S,
  wins: [...STREAM_WIN_CAND],
  label: "",
});

export function pickStreamStrategy(durationSec, { currentDevice, t }) {
  if (!Number.isFinite(durationSec) || durationSec <= MAX_WHOLE_SEC) {
    return STREAM_STRATEGY_DEFAULT;
  }

  const dedupeWins = (wins) => {
    const seen = new Set();
    const out = [];
    for (const w of wins) {
      const key = w.toFixed(2);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(w);
    }
    return out;
  };

  const gpuWins = dedupeWins([18, 12, ...STREAM_WIN_CAND, 4]);
  const gpuWinsLong = dedupeWins([24, 18, 12, ...STREAM_WIN_CAND, 4]);
  const wasmWins = dedupeWins([12, ...STREAM_WIN_CAND, 4]);

  if (currentDevice === "webgpu") {
    if (durationSec >= 600) {
      return { hop: 6, wins: gpuWinsLong, label: t("status.strategyGpu6") };
    }
    return { hop: 4, wins: gpuWins, label: t("status.strategyGpu4") };
  }

  if (durationSec >= 420) {
    return { hop: 4, wins: wasmWins, label: t("status.strategyCpu4") };
  }

  if (durationSec >= 240) {
    return { hop: 3.5, wins: wasmWins, label: t("status.strategyCpu35") };
  }

  return STREAM_STRATEGY_DEFAULT;
}
