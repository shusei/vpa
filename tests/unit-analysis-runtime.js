import assert from "node:assert/strict";
import test from "node:test";

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: { ONNX_MODEL_ID: "test-model" },
});
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { userAgent: "node-test" },
  writable: true,
});

const { analyzeStreamed, analyzeWhole, runStreamedWithWindow } = await import("../assets/js/analysis-core.js");
const { createAnalysisFlowController } = await import("../assets/js/analysis-flow.js");
const { ensurePipeline } = await import("../assets/js/model-core.js");
const { pickStreamStrategy } = await import("../assets/js/stream-strategy.js");

const translate = (key) => key;

test("model runtime selects WASM when WebGPU is unavailable", async () => {
  globalThis.navigator = { userAgent: "node-test" };
  let selectedDevice = null;
  let classifier = null;
  const created = async () => [];

  const result = await ensurePipeline({
    getClf: () => classifier,
    setClf: (value) => { classifier = value; },
    setCurrentDevice: (value) => { selectedDevice = value; },
  }, {
    modelId: "model",
    pipeline: async (_task, _model, options) => {
      assert.equal(options.device, "wasm");
      return created;
    },
    setStatus: () => { },
    t: translate,
  });

  assert.equal(result, created);
  assert.equal(selectedDevice, "wasm");
});

test("model runtime selects WebGPU when navigator.gpu exists", async () => {
  globalThis.navigator = { gpu: {}, userAgent: "node-test" };
  let selectedDevice = null;

  await ensurePipeline({
    getClf: () => null,
    setClf: () => { },
    setCurrentDevice: (value) => { selectedDevice = value; },
  }, {
    modelId: "model",
    pipeline: async (_task, _model, options) => {
      assert.equal(options.device, "webgpu");
      return async () => [];
    },
    setStatus: () => { },
    t: translate,
  });

  assert.equal(selectedDevice, "webgpu");
});

test("stream strategy covers every WASM duration boundary", () => {
  const pick = (durationSec) => pickStreamStrategy(durationSec, { currentDevice: "wasm", t: translate });
  assert.deepEqual(pick(Number.NaN), { hop: 3, wins: [12, 8, 6, 4], label: "" });
  assert.deepEqual(pick(150), { hop: 3, wins: [12, 8, 6, 4], label: "" });
  assert.deepEqual(pick(150.001), { hop: 3, wins: [12, 8, 6, 4], label: "" });
  assert.deepEqual(pick(239.999), { hop: 3, wins: [12, 8, 6, 4], label: "" });
  assert.deepEqual(pick(240), { hop: 3.5, wins: [12, 8, 6, 4], label: "status.strategyCpu35" });
  assert.equal(pick(419.999).hop, 3.5);
  assert.deepEqual(pick(420), { hop: 4, wins: [12, 8, 6, 4], label: "status.strategyCpu4" });
});

test("stream strategy covers every WebGPU duration boundary", () => {
  const pick = (durationSec) => pickStreamStrategy(durationSec, { currentDevice: "webgpu", t: translate });
  assert.deepEqual(pick(150), { hop: 3, wins: [12, 8, 6, 4], label: "" });
  assert.deepEqual(pick(150.001), { hop: 4, wins: [18, 12, 8, 6, 4], label: "status.strategyGpu4" });
  assert.equal(pick(599.999).hop, 4);
  assert.deepEqual(pick(600), { hop: 6, wins: [24, 18, 12, 8, 6, 4], label: "status.strategyGpu6" });
});

test("whole-file OOM switches to streamed analysis", async () => {
  const calls = [];
  await analyzeWhole({ float32: new Float32Array(16), sr: 16, durationSec: 1, token: 1 }, {
    analyzeStreamed: async (...args) => calls.push(args),
    ensurePipeline: async () => async () => { throw new Error("out of memory"); },
    fmtSec: String,
    isAnalysisActive: () => true,
    isOOMError: () => true,
    meter: null,
    render: () => { },
    setStatus: () => { },
    startHeartbeat: () => { },
    stopHeartbeat: () => { },
    t: translate,
    toMap: () => ({}),
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0][3], "status.analyzeWholeOOM");
  assert.equal(calls[0][4], 1);
});

test("streamed OOM downshifts until a window succeeds", async () => {
  const attempted = [];
  const statuses = [];
  await analyzeStreamed({ float32: new Float32Array(16), sr: 16, durationSec: 700, reason: "long", token: 1 }, {
    ensurePipeline: async () => async () => [],
    isAnalysisActive: () => true,
    isOOMError: (error) => error.message === "oom",
    meter: null,
    pickStreamStrategy: () => ({ hop: 6, wins: [24, 18, 12, 8, 6, 4], label: "gpu" }),
    runStreamedWithWindow: async (_model, _audio, _sr, _duration, win) => {
      attempted.push(win);
      if (attempted.length < 3) throw new Error("oom");
    },
    setStatus: (value) => statuses.push(value),
    t: translate,
  });

  assert.deepEqual(attempted, [24, 18, 12]);
  assert.equal(statuses.includes("status.analyzeStreamFailed"), false);
});

test("streamed analysis stops retrying after a non-OOM error", async () => {
  const attempted = [];
  const statuses = [];
  await analyzeStreamed({ float32: new Float32Array(16), sr: 16, durationSec: 700, reason: "long", token: 1 }, {
    ensurePipeline: async () => async () => [],
    isAnalysisActive: () => true,
    isOOMError: () => false,
    meter: null,
    pickStreamStrategy: () => ({ hop: 6, wins: [24, 18, 12], label: "gpu" }),
    runStreamedWithWindow: async (_model, _audio, _sr, _duration, win) => {
      attempted.push(win);
      throw new Error("decoder failure");
    },
    setStatus: (value) => statuses.push(value),
    t: translate,
  });

  assert.deepEqual(attempted, [24]);
  assert.equal(statuses.at(-1), "status.analyzeStreamFailed");
});

test("streamed analysis reports failure after every OOM window is exhausted", async () => {
  const attempted = [];
  const statuses = [];
  await analyzeStreamed({ float32: new Float32Array(16), sr: 16, durationSec: 700, reason: "long", token: 1 }, {
    ensurePipeline: async () => async () => [],
    isAnalysisActive: () => true,
    isOOMError: () => true,
    meter: null,
    pickStreamStrategy: () => ({ hop: 6, wins: [24, 18, 12, 8, 6, 4], label: "gpu" }),
    runStreamedWithWindow: async (_model, _audio, _sr, _duration, win) => {
      attempted.push(win);
      throw new Error("oom");
    },
    setStatus: (value) => statuses.push(value),
    t: translate,
  });

  assert.deepEqual(attempted, [24, 18, 12, 8, 6, 4]);
  assert.equal(statuses.at(-1), "status.analyzeStreamFailed");
});

test("stream aggregation uses duration-weighted log odds", async () => {
  const renders = [];
  let call = 0;
  await runStreamedWithWindow({
    model: async () => {
      call += 1;
      return call === 1
        ? [{ label: "female", score: 0.8 }, { label: "male", score: 0.2 }]
        : [{ label: "female", score: 0.2 }, { label: "male", score: 0.8 }];
    },
    float32: new Float32Array(8),
    sr: 2,
    durationSec: 4,
    WIN_S: 2,
    HOP_S: 2,
    reason: "test",
    token: 1,
  }, {
    clamp01: (value) => Math.max(0, Math.min(1, value)),
    fmtSec: String,
    isAnalysisActive: () => true,
    microYield: async () => { },
    render: (female, male) => renders.push([female, male]),
    setStatus: () => { },
    startHeartbeat: () => { },
    stopHeartbeat: () => { },
    t: translate,
    toMap: (rows) => Object.fromEntries(rows.map((row) => [row.label, row.score])),
  });

  assert.equal(renders.length, 3);
  assert.ok(Math.abs(renders.at(-1)[0] - 0.5) < 1e-12);
  assert.ok(Math.abs(renders.at(-1)[1] - 0.5) < 1e-12);
});

test("analysis flow chooses whole and streamed paths at 150 seconds", async () => {
  const calls = [];
  let nextDuration = 150;
  let nextToken = 0;
  const controller = createAnalysisFlowController({
    analyzeStreamed: async (...args) => calls.push(["streamed", ...args]),
    analyzeWhole: async (...args) => calls.push(["whole", ...args]),
    decodeSmartToFloat32: async () => ({
      float32: new Float32Array(1),
      sr: 16_000,
      durationSec: nextDuration,
    }),
    finishAnalysisRun: () => { },
    finishStreamStats: () => calls.push(["stats"]),
    fmtSec: String,
    isAnalysisActive: () => true,
    MAX_WHOLE_SEC: 150,
    maybeApplyAdaptiveVAD: () => null,
    microYield: async () => { },
    notifyInferenceListeners: () => { },
    offlineExtractStreamMetrics: () => { },
    setPlaybackSource: () => { },
    setStatus: () => { },
    startAnalysisRun: () => ++nextToken,
    t: translate,
    TARGET_SR: 16_000,
    updatePlaybackAvailability: () => { },
    WARN_LONG_SEC: 180,
  });

  await controller.handleFileOrBlob(new Blob(), "upload");
  nextDuration = 150.001;
  await controller.handleFileOrBlob(new Blob(), "upload");

  assert.equal(calls[0][0], "whole");
  assert.equal(calls[2][0], "streamed");
});

test("analysis flow captures extensions from the final VAD-selected audio before inference", async () => {
  const calls = [];
  const original = new Float32Array(16_000);
  const selected = new Float32Array(8_000);
  const controller = createAnalysisFlowController({
    analyzeStreamed: async () => { },
    analyzeWhole: async (samples) => calls.push(["whole", samples]),
    decodeSmartToFloat32: async () => ({
      float32: original,
      sr: 16_000,
      durationSec: 1,
    }),
    finishAnalysisRun: () => { },
    finishStreamStats: () => calls.push(["stats"]),
    fmtSec: String,
    isAnalysisActive: () => true,
    MAX_WHOLE_SEC: 150,
    maybeApplyAdaptiveVAD: () => ({
      arr: selected,
      keptSec: 0.5,
      used: true,
    }),
    microYield: async () => { },
    notifyInferenceListeners: () => { },
    offlineExtractStreamMetrics: (samples, sampleRate, append) => {
      calls.push(["offline", samples, sampleRate, append]);
    },
    runDecodedAudioAnalyzers: async (context) => {
      calls.push(["extensions", context]);
      return { fixture: { ready: true } };
    },
    setAnalysisExtensions: (value) => calls.push(["capture", value]),
    setPlaybackSource: () => { },
    setStatus: () => { },
    startAnalysisRun: () => 1,
    t: translate,
    TARGET_SR: 16_000,
    updatePlaybackAvailability: () => { },
    WARN_LONG_SEC: 180,
  });

  await controller.handleFileOrBlob(new Blob(), "recording");

  assert.deepEqual(calls.map(([name]) => name), [
    "offline",
    "offline",
    "extensions",
    "capture",
    "whole",
    "stats",
  ]);
  assert.equal(calls[2][1].samples, selected);
  assert.equal(calls[2][1].durationSec, 0.5);
  assert.equal(calls[2][1].sampleRate, 16_000);
  assert.equal(calls[2][1].source, "recording");
  assert.deepEqual(calls[3][1], { fixture: { ready: true } });
});

test("analysis flow warns only above the 180 second boundary", async () => {
  async function run(durationSec) {
    const statuses = [];
    let yields = 0;
    const controller = createAnalysisFlowController({
      analyzeStreamed: async () => { },
      analyzeWhole: async () => { },
      decodeSmartToFloat32: async () => ({
        float32: new Float32Array(1),
        sr: 16_000,
        durationSec,
      }),
      finishAnalysisRun: () => { },
      finishStreamStats: () => { },
      fmtSec: String,
      isAnalysisActive: () => true,
      MAX_WHOLE_SEC: 150,
      maybeApplyAdaptiveVAD: () => null,
      microYield: async () => { yields += 1; },
      notifyInferenceListeners: () => { },
      offlineExtractStreamMetrics: () => { },
      setPlaybackSource: () => { },
      setStatus: (value) => statuses.push(value),
      startAnalysisRun: () => 1,
      t: translate,
      TARGET_SR: 16_000,
      updatePlaybackAvailability: () => { },
      WARN_LONG_SEC: 180,
    });
    await controller.handleFileOrBlob(new Blob(), "upload");
    return { statuses, yields };
  }

  const atBoundary = await run(180);
  const aboveBoundary = await run(180.001);
  assert.equal(atBoundary.statuses.includes("status.warnLong"), false);
  assert.equal(atBoundary.yields, 0);
  assert.equal(aboveBoundary.statuses.includes("status.warnLong"), true);
  assert.equal(aboveBoundary.yields, 1);
});
