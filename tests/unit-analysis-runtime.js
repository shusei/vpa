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
const { estimateAcousticPresentation } = await import("../assets/js/acoustic-fast-path.js");
const { createAnalysisEngineBridge } = await import("../assets/js/analysis-engine-bridge.js");
const { createAnalysisSessionController } = await import("../assets/js/analysis-session.js");
const { ensurePipeline } = await import("../assets/js/model-core.js");
const { detectEmbeddedBrowser, openExternalBrowser } = await import("../assets/js/embedded-browser.js");
const {
  mobileInferenceMaxSec,
  selectRepresentativeSamples,
  shouldUseEmbeddedAcousticFastPath,
  shouldUseMobileFastPath,
} = await import("../assets/js/inference-sampling.js");
const { pickStreamStrategy } = await import("../assets/js/stream-strategy.js");

const translate = (key) => key;

test("completed inference events carry the integrated presentation result", () => {
  const session = createAnalysisSessionController();
  const analysis = { analysisId: 42 };
  const presentation = { ready: true, score: 73, version: "advanced-beta-1" };
  let received = null;

  session.onInferenceDone((payload) => {
    received = payload;
  });
  session.notifyInferenceListeners(0.64, 0.36, analysis, presentation);

  assert.deepEqual(received, {
    analysis,
    pf: 0.64,
    pm: 0.36,
    presentation,
  });
});

test("decoded audio analyzers share one registry across cache-busted module URLs", async () => {
  const alternateModule = await import("../assets/js/analysis-flow.js?registry-identity-test");
  alternateModule.resetDecodedAudioAnalyzersForTest();
  const unregister = alternateModule.registerDecodedAudioAnalyzer("voice-age-v2", ({ durationSec }) => ({
    durationSec,
  }));

  assert.deepEqual(await alternateModule.runDecodedAudioAnalyzers({ durationSec: 4 }), {
    "voice-age-v2": { durationSec: 4 },
  });
  const primaryModule = await import("../assets/js/analysis-flow.js");
  assert.deepEqual(await primaryModule.runDecodedAudioAnalyzers({ durationSec: 7 }), {
    "voice-age-v2": { durationSec: 7 },
  });

  unregister();
  alternateModule.resetDecodedAudioAnalyzersForTest();
});

test("detects app webviews without flagging real mobile Safari", () => {
  const line = detectEmbeddedBrowser({
    platform: "Linux armv8l",
    userAgent: "Mozilla/5.0 (Linux; Android 14; wv) AppleWebKit/537.36 Version/4.0 Chrome/124.0 Mobile Safari/537.36 Line/14.9.0",
  });
  assert.deepEqual(
    { app: line.app, embedded: line.embedded, platform: line.platform },
    { app: "line", embedded: true, platform: "android" },
  );

  const safari = detectEmbeddedBrowser({
    platform: "iPhone",
    userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 Version/17.4 Mobile/15E148 Safari/604.1",
  });
  assert.equal(safari.embedded, false);
  assert.equal(safari.platform, "ios");
});

test("detects social app browsers used on Android and iPhone", () => {
  const cases = [
    ["facebook", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Mobile FBAN/EMA;FBAV/470.0"],
    ["instagram", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Mobile Instagram 335.0.0"],
    ["threads", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Mobile Barcelona 335.0.0"],
    ["tiktok", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Mobile TikTok 35.2.0"],
    ["x", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Mobile TwitterAndroid"],
    ["x", "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile Twitter/10.99"],
    ["x", "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile X for iPhone"],
  ];
  cases.forEach(([expectedApp, userAgent]) => {
    const result = detectEmbeddedBrowser({ platform: "mobile", userAgent });
    assert.equal(result.embedded, true);
    assert.equal(result.app, expectedApp);
  });
});
test("external browser opening uses supported LINE and Android handoffs without faking iOS Safari", async () => {
  let liffOptions = null;
  const liffResult = await openExternalBrowser({
    context: { app: "line", embedded: true, platform: "ios" },
    liffLike: {
      isInClient: () => true,
      openWindow: (options) => { liffOptions = options; },
    },
    locationLike: { href: "https://example.com/dev.html" },
  });
  assert.equal(liffResult.method, "liff");
  assert.deepEqual(liffOptions, { external: true, url: "https://example.com/dev.html" });

  let lineAssigned = "";
  const lineResult = await openExternalBrowser({
    context: { app: "line", embedded: true, platform: "ios" },
    liffLike: null,
    locationLike: {
      assign: (value) => { lineAssigned = value; },
      href: "https://example.com/dev.html?mode=quick#result",
    },
  });
  assert.equal(lineResult.method, "line-external-query");
  const lineUrl = new URL(lineAssigned);
  assert.equal(lineUrl.searchParams.get("openExternalBrowser"), "1");
  assert.equal(lineUrl.searchParams.get("mode"), "quick");
  assert.equal(lineUrl.hash, "#result");

  const rejectedRetry = await openExternalBrowser({
    context: { app: "line", embedded: true, platform: "ios" },
    liffLike: null,
    locationLike: { href: lineAssigned },
  });
  assert.equal(rejectedRetry.method, "line-external-unavailable");
  assert.equal(rejectedRetry.opened, false);

  let assigned = "";
  const androidResult = await openExternalBrowser({
    context: { app: "instagram", embedded: true, platform: "android" },
    liffLike: null,
    locationLike: { assign: (value) => { assigned = value; }, href: "https://example.com/dev.html" },
  });
  assert.equal(androidResult.method, "android-intent");
  assert.ok(assigned.startsWith("intent://example.com/dev.html#Intent;"));
  assert.ok(assigned.includes("package=com.android.chrome"));

  let iosWindowOpened = false;
  const iosResult = await openExternalBrowser({
    context: { app: "x", embedded: true, platform: "ios" },
    liffLike: null,
    locationLike: { href: "https://example.com/#vpa-challenge=abc" },
    windowLike: {
      open: () => {
        iosWindowOpened = true;
      },
    },
  });
  assert.equal(iosResult.method, "ios-app-menu-required");
  assert.equal(iosResult.opened, false);
  assert.equal(iosWindowOpened, false);
});
test("embedded inference samples start, middle, and end without changing the recording", () => {
  const originalSamples = Float32Array.from({ length: 200 }, (_value, index) => index);
  const selected = selectRepresentativeSamples(originalSamples, 10, { maxDurationSec: 8 });

  assert.equal(selected.used, true);
  assert.equal(selected.durationSec, 8);
  assert.equal(selected.samples.length, 80);
  assert.equal(selected.samples[0], 0);
  assert.ok(selected.samples.includes(100));
  assert.equal(selected.samples.at(-1), 199);
  assert.equal(originalSamples.length, 200);
});

test("mobile browsers use device-appropriate representative inference windows", () => {
  assert.equal(mobileInferenceMaxSec({ app: "line", embedded: true, platform: "android" }), 4.5);
  assert.equal(mobileInferenceMaxSec({ app: "instagram", embedded: true, platform: "ios" }), 6);
  assert.equal(mobileInferenceMaxSec({ app: "", embedded: false, platform: "ios" }), 8);
  assert.equal(shouldUseMobileFastPath({ embedded: false, platform: "ios" }), true);
  assert.equal(shouldUseMobileFastPath({ embedded: false, platform: "android" }), true);
  assert.equal(shouldUseMobileFastPath({ embedded: false, platform: "desktop" }), false);
});

test("embedded acoustic fast path is limited to X and Threads webviews", () => {
  assert.equal(shouldUseEmbeddedAcousticFastPath({ app: "x", embedded: true }), true);
  assert.equal(shouldUseEmbeddedAcousticFastPath({ app: "threads", embedded: true }), true);
  assert.equal(shouldUseEmbeddedAcousticFastPath({ app: "line", embedded: true }), false);
  assert.equal(shouldUseEmbeddedAcousticFastPath({ app: "x", embedded: false }), false);
});

test("local acoustic estimate separates representative feminine and masculine features", () => {
  function makeStore({ pitch, f2, f3, energy }) {
    const frameCount = 120;
    return {
      energy: Array.from({ length: frameCount }, () => energy),
      formants: Array.from({ length: frameCount }, () => [600, f2, f3]),
      frameSec: 0.05,
      pitchConfidence: Array(frameCount).fill(0.92),
      pitchProcessed: Array(frameCount).fill(pitch),
      voiced: Array(frameCount).fill(true),
    };
  }

  const feminine = estimateAcousticPresentation(makeStore({
    energy: [0.15, 0.55, 0.3],
    f2: 2400,
    f3: 3600,
    pitch: 220,
  }));
  const masculine = estimateAcousticPresentation(makeStore({
    energy: [0.75, 0.2, 0.05],
    f2: 1300,
    f3: 2400,
    pitch: 130,
  }));
  const empty = estimateAcousticPresentation({});

  assert.equal(feminine.ready, true);
  assert.ok(feminine.feminine > 0.7);
  assert.equal(masculine.ready, true);
  assert.ok(masculine.feminine < 0.35);
  assert.equal(empty.ready, false);
  assert.equal(empty.feminine, 0.5);
  assert.ok(Math.abs(feminine.feminine + feminine.masculine - 1) < 1e-12);
});

test("background model preload is deduplicated and reused", async () => {
  let classifier = null;
  let loadCount = 0;
  let releaseLoad;
  const pendingModel = new Promise((resolve) => {
    releaseLoad = () => resolve(async () => []);
  });
  const bridge = createAnalysisEngineBridge({
    MODEL_ID: "model",
    getClf: () => classifier,
    pipeline: () => { },
    setClf: (value) => { classifier = value; },
    setCurrentDevice: () => { },
    setStatus: () => { },
    sharedEnsurePipeline: async (state) => {
      loadCount += 1;
      const model = await pendingModel;
      state.setClf(model);
      return model;
    },
    t: translate,
  });

  const first = bridge.preloadPipeline();
  const second = bridge.preloadPipeline();
  await Promise.resolve();
  assert.equal(loadCount, 1);
  releaseLoad();
  assert.equal(await first, await second);
  await bridge.preloadPipeline();
  assert.equal(loadCount, 1);
});
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

test("embedded fast inference leaves full audio available to acoustic analysis", async () => {
  const full = new Float32Array(16_000 * 20);
  const selected = new Float32Array(16_000 * 8);
  const calls = [];
  const controller = createAnalysisFlowController({
    analyzeStreamed: async () => { },
    analyzeWhole: async (...args) => calls.push(["whole", ...args]),
    decodeSmartToFloat32: async () => ({
      durationSec: 20,
      float32: full,
      sr: 16_000,
    }),
    finishAnalysisRun: () => { },
    finishStreamStats: () => calls.push(["stats"]),
    fmtSec: String,
    isAnalysisActive: () => true,
    MAX_WHOLE_SEC: 150,
    maybeApplyAdaptiveVAD: () => null,
    microYield: async () => { },
    notifyInferenceListeners: () => { },
    offlineExtractStreamMetrics: (samples) => calls.push(["offline", samples]),
    prepareInferenceSamples: (context) => {
      calls.push(["prepare", context.samples]);
      return { durationSec: 8, samples: selected, used: true };
    },
    runDecodedAudioAnalyzers: async (context) => {
      calls.push(["extensions", context.samples]);
      return {};
    },
    setPlaybackSource: () => { },
    setStatus: () => { },
    startAnalysisRun: () => 1,
    t: translate,
    TARGET_SR: 16_000,
    updatePlaybackAvailability: () => { },
    WARN_LONG_SEC: 180,
  });

  await controller.handleFileOrBlob(new Blob(), "recording");

  assert.equal(calls.find(([name]) => name === "offline")[1], full);
  assert.equal(calls.find(([name]) => name === "extensions")[1], full);
  assert.equal(calls.find(([name]) => name === "prepare")[1], full);
  const whole = calls.find(([name]) => name === "whole");
  assert.equal(whole[1], selected);
  assert.equal(whole[2], 16_000);
  assert.equal(whole[3], 8);
});

test("analysis flow can finish from local acoustics without loading the model", async () => {
  const calls = [];
  const controller = createAnalysisFlowController({
    analyzeStreamed: async () => calls.push("streamed"),
    analyzeWhole: async () => calls.push("whole"),
    analyzeWithoutModel: async (context) => {
      calls.push(["acoustic", context.samples]);
      return true;
    },
    decodeSmartToFloat32: async () => ({
      durationSec: 2,
      float32: new Float32Array(32_000),
      sr: 16_000,
    }),
    finishAnalysisRun: () => calls.push("finish"),
    finishStreamStats: () => calls.push("stats"),
    fmtSec: String,
    isAnalysisActive: () => true,
    MAX_WHOLE_SEC: 150,
    maybeApplyAdaptiveVAD: () => null,
    microYield: async () => { },
    notifyInferenceListeners: () => { },
    offlineExtractStreamMetrics: () => calls.push("offline"),
    prepareInferenceSamples: () => {
      calls.push("prepare");
      return null;
    },
    runDecodedAudioAnalyzers: async () => {
      calls.push("extensions");
      return {};
    },
    setPlaybackSource: () => { },
    setStatus: () => { },
    startAnalysisRun: () => 1,
    t: translate,
    TARGET_SR: 16_000,
    updatePlaybackAvailability: () => { },
    WARN_LONG_SEC: 180,
  });

  await controller.handleFileOrBlob(new Blob(), "recording");

  assert.deepEqual(calls.map((entry) => Array.isArray(entry) ? entry[0] : entry), [
    "offline",
    "extensions",
    "acoustic",
    "stats",
    "finish",
  ]);
  assert.equal(calls.some((entry) => entry === "whole" || entry === "streamed" || entry === "prepare"), false);
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
