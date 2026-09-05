import assert from "node:assert/strict";
import { test } from "node:test";
import { createAudioDiagnostics } from "../assets/js/audio-diagnostics.js";
import { createRecordingCoordinator } from "../assets/js/recording-coordinator.js";
import { createRecordingFlowController } from "../assets/js/recording-flow.js";
import { createRealtimePitchStreamController } from "../assets/js/realtime-pitch-stream.js";

for (const fails of [false, true]) {
  test(`browser-initiated stop cleans up and permits another recording (error=${fails})`, async () => {
    const classes = new Set();
    const originalDocument = globalThis.document;
    const originalRecorder = globalThis.MediaRecorder;
    globalThis.document = {
      body: { classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name) } },
      querySelector: () => null,
    };
    class Recorder {
      state = "inactive";
      start() { this.state = "recording"; }
    }
    globalThis.MediaRecorder = Recorder;
    let recorder;
    let coordinator;
    let analyzed = false;
    const track = { readyState: "live", stop() { this.readyState = "ended"; } };
    const flow = createRecordingFlowController({
      createMediaRecorder: () => (recorder = new Recorder()),
      dismissOnboardTip() {},
      handleFileOrBlob: async () => {
        assert.equal(track.readyState, "ended", "microphone must be released before analysis");
        analyzed = true;
        return true;
      },
      MEDIA_RECORDER_DATA_TIMEOUT_MS: 100,
      onStateChange: (state, detail) => coordinator.handleFlowState(state, detail),
      pickSupportedMime: () => "audio/webm",
      refreshAvailability() {},
      requestMicStream: async () => ({ getTracks: () => [track] }),
      setStatus() {},
      startPitchStream: async () => true,
      startRecordingTimer() {},
      stopPitchStream: async () => true,
      stopPlayback() {},
      stopRecordingTimer() {},
      t: (key) => key,
    });
    coordinator = createRecordingCoordinator({ startRecording: flow.startRecording, stopRecording: flow.stopRecording });
    try {
      await coordinator.start({ source: "professional" });
      recorder.state = "inactive";
      if (fails) recorder.onerror({ error: new Error("microphone interrupted") });
      recorder.ondataavailable({ data: new Blob(["recorded data"]) });
      await recorder.onstop();
      assert.equal(classes.has("recording"), false);
      assert.equal(coordinator.getSnapshot().state, fails ? "error" : "idle");
      assert.equal(analyzed, !fails);
      assert.equal(track.readyState, "ended");
      assert.equal(await coordinator.start({ source: "quick" }), true);
    } finally {
      globalThis.document = originalDocument;
      globalThis.MediaRecorder = originalRecorder;
    }
  });
}

test("failed graph connection releases local nodes and their callback", async () => {
  const originalWindow = globalThis.window;
  let sourceDisconnects = 0;
  let processorDisconnects = 0;
  const source = { connect() {}, disconnect() { sourceDisconnects++; } };
  const processor = {
    onaudioprocess: null,
    connect() { throw new Error("graph connection failed"); },
    disconnect() { processorDisconnects++; },
  };
  globalThis.window = { AudioContext: class {
    state = "running";
    destination = {};
    sampleRate = 16000;
    createMediaStreamSource() { return source; }
    createScriptProcessor() { return processor; }
    async suspend() { this.state = "suspended"; }
  } };
  try {
    const controller = createRealtimePitchStreamController({
      arrays: { psDb: [], psHz: [], psHzSmooth: [], psVoiced: [], psConfidence: [] },
      dom: { pitchWrap: {}, pitchCanvas: {} },
      maybeEnableAdvancedPitch() {},
      psRealtimeNoiseTracker: { reset() {} },
      resetPitchPostState() {},
      setRealtimePanelsActive() {},
      startAutoRangeSession() {},
    });
    assert.equal(await controller.startPitchStream({}, { sessionId: 1 }), false);
    await controller.stopPitchStream({ sessionId: 1 });
    assert.equal(sourceDisconnects, 1);
    assert.equal(processorDisconnects, 1);
    assert.equal(processor.onaudioprocess, null);
  } finally {
    globalThis.window = originalWindow;
  }
});

test("disabled diagnostics never inspect panels, pixels, or tracks", () => {
  let reads = 0;
  const diagnostics = createAudioDiagnostics({
    locationObject: { href: "https://example.test/" },
    dom: { pitchCanvas: { get width() { reads++; return 0; } } },
  });
  diagnostics.recordPanel("pitch.raf.frame");
  diagnostics.recordStream("recording.microphone.ready", { getAudioTracks() { reads++; return []; } });
  assert.equal(reads, 0);
});

test("stop cancels a pending resume and suspends a late completion", async () => {
  const originalWindow = globalThis.window;
  let resume;
  let context;
  globalThis.window = { AudioContext: class {
    state = "suspended";
    constructor() { context = this; }
    resume() { return new Promise((resolve) => { resume = () => { this.state = "running"; resolve(); }; }); }
    async suspend() { this.state = "suspended"; }
  } };
  try {
    const controller = createRealtimePitchStreamController({
      arrays: { psDb: [], psHz: [], psHzSmooth: [], psVoiced: [], psConfidence: [] },
      dom: { pitchWrap: {}, pitchCanvas: {} },
      maybeEnableAdvancedPitch() {},
      psRealtimeNoiseTracker: { reset() {} },
      resetPitchPostState() {},
      setRealtimePanelsActive() {},
      startAutoRangeSession() {},
    });
    const preparation = controller.prepareForUserGesture({ sessionId: 1 });
    const start = controller.startPitchStream({}, { preparation, sessionId: 1 });
    await new Promise((resolve) => setTimeout(resolve, 0));
    const stop = controller.stopPitchStream({ sessionId: 1 });
    let timeout;
    try {
      assert.equal(await Promise.race([
        stop,
        new Promise((resolve) => { timeout = setTimeout(() => resolve("blocked"), 200); }),
      ]), true, "stop must not wait for permission/resume to finish");
      assert.equal(await start, false);
      resume();
      await new Promise((resolve) => setTimeout(resolve, 0));
      assert.equal(context.state, "suspended", "late resume must not leave an idle context running");
    } finally {
      clearTimeout(timeout);
      resume();
      await Promise.all([start, stop]);
    }
  } finally {
    globalThis.window = originalWindow;
  }
});
