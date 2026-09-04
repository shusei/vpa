import assert from "node:assert/strict";
import {
  pausePlayback,
  playLastRecording,
  setPlaybackSource,
} from "../assets/js/player-ui.js";
import { createRealtimePitchStreamController } from "../assets/js/realtime-pitch-stream.js";

function deferred() {
  let resolve;
  const promise = new Promise((resolver) => { resolve = resolver; });
  return { promise, resolve };
}

// ----- Playback source races -----

{
  const pendingPlay = deferred();
  const copyStates = [];
  const revokedUrls = [];
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;
  let nextUrl = 1;
  URL.createObjectURL = () => `blob:test-${nextUrl++}`;
  URL.revokeObjectURL = (url) => { revokedUrls.push(url); };

  try {
    const audioEl = {
      currentTime: 0,
      duration: 1,
      ended: false,
      paused: true,
      src: "blob:test-old",
      load() {},
      pause() { this.paused = true; },
      play() {
        this.paused = false;
        return pendingPlay.promise;
      },
      removeAttribute(name) {
        if (name === "src") this.src = "";
      },
    };
    const state = {
      audioEl,
      lastAudioUrl: "blob:test-old",
      playBtn: {},
      playbackGeneration: 0,
    };
    const deps = {
      updatePlaybackAvailability() {},
      updatePlayerCopy(value) { copyStates.push(value); },
    };

    const oldPlay = playLastRecording(state, deps);
    setPlaybackSource(state, new Blob(["new audio"], { type: "audio/webm" }), deps);
    pendingPlay.resolve();

    assert.equal(await oldPlay, false, "a superseded play promise must not reactivate playback UI");
    assert.deepEqual(copyStates, [false], "source replacement must leave playback UI stopped");
    assert.deepEqual(revokedUrls, ["blob:test-old"], "the replaced Blob URL must be revoked once");
    assert.equal(state.lastAudioUrl, "blob:test-1");
  } finally {
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
  }
}

{
  const pendingPlay = deferred();
  const copyStates = [];
  const audioEl = {
    currentTime: 0,
    duration: 1,
    ended: false,
    paused: true,
    src: "blob:test-current",
    pause() { this.paused = true; },
    play() {
      this.paused = false;
      return pendingPlay.promise;
    },
  };
  const state = { audioEl, playbackGeneration: 0 };
  const deps = { updatePlayerCopy(value) { copyStates.push(value); } };

  const oldPlay = playLastRecording(state, deps);
  pausePlayback(state, deps);
  pendingPlay.resolve();

  assert.equal(await oldPlay, false, "a delayed play promise must not override a later pause");
  assert.deepEqual(copyStates, [false]);
}

// ----- Realtime pitch start races -----

{
  const originalGlobals = {
    addEventListener: globalThis.addEventListener,
    cancelAnimationFrame: globalThis.cancelAnimationFrame,
    document: globalThis.document,
    getComputedStyle: globalThis.getComputedStyle,
    removeEventListener: globalThis.removeEventListener,
    requestAnimationFrame: globalThis.requestAnimationFrame,
    window: globalThis.window,
  };
  const contexts = [];
  const panelStates = [];
  let processorCount = 0;
  let processorDisconnectCount = 0;
  let suspendGate = null;
  let requireGestureForResume = false;
  let userGestureActive = false;
  let nextFrame = 1;

  class MockAudioContext {
    constructor() {
      this.closeGate = null;
      this.destination = {};
      this.id = contexts.length;
      this.state = "suspended";
      this.closeCount = 0;
      this.resumeCount = 0;
      this.suspendCount = 0;
      contexts.push(this);
    }

    async close() {
      if (this.state !== "closed") {
        this.closeCount++;
        this.state = "closed";
      }
      if (this.closeGate) await this.closeGate.promise;
    }

    createMediaStreamSource() {
      return {
        connect() {},
        disconnect() {},
      };
    }

    createScriptProcessor() {
      processorCount += 1;
      return {
        connect() {},
        disconnect() { processorDisconnectCount += 1; },
        onaudioprocess: null,
      };
    }

    async resume() {
      this.resumeCount += 1;
      if (requireGestureForResume && !userGestureActive) {
        throw new Error("resume requires a user gesture");
      }
      this.state = "running";
    }

    async suspend() {
      this.suspendCount += 1;
      if (suspendGate) await suspendGate.promise;
      this.state = "suspended";
    }
  }

  const context2d = {
    beginPath() {},
    clearRect() {},
    fillRect() {},
    fillText() {},
    lineTo() {},
    moveTo() {},
    restore() {},
    save() {},
    stroke() {},
  };
  const pitchCanvas = {
    clientHeight: 120,
    getBoundingClientRect: () => ({ height: 120, width: 320 }),
    getContext: () => context2d,
    height: 0,
    width: 0,
  };

  globalThis.window = {
    AudioContext: MockAudioContext,
    devicePixelRatio: 1,
  };
  globalThis.document = { documentElement: { getAttribute: () => "light" } };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
  globalThis.addEventListener = () => {};
  globalThis.removeEventListener = () => {};
  globalThis.requestAnimationFrame = () => nextFrame++;
  globalThis.cancelAnimationFrame = () => {};

  try {
    const arrays = {
      psConfidence: [],
      psDb: [],
      psHz: [],
      psHzSmooth: [],
      psVoiced: [],
    };
    const controller = createRealtimePitchStreamController({
      appendPitchSample: () => ({ processed: null }),
      applyDbCalibration: (value) => ({ value }),
      arrays,
      describeResonanceFromEnergy: () => ({ label: "", pct: null }),
      dom: { pitchCanvas, pitchWrap: {} },
      estimateSpectralFeatures: () => null,
      fmt1: String,
      maybeEnableAdvancedPitch() {},
      normalizeResonanceBands: () => ({}),
      pitchPostState: {},
      psRealtimeNoiseTracker: { capture() {}, reset() {}, shouldDetect: () => ({ ambient: 0, detect: false }) },
      PS_INTERVAL_MS: 50,
      PS_MAX_HZ: 600,
      PS_MIN_HZ: 50,
      resetPitchPostState() {},
      resetRealtimePanels() {},
      runPitchDetection: () => null,
      setRealtimePanelsActive(value) { panelStates.push(value); },
      startAutoRangeSession() {},
      t: () => "",
    });

    requireGestureForResume = true;
    for (let round = 0; round < 8; round += 1) {
      userGestureActive = true;
      const preparation = controller.prepareForUserGesture({ sessionId: round + 1, source: "professional" });
      userGestureActive = false;
      assert.equal(await controller.startPitchStream({}, {
        preparation,
        sessionId: round + 1,
        source: "professional",
      }), true);
      assert.equal(panelStates.at(-1), true, `round ${round + 1} must show realtime panels`);
      assert.equal(await controller.stopPitchStream({ sessionId: round + 1 }), true);
      assert.equal(panelStates.at(-1), false, `round ${round + 1} must hide realtime panels after stop`);
    }

    assert.equal(contexts.length, 1, "repeated recording must reuse one AudioContext");
    assert.equal(contexts[0].closeCount, 0, "the reusable AudioContext must stay available between recordings");
    assert.equal(contexts[0].resumeCount, 8);
    assert.equal(contexts[0].suspendCount, 8);
    assert.equal(processorCount, 8);
    assert.equal(processorDisconnectCount, processorCount);

    requireGestureForResume = false;
    await controller.startPitchStream({});
    suspendGate = deferred();
    const suspendsBeforeRace = contexts[0].suspendCount;
    const delayedStop = controller.stopPitchStream();
    while (contexts[0].suspendCount === suspendsBeforeRace) {
      await Promise.resolve();
    }

    const nextStart = controller.startPitchStream({});
    suspendGate.resolve();
    suspendGate = null;
    await Promise.all([delayedStop, nextStart]);

    assert.equal(contexts.length, 1, "a stop/start race must still reuse the same AudioContext");
    assert.equal(panelStates.at(-1), true, "the newest start must reactivate realtime panels");

    await controller.stopPitchStream();
    assert.equal(panelStates.at(-1), false);
    assert.equal(processorDisconnectCount, processorCount);

    const ownedSession = 101;
    await controller.startPitchStream({}, { sessionId: ownedSession, source: "professional" });
    assert.equal(
      await controller.stopPitchStream({ sessionId: ownedSession - 1 }),
      false,
      "an old session must not stop the active pitch graph",
    );
    assert.equal(panelStates.at(-1), true);
    assert.equal(await controller.stopPitchStream({ sessionId: ownedSession }), true);
    assert.equal(panelStates.at(-1), false);
  } finally {
    Object.assign(globalThis, originalGlobals);
  }
}

console.log("[PASS] Audio playback and realtime pitch lifecycle guards passed.");
