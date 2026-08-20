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
  const firstResume = deferred();
  const contexts = [];
  const panelStates = [];
  let nextFrame = 1;

  class MockAudioContext {
    constructor() {
      this.closeGate = null;
      this.destination = {};
      this.id = contexts.length;
      this.state = "suspended";
      this.closeCount = 0;
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
      return {
        connect() {},
        disconnect() {},
        onaudioprocess: null,
      };
    }

    async resume() {
      if (this.id === 0) await firstResume.promise;
      this.state = "running";
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

    const firstStart = controller.startPitchStream({});
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(contexts.length, 1, "the first start should be waiting for AudioContext.resume()");

    const secondStart = controller.startPitchStream({});
    await secondStart;
    assert.equal(contexts.length, 2);
    assert.equal(contexts[1].closeCount, 0, "the newest pitch session must remain active");

    firstResume.resolve();
    await firstStart;
    assert.equal(contexts[0].closeCount, 1, "the superseded pitch session must close itself");
    assert.equal(contexts[1].closeCount, 0, "an older start must not close the newer pitch session");
    assert.equal(panelStates.at(-1), true);

    const delayedClose = deferred();
    contexts[1].closeGate = delayedClose;
    const delayedStop = controller.stopPitchStream();
    await Promise.resolve();

    const thirdStart = controller.startPitchStream({});
    await thirdStart;
    assert.equal(contexts.length, 3);
    assert.equal(panelStates.at(-1), true, "a newer start must reactivate the realtime panels");

    delayedClose.resolve();
    await delayedStop;
    assert.equal(contexts[1].closeCount, 1);
    assert.equal(panelStates.at(-1), true, "an older stop must not hide a newer pitch session");

    await controller.stopPitchStream();
    assert.equal(contexts[2].closeCount, 1);
    assert.equal(panelStates.at(-1), false);
  } finally {
    Object.assign(globalThis, originalGlobals);
  }
}

console.log("[PASS] Audio playback and realtime pitch lifecycle guards passed.");
