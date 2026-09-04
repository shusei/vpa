import assert from "node:assert/strict";

import { createRecordingFlowController } from "../assets/js/recording-flow.js";

const originalDocument = globalThis.document;
const originalMediaRecorder = globalThis.MediaRecorder;
const states = [];

class MockMediaRecorder {
  constructor() {
    this.mimeType = "audio/webm";
    this.state = "inactive";
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    queueMicrotask(() => {
      this.ondataavailable?.({
        data: new Blob([new Uint8Array([1, 2, 3, 4])], { type: this.mimeType }),
      });
      this.onstop?.();
    });
  }
}

globalThis.MediaRecorder = MockMediaRecorder;
globalThis.document = {
  body: { classList: { add() { }, remove() { } } },
  querySelector: () => null,
};

try {
  const track = { stop() { } };
  const controller = createRecordingFlowController({
    createMediaRecorder: () => new MockMediaRecorder(),
    dismissOnboardTip() { },
    getMicCaptureInfo: () => null,
    handleFileOrBlob: async () => false,
    MEDIA_RECORDER_DATA_TIMEOUT_MS: 1000,
    onStateChange: (state) => states.push(state),
    pickSupportedMime: () => "audio/webm",
    prepareAnalysis: () => null,
    preparePitchStream: () => null,
    refreshAvailability() { },
    requestMicStream: async () => ({ getTracks: () => [track] }),
    setStatus() { },
    startPitchStream: async () => true,
    startRecordingTimer() { },
    stopPitchStream: async () => true,
    stopPlayback() { },
    stopRecordingTimer() { },
    t: (key) => key,
  });

  assert.equal(await controller.startRecording({ sessionId: 1, source: "quick" }), true);
  assert.equal(await controller.stopRecording({ sessionId: 1 }), true);
  for (let attempt = 0; attempt < 10 && !states.includes("error"); attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  assert.deepEqual(states, ["recording", "analyzing", "error"]);
} finally {
  globalThis.document = originalDocument;
  globalThis.MediaRecorder = originalMediaRecorder;
}

console.log("[PASS] Recording flow analysis-failure transition passed.");
