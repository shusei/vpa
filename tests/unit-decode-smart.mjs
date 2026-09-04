import assert from "node:assert/strict";

import { decodeViaWebAudio } from "../assets/js/decode-smart.js";

const originalWindow = globalThis.window;
let legacyDecodeBytes = 0;

class MockAudioContext {
  close() {
    return Promise.resolve();
  }

  createBuffer(_channels, length, sampleRate) {
    const samples = new Float32Array(length);
    return {
      getChannelData: () => samples,
      length,
      sampleRate,
    };
  }

  decodeAudioData(buffer, resolve) {
    if (typeof resolve === "function") {
      legacyDecodeBytes = buffer.byteLength;
      resolve({
        duration: 1,
        getChannelData: () => new Float32Array([0.25, -0.25]),
        length: 2,
        numberOfChannels: 1,
        sampleRate: 16000,
      });
      return undefined;
    }

    structuredClone(buffer, { transfer: [buffer] });
    return Promise.reject(new Error("primary decoder rejected after detaching its input"));
  }
}

globalThis.window = {
  AudioContext: MockAudioContext,
};

try {
  const result = await decodeViaWebAudio(
    new Blob([new Uint8Array([1, 2, 3, 4])]),
    16000,
    (channels, output) => {
      output.set(channels[0]);
      return channels.length;
    },
  );

  assert.equal(legacyDecodeBytes, 4, "legacy decode must receive the copy made before detachment");
  assert.deepEqual([...result.float32], [0.25, -0.25]);
  assert.equal(result.sr, 16000);
} finally {
  globalThis.window = originalWindow;
}

console.log("[PASS] Web Audio detached-buffer fallback passed.");
