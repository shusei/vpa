import assert from "node:assert/strict";

import {
  audioDiagnosticsInternals,
  createAudioDiagnostics,
} from "../assets/js/audio-diagnostics.js";

const originalNavigator = globalThis.navigator;
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    userActivation: { hasBeenActive: true, isActive: false },
    userAgent: "VPA diagnostics unit test",
  },
});

try {
  assert.equal(audioDiagnosticsInternals.audioDebugEnabled({ href: "https://example.test/?vpaAudioDebug=1" }), true);
  assert.equal(audioDiagnosticsInternals.audioDebugEnabled({ href: "https://example.test/" }), false);

  const disabled = createAudioDiagnostics({
    locationObject: { href: "https://example.test/", origin: "https://example.test", pathname: "/" },
  });
  disabled.record("ignored", { value: 1 });
  assert.equal(disabled.enabled, false);
  assert.equal(disabled.getReport().events.length, 0);

  const enabled = createAudioDiagnostics({
    buildVersion: "test-build",
    getExperience: () => "professional",
    locationObject: {
      href: "https://example.test/?vpaAudioDebug=1",
      origin: "https://example.test",
      pathname: "/",
    },
  });
  for (let index = 0; index < audioDiagnosticsInternals.MAX_AUDIO_DEBUG_EVENTS + 25; index += 1) {
    enabled.record("counter", { index });
  }
  const report = enabled.getReport();
  assert.equal(report.buildVersion, "test-build");
  assert.equal(report.currentExperience, "professional");
  assert.equal(report.events.length, audioDiagnosticsInternals.MAX_AUDIO_DEBUG_EVENTS);
  assert.equal(report.events.at(-1).detail.index, audioDiagnosticsInternals.MAX_AUDIO_DEBUG_EVENTS + 24);
  assert.equal(JSON.stringify(report).includes("audioSamples"), false);
} finally {
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: originalNavigator,
  });
}

console.log("[PASS] Audio diagnostics ring buffer guards passed.");
