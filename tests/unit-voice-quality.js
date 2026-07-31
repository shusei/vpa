import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import {
  registerDecodedAudioAnalyzer,
  resetDecodedAudioAnalyzersForTest,
  runDecodedAudioAnalyzers,
} from "../assets/js/analysis-flow.js";
import {
  analyzeVoiceQuality,
  voiceQualityInternals,
} from "../assets/experiments/voice-quality-metrics.js";

const SAMPLE_RATE = 16_000;

function createSignal({
  amplitude = 0.28,
  amplitudeModulation = 0,
  durationSec = 5,
  frequency = 220,
  frequencyModulation = 0,
  noise = 0,
} = {}) {
  const samples = new Float32Array(Math.round(durationSec * SAMPLE_RATE));
  let phase = 0;
  let seed = 0x13579bdf;
  for (let index = 0; index < samples.length; index += 1) {
    const time = index / SAMPLE_RATE;
    const localFrequency = frequency * (1 + (frequencyModulation * Math.sin(2 * Math.PI * 1.7 * time)));
    const localAmplitude = amplitude * (1 + (amplitudeModulation * Math.sin(2 * Math.PI * 2.3 * time)));
    phase += 2 * Math.PI * localFrequency / SAMPLE_RATE;
    seed = ((seed * 1664525) + 1013904223) >>> 0;
    const random = ((seed / 0xffffffff) * 2) - 1;
    samples[index] = (localAmplitude * Math.sin(phase))
      + (localAmplitude * 0.2 * Math.sin(2 * phase))
      + (noise * random);
  }
  return samples;
}

const cleanStart = performance.now();
const clean = analyzeVoiceQuality(createSignal(), SAMPLE_RATE);
const cleanElapsed = performance.now() - cleanStart;
assert.equal(clean.version, voiceQualityInternals.VERSION);
assert.equal(clean.sampleType, "sustainedVowel");
assert.equal(clean.quality.ready, true);
assert.equal(clean.metrics.jitterLocal.reliable, true);
assert.equal(clean.metrics.shimmerLocal.reliable, true);
assert.ok(clean.metrics.jitterLocal.valuePct < 1);
assert.ok(clean.metrics.shimmerLocal.valuePct < 2);
assert.ok(clean.metrics.hnr.valueDb > 10);
assert.ok(clean.metrics.cpp.valueDb > 3);
assert.ok(cleanElapsed < 1500, `clean voice analysis took ${cleanElapsed.toFixed(1)} ms`);

const expressive = analyzeVoiceQuality(createSignal({
  amplitudeModulation: 0.42,
  frequencyModulation: 0.12,
  noise: 0.025,
}), SAMPLE_RATE, {
  sampleType: "connectedSpeech",
});
assert.equal(expressive.sampleType, "connectedSpeech");
assert.equal(expressive.sampleTypeSource, "explicit");
assert.equal(expressive.quality.ready, true);
assert.equal(expressive.metrics.jitterLocal.reliable, false);
assert.equal(expressive.metrics.shimmerLocal.reliable, false);
assert.ok(expressive.metrics.hnr.valueDb < clean.metrics.hnr.valueDb);
assert.ok(Number.isFinite(expressive.metrics.cpp.valueDb));
assert.ok(expressive.metrics.cpp.valueDb > 0);

const unstable = analyzeVoiceQuality(createSignal({
  amplitudeModulation: 0.3,
  frequencyModulation: 0.08,
}), SAMPLE_RATE, {
  sampleType: "sustainedVowel",
});
assert.ok(unstable.metrics.jitterLocal.valuePct > clean.metrics.jitterLocal.valuePct);
assert.ok(unstable.metrics.shimmerLocal.valuePct > clean.metrics.shimmerLocal.valuePct);

const tooShort = analyzeVoiceQuality(createSignal({ durationSec: 1 }), SAMPLE_RATE, {
  sampleType: "connectedSpeech",
});
assert.equal(tooShort.quality.ready, false);
assert.ok(tooShort.quality.reasons.includes("duration"));

const clippedSamples = createSignal({ amplitude: 1.4 });
for (let index = 0; index < clippedSamples.length; index += 1) {
  clippedSamples[index] = Math.max(-1, Math.min(1, clippedSamples[index]));
}
const clipped = analyzeVoiceQuality(clippedSamples, SAMPLE_RATE);
assert.equal(clipped.quality.ready, false);
assert.ok(clipped.quality.reasons.includes("clipping"));

const silent = analyzeVoiceQuality(new Float32Array(SAMPLE_RATE * 5), SAMPLE_RATE);
assert.equal(silent.quality.ready, false);
assert.ok(silent.quality.reasons.includes("level"));
assert.ok(silent.quality.reasons.includes("voicing"));

const longSignal = createSignal({ durationSec: 45 });
const longStart = performance.now();
const longResult = analyzeVoiceQuality(longSignal, SAMPLE_RATE);
const longElapsed = performance.now() - longStart;
assert.equal(longResult.durationSec, 45);
assert.ok(longResult.analyzedDurationSec <= voiceQualityInternals.MAX_ANALYSIS_SEC + 0.01);
assert.ok(longElapsed < 1500, `long voice analysis took ${longElapsed.toFixed(1)} ms`);

resetDecodedAudioAnalyzersForTest();
const unregister = registerDecodedAudioAnalyzer("voice-age-v2", ({ durationSec }) => ({
  durationSec,
}));
assert.deepEqual(await runDecodedAudioAnalyzers({ durationSec: 5 }), {
  "voice-age-v2": {
    durationSec: 5,
  },
});
unregister();
assert.deepEqual(await runDecodedAudioAnalyzers({ durationSec: 5 }), {});
resetDecodedAudioAnalyzersForTest();

console.log(
  `voice quality unit checks passed (${cleanElapsed.toFixed(1)} ms for 5 s, ${longElapsed.toFixed(1)} ms for capped 45 s)`,
);
