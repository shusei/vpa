import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { voiceAgeV2Internals } from "../assets/experiments/voice-age-v2.js";
import { voiceQualityInternals } from "../assets/experiments/voice-quality-metrics.js";

const manifest = JSON.parse(readFileSync(
  resolve("fixtures/voice-age/calibration-manifest.json"),
  "utf8",
));

assert.equal(manifest.schemaVersion, 1);
assert.equal(manifest.ageVersion, voiceAgeV2Internals.VERSION);
assert.equal(manifest.qualityVersion, voiceQualityInternals.VERSION);
assert.equal(manifest.strictScoreVersion, "advanced-beta-1");
assert.equal(manifest.splitRule, "speaker-disjoint");
assert.ok(["research-preview", "calibrated"].includes(manifest.status));
assert.equal(manifest.ageBands.length, 5);
assert.ok(manifest.ageBands.every((band) => (
  Number.isFinite(band.min)
  && Number.isFinite(band.max)
  && band.min < band.max
  && band.minimumSpeakersPerSampleType >= 30
)));
assert.deepEqual(
  manifest.sampleTypes.connectedSpeech.requiredMetrics.sort(),
  ["cpp", "hnr"],
);
assert.deepEqual(
  manifest.sampleTypes.sustainedVowel.requiredMetrics.sort(),
  ["cpp", "hnr", "jitterLocal", "shimmerLocal"].sort(),
);
if (!manifest.humanCalibrationAccepted) {
  assert.equal(manifest.status, "research-preview");
  assert.equal(manifest.maximumConfidence, "medium");
  assert.deepEqual(manifest.committedHumanSamples, []);
}

console.log("voice age calibration contract checks passed");
