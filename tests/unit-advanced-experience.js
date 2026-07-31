import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  advancedEvaluatorInternals,
  evaluateAdvancedExperience,
} from "../assets/experiments/advanced-evaluator.js";
import {
  buildShareTargets,
  buildShareUrl,
} from "../assets/experiments/share-card.js";
import en from "../assets/i18n/en.js";
import zhHans from "../assets/i18n/zh-Hans.js";
import zhHant from "../assets/i18n/zh-Hant.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

const reference = evaluateAdvancedExperience(fixture);
assert.equal(reference.ready, true);
assert.equal(reference.version, "advanced-beta-1");
assert.ok(reference.score >= 60 && reference.score < 100);
assert.ok(reference.score < Math.round(fixture.probabilities.feminine * 100));
assert.ok(reference.voiceAge.min >= 18);
assert.ok(reference.voiceAge.max <= 60);
assert.ok(reference.components.resonance > 0.5);

const resonanceMismatch = structuredClone(fixture);
resonanceMismatch.probabilities.feminine = 0.95;
resonanceMismatch.probabilities.masculine = 0.05;
resonanceMismatch.pitch.stats.med = 280;
resonanceMismatch.advanced.formants.f2.median = 1250;
resonanceMismatch.advanced.formants.f3.median = 2200;
resonanceMismatch.advanced.formants.f2.coverage = 0.9;
resonanceMismatch.advanced.formants.f3.coverage = 0.9;
resonanceMismatch.advanced.energyPct = {
  chest: 0.74,
  head: 0.08,
  mask: 0.18,
};

const mismatch = evaluateAdvancedExperience(resonanceMismatch);
assert.equal(mismatch.ready, true);
assert.equal(mismatch.contradiction, true);
assert.equal(mismatch.archetypeKey, "falsettoExplorer");
assert.equal(mismatch.insightKey, "falsettoContrast");
assert.ok(mismatch.score < 80);
assert.ok(mismatch.score < Math.round(resonanceMismatch.probabilities.feminine * 100));

const insufficient = structuredClone(fixture);
insufficient.volume.snr = 2;
insufficient.summary.voicedRatio = 0.08;
insufficient.offlineSamples.duration = 1;
insufficient.advanced.formants.f2.coverage = 0.05;
insufficient.advanced.formants.f3.coverage = 0.05;

const rejected = evaluateAdvancedExperience(insufficient);
assert.equal(rejected.ready, false);
assert.ok(rejected.insufficientReasons.includes("resonance"));
assert.ok(rejected.insufficientReasons.includes("quality"));
assert.equal(rejected.insightKey, "insufficient");

assert.equal(advancedEvaluatorInternals.interpolate([
  [100, 0],
  [200, 1],
], 150), 0.5);

const shareUrl = buildShareUrl({
  href: "https://example.com/vpa/dev.html?fixture=1#result",
});
assert.equal(shareUrl, "https://example.com/vpa/");

const shareTargets = buildShareTargets({
  caption: "VPA score 72%",
  url: shareUrl,
});
assert.match(shareTargets.x, /^https:\/\/twitter\.com\/intent\/tweet\?/);
assert.match(shareTargets.threads, /^https:\/\/www\.threads\.com\/intent\/post\?/);
assert.match(shareTargets.line, /^https:\/\/social-plugins\.line\.me\/lineit\/share\?/);
assert.match(shareTargets.facebook, /^https:\/\/www\.facebook\.com\/sharer\/sharer\.php\?/);
assert.ok(Object.values(shareTargets).every((target) => target.includes(encodeURIComponent(shareUrl))));

const translationKeys = [
  "archetype.title",
  "archetypes.airySweet",
  "archetypes.balancedIntelligent",
  "archetypes.brightForward",
  "archetypes.falsettoExplorer",
  "archetypes.livelyExpressive",
  "archetypes.matureWarm",
  "archetypes.neutralClear",
  "beta",
  "components.intonation",
  "components.model",
  "components.pitch",
  "components.resonance",
  "components.weight",
  "confidence.high",
  "confidence.label",
  "confidence.low",
  "confidence.medium",
  "contradiction",
  "disclaimer",
  "insight.balancedGrowth",
  "insight.consistencyOpportunity",
  "insight.falsettoContrast",
  "insight.insufficient",
  "insight.label",
  "insight.pitchOpportunity",
  "insight.resonanceOpportunity",
  "insight.strongIntegration",
  "insufficient",
  "mode.advanced",
  "mode.aria",
  "mode.basic",
  "mode.label",
  "mode.note",
  "share.cancelled",
  "share.caption",
  "share.cardDisclaimer",
  "share.challenge",
  "share.copied",
  "share.copy",
  "share.download",
  "share.downloaded",
  "share.failed",
  "share.opened",
  "share.platformAria",
  "share.primary",
  "share.shared",
  "share.title",
  "strictScore",
  "title",
  "voiceAge.title",
  "voiceAge.value",
];

function resolveTranslation(dictionary, key) {
  return `experiment.advanced.${key}`
    .split(".")
    .reduce((value, part) => value?.[part], dictionary);
}

for (const dictionary of [zhHant, zhHans, en]) {
  for (const key of translationKeys) {
    assert.equal(typeof resolveTranslation(dictionary, key), "string", `${dictionary.locale}: ${key}`);
    assert.notEqual(resolveTranslation(dictionary, key), "", `${dictionary.locale}: ${key}`);
  }
}

console.log("advanced experience unit checks passed");
