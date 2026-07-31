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
import {
  compareChallenge,
  createChallengePayload,
  createChallengeUrl,
  decodeChallenge,
  encodeChallenge,
  readChallenge,
} from "../assets/experiments/challenge-link.js";
import {
  audioFileFromUrl,
  shareResultFiles,
} from "../assets/experiments/audio-share.js";
import en from "../assets/i18n/en.js";
import ja from "../assets/i18n/ja.js";
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

const challenge = createChallengePayload(reference, {
  randomUUID: () => "12345678-1234-1234-1234-123456789abc",
});
assert.deepEqual(challenge, {
  ageMax: reference.voiceAge.max,
  ageMin: reference.voiceAge.min,
  archetype: reference.archetypeKey,
  id: "12345678-1234-1234-1234-123456789abc",
  score: reference.score,
  scoreVersion: reference.version,
  schema: 1,
});
assert.deepEqual(decodeChallenge(encodeChallenge(challenge)), challenge);

const challengeUrl = createChallengeUrl(reference, {
  hash: "",
  href: "https://example.com/vpa/dev.html?fixture=1",
  origin: "https://example.com",
  pathname: "/vpa/dev.html",
});
assert.equal(challengeUrl.payload.score, reference.score);
assert.match(challengeUrl.url, /^https:\/\/example\.com\/vpa\/dev\.html#vpa-challenge=/);
assert.deepEqual(readChallenge({ hash: new URL(challengeUrl.url).hash }), challengeUrl.payload);
assert.equal(readChallenge({ hash: "#vpa-challenge=broken" }), null);
assert.deepEqual(compareChallenge(challenge, { ready: true, score: challenge.score + 4 }), {
  difference: 4,
  opponentScore: challenge.score,
  outcome: "beat",
  score: challenge.score + 4,
});
assert.equal(compareChallenge(challenge, { ready: true, score: challenge.score - 1 }).outcome, "behind");
assert.equal(compareChallenge(challenge, { ready: true, score: challenge.score }).outcome, "tied");

const audioFile = await audioFileFromUrl("blob:test", async () => ({
  blob: async () => new Blob(["audio"], { type: "audio/webm" }),
  ok: true,
}));
assert.equal(audioFile.name, "vpa-voice.weba");
assert.equal(audioFile.type, "audio/webm");

const nativeShareCalls = [];
const nativeShare = await shareResultFiles({
  audioFile,
  cardBlob: new Blob(["card"], { type: "image/png" }),
  caption: "VPA result",
  navigatorLike: {
    canShare: ({ files }) => files.length === 2,
    share: async (payload) => nativeShareCalls.push(payload),
  },
  title: "VPA",
  url: "https://example.com/vpa/",
});
assert.equal(nativeShare.method, "files");
assert.equal(nativeShareCalls[0].files.length, 2);

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

const quickTranslationKeys = [
  "analyzing",
  "analyzingHint",
  "challenge.beat",
  "challenge.behind",
  "challenge.comparison",
  "challenge.inviteScore",
  "challenge.received",
  "challenge.tied",
  "errors.analysisTimeout",
  "errors.recordingFailed",
  "footer",
  "history.label",
  "history.none",
  "history.score",
  "localeLabel",
  "prompt",
  "promptHint",
  "promptLabel",
  "recording",
  "recordingHint",
  "requesting",
  "retry",
  "reveal.age",
  "reveal.archetype",
  "reveal.eyebrow",
  "reveal.insight",
  "reveal.score",
  "share.audioDefault",
  "share.audioWarning",
  "share.cancelled",
  "share.copied",
  "share.copyChallenge",
  "share.downloaded",
  "share.downloadedWithAudio",
  "share.failed",
  "share.includeAudio",
  "share.open",
  "share.preview",
  "share.shareTitle",
  "share.shared",
  "share.system",
  "share.title",
  "start",
  "stop",
  "subtitle",
  "title",
  "trust.local",
  "trust.private",
  "trust.strict",
  "viewProfessional",
];

for (const dictionary of [zhHant, zhHans, en, ja]) {
  for (const key of translationKeys) {
    assert.equal(typeof resolveTranslation(dictionary, key), "string", `${dictionary.locale}: ${key}`);
    assert.notEqual(resolveTranslation(dictionary, key), "", `${dictionary.locale}: ${key}`);
  }
  for (const key of quickTranslationKeys) {
    const value = `experiment.quick.${key}`
      .split(".")
      .reduce((result, part) => result?.[part], dictionary);
    assert.equal(typeof value, "string", `${dictionary.locale}: quick.${key}`);
    assert.notEqual(value, "", `${dictionary.locale}: quick.${key}`);
  }
  assert.equal(typeof dictionary.experiment.experience.quick, "string");
  assert.equal(typeof dictionary.experiment.experience.professional, "string");
  assert.equal(typeof dictionary.topbar.localeNames.ja, "string");
}

function flattenDictionary(value, prefix = "", result = {}) {
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (child && typeof child === "object" && !Array.isArray(child)) {
      flattenDictionary(child, path, result);
    } else {
      result[path] = child;
    }
  }
  return result;
}

const flatEnglish = flattenDictionary(en);
const flatJapanese = flattenDictionary(ja);
assert.deepEqual(Object.keys(flatJapanese).sort(), Object.keys(flatEnglish).sort());

const intentionalJapaneseMatches = [
  "analysis.intonation.insufficient.rangeHint",
  "analysis.meter.scale.forty",
  "analysis.meter.scale.full",
  "analysis.meter.scale.sixty",
  "analysis.meter.scale.zero",
  "experiment.advanced.beta",
  "experiment.quick.eyebrow",
  "experiment.quick.reveal.eyebrow",
  "hero.title",
  "meta.title",
  "player.replayHintSpacer",
  "player.replayHintSuffix",
  "realtime.formants.f1Label",
  "realtime.formants.f2Label",
  "realtime.formants.f3Label",
  "realtime.resonance.valuePlaceholder",
  "summary.breathinessDisplay",
  "summary.liaisonDisplay",
  "summary.rangeDisplayHz",
  "topbar.themeLuxeGold",
  "topbar.themePony2026",
].sort();
const actualJapaneseMatches = Object.keys(flatEnglish)
  .filter((key) => (
    typeof flatEnglish[key] === "string"
    && flatEnglish[key] === flatJapanese[key]
  ))
  .sort();
assert.deepEqual(actualJapaneseMatches, intentionalJapaneseMatches);

console.log("advanced experience unit checks passed");
