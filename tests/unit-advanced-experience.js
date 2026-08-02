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
  buildShareText,
  shareResultFiles,
} from "../assets/experiments/audio-share.js";
import {
  defaultClipRange,
  extractWaveform,
  getSupportedVideoProfiles,
} from "../assets/experiments/dynamic-voice-card.js";
import {
  DAILY_PROMPT_IDS,
  getDailyPromptId,
  getStandardPromptId,
  promptTranslationKey,
  STANDARD_PROMPT_IDS,
  STANDARD_TEST_ID,
} from "../assets/experiments/quick-prompts.js";
import {
  aggregateStandardResults,
  standardResultInternals,
} from "../assets/experiments/standard-result.js";
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
assert.equal(reference.voiceAge.ready, true);
assert.equal(reference.voiceAge.version, "voice-age-impression-2.0.0-research");
assert.ok(reference.voiceAge.min >= 18);
assert.ok(reference.voiceAge.max <= 65);
assert.equal(reference.voiceAge.confidenceKey, "medium");
assert.equal(reference.voiceQuality.version, "voice-quality-1.0.0");
assert.ok(reference.components.resonance > 0.5);

const connectedPerturbation = structuredClone(fixture);
connectedPerturbation.offlineSamples.extensions["voice-age-v2"].metrics.jitterLocal.valuePct = 18;
connectedPerturbation.offlineSamples.extensions["voice-age-v2"].metrics.shimmerLocal.valuePct = 30;
const connectedPerturbationResult = evaluateAdvancedExperience(connectedPerturbation);
assert.equal(
  connectedPerturbationResult.voiceAge.youthfulness,
  reference.voiceAge.youthfulness,
);

const missingVoiceQuality = structuredClone(fixture);
delete missingVoiceQuality.offlineSamples.extensions["voice-age-v2"];
const ageRejected = evaluateAdvancedExperience(missingVoiceQuality);
assert.equal(ageRejected.ready, true);
assert.equal(ageRejected.voiceAge.ready, false);
assert.deepEqual(ageRejected.voiceAge.reasons, ["metrics"]);
assert.equal(ageRejected.voiceAge.min, null);
assert.equal(ageRejected.voiceAge.max, null);

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

const promptMorning = getDailyPromptId(new Date(2026, 6, 31, 1, 0));
const promptEvening = getDailyPromptId(new Date(2026, 6, 31, 23, 59));
const promptNextDay = getDailyPromptId(new Date(2026, 7, 1, 0, 1));
assert.equal(promptMorning, promptEvening);
assert.notEqual(promptMorning, promptNextDay);
assert.equal(DAILY_PROMPT_IDS.length, 10);
assert.equal(STANDARD_PROMPT_IDS.length, 3);
assert.deepEqual(
  STANDARD_PROMPT_IDS.map((_, index) => getStandardPromptId(index)),
  STANDARD_PROMPT_IDS,
);
assert.equal(getStandardPromptId(3), null);
assert.equal(promptTranslationKey(promptMorning), `experiment.quick.prompts.${promptMorning}`);

const standardInputs = [64, 81, 72].map((score, index) => ({
  ...structuredClone(reference),
  archetypeKey: index === 1 ? "brightForward" : "airySweet",
  components: {
    ...reference.components,
    pitch: [0.62, 0.88, 0.74][index],
    strict: score / 100,
  },
  pitchHz: [210.2, 238.6, 226.4][index],
  score,
  voiceAge: {
    ...reference.voiceAge,
    max: [30, 24, 28][index],
    min: [22, 18, 20][index],
  },
}));
const standardResult = aggregateStandardResults(standardInputs);
assert.equal(standardResult.ready, true);
assert.equal(standardResult.score, 72);
assert.equal(standardResult.standard.spread, 17);
assert.equal(standardResult.standard.stabilityKey, "low");
assert.deepEqual(standardResult.standard.scores, [64, 81, 72]);
assert.equal(standardResult.archetypeKey, "airySweet");
assert.equal(standardResult.voiceAge.min, 20);
assert.equal(standardResult.voiceAge.max, 28);
assert.equal(standardResult.voiceAge.ready, true);
assert.equal(standardResult.voiceAge.version, reference.voiceAge.version);
assert.equal(standardResult.components.pitch, 0.74);
assert.equal(standardResult.pitchHz, 226.4);
assert.equal(standardResult.version, `${reference.version}.standard3`);
const standardAgeMissing = structuredClone(standardInputs);
standardAgeMissing[1].voiceAge.ready = false;
standardAgeMissing[1].voiceAge.min = null;
standardAgeMissing[1].voiceAge.max = null;
standardAgeMissing[1].voiceAge.reasons = ["quality"];
const standardAgeRejected = aggregateStandardResults(standardAgeMissing);
assert.equal(standardAgeRejected.ready, true);
assert.equal(standardAgeRejected.voiceAge.ready, false);
assert.equal(standardAgeRejected.voiceAge.min, null);
assert.ok(standardAgeRejected.voiceAge.reasons.includes("quality"));
assert.throws(() => aggregateStandardResults(standardInputs.slice(0, 2)), TypeError);
assert.equal(
  standardResultInternals.majority(["first", "second", "third"], "second"),
  "second",
);

class MockMediaRecorder {
  static isTypeSupported(type) {
    return type === "video/mp4" || type === "video/webm";
  }
}

assert.deepEqual(getSupportedVideoProfiles(MockMediaRecorder), [
  { extension: "mp4", mimeType: "video/mp4" },
  { extension: "webm", mimeType: "video/webm" },
]);
assert.deepEqual(getSupportedVideoProfiles(null), []);
assert.deepEqual(defaultClipRange(12), {
  duration: 12,
  end: 12,
  outputDuration: 12,
  start: 0,
});
assert.deepEqual(defaultClipRange(15), {
  duration: 15,
  end: 15,
  outputDuration: 15,
  start: 0,
});
assert.deepEqual(defaultClipRange(30), {
  duration: 30,
  end: 30,
  outputDuration: 30,
  start: 0,
});
assert.deepEqual(defaultClipRange(5), {
  duration: 5,
  end: 5,
  outputDuration: 5,
  start: 0,
});
const waveform = extractWaveform({
  getChannelData: () => Float32Array.from([
    0, 0.2, -0.5, 0.25, 1, -0.75, 0.1, 0,
    0, 0.4, -0.3, 0.2, 0.8, -0.6, 0.2, 0,
  ]),
  length: 16,
  numberOfChannels: 1,
  sampleRate: 8,
}, {
  duration: 2,
  end: 2,
  start: 0,
}, 16);
assert.equal(waveform.length, 16);
assert.equal(Math.max(...waveform), 1);
assert.ok(waveform.every((value) => value >= 0.04 && value <= 1));


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
assert.match(shareTargets.line, /^https:\/\/line\.me\/R\/share\?/);
assert.ok(Object.values(shareTargets).every((target) => target.includes(encodeURIComponent(shareUrl))));
const xTarget = new URL(shareTargets.x);
assert.equal(xTarget.searchParams.get("text"), "VPA score 72%");
assert.equal(xTarget.searchParams.get("url"), shareUrl);
assert.equal(xTarget.searchParams.get("hashtags"), "VoicePresentationAnalyzer");
const lineTarget = new URL(shareTargets.line);
assert.equal(lineTarget.searchParams.get("text"), `VPA score 72%\n${shareUrl}`);
assert.equal(lineTarget.searchParams.has("url"), false);
assert.equal(buildShareText("VPA score 72%", shareUrl), `VPA score 72%\n${shareUrl}`);

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
assert.deepEqual(compareChallenge(challenge, {
  ready: true,
  score: challenge.score + 4,
  version: reference.version,
}), {
  difference: 4,
  opponentScore: challenge.score,
  outcome: "beat",
  score: challenge.score + 4,
});
assert.equal(compareChallenge(challenge, {
  ready: true,
  score: challenge.score - 1,
  version: reference.version,
}).outcome, "behind");
assert.equal(compareChallenge(challenge, {
  ready: true,
  score: challenge.score,
  version: reference.version,
}).outcome, "tied");
assert.equal(compareChallenge(challenge, {
  ready: true,
  score: challenge.score,
  version: "advanced-beta-2",
}), null);

const dailyResult = {
  ...reference,
  quickTest: {
    mode: "daily",
    promptId: promptMorning,
  },
};
const dailyChallenge = createChallengePayload(dailyResult, {
  randomUUID: () => "87654321-4321-4321-4321-cba987654321",
});
assert.equal(dailyChallenge.schema, 3);
assert.equal(dailyChallenge.testMode, "daily");
assert.equal(dailyChallenge.promptId, promptMorning);
assert.equal(dailyChallenge.ageVersion, reference.voiceAge.version);
assert.deepEqual(decodeChallenge(encodeChallenge(dailyChallenge)), dailyChallenge);
assert.equal(compareChallenge(dailyChallenge, dailyResult).outcome, "tied");
assert.equal(compareChallenge(dailyChallenge, {
  ...dailyResult,
  quickTest: {
    mode: "daily",
    promptId: promptNextDay,
  },
}), null);

const noAgeChallenge = createChallengePayload({
  ...ageRejected,
  quickTest: {
    mode: "daily",
    promptId: promptMorning,
  },
}, {
  randomUUID: () => "99887766-1234-1234-1234-123456789abc",
});
assert.equal(noAgeChallenge.schema, 3);
assert.equal("ageMin" in noAgeChallenge, false);
assert.equal("ageMax" in noAgeChallenge, false);
assert.equal("ageVersion" in noAgeChallenge, false);
assert.deepEqual(decodeChallenge(encodeChallenge(noAgeChallenge)), noAgeChallenge);

const standardChallenge = createChallengePayload({
  ...standardResult,
  quickTest: {
    mode: "standard",
    promptId: STANDARD_TEST_ID,
  },
}, {
  randomUUID: () => "11223344-1234-1234-1234-123456789abc",
});
assert.equal(standardChallenge.schema, 3);
assert.equal(standardChallenge.testMode, "standard");
assert.equal(standardChallenge.promptId, STANDARD_TEST_ID);

const audioFile = new File(
  [new Blob(["audio"], { type: "audio/wav" })],
  "vpa-voice-clip.wav",
  { type: "audio/wav" },
);
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
  url: "https://example.com/vpa/#vpa-challenge=test",
});
assert.equal(nativeShare.method, "files");
assert.equal(nativeShareCalls[0].files.length, 2);
assert.equal(nativeShareCalls[0].text, "VPA result\nhttps://example.com/vpa/#vpa-challenge=test");
assert.equal(Object.hasOwn(nativeShareCalls[0], "url"), false);

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
  "share.captionWithoutAge",
  "share.cardDisclaimer",
  "share.challenge",
  "share.copied",
  "share.copy",
  "share.download",
  "share.downloaded",
  "share.failed",
  "share.opened",
  "share.platformAria",
  "share.platformHint",
  "share.primary",
  "share.shared",
  "share.title",
  "strictScore",
  "title",
  "voiceAge.title",
  "voiceAge.unavailable",
  "voiceAge.value",
  "voiceAgeV2.confidence",
  "voiceAgeV2.connectedReference",
  "voiceAgeV2.dbValue",
  "voiceAgeV2.eyebrow",
  "voiceAgeV2.metrics.cpp",
  "voiceAgeV2.metrics.hnr",
  "voiceAgeV2.metrics.jitter",
  "voiceAgeV2.metrics.shimmer",
  "voiceAgeV2.notAvailable",
  "voiceAgeV2.note",
  "voiceAgeV2.percentValue",
  "voiceAgeV2.rejected",
  "voiceAgeV2.sampleType",
  "voiceAgeV2.sampleTypes.connectedSpeech",
  "voiceAgeV2.sampleTypes.sustainedVowel",
  "voiceAgeV2.sampleTypes.unknown",
  "voiceAgeV2.title",
  "voiceAgeV2.used",
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
  "dynamic.audioFailed",
  "dynamic.brand",
  "dynamic.challenge",
  "dynamic.challengeLabel",
  "dynamic.close",
  "dynamic.download",
  "dynamic.downloaded",
  "dynamic.eyebrow",
  "dynamic.failed",
  "dynamic.fallbackAlt",
  "dynamic.fallbackHint",
  "dynamic.loadingAudio",
  "dynamic.noAudio",
  "dynamic.open",
  "dynamic.progress.decoding",
  "dynamic.progress.encoding",
  "dynamic.progress.keepOpen",
  "dynamic.readyFallback",
  "dynamic.readyVideo",
  "dynamic.retry",
  "dynamic.share",
  "dynamic.shared",
  "dynamic.shareFailed",
  "dynamic.shareTitle",
  "dynamic.subtitle",
  "dynamic.title",
  "dynamic.waveform",
  "footer",
  "history.label",
  "history.none",
  "history.score",
  "localeLabel",
  "prompt",
  "promptHint",
  "promptLabel",
  "prompts.arrival",
  "prompts.commute",
  "prompts.dinner",
  "prompts.directions",
  "prompts.repeat",
  "prompts.shopping",
  "prompts.timing",
  "prompts.weather",
  "prompts.weekend",
  "prompts.workload",
  "recording",
  "recordingHint",
  "requesting",
  "retry",
  "reveal.age",
  "reveal.archetype",
  "reveal.eyebrow",
  "reveal.feminine",
  "reveal.insight",
  "reveal.masculine",
  "reveal.score",
  "reveal.singleScore",
  "reveal.standardScore",
  "reveal.tendency",
  "reveal.tendencyAria",
  "share.audioDefault",
  "share.audioWarning",
  "share.cancelled",
  "share.copied",
  "share.copyChallenge",
  "share.directAria",
  "share.directHint",
  "share.directTitle",
  "share.downloaded",
  "share.failed",
  "share.includeAudio",
  "share.open",
  "share.shareTitle",
  "share.shared",
  "share.system",
  "share.title",
  "start",
  "standard.audioDefault",
  "standard.backDaily",
  "standard.cta",
  "standard.next",
  "standard.progress",
  "standard.scoresLabel",
  "standard.spread",
  "standard.stability",
  "standard.stabilityValues.high",
  "standard.stabilityValues.low",
  "standard.stabilityValues.medium",
  "standard.stepComplete",
  "standard.stepScore",
  "standard.subtitle",
  "standard.title",
  "standard.viewProfessional",
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
  "experiment.advanced.voiceAgeV2.dbValue",
  "experiment.advanced.voiceAgeV2.eyebrow",
  "experiment.advanced.voiceAgeV2.percentValue",
  "experiment.quick.dynamic.brand",
  "experiment.quick.dynamic.eyebrow",
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
