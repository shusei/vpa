import assert from "node:assert/strict";
import en from "../assets/i18n/en.js";
import ja from "../assets/i18n/ja.js";
import zhHans from "../assets/i18n/zh-Hans.js";
import zhHant from "../assets/i18n/zh-Hant.js";
import { MANUAL_DATA, MANUAL_SOURCE_URLS } from "../assets/js/manual-data.js";
import { MEDICAL_SAFETY_SOURCES } from "../assets/i18n/safety-copy.js";
import { ANALYSIS_GUIDANCE_TEXT } from "../assets/i18n/analysis-guidance.js";

const dictionaries = { en, ja, "zh-Hans": zhHans, "zh-Hant": zhHant };
const officialUrls = [
  MEDICAL_SAFETY_SOURCES.asha,
  MEDICAL_SAFETY_SOURCES.nidcdCare,
  MEDICAL_SAFETY_SOURCES.nidcdHoarseness,
  MEDICAL_SAFETY_SOURCES.ucsf,
];

assert.deepEqual(MANUAL_SOURCE_URLS, MEDICAL_SAFETY_SOURCES);

const unsafePrescriptions = [
  /recommended 8%[–-]18%/iu,
  /add gentle cord closure/iu,
  /vocal fold contact/iu,
  /short [“"]uh[”"] bursts/iu,
  /typical feminine range/iu,
  /female range/iu,
  /high priority/iu,
  /needs work/iu,
  /engage (?:the )?cords/iu,
  /加強聲帶閉合/u,
  /加强声带闭合/u,
  /女性常見(?:區|範圍)/u,
  /女性常见(?:区|范围)/u,
  /高優先/u,
  /高优先/u,
  /声帯閉鎖/u,
];

const safetySignals = {
  en: [/pain/iu, /hoarseness/iu, /stop/iu, /not (?:a )?medical/iu],
  ja: [/痛み/u, /かすれ/u, /中止/u, /医療/u],
  "zh-Hans": [/疼痛/u, /沙哑/u, /停止/u, /不是医疗/u],
  "zh-Hant": [/疼痛/u, /沙啞/u, /停止/u, /不是醫療/u],
};

for (const [locale, dictionary] of Object.entries(dictionaries)) {
  const guidanceLabels = ANALYSIS_GUIDANCE_TEXT[locale].labels;
  const separator = locale === "en" ? ": " : "：";
  const assertGuidance = (value, path) => {
    assert.equal(typeof value, "string", `${locale} ${path} must be a string`);
    for (const label of guidanceLabels) {
      assert.match(value, new RegExp(`${label}${separator}`), `${locale} ${path} is missing ${label}`);
    }
    assert.equal(value.split(/\r?\n/).filter(Boolean).length, 3, `${locale} ${path} must have exactly three guidance rows`);
  };

  const visibleCopy = JSON.stringify({
    analysis: dictionary.analysis,
    pitchBands: dictionary.pitchBands,
    experiment: dictionary.experiment,
    help: dictionary.help,
    helpDialog: dictionary.helpDialog,
    practice: dictionary.practice,
    realtime: dictionary.realtime,
    summary: dictionary.summary,
    manual: MANUAL_DATA[locale],
  });

  for (const pattern of unsafePrescriptions) {
    assert.doesNotMatch(visibleCopy, pattern, `${locale} still contains unsafe prescriptive copy: ${pattern}`);
  }
  for (const pattern of safetySignals[locale]) {
    assert.match(visibleCopy, pattern, `${locale} is missing required safety signal: ${pattern}`);
  }

  assert.ok(dictionary.experiment.quick.safety.title, `${locale} quick result lacks safety title`);
  assert.ok(dictionary.experiment.quick.safety.body, `${locale} quick result lacks safety explanation`);
  assert.ok(dictionary.experiment.quick.safety.stop, `${locale} quick result lacks stop warning`);
  assert.ok(dictionary.experiment.advanced.safety.care, `${locale} advanced result lacks care guidance`);
  assert.ok(dictionary.analysis.safety.focus, `${locale} granular focus cards lack safety boundary`);
  assert.ok(dictionary.practice.warmup.title, `${locale} pre-recording comfort check is missing`);
  assert.ok(MANUAL_DATA[locale]?.html, `${locale} manual is missing`);

  assert.doesNotMatch(
    JSON.stringify(dictionary.practice.warmup),
    /bridge of (?:your )?nose|chest-heavy|tongue forward|relax the larynx|vowel focus metric|鼻梁|舌頭前放|舌头前放|胸腔偏重|喉頭を緩|喉头放松/iu,
    `${locale} pre-recording card still contains anatomical or score-target training`,
  );

  for (const url of officialUrls) {
    assert.ok(
      dictionary.helpDialog.dialogHtml.includes(url),
      `${locale} Help is missing official source ${url}`,
    );
    assert.ok(MANUAL_DATA[locale].html.includes(url), `${locale} Manual is missing official source ${url}`);
  }

  assert.doesNotMatch(
    dictionary.experiment.quick.refine.retryAria,
    /higher score|更高分|高い点/u,
    `${locale} retry control still pressures users to chase a score`,
  );

  const guidedMetricGroups = {
    resonanceBalance: dictionary.analysis.resonanceBalance,
    tilt: dictionary.analysis.tilt,
    breathiness: dictionary.analysis.breathiness,
    brightness: dictionary.analysis.brightness,
    vowelFocus: dictionary.analysis.vowelFocus,
    speechRate: dictionary.analysis.speechRate,
    liaison: dictionary.analysis.liaison,
    intonationSlope: dictionary.analysis.intonation.slope,
    intonationRange: dictionary.analysis.intonation.range,
  };
  const metricNextSteps = [];
  for (const [groupName, states] of Object.entries(guidedMetricGroups)) {
    const hints = Object.entries(states).filter(([, entry]) => entry && typeof entry === "object" && "hint" in entry).map(([state, entry]) => {
      assertGuidance(entry.hint, `analysis.${groupName}.${state}.hint`);
      return entry.hint;
    });
    assert.ok(new Set(hints).size > 1, `${locale} ${groupName} still repeats one generic hint for every result`);
    const representative = Object.entries(states).find(([state, entry]) => state !== "insufficient" && entry?.hint)?.[1]?.hint;
    if (representative) metricNextSteps.push(representative.split(/\r?\n/)[2]);
  }
  assert.ok(new Set(metricNextSteps).size >= 8, `${locale} detailed metrics still share generic next steps instead of metric-specific guidance`);

  const formantNextSteps = [];
  for (const [formant, states] of Object.entries(dictionary.analysis.formant.guidance)) {
    for (const [state, value] of Object.entries(states)) {
      assertGuidance(value, `analysis.formant.guidance.${formant}.${state}`);
    }
    formantNextSteps.push(states.low.split(/\r?\n/)[2]);
  }
  assert.equal(new Set(formantNextSteps).size, 3, `${locale} F1/F2/F3 still share one generic next step`);
  assertGuidance(dictionary.analysis.meter.hint, "analysis.meter.hint");
  for (const [key, value] of Object.entries(dictionary.experiment.quick.insight)) {
    assert.equal(typeof value, "string", `${locale} experiment.quick.insight.${key} must be a string`);
    assert.ok(value.trim(), `${locale} experiment.quick.insight.${key} must not be empty`);
    assert.equal(value.split(/\r?\n/).filter(Boolean).length, 1, `${locale} experiment.quick.insight.${key} must be one concise line`);
    for (const label of guidanceLabels) {
      assert.doesNotMatch(value, new RegExp(`^${label}${separator}`), `${locale} experiment.quick.insight.${key} still starts with a guidance label`);
    }
  }
  assertGuidance(dictionary.experiment.advanced.pitchMedianHint, "experiment.advanced.pitchMedianHint");
  for (const [key, value] of Object.entries(dictionary.experiment.advanced.components.guidance)) {
    assertGuidance(value, `experiment.advanced.components.guidance.${key}`);
  }
  for (const [key, value] of Object.entries(dictionary.experiment.advanced.voiceAgeV2.metricGuidance)) {
    assertGuidance(value, `experiment.advanced.voiceAgeV2.metricGuidance.${key}`);
  }
  for (const [key, value] of Object.entries(dictionary.experiment.advanced.insight)) {
    if (key !== "label") assertGuidance(value, `experiment.advanced.insight.${key}`);
  }
  for (const [key, value] of Object.entries(dictionary.summary.focus.items)) {
    assertGuidance(value, `summary.focus.items.${key}`);
  }
}

console.log("[PASS] Compact Quick summaries plus Advanced, Help, and Manual safety-guidance audits passed in all 4 locales.");
