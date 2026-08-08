import assert from "node:assert/strict";
import en from "../assets/i18n/en.js";
import ja from "../assets/i18n/ja.js";
import zhHans from "../assets/i18n/zh-Hans.js";
import zhHant from "../assets/i18n/zh-Hant.js";
import { MANUAL_DATA, MANUAL_SOURCE_URLS } from "../assets/js/manual-data.js";
import { MEDICAL_SAFETY_SOURCES } from "../assets/i18n/safety-copy.js";

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
  const visibleCopy = JSON.stringify({
    analysis: dictionary.analysis,
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
}

console.log("[PASS] Medical-safety copy audit passed for Quick, Advanced, Help, and Manual in all 4 locales.");
