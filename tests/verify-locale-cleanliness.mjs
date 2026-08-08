import fs from 'node:fs';
import path from 'node:path';

// Minimal DOM mock required by i18n.js
const mockElement = {
  getAttribute: () => null,
  setAttribute: () => {},
  querySelectorAll: () => [],
  querySelector: () => null,
};

global.document = {
  querySelectorAll: () => [],
  querySelector: () => null,
  getElementById: () => null,
  documentElement: {
    setAttribute: () => {},
    getAttribute: (attr) => (attr === 'lang' ? 'en' : null),
  },
  body: mockElement,
  title: '',
};
global.window = global;
global.localStorage = { getItem: () => 'en', setItem: () => {} };

import zhHant from '../assets/i18n/zh-Hant.js';
import zhHans from '../assets/i18n/zh-Hans.js';
import en from '../assets/i18n/en.js';
import ja from '../assets/i18n/ja.js';

const packageVersion = JSON.parse(fs.readFileSync(path.resolve('package.json'), 'utf8')).version;
const { t, setLocale } = await import(`../assets/js/i18n.js?v=${packageVersion}`);
import { bandOf } from '../assets/js/summary-helpers.js';
import { createFocusHelpers } from '../assets/js/focus-insights.js';
import { computeAdvancedSummary } from '../assets/js/advanced-summary-core.js';
import {
  averageFinite,
  averageEnergy,
  summarizeBreathiness,
  buildEligibleFrameMask,
  categorizeBrightness,
  detectVoiceLeaning,
  normalizeResonanceBands,
  describeResonanceFromEnergy,
  categorizeTilt,
  categorizeBreathiness,
  makeFormantHint,
  summarizeFormantTrends,
  buildFormantTrendDisplay,
  analyzeVowelFocus,
  analyzeSpeechRate,
  analyzeConnectedSpeech,
  analyzeIntonation,
} from '../assets/js/advanced-metrics.js';

console.log('=== [Locale Cleanliness & Coverage Regression Test] ===');

const dictionaries = { 'zh-Hant': zhHant, 'zh-Hans': zhHans, 'en': en, 'ja': ja };

// Step 1: Verify Dictionary Keys Symmetry across all 4 locales
function getFlatKeys(obj, prefix = '') {
  let keys = {};
  for (const k in obj) {
    const p = prefix ? prefix + '.' + k : k;
    if (typeof obj[k] === 'object' && obj[k] !== null && !Array.isArray(obj[k])) {
      Object.assign(keys, getFlatKeys(obj[k], p));
    } else {
      keys[p] = obj[k];
    }
  }
  return keys;
}

const flatDicts = {};
for (const [lang, dict] of Object.entries(dictionaries)) {
  flatDicts[lang] = getFlatKeys(dict);
}

const baseKeys = Object.keys(flatDicts['zh-Hant']);
let keyErrors = 0;

for (const lang of ['zh-Hans', 'en', 'ja']) {
  const missing = baseKeys.filter(k => !(k in flatDicts[lang]));
  if (missing.length) {
    console.error(`❌ [FAIL] Missing ${missing.length} keys in ${lang} dictionary:`, missing);
    keyErrors += missing.length;
  }
}

if (keyErrors === 0) {
  console.log(`✅ [PASS] All 4 locale dictionaries have 100% symmetric key coverage (${baseKeys.length} keys each).`);
}

// Step 2: Verify EN dictionary contains ZERO CJK Chinese characters
const enFlat = flatDicts['en'];
const chineseInEn = [];
const allowedEnExceptions = new Set([
  'topbar.localeNames.zhHant',
  'topbar.localeNames.zhHans',
  'topbar.localeNames.ja'
]);

for (const [key, val] of Object.entries(enFlat)) {
  if (typeof val === 'string' && /[\u4e00-\u9fa5]/.test(val)) {
    if (!allowedEnExceptions.has(key)) {
      chineseInEn.push({ key, val });
    }
  }
}

if (chineseInEn.length > 0) {
  console.error(`❌ [FAIL] Found ${chineseInEn.length} Chinese text contamination in EN dictionary:`);
  chineseInEn.forEach(item => console.error(`  - ${item.key}: "${item.val}"`));
} else {
  console.log(`✅ [PASS] EN dictionary contains ZERO unintended Chinese characters.`);
}

// Step 3: Verify every production import resolves to one cache-busted i18n module instance
const assetsRoot = path.resolve('assets');
const canonicalI18nPath = path.resolve('assets/js/i18n.js');
const expectedI18nQuery = `v=${packageVersion}`;
const i18nImportErrors = [];

function listJavaScriptFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(dir, entry.name);
    return entry.isDirectory() ? listJavaScriptFiles(target) : (entry.name.endsWith('.js') ? [target] : []);
  });
}

for (const file of listJavaScriptFiles(assetsRoot)) {
  const source = fs.readFileSync(file, 'utf8');
  const importPattern = /from\s+["']([^"']*i18n\.js(?:\?[^"']*)?)["']/g;
  for (const match of source.matchAll(importPattern)) {
    const [modulePath, query = ''] = match[1].split('?');
    const resolvedModule = path.resolve(path.dirname(file), modulePath);
    if (resolvedModule !== canonicalI18nPath || query !== expectedI18nQuery) {
      i18nImportErrors.push({
        file: path.relative(process.cwd(), file),
        specifier: match[1],
      });
    }
  }
}

if (i18nImportErrors.length > 0) {
  console.error('❌ [FAIL] Production modules do not share one canonical i18n instance:', i18nImportErrors);
} else {
  console.log(`✅ [PASS] Every production module shares assets/js/i18n.js?${expectedI18nQuery}.`);
}

// Step 4: Verify index.html static HTML elements have data-i18n attributes or container
const htmlPath = path.resolve('index.html');
const html = fs.readFileSync(htmlPath, 'utf8');

const requiredInfoBindings = [
  'info.interfaceHtml',
  'info.modelHtml',
  'info.accuracyHtml',
  'info.methodHtml',
  'info.ethicsHtml',
  'info.compatHtml',
  'info.versionHtml',
];
const missingInfoBindings = requiredInfoBindings.filter((key) => (
  !html.includes(`data-i18n-html="${key}"`)
));
if (missingInfoBindings.length > 0) {
  console.error('❌ [FAIL] Full information sections are missing translation containers:', missingInfoBindings);
} else {
  console.log('✅ [PASS] Every full information section is translated as one complete container.');
}
const tagRegex = /<([a-z1-6]+)([^>]*)>([^<]*[\u4e00-\u9fa5][^<]*)<\/\1>/gi;
let match;
const missingI18nTags = [];
while ((match = tagRegex.exec(html)) !== null) {
  const [full, tagName, attrs, text] = match;
  if (attrs.includes('data-i18n') || attrs.includes('data-i18n-html') || attrs.includes('data-i18n-attrs')) {
    continue;
  }
  if (['script', 'style', 'option'].includes(tagName.toLowerCase())) continue;

  const tagPos = match.index;
  const contentBefore = html.slice(Math.max(0, tagPos - 500), tagPos);
  const containerMatch = contentBefore.match(/data-i18n-html="[^"]*"/g);
  if (containerMatch && contentBefore.lastIndexOf('data-i18n-html') > contentBefore.lastIndexOf('</div>')) {
    continue;
  }

  missingI18nTags.push({ tagName, text: text.trim(), full: full.slice(0, 100) });
}

if (missingI18nTags.length > 0) {
  console.error(`❌ [FAIL] Found ${missingI18nTags.length} static HTML tags containing Chinese WITHOUT data-i18n in index.html:`);
  missingI18nTags.forEach(t => console.error(`  - <${t.tagName}> "${t.text}"`));
} else {
  console.log(`✅ [PASS] All static HTML tags with text in index.html are 100% bound with data-i18n attributes.`);
}

// Step 5: Dynamic Analysis Metrics Evaluation in EN Mode
await setLocale('en');

const mockStore = {
  duration: 10,
  frameSec: 0.05,
  pitchProcessed: new Array(200).fill(220),
  pitchRaw: new Array(200).fill(220),
  pitchConfidence: new Array(200).fill(0.9),
  voiced: new Array(200).fill(true),
  db: new Array(200).fill(65),
  formants: new Array(200).fill([400, 1800, 2800]),
  tilt: new Array(200).fill(-12),
  breathiness: new Array(200).fill(0.12),
  energy: new Array(200).fill([0.2, 0.5, 0.3]),
};

const advDeps = {
  analyzeConnectedSpeech,
  analyzeIntonation,
  analyzeSpeechRate,
  analyzeVowelFocus,
  averageEnergy,
  averageFinite,
  buildEligibleFrameMask,
  categorizeBreathiness,
  categorizeBrightness,
  categorizeTilt,
  describeResonanceFromEnergy,
  detectVoiceLeaning,
  FORMANT_CONFIDENCE_THRESHOLD: 0.5,
  FORMANT_MAX_GAP_FRAMES: 8,
  lastPf: 0.7,
  lastPm: 0.3,
  makeStats: (arr) => ({ avg: 220, med: 220, p05: 190, p95: 240, sd: 10 }),
  offlineFeatureStore: mockStore,
  percentileSorted: (arr, p) => 50,
  PS_INTERVAL_MS: 50,
  summarizeBreathiness,
  summarizeFormantTrends,
};

const advSummary = computeAdvancedSummary(advDeps);
const advJson = JSON.stringify(advSummary || {});
const advChinese = advJson.match(/[\u4e00-\u9fa5]+/g);

const { buildFocusInsights, renderFocusBlock } = createFocusHelpers({
  fmt1: (v) => String(v),
  getSummaryText: () => en.summary,
  summaryString: (key, params) => t('summary.' + key, params),
  t: t,
});

const focus = buildFocusInsights({
  band: bandOf(220, { t, PS_MAX_HZ: 600 }),
  stabilityKey: 'wide',
  stabilityLabel: t('summary.stability.wide'),
  spread: 85,
  snr: 10,
  snrDisplay: '10 dB',
  snrKey: 'noisy',
  snrLabel: t('summary.snrTags.noisy'),
  diverge: true,
  trendLabel: t('realtime.meter.feminine'),
  advSummary,
  voicedHintKey: 'low',
  voicedHintLabel: t('summary.voicedHint.low'),
});

const focusJson = JSON.stringify(focus || {});
const focusChinese = focusJson.match(/[\u4e00-\u9fa5]+/g);

let dynamicErrors = 0;
if (advChinese) {
  console.error('❌ [FAIL] Chinese text found in EN mode computeAdvancedSummary output:', advChinese);
  dynamicErrors++;
}
if (focusChinese) {
  console.error('❌ [FAIL] Chinese text found in EN mode Focus Insights output:', focusChinese);
  dynamicErrors++;
}

if (dynamicErrors === 0) {
  console.log('✅ [PASS] Dynamic Analysis Metrics & Focus Insights in EN mode are 100% free of Chinese characters.');
}

// Final assertion
if (keyErrors > 0 || chineseInEn.length > 0 || i18nImportErrors.length > 0 || missingInfoBindings.length > 0 || missingI18nTags.length > 0 || dynamicErrors > 0) {
  console.error('\n❌ Locale Cleanliness Test Failed!');
  process.exit(1);
} else {
  console.log('\n🎉 ALL Locale Cleanliness & Coverage Tests PASSED SUCCESSFULLY!\n');
}
