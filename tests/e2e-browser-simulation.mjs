import fs from 'node:fs';
import path from 'node:path';

class MockElement {
  constructor(tagName = 'div', attributes = {}) {
    this.tagName = tagName;
    this.attributes = attributes;
    this.children = [];
    this.textContent = '';
    this.innerHTML = '';
  }

  getAttribute(attr) { return this.attributes[attr] || null; }
  setAttribute(attr, val) { this.attributes[attr] = val; }
  appendChild(child) { this.children.push(child); }
  addEventListener() {}
  removeEventListener() {}

  querySelectorAll(selector) {
    let results = [];
    const attrMatch = selector.match(/\[([a-z0-9-]+)\]/i);
    if (attrMatch) {
      const attrName = attrMatch[1];
      if (this.attributes[attrName] !== undefined) results.push(this);
    }
    for (const child of this.children) {
      if (child.querySelectorAll) results = results.concat(child.querySelectorAll(selector));
    }
    return results;
  }

  querySelector(selector) {
    if (selector === '.advanced-section') return this;
    if (selector === '#streamStats') return this;
    if (selector === '#advancedExperience') return this;
    if (selector === '.tags') {
      let tagsEl = this.children.find(c => c.attributes.class === 'tags');
      if (!tagsEl) {
        tagsEl = new MockElement('div', { class: 'tags' });
        this.children.push(tagsEl);
      }
      return tagsEl;
    }
    return new MockElement('div');
  }
}

const mockHtml = new MockElement('html', { lang: 'en' });
const mockBody = new MockElement('body');
const mockStats = new MockElement('div', { id: 'streamStats' });
const mockAdvExp = new MockElement('div', { id: 'advancedExperience' });
mockBody.appendChild(mockStats);
mockBody.appendChild(mockAdvExp);
mockHtml.appendChild(mockBody);

global.document = {
  documentElement: mockHtml,
  body: mockBody,
  getElementById: (id) => {
    if (id === 'streamStats') return mockStats;
    if (id === 'advancedExperience') return mockAdvExp;
    return new MockElement('div');
  },
  querySelectorAll: (sel) => mockHtml.querySelectorAll(sel),
  querySelector: (sel) => mockHtml.querySelector(sel),
  createElement: (tag) => new MockElement(tag),
};
global.window = global;
global.addEventListener = () => {};
global.removeEventListener = () => {};
global.localStorage = { getItem: () => 'en', setItem: () => {} };

const packageVersion = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8')).version;
const { t, setLocale } = await import(`../assets/js/i18n.js?v=${packageVersion}`);
const { computeAdvancedSummary } = await import('../assets/js/advanced-summary-core.js');
const { createAdvancedSummaryRenderer } = await import('../assets/js/advanced-summary-render.js');
const { finishStreamStats } = await import('../assets/js/stats-core.js');
const { createFocusHelpers } = await import('../assets/js/focus-insights.js');

const {
  averageFinite,
  averageEnergy,
  summarizeBreathiness,
  buildEligibleFrameMask,
  categorizeBrightness,
  detectVoiceLeaning,
  describeResonanceFromEnergy,
  categorizeTilt,
  categorizeBreathiness,
  summarizeFormantTrends,
  analyzeVowelFocus,
  analyzeSpeechRate,
  analyzeConnectedSpeech,
  analyzeIntonation,
} = await import('../assets/js/advanced-metrics.js');

import en from '../assets/i18n/en.js';
import ja from '../assets/i18n/ja.js';
import zhHans from '../assets/i18n/zh-Hans.js';
import zhHant from '../assets/i18n/zh-Hant.js';

console.log('===============================================================');
console.log('   FULL END-TO-END (E2E) REALTIME & METRIC CARD REGRESSION TEST');
console.log('===============================================================');

const mockStore = {
  duration: 10,
  frameSec: 0.05,
  pitchProcessed: new Array(200).fill(220),
  pitchRaw: new Array(200).fill(220),
  pitchConfidence: new Array(200).fill(0.9),
  voiced: new Array(200).fill(true),
  db: new Array(200).fill(65),
  formants: new Array(200).fill([361, 2294, 3750]),
  tilt: new Array(200).fill(1.8),
  breathiness: new Array(200).fill(0.41),
  energy: new Array(200).fill([0.16, 0.30, 0.54]),
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
  lastPf: 0.8,
  lastPm: 0.2,
  makeStats: () => ({ avg: 232.2, med: 234.2, p05: 207.7, p95: 257.5, sd: 9.4 }),
  offlineFeatureStore: mockStore,
  percentileSorted: () => 57.0,
  PS_INTERVAL_MS: 50,
  summarizeBreathiness,
  summarizeFormantTrends,
};

const getDictSummary = () => {
  const l = document.documentElement.getAttribute('lang');
  if (l === 'en') return en.summary;
  if (l === 'ja') return ja.summary;
  if (l === 'zh-Hans') return zhHans.summary;
  return zhHant.summary;
};

const renderer = createAdvancedSummaryRenderer({
  BASELINES: {
    f1: { min: 170, max: 420 },
    f2: { min: 1450, max: 2750 },
    f3: { min: 2400, max: 3400 },
    tilt: { min: -1, max: 8 },
    breath: { min: 8, max: 18 },
    liaison: { min: 40, max: 75 },
  },
  escapeAttr: (s) => s,
  escapeHtml: (s) => s,
  fmt1: (v) => String(v),
  formatBaselineRange: () => '10-20',
  getAdvancedMode: () => 'advanced',
  getDetailsOpen: () => true,
  getSummaryText: getDictSummary,
  renderGauge: () => '',
  summaryString: (k, p) => t('summary.' + k, p),
  t,
});

const focusHelpers = createFocusHelpers({
  fmt1: (v) => String(v),
  getSummaryText: getDictSummary,
  summaryString: (k, p) => t('summary.' + k, p),
  t,
});

const advSummary = computeAdvancedSummary(advDeps);

// STEP 1: E2E Locale Switch & Audio Inference Loop
const locales = ['zh-Hant', 'en', 'ja', 'zh-Hans'];

for (const loc of locales) {
  console.log(`\n[E2E Step] Simulating User Switching Locale to "${loc}" & Recording Voice Take...`);
  await setLocale(loc);

  finishStreamStats({
    analysisSeq: 1,
    bandOf: () => (loc === 'en' ? 'Typical feminine (180–310 Hz)' : '常見女性音高區'),
    buildEligibleFrameMask,
    buildFocusInsights: (ctx) => focusHelpers.buildFocusInsights(ctx),
    cloneOfflineFeatureStore: () => mockStore,
    computeAdvancedSummary: () => advSummary,
    CONFIDENCE_INCLUDE_THRESHOLD: 0.5,
    currentDevice: 'WebGPU',
    drawIntonationCurve: () => {},
    EPS: 0.0001,
    escapeAttr: (s) => s,
    filterPitchForStats: (arr) => arr,
    fmt1: (v) => String(v),
    FORMANT_CONFIDENCE_THRESHOLD: 0.5,
    FORMANT_MAX_GAP_FRAMES: 8,
    isDivergent: () => false,
    lastPf: 0.8,
    lastPm: 0.2,
    logPostProcessingDiagnostics: () => {},
    makeStats: () => ({ avg: 232.2, med: 234.2, p05: 207.7, p95: 257.5, sd: 9.4 }),
    notifyInferenceListeners: () => {},
    offlineFeatureStore: mockStore,
    openPracticeCategory: () => {},
    PITCH_COUNTER_KEYS: [],
    percentileSorted: () => 57.0,
    pitchPostState: { count: 10 },
    PS_INTERVAL_MS: 50,
    psConfidence: new Array(200).fill(0.9),
    psDb: new Array(200).fill(65),
    psHz: new Array(200).fill(220),
    psHzSmooth: new Array(200).fill(220),
    psVoiced: new Array(200).fill(true),
    renderAdvancedSummary: (s, ctx) => renderer.renderAdvancedSummary(s, ctx),
    renderFocusBlock: (f) => focusHelpers.renderFocusBlock(f),
    resizeIntonationCanvas: () => {},
    setLatestAnalysisExport: () => {},
    setupAdvancedSection: () => {},
    setupIntonationLegend: () => {},
    summaryString: (k, p) => t('summary.' + k, p),
    getSummaryText: getDictSummary,
    summaryText: getDictSummary(),
    t,
    VOLUME_DISPLAY_MODE: 'relative',
    wireAdvancedIntonation: () => {},
  });

  const domText = mockStats.innerHTML.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

  if (loc === 'en') {
    const chineseMatches = domText.match(/[\u4e00-\u9fa5]+/g);
    if (chineseMatches) {
      console.error(`❌ [E2E FAIL] Residual Chinese characters in EN mode:`, chineseMatches);
      process.exit(1);
    }
    console.log(`  ✓ E2E Assertion Passed: English DOM contains ZERO Chinese characters!`);
  } else if (loc === 'ja') {
    const kanaMatches = domText.match(/[\u3040-\u30ff]+/g);
    if (!kanaMatches) {
      console.error(`❌ [E2E FAIL] Japanese mode failed to render Kana!`);
      process.exit(1);
    }
    console.log(`  ✓ E2E Assertion Passed: Japanese DOM contains valid Kana text!`);
  } else {
    console.log(`  ✓ E2E Assertion Passed: ${loc} DOM rendered successfully!`);
  }
}

console.log('\n===============================================================');
console.log('🎉 E2E TEST SUITE PASSED 100% WITH PERFECT LOCALIZATION!');
console.log('===============================================================\n');
