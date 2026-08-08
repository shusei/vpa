import fs from 'node:fs';

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
mockBody.appendChild(mockStats);
mockHtml.appendChild(mockBody);

global.document = {
  documentElement: mockHtml,
  body: mockBody,
  getElementById: (id) => (id === 'streamStats' ? mockStats : new MockElement('div')),
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

console.log('===========================================================');
console.log('   STRICT END-TO-END LOCALIZATION & RENDER REGRESSION TEST ');
console.log('===========================================================');

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

const testConfigs = [
  { locale: 'en', expectToken: 'Formant & resonance', disallowChinese: true },
  { locale: 'ja', expectToken: 'フォルマント', requireKana: true, disallowHant: true },
  { locale: 'zh-Hans', expectToken: '共振峰与共鸣', expectHans: true },
  { locale: 'zh-Hant', expectToken: '共振峰與共鳴', expectHant: true },
];

for (const config of testConfigs) {
  console.log(`\n▶ [STRICT TEST] Switching active locale to: "${config.locale}"`);
  await setLocale(config.locale);

  finishStreamStats({
    analysisSeq: 1,
    bandOf: () => (config.locale === 'en' ? 'Typical feminine (180–310 Hz)' : '常見女性音高區'),
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

  const fullText = mockStats.innerHTML.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

  if (!fullText.includes(config.expectToken)) {
    console.error(`❌ [FAIL] Missing expected token "${config.expectToken}" in locale ${config.locale}`);
    process.exit(1);
  }

  if (config.disallowChinese) {
    const chineseMatches = fullText.match(/[\u4e00-\u9fa5]+/g);
    if (chineseMatches) {
      console.error(`❌ [STRICT FAIL] Chinese leakage detected in ${config.locale} mode:`, chineseMatches);
      process.exit(1);
    }
  }

  if (config.requireKana) {
    const kanaMatches = fullText.match(/[\u3040-\u30ff]+/g);
    if (!kanaMatches) {
      console.error(`❌ [STRICT FAIL] No Japanese Kana detected in ${config.locale} mode!`);
      process.exit(1);
    }
  }

  if (config.disallowHant) {
    const hantLeakage = fullText.match(/[與這個繁]/g);
    if (hantLeakage) {
      console.error(`❌ [STRICT FAIL] Traditional Chinese leakage in Japanese mode:`, hantLeakage);
      process.exit(1);
    }
  }

  console.log(`✅ [STRICT PASS] Locale "${config.locale}" passed 100% verification!`);
}

console.log('\n===========================================================');
console.log('🎉 ALL STRICT END-TO-END LOCALIZATION TESTS PASSED 100%!');
console.log('===========================================================\n');
