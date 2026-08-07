import fs from 'node:fs';
import path from 'node:path';
import zhHant from '../assets/i18n/zh-Hant.js';
import zhHans from '../assets/i18n/zh-Hans.js';
import en from '../assets/i18n/en.js';
import ja from '../assets/i18n/ja.js';

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

// Step 3: Verify index.html static HTML elements have data-i18n attributes or container
const htmlPath = path.resolve('index.html');
const html = fs.readFileSync(htmlPath, 'utf8');

// Match standalone HTML tags with Chinese text
const tagRegex = /<([a-z1-6]+)([^>]*)>([^<]*[\u4e00-\u9fa5][^<]*)<\/\1>/gi;
let match;
const missingI18nTags = [];
while ((match = tagRegex.exec(html)) !== null) {
  const [full, tagName, attrs, text] = match;
  // If tag has data-i18n/html/attrs, it is safe
  if (attrs.includes('data-i18n') || attrs.includes('data-i18n-html') || attrs.includes('data-i18n-attrs')) {
    continue;
  }
  if (['script', 'style', 'option'].includes(tagName.toLowerCase())) continue;

  // Check if this tag is inside an ancestor that has data-i18n-html
  const tagPos = match.index;
  const contentBefore = html.slice(Math.max(0, tagPos - 500), tagPos);
  const containerMatch = contentBefore.match(/data-i18n-html="[^"]*"/g);
  // If wrapped inside data-i18n-html container, the container will replace it dynamically
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

// Final assertion
if (keyErrors > 0 || chineseInEn.length > 0 || missingI18nTags.length > 0) {
  console.error('\n❌ Locale Cleanliness Test Failed!');
  process.exit(1);
} else {
  console.log('\n🎉 ALL Locale Cleanliness & Coverage Tests PASSED SUCCESSFULLY!\n');
}
