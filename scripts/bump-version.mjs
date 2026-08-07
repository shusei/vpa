import fs from 'node:fs';
import path from 'node:path';

const projectRoot = path.resolve(import.meta.dirname, '..');
const pkgPath = path.resolve(projectRoot, 'package.json');
const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

const currentVersion = pkg.version;
const type = process.argv[2] || 'patch';

const parts = currentVersion.split('.').map(Number);
if (type === 'minor') {
  parts[1]++;
  parts[2] = 0;
} else if (type === 'major') {
  parts[0]++;
  parts[1] = 0;
  parts[2] = 0;
} else {
  parts[2]++;
}

const nextVersion = parts.join('.');
pkg.version = nextVersion;
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n', 'utf8');

console.log(`=== Bumping Version: ${currentVersion} -> ${nextVersion} ===`);

const filesToUpdate = [
  'index.html',
  'dev.html',
  'assets/app.js',
  'assets/app-core.js',
  'assets/js/i18n.js',
  'assets/experiments/experience-shell.js',
  'assets/experiments/advanced-experience.js',
  'assets/experiments/dynamic-card-controller.js',
  'assets/experiments/public-share.js',
  'assets/experiments/share-card.js',
  'assets/i18n/zh-Hant.js',
  'assets/i18n/zh-Hans.js',
  'assets/i18n/en.js',
  'assets/i18n/ja.js',
];

const newVersionTag = `${nextVersion}`;

for (const relPath of filesToUpdate) {
  const absPath = path.resolve(projectRoot, relPath);
  if (!fs.existsSync(absPath)) continue;

  let content = fs.readFileSync(absPath, 'utf8');
  content = content.replace(/\?v=[a-zA-Z0-9.-]+/g, `?v=${newVersionTag}`);
  fs.writeFileSync(absPath, content, 'utf8');
  console.log(`✅ Cache-busting updated: ${relPath}`);
}

// Update verify-social-preview.mjs expected tokens
const socialTestPath = path.resolve(projectRoot, 'tests/verify-social-preview.mjs');
if (fs.existsSync(socialTestPath)) {
  let content = fs.readFileSync(socialTestPath, 'utf8');
  content = content.replace(/\?v=[a-zA-Z0-9.-]+/g, `?v=${newVersionTag}`);
  fs.writeFileSync(socialTestPath, content, 'utf8');
  console.log(`✅ Updated tests/verify-social-preview.mjs expected tokens`);
}

console.log(`\n🎉 Successfully bumped version to ${nextVersion} and updated all cache-busting tags!\n`);
