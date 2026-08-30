import assert from 'node:assert/strict';
import { access, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (process.env.SKIP_VENDOR_FFMPEG === '1') {
  console.warn('[ffmpeg-check] Skipping ffmpeg asset verification (SKIP_VENDOR_FFMPEG=1).');
  process.exit(0);
}

const REQUIRED_FILES = [
  {
    label: 'ffmpeg worker',
    relativePath: 'assets/vendor/ffmpeg/worker.js',
    minimumBytes: 1024,
  },
  {
    label: 'ffmpeg core js',
    relativePath: 'assets/vendor/ffmpeg/ffmpeg-core.js',
    minimumBytes: 1024,
  },
];

const OPTIONAL_FILES = [
  {
    label: 'ffmpeg core wasm',
    relativePath: 'assets/vendor/ffmpeg/ffmpeg-core.wasm',
    minimumBytes: 1_000_000,
  },
];

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

async function ensureFile({ label, relativePath, minimumBytes }) {
  const fullPath = path.resolve(ROOT, relativePath);
  await access(fullPath);
  const info = await stat(fullPath);
  assert.ok(info.isFile(), `${label} is not a regular file (${relativePath})`);
  assert.ok(
    info.size >= minimumBytes,
    `${label} appears truncated — expected at least ${minimumBytes} bytes, saw ${info.size}`,
  );
  console.log(`[ffmpeg-check] Verified ${label} at ${relativePath} (${info.size} bytes)`);
}

async function ensureOptionalFile(file) {
  try {
    await ensureFile(file);
  } catch (error) {
    if (error.code === 'ENOENT') {
      console.warn(
        `[ffmpeg-check] Optional ${file.label} missing (${file.relativePath}); skipping size verification.`,
      );
      return;
    }

    if (error.name === 'AssertionError' && /appears truncated/.test(error.message)) {
      console.warn(
        `[ffmpeg-check] Optional ${file.label} appears truncated (${file.relativePath}); skipping size verification.`,
      );
      return;
    }

    throw error;
  }
}

async function main() {
  const deploymentWorkflow = await readFile(
    path.resolve(ROOT, '.github/workflows/pages.yml'),
    'utf8',
  );
  const fetchStep = deploymentWorkflow.indexOf('run: npm run fetch:ffmpeg');
  const testStep = deploymentWorkflow.indexOf('run: npm test');
  const buildStep = deploymentWorkflow.indexOf('run: npm run build');
  const deployStep = deploymentWorkflow.indexOf('uses: actions/deploy-pages@v4');
  const deployedAssetCheck = deploymentWorkflow.indexOf('/assets/vendor/ffmpeg/ffmpeg-core.wasm" --output /dev/null');
  assert.ok(fetchStep >= 0, 'Pages deployment must download the ffmpeg WASM asset.');
  assert.ok(testStep > fetchStep, 'Pages deployment must run the complete test suite after preparing ffmpeg.');
  assert.ok(buildStep > testStep, 'Pages deployment must pass the complete test suite before building dist.');
  assert.ok(buildStep > fetchStep, 'Pages deployment must download ffmpeg before building dist.');
  assert.ok(deployedAssetCheck > deployStep, 'Pages deployment must verify the hosted ffmpeg WASM asset.');
  console.log('[ffmpeg-check] Verified Pages prepares ffmpeg and passes all tests before deployment.');

  for (const file of REQUIRED_FILES) {
    await ensureFile(file);
  }

  for (const file of OPTIONAL_FILES) {
    await ensureOptionalFile(file);
  }
}

main().catch((err) => {
  console.error('[ffmpeg-check] Missing ffmpeg vendored asset.');
  console.error(err);
  console.error('Run "npm run fetch:ffmpeg" to download the expected files.');
  process.exitCode = 1;
});
