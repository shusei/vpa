import assert from 'node:assert/strict';
import { access, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (process.env.SKIP_VENDOR_FFMPEG === '1') {
  console.warn('[ffmpeg-check] Skipping ffmpeg asset verification (SKIP_VENDOR_FFMPEG=1).');
  process.exit(0);
}

const EXPECTED_FILES = [
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

async function main() {
  for (const file of EXPECTED_FILES) {
    await ensureFile(file);
  }
}

main().catch((err) => {
  console.error('[ffmpeg-check] Missing ffmpeg vendored asset.');
  console.error(err);
  console.error('Run "npm run fetch:ffmpeg" to download the expected files.');
  process.exitCode = 1;
});
