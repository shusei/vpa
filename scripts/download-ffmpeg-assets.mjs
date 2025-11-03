#!/usr/bin/env node
import assert from 'node:assert/strict';
import { createWriteStream } from 'node:fs';
import { mkdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import https from 'node:https';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const TARGET_DIR = path.join(ROOT, 'assets', 'vendor', 'ffmpeg');

const FILES = [
  {
    name: 'worker.js',
    url: 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm/worker.js',
    minimumBytes: 1024,
  },
  {
    name: 'ffmpeg-core.js',
    url: 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.js',
    minimumBytes: 1024,
  },
  {
    name: 'ffmpeg-core.wasm',
    url: 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm/ffmpeg-core.wasm',
    minimumBytes: 1_000_000,
  },
];

function fetchToFile(url, destination) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, (response) => {
      if (response.statusCode && response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        const nextUrl = new URL(response.headers.location, url).toString();
        response.resume();
        fetchToFile(nextUrl, destination).then(resolve, reject);
        return;
      }

      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Unexpected status ${response.statusCode} for ${url}`));
        return;
      }

      const file = createWriteStream(destination);
      response.pipe(file);
      file.on('finish', () => {
        file.close(resolve);
      });
      file.on('error', (error) => {
        file.close(() => reject(error));
      });
    });

    request.setTimeout(60_000, () => {
      request.destroy(new Error(`Request timed out while downloading ${url}`));
    });
    request.on('error', reject);
  });
}

async function ensureFile({ name, url, minimumBytes }) {
  const destination = path.join(TARGET_DIR, name);
  let needsDownload = true;
  try {
    const existing = await stat(destination);
    if (existing.isFile() && existing.size >= minimumBytes) {
      console.log(`[ffmpeg-fetch] ${name} already present (${existing.size} bytes)`);
      needsDownload = false;
    } else {
      console.warn(`[ffmpeg-fetch] ${name} present but too small (${existing.size} bytes), re-downloading`);
    }
  } catch (error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }
  }

  if (needsDownload) {
    console.log(`[ffmpeg-fetch] Downloading ${name} from ${url}`);
    await fetchToFile(url, destination);
  }

  const info = await stat(destination);
  assert.ok(info.isFile(), `${name} was not written as a regular file`);
  assert.ok(
    info.size >= minimumBytes,
    `${name} appears truncated — expected at least ${minimumBytes} bytes, saw ${info.size}`,
  );
  console.log(`[ffmpeg-fetch] Ready: ${name} (${info.size} bytes)`);
}

async function main() {
  await mkdir(TARGET_DIR, { recursive: true });
  for (const file of FILES) {
    await ensureFile(file);
  }
}

main().catch((error) => {
  console.error('[ffmpeg-fetch] Failed to download ffmpeg assets');
  console.error(error);
  process.exitCode = 1;
});
