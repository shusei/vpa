import assert from 'node:assert/strict';
import dns from 'node:dns/promises';

const CHECKS = [
  {
    url: 'https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.15/dist/umd/index.js',
    expect: 'javascript',
    label: '@ffmpeg/util',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/umd/ffmpeg.min.js',
    expect: 'javascript',
    label: '@ffmpeg/ffmpeg',
  },
  {
    url: 'https://cdn.jsdelivr.net/npm/@ffmpeg/core-mt@0.12.15/dist/umd/ffmpeg-core.wasm',
    expect: 'application/wasm',
    label: '@ffmpeg/core-mt wasm',
  },
];
const HOST = new URL(CHECKS[0].url).hostname;

function isSkippableError(error) {
  const codes = new Set(['ENOTFOUND', 'ENETUNREACH', 'ECONNRESET', 'ECONNREFUSED', 'EAI_AGAIN', 'ETIMEDOUT']);
  if (!error) return false;
  if (codes.has(error.code) || codes.has(error.errno)) {
    return true;
  }
  if (error.cause && isSkippableError(error.cause)) {
    return true;
  }
  if (error instanceof AggregateError) {
    return error.errors.every((inner) => isSkippableError(inner));
  }
  return false;
}

async function main() {
  try {
    await dns.lookup(HOST);
  } catch (err) {
    console.warn(`[ffmpeg-check] DNS lookup failed for ${HOST}: ${err?.code || err?.message}`);
    console.warn('[ffmpeg-check] Skipping ffmpeg download verification (likely offline environment).');
    return;
  }

  for (const { url, expect, label } of CHECKS) {
    const response = await performHeadRequest(url);
    if (!response) {
      return;
    }

    assert.ok(response.ok, `HEAD request for ${label} failed with ${response.status} ${response.statusText}`);
    const allowOrigin = response.headers.get('access-control-allow-origin');
    assert.ok(allowOrigin && allowOrigin.trim() !== '', `Access-Control-Allow-Origin header missing for ${label}`);
    const contentType = response.headers.get('content-type');
    assert.ok(
      contentType && contentType.toLowerCase().includes(expect),
      `Unexpected content-type for ${label}: ${contentType}`,
    );
    const contentLength = response.headers.get('content-length');
    assert.ok(
      contentLength && Number.parseInt(contentLength, 10) > 0,
      `Content-Length header missing or zero for ${label}`,
    );

    console.log(
      `[ffmpeg-check] Verified ${label} (${url}) — ${response.status} ${response.statusText}, CORS: ${allowOrigin}`,
    );
  }
}

main().catch((err) => {
  if (isSkippableError(err)) {
    const causeCode = err?.cause?.code || err?.cause?.errno;
    const detail = err?.code || causeCode || err?.message;
    console.warn(`[ffmpeg-check] Network unavailable (${detail}); skipping ffmpeg download verification.`);
    return;
  }
  console.error('[ffmpeg-check] Failed to verify ffmpeg-core.wasm download.');
  console.error(err);
  process.exitCode = 1;
});

async function performHeadRequest(url) {
  try {
    const response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    if (response.status === 405) {
      return fetch(url, { method: 'GET', cache: 'no-store' });
    }
    return response;
  } catch (err) {
    if (isSkippableError(err)) {
      const causeCode = err?.cause?.code || err?.cause?.errno;
      const detail = err?.code || causeCode || err?.message;
      console.warn(`[ffmpeg-check] Network unavailable (${detail}); skipping ffmpeg download verification.`);
      return null;
    }
    throw err;
  }
}
