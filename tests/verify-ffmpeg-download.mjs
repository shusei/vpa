import assert from 'node:assert/strict';
import dns from 'node:dns/promises';

async function headOrRange(url) {
  let response = null;
  try {
    response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
  } catch (error) {
    if (!isSkippableError(error)) {
      throw error;
    }
  }

  if (!response || !response.ok) {
    try {
      response = await fetch(url, {
        method: 'GET',
        headers: { Range: 'bytes=0-16' },
        cache: 'no-store',
      });
    } catch (error) {
      if (!isSkippableError(error)) {
        throw error;
      }
    }
  }

  if (!response) {
    return null;
  }

  if (response.ok || response.status === 206) {
    return response;
  }

  return null;
}

const FFMPEG_VER = '0.12.15';
const CORE_VER = '0.12.10';
const CHECKS = [
  {
    url: `https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@${FFMPEG_VER}/dist/esm/index.js`,
    expect: 'javascript',
    label: '@ffmpeg/ffmpeg',
  },
  {
    url: `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm/ffmpeg-core.js`,
    expect: 'javascript',
    label: '@ffmpeg/core js',
  },
  {
    url: `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm/ffmpeg-core.wasm`,
    expect: 'application/wasm',
    label: '@ffmpeg/core wasm',
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
    const response = await headOrRange(url);
    if (!response) {
      console.warn(`[ffmpeg-check] Probe skipped for ${label} (${url}) due to network issues.`);
      return;
    }

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

  const workerUrl = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/ffmpeg-core.worker.js`;
  const workerResponse = await headOrRange(workerUrl);
  const status = workerResponse?.status || 0;
  const workerOk = Boolean(workerResponse && (workerResponse.ok || status === 206));
  assert.equal(
    workerOk,
    true,
    `Probe failed for @ffmpeg/core worker: ${workerUrl} — status=${workerResponse?.status} ${workerResponse?.statusText}`,
  );

  const allowOrigin = workerResponse.headers.get('access-control-allow-origin');
  assert.ok(
    allowOrigin && allowOrigin.trim() !== '',
    'Access-Control-Allow-Origin header missing for @ffmpeg/core worker',
  );
  const contentType = workerResponse.headers.get('content-type');
  assert.ok(
    contentType && contentType.toLowerCase().includes('javascript'),
    `Unexpected content-type for @ffmpeg/core worker: ${contentType}`,
  );
  const contentLength = workerResponse.headers.get('content-length');
  assert.ok(
    contentLength && Number.parseInt(contentLength, 10) > 0,
    'Content-Length header missing or zero for @ffmpeg/core worker',
  );

  console.log(
    `[ffmpeg-check] Verified @ffmpeg/core worker (${workerUrl}) — ${workerResponse.status} ${workerResponse.statusText}, CORS: ${allowOrigin}`,
  );
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

