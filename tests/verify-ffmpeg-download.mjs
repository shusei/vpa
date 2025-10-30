import assert from 'node:assert/strict';
import dns from 'node:dns/promises';

const TARGET_URL = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core-mt@0.12.10/dist/umd/ffmpeg-core.wasm';
const HOST = new URL(TARGET_URL).hostname;

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

  let response;
  try {
    response = await fetch(TARGET_URL, { method: 'HEAD', cache: 'no-store' });
  } catch (err) {
    if (isSkippableError(err)) {
      const causeCode = err?.cause?.code || err?.cause?.errno;
      const detail = err?.code || causeCode || err?.message;
      console.warn(`[ffmpeg-check] Network unavailable (${detail}); skipping ffmpeg download verification.`);
      return;
    }
    throw err;
  }

  assert.ok(response.ok, `HEAD request failed with ${response.status} ${response.statusText}`);
  const allowOrigin = response.headers.get('access-control-allow-origin');
  assert.ok(allowOrigin && allowOrigin.trim() !== '', 'Access-Control-Allow-Origin header missing in response');
  const contentType = response.headers.get('content-type');
  assert.ok(contentType && contentType.includes('application/wasm'), `Unexpected content-type: ${contentType}`);
  const contentLength = response.headers.get('content-length');
  assert.ok(contentLength && Number.parseInt(contentLength, 10) > 0, 'Content-Length header missing or zero');

  console.log(`[ffmpeg-check] Verified ${TARGET_URL} — ${response.status} ${response.statusText}, CORS: ${allowOrigin}`);
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
