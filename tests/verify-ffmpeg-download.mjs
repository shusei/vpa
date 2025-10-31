import assert from 'node:assert/strict';
import dns from 'node:dns/promises';

async function headOrRange(url) {
  let response = null;
  let networkIssue = false;
  try {
    response = await fetch(url, { method: 'HEAD', cache: 'no-store' });
  } catch (error) {
    if (!isSkippableError(error)) {
      throw error;
    }
    networkIssue = true;
  }

  if (!response || !(response.ok || response.status === 206)) {
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
      return { response: null, status: response?.status, networkIssue: true };
    }
  }

  if (!response) {
    return { response: null, status: undefined, networkIssue };
  }

  if (response.ok || response.status === 206) {
    return { response, status: response.status, networkIssue };
  }

  return { response: null, status: response.status, networkIssue };
}

async function pickReachableUrl(urls) {
  let networkIssue = false;
  let lastStatus;
  for (const url of urls) {
    const { response, status, networkIssue: candidateIssue } = await headOrRange(url);
    if (response) {
      return { url, response, networkIssue: false };
    }
    if (candidateIssue) {
      networkIssue = true;
    }
    if (typeof status === 'number') {
      lastStatus = status;
    }
  }

  return { url: urls[0], response: null, networkIssue, status: lastStatus };
}

const FFMPEG_VER = '0.12.15';
const CORE_VER = '0.12.15';
const FFMPEG_BASE = `https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@${FFMPEG_VER}/dist/esm`;
const CORE_BASE_NEW = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist`;
const CORE_BASE_OLD = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm`;

const FFMPEG_CHECKS = [
  {
    label: '@ffmpeg/ffmpeg',
    expect: 'javascript',
    urls: [`${FFMPEG_BASE}/index.js`],
  },
];

const CORE_CANDIDATE_CHECKS = [
  {
    label: '@ffmpeg/core js',
    expect: 'javascript',
    urls: [`${CORE_BASE_NEW}/ffmpeg-core.js`, `${CORE_BASE_OLD}/ffmpeg-core.js`],
  },
  {
    label: '@ffmpeg/core wasm',
    expect: 'application/wasm',
    urls: [`${CORE_BASE_NEW}/ffmpeg-core.wasm`, `${CORE_BASE_OLD}/ffmpeg-core.wasm`],
  },
  {
    label: '@ffmpeg/core worker',
    expect: 'javascript',
    urls: [`${CORE_BASE_NEW}/ffmpeg-core.worker.js`, `${CORE_BASE_OLD}/ffmpeg-core.worker.js`],
  },
];

const HOST = new URL(FFMPEG_CHECKS[0].urls[0]).hostname;

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

function validateResponse(response, expect, label) {
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

  return { allowOrigin };
}

async function main() {
  try {
    await dns.lookup(HOST);
  } catch (err) {
    console.warn(`[ffmpeg-check] DNS lookup failed for ${HOST}: ${err?.code || err?.message}`);
    console.warn('[ffmpeg-check] Skipping ffmpeg download verification (likely offline environment).');
    return;
  }

  for (const { label, expect, urls } of FFMPEG_CHECKS) {
    const { url, response, networkIssue } = await pickReachableUrl(urls);
    if (!response) {
      if (networkIssue) {
        console.warn(`[ffmpeg-check] Probe skipped for ${label} (${urls.join(' or ')}) due to network issues.`);
        return;
      }
      throw new Error(`Unable to reach ${label} asset (tried ${urls.join(', ')})`);
    }

    const { allowOrigin } = validateResponse(response, expect, label);

    console.log(
      `[ffmpeg-check] Verified ${label} (${url}) — ${response.status} ${response.statusText}, CORS: ${allowOrigin}`,
    );
  }

  for (const { label, expect, urls } of CORE_CANDIDATE_CHECKS) {
    const { url, response, networkIssue, status } = await pickReachableUrl(urls);
    if (!response) {
      if (networkIssue) {
        console.warn(`[ffmpeg-check] Probe skipped for ${label} (${urls.join(' or ')}) due to network issues.`);
        return;
      }
      throw new Error(
        `Unable to reach ${label} asset (tried ${urls.join(', ')}, last status: ${status ?? 'unknown'})`,
      );
    }

    const { allowOrigin } = validateResponse(response, expect, label);

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

