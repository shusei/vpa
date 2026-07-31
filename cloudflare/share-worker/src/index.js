const DEFAULT_APP_URL = "https://shusei.github.io/vpa/";
const DEFAULT_SHARE_TTL_DAYS = 365;
const MAX_IMAGE_BYTES = 400_000;
const MAX_METADATA_BYTES = 8_000;
const SHARE_ID_PATTERN = /^[A-Za-z0-9_-]{16}$/;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function truncate(value, limit) {
  return Array.from(String(value ?? "").trim()).slice(0, limit).join("");
}

function allowedOrigins(env) {
  return String(env.SITE_ORIGINS || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const headers = new Headers({
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  });
  if (allowedOrigins(env).includes(origin)) {
    headers.set("Access-Control-Allow-Origin", origin);
  }
  return headers;
}

function jsonResponse(request, env, value, status = 200) {
  const headers = corsHeaders(request, env);
  headers.set("Cache-Control", "no-store");
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("X-Content-Type-Options", "nosniff");
  return new Response(JSON.stringify(value), { headers, status });
}

function isAllowedUploadOrigin(request, env) {
  const origin = request.headers.get("Origin") || "";
  return Boolean(origin && allowedOrigins(env).includes(origin));
}

function publicAppUrl(env) {
  try {
    return new URL(env.PUBLIC_APP_URL || DEFAULT_APP_URL);
  } catch {
    return new URL(DEFAULT_APP_URL);
  }
}

function normalizeTargetUrl(value, env) {
  const appUrl = publicAppUrl(env);
  const target = new URL(String(value || ""), appUrl);
  if (target.protocol !== "https:" && target.protocol !== "http:") {
    throw new TypeError("unsupported target URL");
  }
  if (target.origin !== appUrl.origin || !target.pathname.startsWith(appUrl.pathname)) {
    throw new TypeError("target URL is outside the configured app");
  }
  return target.toString();
}

function normalizeMetadata(value, env) {
  if (!value || typeof value !== "object" || Number(value.schema) !== 1) {
    throw new TypeError("invalid metadata schema");
  }
  const locale = ["zh-Hant", "zh-Hans", "en", "ja"].includes(value.locale)
    ? value.locale
    : "zh-Hant";
  const title = truncate(value.title, 100);
  const description = truncate(value.description, 240);
  const alt = truncate(value.alt || description, 200);
  if (!title || !description) throw new TypeError("missing share copy");
  return {
    alt,
    description,
    locale,
    schema: 1,
    targetUrl: normalizeTargetUrl(value.targetUrl, env),
    title,
  };
}

function shareTtlSeconds(env) {
  const configured = Math.floor(Number(env.SHARE_TTL_DAYS));
  const days = Number.isFinite(configured)
    ? Math.max(1, Math.min(DEFAULT_SHARE_TTL_DAYS, configured))
    : DEFAULT_SHARE_TTL_DAYS;
  return days * 24 * 60 * 60;
}

function createShareId(cryptoLike = globalThis.crypto) {
  const bytes = new Uint8Array(12);
  cryptoLike.getRandomValues(bytes);
  let binary = "";
  bytes.forEach((value) => {
    binary += String.fromCharCode(value);
  });
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
}

function hasJpegSignature(bytes) {
  return bytes.length >= 4
    && bytes[0] === 0xff
    && bytes[1] === 0xd8
    && bytes[2] === 0xff;
}

function imageBytesFromRow(value) {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  if (Array.isArray(value)) return Uint8Array.from(value);
  return new Uint8Array();
}

async function createShare(request, env) {
  if (!env.SHARE_DB) return jsonResponse(request, env, { error: "storage unavailable" }, 503);
  if (!isAllowedUploadOrigin(request, env)) {
    return jsonResponse(request, env, { error: "origin not allowed" }, 403);
  }
  const contentLength = Number(request.headers.get("Content-Length") || 0);
  if (contentLength > MAX_IMAGE_BYTES + MAX_METADATA_BYTES + 64_000) {
    return jsonResponse(request, env, { error: "request too large" }, 413);
  }

  let form;
  try {
    form = await request.formData();
  } catch {
    return jsonResponse(request, env, { error: "invalid form data" }, 400);
  }
  const image = form.get("image");
  const metadataText = String(form.get("metadata") || "");
  if (!(image instanceof Blob) || image.type !== "image/jpeg") {
    return jsonResponse(request, env, { error: "a JPEG image is required" }, 400);
  }
  if (!metadataText || new TextEncoder().encode(metadataText).byteLength > MAX_METADATA_BYTES) {
    return jsonResponse(request, env, { error: "invalid metadata size" }, 400);
  }

  let metadata;
  try {
    metadata = normalizeMetadata(JSON.parse(metadataText), env);
  } catch (error) {
    return jsonResponse(request, env, { error: error.message || "invalid metadata" }, 400);
  }

  const imageBytes = new Uint8Array(await image.arrayBuffer());
  if (!imageBytes.length || imageBytes.length > MAX_IMAGE_BYTES || !hasJpegSignature(imageBytes)) {
    return jsonResponse(request, env, { error: "invalid JPEG image" }, 400);
  }

  const id = createShareId();
  const createdAt = Math.floor(Date.now() / 1000);
  const expiresAt = createdAt + shareTtlSeconds(env);
  try {
    await env.SHARE_DB.prepare(`
      INSERT INTO shares (
        id, created_at, expires_at, locale, title, description, alt, target_url, image
      ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
    `).bind(
      id,
      createdAt,
      expiresAt,
      metadata.locale,
      metadata.title,
      metadata.description,
      metadata.alt,
      metadata.targetUrl,
      imageBytes.buffer,
    ).run();
  } catch (error) {
    console.error("[share-worker] D1 write failed", error);
    return jsonResponse(request, env, { error: "unable to save share" }, 500);
  }

  const origin = new URL(request.url).origin;
  return jsonResponse(request, env, {
    expiresAt: new Date(expiresAt * 1000).toISOString(),
    id,
    imageUrl: `${origin}/i/${id}.jpg`,
    url: `${origin}/r/${id}`,
  }, 201);
}

async function readShare(id, env) {
  if (!SHARE_ID_PATTERN.test(id) || !env.SHARE_DB) return null;
  let row;
  try {
    row = await env.SHARE_DB.prepare(`
      SELECT locale, title, description, alt, target_url, image, expires_at
      FROM shares
      WHERE id = ?1 AND expires_at > ?2
    `).bind(id, Math.floor(Date.now() / 1000)).first();
  } catch (error) {
    console.error("[share-worker] D1 read failed", error);
    return null;
  }
  if (!row) return null;
  const imageBytes = imageBytesFromRow(row.image);
  if (!hasJpegSignature(imageBytes) || imageBytes.length > MAX_IMAGE_BYTES) return null;
  try {
    return {
      expiresAt: Number(row.expires_at),
      imageBytes,
      metadata: normalizeMetadata({
        alt: row.alt,
        description: row.description,
        locale: row.locale,
        schema: 1,
        targetUrl: row.target_url,
        title: row.title,
      }, env),
    };
  } catch {
    return null;
  }
}

function resultHtml({ imageUrl, metadata, resultUrl }) {
  const title = escapeHtml(metadata.title);
  const description = escapeHtml(metadata.description);
  const alt = escapeHtml(metadata.alt);
  const targetUrl = escapeHtml(metadata.targetUrl);
  const safeImageUrl = escapeHtml(imageUrl);
  const safeResultUrl = escapeHtml(resultUrl);
  return `<!doctype html>
<html lang="${escapeHtml(metadata.locale)}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>${title}</title>
  <link rel="canonical" href="${safeResultUrl}" />
  <meta name="description" content="${description}" />
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Voice Presentation Analyzer" />
  <meta property="og:title" content="${title}" />
  <meta property="og:description" content="${description}" />
  <meta property="og:url" content="${safeResultUrl}" />
  <meta property="og:image" content="${safeImageUrl}" />
  <meta property="og:image:secure_url" content="${safeImageUrl}" />
  <meta property="og:image:type" content="image/jpeg" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta property="og:image:alt" content="${alt}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="${title}" />
  <meta name="twitter:description" content="${description}" />
  <meta name="twitter:image" content="${safeImageUrl}" />
  <meta name="twitter:image:alt" content="${alt}" />
  <style>
    :root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#0d2026;color:#f7f2e8}
    *{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:radial-gradient(circle at top,#27444d,#0d2026 65%)}
    main{width:min(100%,900px);display:grid;gap:20px;text-align:center}img{display:block;width:100%;height:auto;border-radius:24px;box-shadow:0 24px 70px rgba(0,0,0,.34)}
    p{margin:0;color:#c9dbd8;font-size:1.05rem;line-height:1.7}a{justify-self:center;padding:13px 22px;border-radius:999px;background:#f2c978;color:#17272c;font-weight:800;text-decoration:none}
  </style>
</head>
<body>
  <main>
    <img src="${safeImageUrl}" width="1200" height="630" alt="${alt}" />
    <p>${description}</p>
    <a href="${targetUrl}">Voice Presentation Analyzer</a>
  </main>
</body>
</html>`;
}

function cacheControl(expiresAt) {
  const remaining = Math.max(60, Number(expiresAt) - Math.floor(Date.now() / 1000));
  return `public, max-age=${Math.min(86_400, remaining)}`;
}

async function showResult(request, env, id) {
  const share = await readShare(id, env);
  if (!share) return Response.redirect(publicAppUrl(env).toString(), 302);
  const url = new URL(request.url);
  const resultUrl = `${url.origin}/r/${id}`;
  const imageUrl = `${url.origin}/i/${id}.jpg`;
  return new Response(resultHtml({ imageUrl, metadata: share.metadata, resultUrl }), {
    headers: {
      "Cache-Control": cacheControl(share.expiresAt),
      "Content-Security-Policy": "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
      "Content-Type": "text/html; charset=utf-8",
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

async function showImage(request, env, id) {
  const share = await readShare(id, env);
  if (!share) return new Response("Not found", { status: 404 });
  return new Response(request.method === "HEAD" ? null : share.imageBytes, {
    headers: {
      "Cache-Control": cacheControl(share.expiresAt),
      "Content-Length": String(share.imageBytes.byteLength),
      "Content-Type": "image/jpeg",
      ETag: `"${id}"`,
      "X-Content-Type-Options": "nosniff",
    },
  });
}

async function deleteExpiredShares(env) {
  if (!env.SHARE_DB) return;
  await env.SHARE_DB.prepare(`
    DELETE FROM shares
    WHERE id IN (
      SELECT id FROM shares
      WHERE expires_at <= ?1
      ORDER BY expires_at
      LIMIT 5000
    )
  `).bind(Math.floor(Date.now() / 1000)).run();
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS" && url.pathname === "/api/shares") {
      return new Response(null, { headers: corsHeaders(request, env), status: 204 });
    }
    if (request.method === "POST" && url.pathname === "/api/shares") {
      return createShare(request, env);
    }
    const resultMatch = url.pathname.match(/^\/r\/([A-Za-z0-9_-]{16})\/?$/);
    if (request.method === "GET" && resultMatch) {
      return showResult(request, env, resultMatch[1]);
    }
    const imageMatch = url.pathname.match(/^\/i\/([A-Za-z0-9_-]{16})\.jpg$/);
    if ((request.method === "GET" || request.method === "HEAD") && imageMatch) {
      return showImage(request, env, imageMatch[1]);
    }
    if (request.method === "GET" && url.pathname === "/") {
      return new Response("VPA share service", {
        headers: { "Cache-Control": "no-store", "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    return new Response("Not found", { status: 404 });
  },

  async scheduled(_controller, env, ctx) {
    ctx.waitUntil(deleteExpiredShares(env));
  },
};

export const shareWorkerInternals = {
  createShareId,
  deleteExpiredShares,
  hasJpegSignature,
  imageBytesFromRow,
  normalizeMetadata,
  resultHtml,
  shareTtlSeconds,
};
