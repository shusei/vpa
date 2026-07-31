const MAX_IMAGE_BYTES = 400_000;

function normalizeOrigin(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:" && url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      return "";
    }
    return url.origin;
  } catch {
    return "";
  }
}

export function getShareServiceOrigin(globalLike = window) {
  return normalizeOrigin(globalLike.VPA_SHARE_SERVICE_ORIGIN);
}

export function isShareServiceConfigured(globalLike = window) {
  return Boolean(getShareServiceOrigin(globalLike));
}

export function getPublicAppUrl(
  locationLike = window.location,
  globalLike = window,
) {
  const configured = String(globalLike.VPA_PUBLIC_APP_URL || "").trim();
  if (configured) {
    try {
      return new URL(configured).toString();
    } catch {
      // Fall back to the current deployment.
    }
  }
  const url = new URL(locationLike.href);
  url.hash = "";
  url.search = "";
  url.pathname = url.pathname.replace(/dev\.html$/, "");
  return url.toString();
}

export async function publishShareResult({
  fetchLike = fetch,
  imageBlob,
  metadata,
  serviceOrigin = getShareServiceOrigin(),
}) {
  if (!serviceOrigin) throw new Error("share service is not configured");
  if (!(imageBlob instanceof Blob) || imageBlob.type !== "image/jpeg") {
    throw new TypeError("a JPEG share image is required");
  }
  if (!imageBlob.size || imageBlob.size > MAX_IMAGE_BYTES) {
    throw new RangeError("share image is too large");
  }
  const form = new FormData();
  form.append("image", imageBlob, "vpa-result.jpg");
  form.append("metadata", JSON.stringify({
    ...metadata,
    schema: 1,
  }));
  const response = await fetchLike(`${serviceOrigin}/api/shares`, {
    body: form,
    cache: "no-store",
    credentials: "omit",
    method: "POST",
  });
  if (!response.ok) {
    let detail = "";
    try {
      detail = String((await response.json())?.error || "");
    } catch {
      // The status code remains useful if the body is not JSON.
    }
    throw new Error(detail || `share service returned ${response.status}`);
  }
  const result = await response.json();
  const url = new URL(String(result?.url || ""));
  if (url.origin !== serviceOrigin || !/^\/r\/[A-Za-z0-9_-]{16}$/.test(url.pathname)) {
    throw new Error("share service returned an invalid URL");
  }
  return {
    id: String(result.id || ""),
    imageUrl: String(result.imageUrl || ""),
    url: url.toString(),
  };
}

export const shareServiceInternals = {
  MAX_IMAGE_BYTES,
  normalizeOrigin,
};
