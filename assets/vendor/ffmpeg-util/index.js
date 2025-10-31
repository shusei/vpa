const HAS_BUFFER = typeof Buffer !== "undefined" && typeof Buffer.from === "function";

function isArrayBufferLike(value) {
  return value instanceof ArrayBuffer || (value && typeof value === "object" && value.constructor?.name === "SharedArrayBuffer");
}

function isArrayBufferView(value) {
  return typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView && ArrayBuffer.isView(value);
}

function isBlobLike(value) {
  return (
    value &&
    typeof value === "object" &&
    typeof value.arrayBuffer === "function" &&
    (typeof value.type === "string" || typeof value.size === "number")
  );
}

function isResponse(value) {
  return value && typeof value === "object" && typeof value.arrayBuffer === "function" && typeof value.blob === "function";
}

function cloneUint8Array(view) {
  const array = new Uint8Array(view.byteLength);
  array.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return array;
}

async function convertToUint8Array(source) {
  if (source == null) {
    throw new TypeError("fetchFile: input must not be null or undefined");
  }

  if (source instanceof Uint8Array) {
    return cloneUint8Array(source);
  }

  if (isArrayBufferView(source)) {
    return cloneUint8Array(source);
  }

  if (isArrayBufferLike(source)) {
    return new Uint8Array(source);
  }

  if (HAS_BUFFER && Buffer.isBuffer(source)) {
    return Uint8Array.from(source);
  }

  if (isBlobLike(source) || isResponse(source)) {
    const buffer = await source.arrayBuffer();
    return new Uint8Array(buffer);
  }

  return null;
}

async function fetchFromUrl(url) {
  if (typeof fetch !== "function") {
    throw new TypeError("fetchFile: global fetch API is not available");
  }

  const response = await fetch(url);
  if (!response || !response.ok) {
    throw new Error(`fetchFile: failed to load resource \"${url}\": ${response?.status || "unknown"}`);
  }
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}

export async function fetchFile(input) {
  const localResult = await convertToUint8Array(input);
  if (localResult) {
    return localResult;
  }

  if (input instanceof URL) {
    return fetchFromUrl(input.href);
  }

  if (typeof input === "string") {
    const trimmed = input.trim();
    if (trimmed === "") {
      throw new TypeError("fetchFile: empty string is not a valid resource");
    }
    return fetchFromUrl(trimmed);
  }

  if (typeof File !== "undefined" && input instanceof File) {
    const buffer = await input.arrayBuffer();
    return new Uint8Array(buffer);
  }

  throw new TypeError("fetchFile: unsupported input type");
}

export async function toBlobURL(url, mimeType) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response || !response.ok) {
    const status = response ? `${response.status} ${response.statusText}` : "unknown";
    throw new Error(`toBlobURL fetch failed: ${status}`);
  }
  const buffer = await response.arrayBuffer();
  const blob = new Blob([buffer], { type: mimeType });
  return URL.createObjectURL(blob);
}

export default fetchFile;
