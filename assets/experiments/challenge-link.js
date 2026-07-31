const HASH_KEY = "vpa-challenge";
const SCHEMA_VERSION = 1;
const SCORE_VERSION_PATTERN = /^[a-z0-9._-]{1,48}$/i;
const ARCHETYPE_PATTERN = /^[a-z][a-zA-Z0-9]{1,47}$/;
const CHALLENGE_ID_PATTERN = /^[a-zA-Z0-9-]{8,64}$/;

function bytesToBase64Url(bytes) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
}

function base64UrlToBytes(value) {
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function makeChallengeId(cryptoLike = globalThis.crypto) {
  if (typeof cryptoLike?.randomUUID === "function") {
    return cryptoLike.randomUUID();
  }
  const random = Math.random().toString(36).slice(2);
  return `${Date.now().toString(36)}-${random}`;
}

function normalizeChallenge(value) {
  if (!value || typeof value !== "object") return null;
  const score = Math.round(Number(value.score));
  const ageMin = Math.round(Number(value.ageMin));
  const ageMax = Math.round(Number(value.ageMax));
  const archetype = String(value.archetype || "");
  const scoreVersion = String(value.scoreVersion || "");
  const id = String(value.id || "");
  if (value.schema !== SCHEMA_VERSION) return null;
  if (!Number.isFinite(score) || score < 0 || score > 100) return null;
  if (!Number.isFinite(ageMin) || !Number.isFinite(ageMax)) return null;
  if (ageMin < 10 || ageMax > 90 || ageMin >= ageMax) return null;
  if (!ARCHETYPE_PATTERN.test(archetype)) return null;
  if (!SCORE_VERSION_PATTERN.test(scoreVersion)) return null;
  if (!CHALLENGE_ID_PATTERN.test(id)) return null;
  return {
    ageMax,
    ageMin,
    archetype,
    id,
    schema: SCHEMA_VERSION,
    score,
    scoreVersion,
  };
}

export function createChallengePayload(result, cryptoLike = globalThis.crypto) {
  return normalizeChallenge({
    ageMax: result?.voiceAge?.max,
    ageMin: result?.voiceAge?.min,
    archetype: result?.archetypeKey,
    id: makeChallengeId(cryptoLike),
    schema: SCHEMA_VERSION,
    score: result?.score,
    scoreVersion: result?.version,
  });
}

export function encodeChallenge(payload) {
  const normalized = normalizeChallenge(payload);
  if (!normalized) throw new TypeError("invalid challenge payload");
  const json = JSON.stringify(normalized);
  return bytesToBase64Url(new TextEncoder().encode(json));
}

export function decodeChallenge(encoded) {
  if (typeof encoded !== "string" || !encoded || encoded.length > 1024) return null;
  try {
    const json = new TextDecoder().decode(base64UrlToBytes(encoded));
    return normalizeChallenge(JSON.parse(json));
  } catch {
    return null;
  }
}

export function createChallengeUrl(result, locationLike = window.location) {
  const payload = createChallengePayload(result);
  if (!payload) throw new TypeError("result cannot create a challenge");
  const url = new URL(locationLike.href);
  url.search = "";
  url.hash = `${HASH_KEY}=${encodeChallenge(payload)}`;
  return {
    payload,
    url: url.toString(),
  };
}

export function readChallenge(locationLike = window.location) {
  try {
    const hash = String(locationLike.hash || "").replace(/^#/, "");
    const params = new URLSearchParams(hash);
    return decodeChallenge(params.get(HASH_KEY));
  } catch {
    return null;
  }
}

export function compareChallenge(challenge, result) {
  const normalized = normalizeChallenge(challenge);
  const score = Math.round(Number(result?.score));
  if (!normalized || !Number.isFinite(score)) return null;
  const difference = score - normalized.score;
  return {
    difference,
    outcome: difference > 0 ? "beat" : difference < 0 ? "behind" : "tied",
    opponentScore: normalized.score,
    score,
  };
}

export const challengeLinkInternals = {
  HASH_KEY,
  SCHEMA_VERSION,
  normalizeChallenge,
};
