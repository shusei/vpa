import {
  isQuickPromptId,
  STANDARD_TEST_ID,
} from "./quick-prompts.js";

const HASH_KEY = "vpa-challenge";
const LEGACY_SCHEMA_VERSION = 1;
const PROMPT_SCHEMA_VERSION = 2;
const SCHEMA_VERSION = 3;
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
  const schema = Number(value.schema);
  const score = Math.round(Number(value.score));
  const ageMin = Math.round(Number(value.ageMin));
  const ageMax = Math.round(Number(value.ageMax));
  const ageVersion = String(value.ageVersion || "");
  const archetype = String(value.archetype || "");
  const scoreVersion = String(value.scoreVersion || "");
  const id = String(value.id || "");
  if (
    schema !== LEGACY_SCHEMA_VERSION
    && schema !== PROMPT_SCHEMA_VERSION
    && schema !== SCHEMA_VERSION
  ) return null;
  if (!Number.isFinite(score) || score < 0 || score > 100) return null;
  const hasAge = Number.isFinite(ageMin) && Number.isFinite(ageMax);
  if (schema !== SCHEMA_VERSION && !hasAge) return null;
  if (hasAge && (ageMin < 10 || ageMax > 90 || ageMin >= ageMax)) return null;
  if (schema === SCHEMA_VERSION && hasAge && !SCORE_VERSION_PATTERN.test(ageVersion)) return null;
  if (!ARCHETYPE_PATTERN.test(archetype)) return null;
  if (!SCORE_VERSION_PATTERN.test(scoreVersion)) return null;
  if (!CHALLENGE_ID_PATTERN.test(id)) return null;
  const normalized = {
    archetype,
    id,
    schema,
    score,
    scoreVersion,
  };
  if (hasAge) {
    normalized.ageMax = ageMax;
    normalized.ageMin = ageMin;
    if (schema === SCHEMA_VERSION) normalized.ageVersion = ageVersion;
  }
  if (schema === LEGACY_SCHEMA_VERSION) return normalized;

  const promptId = String(value.promptId || "");
  const testMode = String(value.testMode || "");
  const validPrompt = testMode === "daily"
    ? isQuickPromptId(promptId)
    : testMode === "standard" && promptId === STANDARD_TEST_ID;
  if (!validPrompt) return null;
  return {
    ...normalized,
    promptId,
    testMode,
  };
}

export function createChallengePayload(result, cryptoLike = globalThis.crypto) {
  const quickTest = result?.quickTest;
  const hasQuickTest = quickTest?.mode === "daily"
    ? isQuickPromptId(quickTest.promptId)
    : quickTest?.mode === "standard" && quickTest.promptId === STANDARD_TEST_ID;
  const hasAge = result?.voiceAge?.ready !== false
    && Number.isFinite(Number(result?.voiceAge?.min))
    && Number.isFinite(Number(result?.voiceAge?.max));
  return normalizeChallenge({
    ageMax: hasAge ? result.voiceAge.max : undefined,
    ageMin: hasAge ? result.voiceAge.min : undefined,
    ageVersion: hasAge ? result.voiceAge.version : undefined,
    archetype: result?.archetypeKey,
    id: makeChallengeId(cryptoLike),
    promptId: hasQuickTest ? quickTest.promptId : undefined,
    schema: hasQuickTest ? SCHEMA_VERSION : LEGACY_SCHEMA_VERSION,
    score: result?.score,
    scoreVersion: result?.version,
    testMode: hasQuickTest ? quickTest.mode : undefined,
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
  const configuredBase = locationLike === globalThis.location
    ? String(globalThis.VPA_PUBLIC_APP_URL || "").trim()
    : "";
  const url = new URL(configuredBase || locationLike.href);
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
  if (String(result?.version || "") !== normalized.scoreVersion) return null;
  if (normalized.schema === SCHEMA_VERSION) {
    if (result?.quickTest?.mode !== normalized.testMode) return null;
    if (result?.quickTest?.promptId !== normalized.promptId) return null;
  }
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
  LEGACY_SCHEMA_VERSION,
  PROMPT_SCHEMA_VERSION,
  SCHEMA_VERSION,
  normalizeChallenge,
};
