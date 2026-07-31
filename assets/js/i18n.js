import zhHant from "../i18n/zh-Hant.js";

const STORAGE_KEY = "vpa.locale";
const BASE_LOCALES = ["zh-Hant", "zh-Hans", "en"];
const EXPERIMENTAL_LOCALES = typeof window !== "undefined" && Array.isArray(window.VPA_EXPERIMENT_LOCALES)
  ? window.VPA_EXPERIMENT_LOCALES
  : [];
const SUPPORTED_LOCALES = Array.from(new Set([
  ...BASE_LOCALES,
  ...EXPERIMENTAL_LOCALES.filter((locale) => locale === "ja"),
]));
const LOADERS = {
  "zh-Hans": () => import("../i18n/zh-Hans.js"),
  en: () => import("../i18n/en.js"),
  ja: () => import("../i18n/ja.js"),
};

let currentLocale = "zh-Hant";
let dictionary = zhHant;
const fallbackDictionary = zhHant;
let initPromise = null;
const listeners = new Set();

function resolve(dict, key) {
  if (!dict) return undefined;
  return key.split(".").reduce((acc, part) => (acc != null ? acc[part] : undefined), dict);
}

function interpolate(template, params = {}) {
  if (typeof template !== "string") return template;
  return template.replace(/\{\{(.*?)\}\}/g, (_, rawKey) => {
    const trimmed = rawKey.trim();
    if (Object.prototype.hasOwnProperty.call(params, trimmed)) {
      const value = params[trimmed];
      return value == null ? "" : String(value);
    }
    return "";
  });
}

function applyDomTranslations(root = document) {
  if (!root) return;

  root.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (!key) return;
    const text = t(key);
    if (typeof text === "string") {
      el.textContent = text;
    }
  });

  root.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const key = el.getAttribute("data-i18n-html");
    if (!key) return;
    const html = t(key);
    if (typeof html === "string") {
      el.innerHTML = html;
    }
  });

  root.querySelectorAll("[data-i18n-attrs]").forEach((el) => {
    const attrSpec = el.getAttribute("data-i18n-attrs");
    if (!attrSpec) return;
    attrSpec.split(",").forEach((pair) => {
      const [attr, key] = pair.split(":").map((part) => part.trim());
      if (!attr || !key) return;
      const value = t(key);
      if (typeof value === "string") {
        el.setAttribute(attr, value);
      }
    });
  });

  const title = t("meta.title");
  if (typeof title === "string" && title) {
    document.title = title;
  }
  const desc = t("meta.description");
  if (typeof desc === "string") {
    const meta = document.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute("content", desc);
  }
}

function mapCandidateLocale(candidate) {
  if (!candidate) return null;
  const lower = candidate.toLowerCase();
  if (lower.startsWith("zh-hant") || lower === "zh-tw" || lower === "zh-hk" || lower === "zh-mo") {
    return "zh-Hant";
  }
  if (lower.startsWith("zh")) {
    return "zh-Hans";
  }
  if (lower.startsWith("en")) {
    return "en";
  }
  if (lower.startsWith("ja") && SUPPORTED_LOCALES.includes("ja")) {
    return "ja";
  }
  return null;
}

function detectPreferredLocale() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED_LOCALES.includes(saved)) {
      return saved;
    }
  } catch {
    // ignore storage errors
  }

  const nav = typeof navigator !== "undefined" ? navigator : null;
  if (nav) {
    const langs = Array.isArray(nav.languages) && nav.languages.length ? nav.languages : [nav.language];
    for (const lang of langs) {
      const mapped = mapCandidateLocale(lang);
      if (mapped) return mapped;
    }
  }

  return "en";
}

async function loadDictionary(locale) {
  if (locale === "zh-Hant") return zhHant;
  const loader = LOADERS[locale];
  if (!loader) return zhHant;
  try {
    const mod = await loader();
    return mod?.default || zhHant;
  } catch (err) {
    console.error("[i18n] Failed to load locale", locale, err);
    return zhHant;
  }
}

async function internalSetLocale(locale, persist = true) {
  const target = SUPPORTED_LOCALES.includes(locale) ? locale : "en";
  if (target === currentLocale && dictionary) {
    applyDomTranslations();
    return currentLocale;
  }
  const nextDict = await loadDictionary(target);
  dictionary = nextDict || zhHant;
  currentLocale = target;
  document.documentElement.setAttribute("lang", target);
  if (persist) {
    try {
      localStorage.setItem(STORAGE_KEY, target);
    } catch {
      // ignore storage write issues
    }
  }
  applyDomTranslations();
  listeners.forEach((fn) => {
    try {
      fn(currentLocale);
    } catch (err) {
      console.error("[i18n] listener error", err);
    }
  });
  return currentLocale;
}

export function t(key, params) {
  const value = resolve(dictionary, key);
  const fallback = value === undefined ? resolve(fallbackDictionary, key) : value;
  if (fallback === undefined || fallback === null) {
    return "";
  }
  if (typeof fallback === "string") {
    return interpolate(fallback, params);
  }
  return fallback;
}

export function getLocaleValue(key) {
  const value = resolve(dictionary, key);
  if (value !== undefined) return value;
  return resolve(fallbackDictionary, key);
}

export function onLocaleChange(fn) {
  if (typeof fn !== "function") return () => { };
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function getCurrentLocale() {
  return currentLocale;
}

export async function initI18n() {
  if (!initPromise) {
    initPromise = (async () => {
      const locale = detectPreferredLocale();
      await internalSetLocale(locale, false);
      return currentLocale;
    })();
  }
  return initPromise;
}

export async function setLocale(locale) {
  return internalSetLocale(locale, true);
}

export const i18nInternals = {
  detectPreferredLocale,
  mapCandidateLocale,
  supportedLocales: [...SUPPORTED_LOCALES],
};

applyDomTranslations();
