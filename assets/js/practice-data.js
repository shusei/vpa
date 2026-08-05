const FALLBACK = { version: 1, categories: [], phrases: [] };

function normalizeLocale(locale) {
  if (!locale) return "zh-Hant";
  const map = {
    "zh-TW": "zh-Hant",
    "zh-Hant": "zh-Hant",
    "zh-CN": "zh-Hans",
    "zh-Hans": "zh-Hans",
    "en-US": "en",
    en: "en",
    "en-GB": "en",
    "ja-JP": "ja",
    ja: "ja"
  };
  return map[locale] || "zh-Hant";
}

export async function loadPracticeData(locale = "zh-Hant") {
  const key = normalizeLocale(locale);
  const paths = [
    `assets/data/practice/${key}.json`,
    "assets/data/practice/zh-Hant.json",
  ];
  for (const path of paths) {
    try {
      const res = await fetch(path);
      if (!res?.ok) {
        console.error("[practice-data] fetch not ok for path:", path, "status:", res?.status);
        continue;
      }
      const data = await res.json();
      if (data && typeof data === "object") {
        return {
          version: Number.isFinite(data.version) ? data.version : 1,
          categories: Array.isArray(data.categories) ? data.categories : [],
          phrases: Array.isArray(data.phrases) ? data.phrases : []
        };
      }
    } catch (error) {
      console.error("[practice-data] load error", path, error);
    }
  }
  return { ...FALLBACK };
}
