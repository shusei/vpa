function readCachedThreadCount(storageKey) {
  try {
    if (typeof localStorage === "undefined") {
      return null;
    }
    const raw = localStorage.getItem(storageKey);
    if (raw == null) {
      return null;
    }
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 1) {
      return null;
    }
    return parsed;
  } catch (_error) {
    return null;
  }
}

function writeCachedThreadCount(storageKey, value) {
  try {
    if (typeof localStorage === "undefined") {
      return;
    }
    localStorage.setItem(storageKey, String(value));
  } catch (_error) {
    // 忽略快取失敗，Safari 無痕模式可能會丟例外
  }
}

export function detectThreadCount({ storageKey }) {
  const fallback = 1;
  const nav = typeof navigator !== "undefined" ? navigator : null;

  if (!nav) {
    return { threads: fallback, reason: "no-navigator" };
  }

  try {
    const concurrencyRaw = nav.hardwareConcurrency;
    const concurrency = Number.isFinite(concurrencyRaw) && concurrencyRaw > 0
      ? Math.floor(concurrencyRaw)
      : fallback;

    const ua = String(nav.userAgent || "").toLowerCase();
    const platform = String(nav.platform || "").toLowerCase();
    const vendor = String(nav.vendor || "").toLowerCase();
    const maxTouchPoints = Number(nav.maxTouchPoints || 0);

    const isAndroid = ua.includes("android");
    const isiOS = /iphone|ipad|ipod/.test(ua) || (platform === "macintel" && maxTouchPoints > 1);
    const isMobile = isAndroid || isiOS || ua.includes("mobile");

    const isSafari = (() => {
      if (!ua.includes("safari")) {
        return false;
      }
      const disqualifiers = [
        "chrome",
        "crios",
        "crmo",
        "android",
        "edge",
        "edg",
        "opr",
        "opera",
        "firefox",
        "fxios",
      ];
      return !disqualifiers.some((token) => ua.includes(token)) && !vendor.includes("google");
    })();

    if (isSafari) {
      return { threads: fallback, reason: "safari" };
    }

    const cachedThreads = readCachedThreadCount(storageKey);
    if (cachedThreads !== null) {
      const sanitized = Math.min(Math.max(fallback, cachedThreads), Math.max(fallback, concurrency));
      if (sanitized !== cachedThreads) {
        writeCachedThreadCount(storageKey, sanitized);
      }
      return { threads: sanitized, reason: "cached" };
    }

    const desktopCap = Math.min(4, Math.max(fallback, concurrency));
    const threads = isMobile ? fallback : Math.max(fallback, desktopCap);

    writeCachedThreadCount(storageKey, threads);
    return { threads, reason: isMobile ? "mobile" : "desktop" };
  } catch (error) {
    return { threads: fallback, reason: "error", error };
  }
}
