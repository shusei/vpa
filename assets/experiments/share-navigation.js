const ANDROID_APP_PACKAGES = Object.freeze({
  line: "jp.naver.line.android",
  threads: "com.instagram.barcelona",
  x: "com.twitter.android",
});
export function prefersCurrentTab(windowLike) {
  const navigatorLike = windowLike?.navigator || {};
  if (navigatorLike.userAgentData?.mobile === true) return true;
  const userAgent = String(navigatorLike.userAgent || "");
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(userAgent)) return true;
  return /Macintosh/i.test(userAgent) && Number(navigatorLike.maxTouchPoints) > 1;
}

function androidAppIntent(target, packageName) {
  try {
    const url = new URL(target);
    if (url.protocol !== "https:" && url.protocol !== "http:") return target;
    const scheme = url.protocol.slice(0, -1);
    const data = `${url.host}${url.pathname}${url.search}${url.hash}`;
    const fallback = encodeURIComponent(url.toString());
    return `intent://${data}#Intent;scheme=${scheme};package=${packageName};action=android.intent.action.VIEW;category=android.intent.category.BROWSABLE;S.browser_fallback_url=${fallback};end`;
  } catch {
    return target;
  }
}

export function buildAppFirstTarget({ platform, target, windowLike }) {
  const userAgent = String(windowLike?.navigator?.userAgent || "");
  if (!/Android/i.test(userAgent)) return target;
  const packageName = ANDROID_APP_PACKAGES[platform];
  if (!packageName) return target;
  return androidAppIntent(target, packageName);
}

function closePopup(popup) {
  if (!popup) return;
  try {
    if (!popup.closed) popup.close();
  } catch {
    // A cross-origin share page is already active and must stay open.
  }
}

function closeOrphanedPlaceholder(popup, windowLike) {
  if (!popup || typeof windowLike?.setTimeout !== "function") return;
  windowLike.setTimeout(() => {
    try {
      const href = String(popup.location?.href || "");
      if (!popup.closed && (!href || href === "about:blank")) popup.close();
    } catch {
      // Cross-origin access means the share page loaded successfully.
    }
  }, 1500);
}

export function navigate(popup, target, windowLike, { currentTab = false, platform = "" } = {}) {
  if (currentTab) {
    closePopup(popup);
    const destination = buildAppFirstTarget({ platform, target, windowLike });
    if (typeof windowLike.location?.assign === "function") {
      windowLike.location.assign(destination);
    } else {
      windowLike.location.href = destination;
    }
    return "current-tab";
  }
  if (popup) {
    try {
      popup.location.replace(target);
      closeOrphanedPlaceholder(popup, windowLike);
      return "pending-tab";
    } catch {
      closePopup(popup);
    }
  }
  windowLike.open(target, "_blank", "noopener,noreferrer");
  return "new-tab";
}
