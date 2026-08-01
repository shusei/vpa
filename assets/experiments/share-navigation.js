export function prefersCurrentTab(windowLike) {
  const navigatorLike = windowLike?.navigator || {};
  if (navigatorLike.userAgentData?.mobile === true) return true;
  const userAgent = String(navigatorLike.userAgent || "");
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(userAgent)) return true;
  return /Macintosh/i.test(userAgent) && Number(navigatorLike.maxTouchPoints) > 1;
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

export function navigate(popup, target, windowLike, { currentTab = false } = {}) {
  if (currentTab) {
    closePopup(popup);
    if (typeof windowLike.location?.assign === "function") {
      windowLike.location.assign(target);
    } else {
      windowLike.location.href = target;
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
