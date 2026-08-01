import { getCurrentLocale, t } from "../js/i18n.js";
import { buildShareTargets } from "./share-card.js?v=20260801-appopen1";
import {
  isShareServiceConfigured,
  publishShareResult,
} from "./share-service.js?v=20260801-sharefix1";
import { createSocialPreviewBlob } from "./social-preview-card.js?v=20260801-sharefix1";
import { navigate, prefersCurrentTab } from "./share-navigation.js?v=20260801-appopen1";

let publishedShareCache = null;

function labels() {
  return {
    brand: "VPA / VOICE IMPRESSION",
    cta: "Voice Presentation Analyzer",
    disclaimer: t("experiment.advanced.share.cardDisclaimer"),
    feminine: t("experiment.quick.reveal.feminine"),
    heading: t("experiment.quick.reveal.tendency"),
    insight: t("experiment.quick.reveal.insight"),
    masculine: t("experiment.quick.reveal.masculine"),
    pitch: t("experiment.advanced.components.pitch"),
    voiceAge: t("experiment.advanced.voiceAge.title"),
  };
}

function createPendingWindow(windowLike) {
  const popup = windowLike.open("about:blank", "_blank");
  if (!popup) return null;
  try {
    popup.opener = null;
    popup.document.title = "VPA";
    popup.document.body.textContent = "Preparing VPA share…";
    popup.document.body.style.cssText = "font:600 18px system-ui;padding:32px;color:#173741";
  } catch {
    // Navigation still works if the placeholder document cannot be styled.
  }
  return popup;
}

function cacheResult(cacheKey, promise) {
  promise.then((result) => {
    if (publishedShareCache?.key === cacheKey && publishedShareCache.promise === promise) {
      publishedShareCache.result = result;
    }
  }).catch(() => {
    // The caller handles publication errors and clears a failed cache entry.
  });
}

function shareCacheKey(challenge) {
  const theme = document.documentElement.getAttribute("data-faction") === "light"
    ? "light"
    : "dark";
  return [challenge.payload.id, getCurrentLocale(), theme].join(":");
}

async function publish({ analysis, challenge, formatted, result }) {
  const cacheKey = shareCacheKey(challenge);
  if (publishedShareCache?.key === cacheKey) return publishedShareCache.promise;

  const promise = (async () => {
    const cardLabels = labels();
    const theme = document.documentElement.getAttribute("data-faction") === "light"
      ? "light"
      : "dark";
    const locale = getCurrentLocale();
    const pitchHz = Number(analysis?.pitch?.stats?.med);
    const imageBlob = await createSocialPreviewBlob({
      formatted,
      labels: cardLabels,
      pitchHz,
      result,
      theme,
    });
    const pitchText = Number.isFinite(pitchHz) ? `${pitchHz.toFixed(1)} Hz` : "";
    return publishShareResult({
      imageBlob,
      metadata: {
        alt: [
          formatted.archetype,
          `${cardLabels.feminine} ${result.score}%`,
          pitchText,
          formatted.age,
        ].filter(Boolean).join("，"),
        description: formatted.caption,
        locale,
        targetUrl: challenge.url,
        title: `VPA｜${formatted.archetype} ${result.score}%`,
      },
    });
  })();
  publishedShareCache = { key: cacheKey, promise, result: null };
  cacheResult(cacheKey, promise);
  try {
    return await promise;
  } catch (error) {
    if (publishedShareCache?.promise === promise) publishedShareCache = null;
    throw error;
  }
}

export async function getPublicShareResult({
  analysis,
  challenge,
  formatted,
  result,
  windowLike = window,
}) {
  if (!isShareServiceConfigured(windowLike)) return null;
  return publish({ analysis, challenge, formatted, result });
}

export async function openPublicPlatformShare({
  analysis,
  challenge,
  formatted,
  platform,
  result,
  windowLike = window,
}) {
  const fallbackTarget = buildShareTargets({
    caption: formatted.caption,
    url: challenge.url,
  })[platform];
  if (!fallbackTarget) return { method: "unsupported" };
  const currentTab = prefersCurrentTab(windowLike);
  if (!isShareServiceConfigured(windowLike)) {
    navigate(null, fallbackTarget, windowLike, { currentTab, platform });
    return { method: "unconfigured" };
  }
  if (publishedShareCache?.key === shareCacheKey(challenge) && publishedShareCache.result) {
    const target = buildShareTargets({
      caption: formatted.caption,
      url: publishedShareCache.result.url,
    })[platform];
    navigate(null, target, windowLike, { currentTab, platform });
    return { method: "public-result", url: publishedShareCache.result.url };
  }

  const popup = currentTab ? null : createPendingWindow(windowLike);
  try {
    const publicShare = await getPublicShareResult({
      analysis,
      challenge,
      formatted,
      result,
      windowLike,
    });
    const target = buildShareTargets({
      caption: formatted.caption,
      url: publicShare.url,
    })[platform];
    navigate(popup, target, windowLike, { currentTab, platform });
    return { method: "public-result", url: publicShare.url };
  } catch (error) {
    console.error("[public-share] result publishing failed", error);
    navigate(popup, fallbackTarget, windowLike, { currentTab, platform });
    return { error, method: "fallback" };
  }
}

export function resetPublicShareCache() {
  publishedShareCache = null;
}
