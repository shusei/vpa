const APP_RULES = [
  ["line", /\bLine\//i],
  ["facebook", /\bFBAN\/|\bFBAV\//i],
  ["instagram", /\bInstagram\b/i],
  ["threads", /\bBarcelona\b|\bThreads\b/i],
  ["tiktok", /\bTikTok\b|\bmusical_ly\b|\bBytedanceWebview\b/i],
  ["x", /\bTwitter for iPhone\b|\bTwitterAndroid\b/i],
  ["wechat", /\bMicroMessenger\b/i],
  ["snapchat", /\bSnapchat\b/i],
];

const COPY = {
  en: {
    body: "This link is open inside an app browser, where voice analysis can be much slower. Use the app menu to open it in Safari or Chrome. You may also continue here with automatic fast analysis.",
    close: "Continue here",
    copied: "Link copied",
    failed: "Your browser could not be opened automatically. Use Copy link, or choose Open in browser from the current app menu.",
    copy: "Copy link",
    open: "Open in browser",
    opening: "Opening browser…",
    title: "Open in your browser for faster analysis",
  },
  ja: {
    body: "アプリ内ブラウザでは音声分析が非常に遅くなることがあります。アプリのメニューから Safari または Chrome で開いてください。このまま続ける場合は自動で高速分析を使用します。",
    close: "このまま続ける",
    copied: "リンクをコピーしました",
    failed: "ブラウザを自動で開けませんでした。リンクをコピーするか、現在のアプリのメニューからブラウザで開いてください。",
    copy: "リンクをコピー",
    open: "ブラウザで開く",
    opening: "ブラウザを開いています…",
    title: "ブラウザで開くと分析が速くなります",
  },
  "zh-Hans": {
    body: "目前使用的是 App 内置浏览器，语音分析可能会非常慢。请从 App 菜单选择用 Safari 或 Chrome 打开；留在这里也可以，系统会自动使用快速分析。",
    close: "继续在这里使用",
    copied: "链接已复制",
    failed: "无法自动打开浏览器。请复制链接，或从当前 App 菜单选择用浏览器打开。",
    copy: "复制链接",
    open: "用浏览器打开",
    opening: "正在打开浏览器…",
    title: "用浏览器打开，分析会更快",
  },
  "zh-Hant": {
    body: "目前使用的是 App 內建瀏覽器，語音分析可能會非常慢。請從 App 選單選擇用 Safari 或 Chrome 開啟；留在這裡也可以，系統會自動使用快速分析。",
    close: "繼續在這裡使用",
    copied: "連結已複製",
    failed: "無法自動開啟瀏覽器。請複製連結，或從目前 App 選單選擇用瀏覽器開啟。",
    copy: "複製連結",
    open: "用瀏覽器開啟",
    opening: "正在開啟瀏覽器…",
    title: "用瀏覽器開啟，分析會更快",
  },
};

function localeKey(documentLike, navigatorLike) {
  const raw = String(documentLike?.documentElement?.lang || navigatorLike?.language || "").toLowerCase();
  if (raw.startsWith("ja")) return "ja";
  if (raw.startsWith("zh-cn") || raw.startsWith("zh-sg") || raw.includes("hans")) return "zh-Hans";
  if (raw.startsWith("zh")) return "zh-Hant";
  return "en";
}

export function detectEmbeddedBrowser(navigatorLike = globalThis.navigator) {
  const ua = String(navigatorLike?.userAgent || "");
  const lower = ua.toLowerCase();
  const platform = /android/i.test(ua)
    ? "android"
    : /iphone|ipad|ipod/i.test(ua) || (
      String(navigatorLike?.platform || "").toLowerCase() === "macintel"
      && Number(navigatorLike?.maxTouchPoints || 0) > 1
    )
      ? "ios"
      : /mobile/i.test(ua)
        ? "mobile"
        : "desktop";
  const matched = APP_RULES.find(([, pattern]) => pattern.test(ua));
  const androidWebView = platform === "android" && (
    /;\s*wv\)/i.test(ua)
    || (/\bVersion\/4\.0\b/i.test(ua) && /\bChrome\//i.test(ua))
  );
  const iosWebView = platform === "ios"
    && lower.includes("applewebkit")
    && lower.includes("mobile")
    && !lower.includes("safari")
    && !/crios|fxios|edgios|opios/i.test(ua)
    && navigatorLike?.standalone !== true;
  const app = matched?.[0] || (androidWebView || iosWebView ? "webview" : "");
  return {
    app,
    embedded: Boolean(app),
    platform,
    userAgent: ua,
  };
}

function androidBrowserIntent(url) {
  try {
    const target = new URL(url);
    if (target.protocol !== "https:" && target.protocol !== "http:") return "";
    const scheme = target.protocol.slice(0, -1);
    const data = `${target.host}${target.pathname}${target.search}`;
    const fallback = encodeURIComponent(target.toString());
    return `intent://${data}#Intent;scheme=${scheme};action=android.intent.action.VIEW;category=android.intent.category.BROWSABLE;S.browser_fallback_url=${fallback};end`;
  } catch {
    return "";
  }
}

export async function openExternalBrowser({
  context,
  liffLike = globalThis.liff,
  locationLike = globalThis.location,
  windowLike = globalThis.window,
} = {}) {
  const url = String(locationLike?.href || "");
  if (!url) return { method: "unavailable", opened: false };

  try {
    if (typeof liffLike?.isInClient === "function"
      && liffLike.isInClient()
      && typeof liffLike.openWindow === "function") {
      await liffLike.openWindow({ external: true, url });
      return { method: "liff", opened: true };
    }
  } catch (error) {
    console.warn("[external-browser] LIFF open failed; using platform fallback.", error);
  }

  if (context?.platform === "android") {
    const intent = androidBrowserIntent(url);
    if (intent && typeof locationLike?.assign === "function") {
      locationLike.assign(intent);
      return { method: "android-intent", opened: true };
    }
  }

  try {
    const opened = windowLike?.open?.(url, "_blank");
    try { if (opened) opened.opener = null; } catch { }
    return { method: "new-window", opened: Boolean(opened) };
  } catch {
    return { method: "unavailable", opened: false };
  }
}
async function copyCurrentUrl({ documentLike, locationLike, navigatorLike }) {
  const value = String(locationLike?.href || "");
  if (!value) return false;
  try {
    if (typeof navigatorLike?.clipboard?.writeText === "function") {
      await navigatorLike.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Use the selection fallback below.
  }
  try {
    const input = documentLike.createElement("textarea");
    input.value = value;
    input.setAttribute("readonly", "");
    input.style.cssText = "position:fixed;opacity:0;pointer-events:none";
    documentLike.body.append(input);
    input.select();
    const copied = documentLike.execCommand("copy");
    input.remove();
    return copied;
  } catch {
    return false;
  }
}

export function installEmbeddedBrowserGuard({
  documentLike = globalThis.document,
  locationLike = globalThis.location,
  navigatorLike = globalThis.navigator,
  sessionStorageLike = globalThis.sessionStorage,
  windowLike = globalThis.window,
} = {}) {
  const context = detectEmbeddedBrowser(navigatorLike);
  if (!context.embedded || !documentLike?.body) return { context, element: null };
  try {
    if (sessionStorageLike?.getItem("vpa::embedded-browser-dismissed") === "1") {
      return { context, element: null };
    }
  } catch {
    // Storage is optional in embedded browsers.
  }

  const copy = COPY[localeKey(documentLike, navigatorLike)];
  const element = documentLike.createElement("aside");
  element.dataset.embeddedBrowserGuard = context.app;
  element.setAttribute("role", "alert");
  element.style.cssText = [
    "position:fixed",
    "z-index:2147483000",
    "left:12px",
    "right:12px",
    "bottom:max(12px,env(safe-area-inset-bottom))",
    "max-width:620px",
    "margin:auto",
    "padding:16px",
    "border:1px solid rgba(255,255,255,.28)",
    "border-radius:18px",
    "background:#132f38",
    "color:#fff",
    "box-shadow:0 18px 55px rgba(0,0,0,.4)",
    "font:500 15px/1.55 system-ui,-apple-system,sans-serif",
  ].join(";");

  const title = documentLike.createElement("strong");
  title.textContent = copy.title;
  title.style.cssText = "display:block;margin-bottom:5px;font-size:17px";
  const body = documentLike.createElement("span");
  body.textContent = copy.body;
  const actions = documentLike.createElement("div");
  actions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;margin-top:12px";
  const openButton = documentLike.createElement("button");
  openButton.type = "button";
  openButton.textContent = copy.open;
  openButton.style.cssText = "border:0;border-radius:999px;padding:9px 14px;background:#f2c978;color:#17272c;font:700 14px system-ui";
  const copyButton = documentLike.createElement("button");
  copyButton.type = "button";
  copyButton.textContent = copy.copy;
  copyButton.style.cssText = "border:1px solid rgba(255,255,255,.45);border-radius:999px;padding:8px 13px;background:transparent;color:#fff;font:700 14px system-ui";
  const closeButton = documentLike.createElement("button");
  closeButton.type = "button";
  closeButton.textContent = copy.close;
  closeButton.style.cssText = "border:1px solid rgba(255,255,255,.45);border-radius:999px;padding:8px 13px;background:transparent;color:#fff;font:700 14px system-ui";

  openButton.addEventListener("click", async () => {
    openButton.disabled = true;
    openButton.textContent = copy.opening;
    const response = await openExternalBrowser({ context, locationLike, windowLike });
    if (!response.opened) {
      body.textContent = copy.failed;
      openButton.disabled = false;
      openButton.textContent = copy.open;
    }
  });
  copyButton.addEventListener("click", async () => {
    if (await copyCurrentUrl({ documentLike, locationLike, navigatorLike })) {
      copyButton.textContent = copy.copied;
    }
  });
  closeButton.addEventListener("click", () => {
    try {
      sessionStorageLike?.setItem("vpa::embedded-browser-dismissed", "1");
    } catch {
      // Dismissal still works for the current page.
    }
    element.remove();
  });

  actions.append(openButton, copyButton, closeButton);
  element.append(title, body, actions);
  documentLike.body.append(element);
  return { context, element };
}
