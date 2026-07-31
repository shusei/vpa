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
    copy: "Copy link",
    title: "Open in your browser for faster analysis",
  },
  ja: {
    body: "アプリ内ブラウザでは音声分析が非常に遅くなることがあります。アプリのメニューから Safari または Chrome で開いてください。このまま続ける場合は自動で高速分析を使用します。",
    close: "このまま続ける",
    copied: "リンクをコピーしました",
    copy: "リンクをコピー",
    title: "ブラウザで開くと分析が速くなります",
  },
  "zh-Hans": {
    body: "目前使用的是 App 内置浏览器，语音分析可能会非常慢。请从 App 菜单选择用 Safari 或 Chrome 打开；留在这里也可以，系统会自动使用快速分析。",
    close: "继续在这里使用",
    copied: "链接已复制",
    copy: "复制链接",
    title: "用浏览器打开，分析会更快",
  },
  "zh-Hant": {
    body: "目前使用的是 App 內建瀏覽器，語音分析可能會非常慢。請從 App 選單選擇用 Safari 或 Chrome 開啟；留在這裡也可以，系統會自動使用快速分析。",
    close: "繼續在這裡使用",
    copied: "連結已複製",
    copy: "複製連結",
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
  const copyButton = documentLike.createElement("button");
  copyButton.type = "button";
  copyButton.textContent = copy.copy;
  copyButton.style.cssText = "border:0;border-radius:999px;padding:9px 14px;background:#f2c978;color:#17272c;font:700 14px system-ui";
  const closeButton = documentLike.createElement("button");
  closeButton.type = "button";
  closeButton.textContent = copy.close;
  closeButton.style.cssText = "border:1px solid rgba(255,255,255,.45);border-radius:999px;padding:8px 13px;background:transparent;color:#fff;font:700 14px system-ui";

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

  actions.append(copyButton, closeButton);
  element.append(title, body, actions);
  documentLike.body.append(element);
  return { context, element };
}
