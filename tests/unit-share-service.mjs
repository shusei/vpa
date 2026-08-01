import assert from "node:assert/strict";
import test from "node:test";
import {
  getPublicAppUrl,
  publishShareResult,
  shareServiceInternals,
} from "../assets/experiments/share-service.js";
import * as shareNavigation from "../assets/experiments/share-navigation.js";

test("normalizes the configured service origin and rejects insecure remote endpoints", () => {
  assert.equal(
    shareServiceInternals.normalizeOrigin("https://share.example/path"),
    "https://share.example",
  );
  assert.equal(shareServiceInternals.normalizeOrigin("http://share.example"), "");
  assert.equal(
    shareServiceInternals.normalizeOrigin("http://127.0.0.1:8787/path"),
    "http://127.0.0.1:8787",
  );
});

test("removes dev.html when deriving the public app URL", () => {
  assert.equal(getPublicAppUrl({
    href: "https://shusei.github.io/vpa/dev.html?test=1#result",
  }, {}), "https://shusei.github.io/vpa/");
});

test("publishes one compressed JPEG and validates the returned short URL", async () => {
  const imageBlob = new Blob([Uint8Array.from([0xff, 0xd8, 0xff])], { type: "image/jpeg" });
  let request;
  const result = await publishShareResult({
    fetchLike: async (url, options) => {
      request = { options, url };
      return new Response(JSON.stringify({
        id: "abcdefghijklmnop",
        imageUrl: "https://share.example/i/abcdefghijklmnop.jpg",
        url: "https://share.example/r/abcdefghijklmnop",
      }), {
        headers: { "Content-Type": "application/json" },
        status: 201,
      });
    },
    imageBlob,
    metadata: {
      description: "Result",
      targetUrl: "https://shusei.github.io/vpa/",
      title: "VPA",
    },
    serviceOrigin: "https://share.example",
  });
  assert.equal(request.url, "https://share.example/api/shares");
  assert.equal(request.options.method, "POST");
  assert.ok(request.options.body instanceof FormData);
  assert.equal(result.url, "https://share.example/r/abcdefghijklmnop");
});

test("mobile sharing navigates the current tab without opening an empty tab", () => {
  const opened = [];
  const assigned = [];
  const windowLike = {
    location: { assign: (url) => assigned.push(url) },
    navigator: { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile" },
    open: (...args) => opened.push(args),
  };
  const currentTab = shareNavigation.prefersCurrentTab(windowLike);
  assert.equal(currentTab, true);
  const universalTargets = [
    ["x", "https://twitter.com/intent/tweet?text=VPA"],
    ["threads", "https://www.threads.com/intent/post?text=VPA"],
    ["line", "https://line.me/R/share?text=VPA"],
  ];
  universalTargets.forEach(([platform, target]) => {
    assert.equal(shareNavigation.buildAppFirstTarget({ platform, target, windowLike }), target);
  });
  assert.equal(
    shareNavigation.navigate(null, "https://twitter.com/intent/tweet?text=VPA", windowLike, { currentTab, platform: "x" }),
    "current-tab",
  );
  assert.deepEqual(assigned, ["https://twitter.com/intent/tweet?text=VPA"]);
  assert.deepEqual(opened, []);
});

test("Android sharing preserves X App Links while targeting verified LINE and Threads apps", () => {
  const windowLike = {
    navigator: { userAgent: "Mozilla/5.0 (Linux; Android 15) AppleWebKit/537.36 Mobile" },
  };
  const cases = [
    ["threads", "https://www.threads.com/intent/post?text=VPA", "com.instagram.barcelona"],
    ["line", "https://line.me/R/share?text=VPA", "jp.naver.line.android"],
  ];

  cases.forEach(([platform, target, packageName]) => {
    const appTarget = shareNavigation.buildAppFirstTarget({ platform, target, windowLike });
    assert.match(appTarget, /^intent:\/\//);
    assert.ok(appTarget.includes(`package=${packageName}`));
    assert.ok(appTarget.includes(`S.browser_fallback_url=${encodeURIComponent(target)}`));
  });

  const xTarget = "https://twitter.com/intent/tweet?text=VPA";
  const xAppTarget = shareNavigation.buildAppFirstTarget({ platform: "x", target: xTarget, windowLike });
  assert.equal(xAppTarget, xTarget);
  assert.ok(!xAppTarget.startsWith("intent://"));
  assert.ok(!xAppTarget.includes("package=com.twitter.android"));

  const assigned = [];
  const navigationWindow = {
    ...windowLike,
    location: { assign: (url) => assigned.push(url) },
    open: () => { throw new Error("Android app sharing must not open a browser tab"); },
  };
  shareNavigation.navigate(null, cases[1][1], navigationWindow, { currentTab: true, platform: "line" });
  assert.ok(assigned[0].includes("package=jp.naver.line.android"));
});

test("desktop sharing closes a placeholder that remains blank after app handoff", () => {
  const popup = {
    close() { this.closed = true; },
    closed: false,
    location: {
      href: "about:blank",
      replace() {},
    },
  };
  const windowLike = {
    navigator: { userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
    open() { throw new Error("A second tab must not be opened"); },
    setTimeout(callback) { callback(); },
  };
  assert.equal(shareNavigation.prefersCurrentTab(windowLike), false);
  assert.equal(
    shareNavigation.navigate(popup, "https://www.threads.com/intent/post?text=VPA", windowLike),
    "pending-tab",
  );
  assert.equal(popup.closed, true);
});
