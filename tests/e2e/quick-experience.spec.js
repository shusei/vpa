import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

async function openDevelopmentPage(page, { shareServiceOrigin = "", ...runtimeOptions } = {}) {
  await installDeterministicRuntime(page, runtimeOptions);
  await page.addInitScript((origin) => {
    window.VPA_SHARE_SERVICE_ORIGIN = origin;
    window.VPA_PUBLIC_APP_URL = "";
  }, shareServiceOrigin);
  await page.goto("/dev.html");
  await expect.poll(() => page.evaluate(() => Boolean(
    window.vpaAdvancedExperience && window.vpaExperience
  ))).toBe(true);
}

async function inspectVoiceCardVideo(video) {
  return video.evaluate(async (element) => {
    if (element.readyState < HTMLMediaElement.HAVE_METADATA) {
      await new Promise((resolvePromise, reject) => {
        element.addEventListener("loadedmetadata", resolvePromise, { once: true });
        element.addEventListener("error", reject, { once: true });
      });
    }
    await element.play();
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
    const captured = typeof element.captureStream === "function"
      ? element.captureStream()
      : null;
    const blob = await fetch(element.currentSrc || element.src).then((response) => response.blob());
    const details = {
      audioTracks: captured?.getAudioTracks().length || 0,
      defaultMuted: element.defaultMuted,
      duration: element.duration,
      height: element.videoHeight,
      muted: element.muted,
      size: blob.size,
      type: blob.type,
      volume: element.volume,
      videoTracks: captured?.getVideoTracks().length || 0,
      width: element.videoWidth,
    };
    captured?.getTracks().forEach((track) => track.stop());
    element.pause();
    return details;
  });
}

test("quick and professional experiences share one analysis result", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await expect(page.locator("html")).toHaveAttribute("data-experience", "quick");
  await expect(page.locator("#quickExperience")).toBeVisible();
  await expect(page.locator("body > .hero")).toBeHidden();
  await expect(page.locator(".quick-landing .quick-primary")).toHaveCount(1);
  await expect(page.locator(".quick-standard-cta")).toBeVisible();
  if (process.env.VPA_QUICK_LANDING_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_QUICK_LANDING_CAPTURE),
    });
  }

  const result = await page.evaluate((analysis) => {
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);
  await expect(page.locator(".quick-result__feminine strong")).toHaveText(`${result.score}%`);
  await expect(page.locator(".quick-result__masculine strong")).toHaveText(`${100 - result.score}%`);

  await page.locator('.quick-result [data-experience-target="professional"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-experience", "professional");
  await expect(page.locator("body > .hero")).toBeVisible();
  await expect(page.locator("#advancedExperience")).toBeVisible();
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0)).toBe(0);

  await page.locator('#experienceNav [data-experience-target="quick"]').click();
  await expect(page.locator(".quick-result__feminine strong")).toHaveText(`${result.score}%`);
  if (process.env.VPA_QUICK_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_QUICK_CAPTURE),
    });
  }
  expect(runtimeErrors).toEqual([]);
});

test("quick experience text keeps accessible contrast across every theme", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  const themes = await page.locator(".theme-item").evaluateAll((buttons) => {
    return buttons.map((button) => button.dataset.theme);
  });
  const failures = [];

  for (const theme of themes) {
    await page.locator(`.theme-item[data-theme="${theme}"]`).evaluate((button) => button.click());
    const audit = await page.evaluate(() => {
      const host = document.querySelector("#quickExperience");
      const canvas = document.createElement("canvas");
      canvas.width = 1;
      canvas.height = 1;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      function rgba(value) {
        ctx.clearRect(0, 0, 1, 1);
        ctx.fillStyle = "rgba(0, 0, 0, 0)";
        ctx.fillRect(0, 0, 1, 1);
        ctx.fillStyle = value;
        ctx.fillRect(0, 0, 1, 1);
        return Array.from(ctx.getImageData(0, 0, 1, 1).data);
      }

      function luminance(color) {
        const channels = color.slice(0, 3).map((channel) => {
          const normalized = channel / 255;
          return normalized <= 0.04045
            ? normalized / 12.92
            : ((normalized + 0.055) / 1.055) ** 2.4;
        });
        return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2]);
      }

      function contrast(foreground, background) {
        const foregroundLuminance = luminance(rgba(foreground));
        const backgroundLuminance = luminance(rgba(background));
        return (Math.max(foregroundLuminance, backgroundLuminance) + 0.05)
          / (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
      }

      function resolvedColor(value) {
        const probe = document.createElement("span");
        probe.style.color = value;
        host.append(probe);
        const color = getComputedStyle(probe).color;
        probe.remove();
        return color;
      }

      const pairs = [
        ["body", "var(--quick-ink)", "var(--quick-bg)"],
        ["body-gradient-end", "var(--quick-ink)", "var(--quick-bg-end)"],
        ["muted-body", "var(--quick-muted)", "var(--quick-bg)"],
        ["muted-body-gradient-end", "var(--quick-muted)", "var(--quick-bg-end)"],
        ["card", "var(--quick-ink)", "var(--quick-surface)"],
        ["muted-card", "var(--quick-muted)", "var(--quick-surface)"],
        ["strong-card", "var(--quick-ink)", "var(--quick-surface-strong)"],
        ["muted-strong-card", "var(--quick-muted)", "var(--quick-surface-strong)"],
        ["primary", "var(--quick-action-ink)", "var(--quick-action)"],
        ["primary-gradient-end", "var(--quick-action-ink)", "var(--quick-action-end)"],
        ["emphasis-body", "var(--quick-emphasis)", "var(--quick-bg)"],
        ["emphasis-card", "var(--quick-emphasis)", "var(--quick-surface)"],
        ["brand-accent", "var(--quick-accent-ink)", "var(--quick-accent)"],
        ["brand-accent-2", "var(--quick-accent-ink)", "var(--quick-accent-2)"],
        ["stop-start", "var(--quick-danger-ink)", "var(--quick-danger-a)"],
        ["stop-end", "var(--quick-danger-ink)", "var(--quick-danger-b)"],
      ].map(([label, foreground, background]) => {
        const resolvedForeground = resolvedColor(foreground);
        const resolvedBackground = resolvedColor(background);
        return {
          background: resolvedBackground,
          foreground: resolvedForeground,
          label,
          ratio: contrast(resolvedForeground, resolvedBackground),
        };
      });

      return {
        faction: document.documentElement.dataset.faction,
        pairs,
      };
    });

    for (const pair of audit.pairs) {
      if (pair.ratio < 4.5) {
        failures.push({
          ...pair,
          faction: audit.faction,
          ratio: Number(pair.ratio.toFixed(2)),
          theme,
        });
      }
    }
  }

  expect(themes.length).toBeGreaterThanOrEqual(30);
  expect(failures).toEqual([]);
  expect(runtimeErrors).toEqual([]);
});

test("quick recording delegates to the production recording and analysis path", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.locator("[data-quick-record]").click();
  await expect(page.locator('[data-quick-stage="recording"]')).toBeVisible({ timeout: 30_000 });
  await page.waitForTimeout(1200);
  await page.locator("[data-quick-record]").click();
  await expect(page.locator('[data-quick-stage="result"]')).toBeVisible({ timeout: 60_000 });

  expect(await page.evaluate(() => window.vpaExperience.getLatestAnalysis()?.analysisId || 0))
    .toBeGreaterThan(0);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0))
    .toBeGreaterThan(0);
  expect(await page.evaluate(() => {
    return (window.dataLayer || [])
      .map((entry) => Array.from(entry))
      .filter((entry) => entry[0] === "event")
      .map((entry) => entry[1]);
  })).toEqual(expect.arrayContaining(["quick_test_started", "quick_test_completed"]));
  expect(runtimeErrors).toEqual([]);
});

test("three-line standard test runs the production path three times and uses the median", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.locator("[data-quick-standard]").click();
  await expect(page.locator(".quick-standard h1")).toHaveText("三句標準測試");
  await expect(page.locator(".quick-prompt > span")).toContainText("1 / 3");
  if (process.env.VPA_STANDARD_LANDING_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_STANDARD_LANDING_CAPTURE),
    });
  }

  for (let step = 1; step <= 3; step += 1) {
    await page.locator("[data-quick-record]").click();
    await expect(page.locator('[data-quick-stage="recording"]')).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.locator(".quick-progress__prompt")).not.toBeEmpty();
    await page.waitForTimeout(1200);
    await page.locator("[data-quick-record]").click();

    if (step < 3) {
      await expect(page.locator('[data-quick-stage="standard-next"]')).toBeVisible({
        timeout: 60_000,
      });
      await expect(page.locator(".quick-standard-scores span")).toHaveCount(step);
      await page.locator("[data-quick-standard-next]").click();
      await expect(page.locator(".quick-prompt > span")).toContainText(`${step + 1} / 3`);
    }
  }

  await expect(page.locator('[data-quick-stage="result"]')).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.locator(".quick-standard-summary")).toBeVisible();
  const result = await page.evaluate(() => window.vpaExperience.getLatestResult());
  expect(result.quickTest).toEqual({
    mode: "standard",
    promptId: "standard-v1",
  });
  expect(result.standard.scores).toHaveLength(3);
  expect(result.score).toBe([...result.standard.scores].sort((a, b) => a - b)[1]);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0)).toBeGreaterThanOrEqual(3);
  const standardChallenge = await page.evaluate(async () => {
    const { createChallengeUrl } = await import("/assets/experiments/challenge-link.js");
    return createChallengeUrl(window.vpaExperience.getLatestResult());
  });
  expect(standardChallenge.payload).toMatchObject({
    promptId: "standard-v1",
    schema: 3,
    testMode: "standard",
  });
  if (process.env.VPA_STANDARD_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_STANDARD_CAPTURE),
    });
  }

  await page.goto(standardChallenge.url);
  await page.reload();
  await expect.poll(() => page.evaluate(() => window.vpaExperience?.getQuickTestMode()))
    .toBe("standard");
  await expect(page.locator(".quick-challenge-invite")).toContainText(`${result.score}%`);
  await expect(page.locator(".quick-standard h1")).toBeVisible();
  expect(runtimeErrors).toEqual([]);
});

test("Japanese browser language becomes the default locale and a saved choice wins", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page, {
    navigatorLanguages: ["ja-JP", "en-US"],
    storedLocale: null,
  });

  await expect(page.locator("html")).toHaveAttribute("lang", "ja");
  await expect(page.locator(".quick-landing h1")).toHaveText("あなたの声は、どんな印象？");
  await expect(page.locator("[data-quick-locale]")).toHaveValue("ja");
  if (process.env.VPA_JA_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_JA_CAPTURE),
    });
  }

  await page.locator("[data-quick-locale]").selectOption("zh-Hant");
  await expect(page.locator(".quick-landing h1")).toHaveText("你的聲音，給人什麼印象？");
  expect(await page.evaluate(() => localStorage.getItem("vpa.locale"))).toBe("zh-Hant");

  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "zh-Hant");
  await expect(page.locator("[data-quick-locale]")).toHaveValue("zh-Hant");

  expect(await page.evaluate(async () => {
    const { i18nInternals } = await import("/assets/js/i18n.js");
    return ["zh-TW", "zh-CN", "en-US", "ja-JP", "fr-FR"]
      .map((locale) => i18nInternals.mapCandidateLocale(locale));
  })).toEqual(["zh-Hant", "zh-Hans", "en", "ja", null]);

  await page.locator("[data-quick-locale]").selectOption("ja");
  await page.locator('#experienceNav [data-experience-target="professional"]').click();
  await page.locator("#practiceToggle").click();
  await expect(page.locator("#practiceList")).toContainText("すみません、少しお時間をいただけますか。");
  await page.locator("#helpBtn").click();
  await expect(page.locator("#helpOverlay")).toContainText("使い方ガイド");
  await expect(page.locator("#helpOverlay")).toContainText("パネルの見方");
  expect(runtimeErrors).toEqual([]);
});

test("challenge link carries only summary data and compares the next result", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  const challenge = await page.evaluate(async (analysis) => {
    window.vpaAdvancedExperience.renderAnalysis(analysis);
    const result = window.vpaExperience.getLatestResult();
    const { createChallengeUrl } = await import("/assets/experiments/challenge-link.js");
    return createChallengeUrl(result);
  }, fixture);

  expect(challenge.url).not.toContain("audio");
  expect(Object.keys(challenge.payload).sort()).toEqual([
    "ageMax",
    "ageMin",
    "ageVersion",
    "archetype",
    "id",
    "promptId",
    "schema",
    "score",
    "scoreVersion",
    "testMode",
  ]);
  expect(challenge.payload.schema).toBe(3);
  expect(challenge.payload.testMode).toBe("daily");

  await page.goto(challenge.url);
  await page.reload();
  await expect(page.locator(".quick-challenge-invite")).toContainText(`${challenge.payload.score}`);
  expect(await page.evaluate(() => window.vpaExperience.getLatestResult())).toBe(null);
  await page.evaluate((analysis) => {
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);
  await expect(page.locator(".quick-comparison")).toContainText("百分比相同");
  await expect(page.locator(".quick-comparison")).toContainText(
    `${challenge.payload.score}%`,
  );
  expect(runtimeErrors).toEqual([]);
});

test("result page offers only the three verified direct sharing platforms", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);
  await page.evaluate(async (analysis) => {
    const { recorderCtl } = await import("/assets/app.js");
    recorderCtl.getLastRecordingUrl = () => "/tests/.generated-media/tone.wav";
    window.vpaAdvancedExperience.renderAnalysis(analysis);
    window.open = (url) => {
      window.__vpaOpenedShareUrl = String(url);
      return null;
    };
  }, fixture);

  await expect(page.locator("[data-quick-platform]")).toHaveCount(3);
  await page.locator('[data-quick-platform="threads"]').click();
  await expect.poll(() => page.evaluate(() => window.__vpaOpenedShareUrl)).toContain(
    "https://www.threads.com/intent/post?text=",
  );
  expect(decodeURIComponent(await page.evaluate(() => window.__vpaOpenedShareUrl)))
    .toContain("#vpa-challenge=");
  await page.locator('[data-quick-platform="line"]').click();
  await expect.poll(() => page.evaluate(() => window.__vpaOpenedShareUrl)).toContain(
    "https://line.me/R/share?text=",
  );
  const lineTarget = new URL(await page.evaluate(() => window.__vpaOpenedShareUrl));
  expect(lineTarget.searchParams.get("text")?.length).toBeGreaterThan(10);
  expect(lineTarget.searchParams.get("text")).toContain("#vpa-challenge=");
  await page.locator('[data-quick-platform="x"]').click();
  await expect.poll(() => page.evaluate(() => window.__vpaOpenedShareUrl)).toContain(
    "https://twitter.com/intent/tweet?",
  );
  const xTarget = new URL(await page.evaluate(() => window.__vpaOpenedShareUrl));
  expect(xTarget.searchParams.get("url")).toContain("#vpa-challenge=");
  expect(xTarget.searchParams.get("hashtags")).toBe("VoicePresentationAnalyzer");
  expect(xTarget.toString()).not.toContain("package=com.twitter.android");
  await expect(page.locator('[data-quick-platform="facebook"], [data-quick-platform="tiktok"], [data-quick-platform="instagram"]')).toHaveCount(0);
  expect(runtimeErrors).toEqual([]);
});

test("direct LINE sharing publishes a personalized image result with copy", async ({ page }) => {
  let uploadedRequest;
  await page.route("https://share.example/api/shares", async (route) => {
    uploadedRequest = route.request();
    await route.fulfill({
      body: JSON.stringify({
        id: "abcdefghijklmnop",
        imageUrl: "https://share.example/i/abcdefghijklmnop.jpg",
        url: "https://share.example/r/abcdefghijklmnop",
      }),
      contentType: "application/json",
      status: 201,
    });
  });
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page, { shareServiceOrigin: "https://share.example" });
  await page.evaluate((analysis) => {
    window.vpaAdvancedExperience.renderAnalysis(analysis);
    window.__vpaCopiedText = "";
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (value) => { window.__vpaCopiedText = String(value); },
      },
    });
    window.open = (url) => {
      if (String(url) !== "about:blank") window.__vpaOpenedShareUrl = String(url);
      return null;
    };
  }, fixture);

  await page.locator('[data-quick-platform="line"]').click();
  await expect.poll(() => page.evaluate(() => window.__vpaOpenedShareUrl || ""))
    .toContain("https://line.me/R/share?text=");
  const opened = new URL(await page.evaluate(() => window.__vpaOpenedShareUrl));
  expect(opened.searchParams.get("text")).toContain("https://share.example/r/abcdefghijklmnop");
  expect(opened.searchParams.get("text")?.length).toBeGreaterThan(10);
  expect(opened.toString()).not.toContain("#vpa-challenge=");
  const uploadContentType = uploadedRequest.headers()["content-type"];
  const uploadForm = await new Request("https://share.example/api/shares", {
    body: uploadedRequest.postDataBuffer(),
    headers: { "Content-Type": uploadContentType },
    method: "POST",
  }).formData();
  const uploadedImage = uploadForm.get("image");
  expect(uploadedImage.type).toBe("image/jpeg");
  expect(uploadedImage.size).toBeGreaterThan(10_000);
  expect(uploadedImage.size).toBeLessThanOrEqual(400_000);
  await page.locator("[data-quick-share]").click();
  await page.locator("[data-quick-copy-challenge]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaCopiedText))
    .toBe("https://share.example/r/abcdefghijklmnop");
  expect(runtimeErrors).toEqual([]);
});

test("image sharing is default and dynamic video requires explicit voice opt-in", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "canShare", {
      configurable: true,
      value: ({ files }) => Array.isArray(files) && files.length > 0,
    });
    Object.defineProperty(navigator, "share", {
      configurable: true,
      value: async (payload) => {
        window.__vpaLastShare = {
          files: (payload.files || []).map((file) => ({
            name: file.name,
            type: file.type,
          })),
          text: payload.text,
          hasUrl: Object.hasOwn(payload, "url"),
        };
      },
    });
  });
  await openDevelopmentPage(page);

  await page.evaluate(async (analysis) => {
    const { recorderCtl } = await import("/assets/app.js");
    recorderCtl.getLastRecordingUrl = () => "/tests/.generated-media/tone.wav";
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  await page.locator("[data-quick-share]").click();
  await expect(page.locator("[data-quick-audio]")).not.toBeChecked();
  await expect(page.locator("[data-quick-system-share]")).toHaveText("分享圖片＋文字");
  await expect(page.locator(".quick-dynamic-card")).toHaveCount(0);
  await page.locator("[data-quick-system-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(1);
  expect(await page.evaluate(() => window.__vpaLastShare.files[0].name)).toBe("vpa-result.png");
  expect(await page.evaluate(() => window.__vpaLastShare.text.length)).toBeGreaterThan(10);
  expect(await page.evaluate(() => window.__vpaLastShare.hasUrl)).toBe(false);
  expect(await page.evaluate(() => window.__vpaLastShare.text)).toContain("#vpa-challenge=");

  await page.locator("[data-quick-audio]").check();
  await expect(page.locator("[data-quick-system-share]")).toHaveCount(0);
  await expect(page.locator(".quick-share-audio-warning")).toBeVisible();
  await expect(page.locator(".quick-dynamic-card")).toBeVisible();
  await expect(page.locator(".quick-dynamic-progress")).toBeVisible();
  await expect(page.locator("[data-dynamic-start], [data-dynamic-end]")).toHaveCount(0);
  await expect(page.locator("[data-dynamic-preview-play]")).toHaveCount(0);
  const optInEvents = await page.evaluate(() => {
    return (window.dataLayer || [])
      .map((entry) => Array.from(entry))
      .filter((entry) => entry[0] === "event" && entry[1] === "audio_share_opt_in")
      .map((entry) => entry[2]);
  });
  expect(optInEvents).toEqual([
    expect.objectContaining({
      enabled: true,
      source: "dynamic_card",
    }),
  ]);

  await page.locator("[data-quick-audio]").uncheck();
  await expect(page.locator("[data-quick-system-share]")).toBeVisible();
  await expect(page.locator(".quick-dynamic-card")).toHaveCount(0);
  expect(await page.evaluate(() => Object.keys(localStorage).some((key) => key.includes("audio"))))
    .toBe(false);
  expect(runtimeErrors).toEqual([]);
});

test("dynamic video keeps a full untrimmed 30 second recording", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);
  await page.evaluate(async (analysis) => {
    const { recorderCtl } = await import("/assets/app.js");
    recorderCtl.getLastRecordingUrl = () => "/tests/.generated-media/tone-30.wav";
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  await page.locator("[data-quick-share]").click();
  await page.locator("[data-quick-audio]").check();
  await expect(page.locator(".quick-dynamic-editor")).toHaveCount(0);
  await expect(page.locator("[data-dynamic-start], [data-dynamic-end]")).toHaveCount(0);
  const video = page.locator(".quick-dynamic-output video");
  await expect(video).toBeVisible({ timeout: 50_000 });
  const media = await inspectVoiceCardVideo(video);
  expect(media.duration).toBeGreaterThanOrEqual(29.5);
  expect(media.duration).toBeLessThanOrEqual(30.5);
  expect(media.audioTracks).toBe(1);
  expect(media).toMatchObject({
    defaultMuted: false,
    muted: false,
    volume: 1,
  });
  await expect(page.locator("[data-dynamic-preview-play]")).toHaveCount(0);
  expect(media.videoTracks).toBe(1);
  expect(runtimeErrors).toEqual([]);
});
test("dynamic voice card exports the full recording and preserves the fallback chain", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await page.route("https://share.example/api/shares", async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        id: "abcdefghijklmnop",
        imageUrl: "https://share.example/i/abcdefghijklmnop.jpg",
        url: "https://share.example/r/abcdefghijklmnop",
      }),
      contentType: "application/json",
      status: 201,
    });
  });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "canShare", {
      configurable: true,
      value: ({ files }) => Array.isArray(files) && files.length > 0,
    });
    Object.defineProperty(navigator, "share", {
      configurable: true,
      value: async (payload) => {
        window.__vpaLastShare = {
          files: (payload.files || []).map((file) => ({
            name: file.name,
            size: file.size,
            type: file.type,
          })),
          text: payload.text,
          hasUrl: Object.hasOwn(payload, "url"),
        };
      },
    });
  });
  await openDevelopmentPage(page, { shareServiceOrigin: "https://share.example" });

  const expectedScoreBand = await page.evaluate(async (analysis) => {
    const { recorderCtl } = await import("/assets/app.js");
    recorderCtl.getLastRecordingUrl = () => "/tests/.generated-media/tone.wav";
    window.vpaAdvancedExperience.renderAnalysis(analysis);
    return Math.floor(window.vpaExperience.getLatestResult().score / 10) * 10;
  }, fixture);

  await page.locator("[data-quick-share]").click();
  await page.locator("[data-quick-audio]").check();
  await expect(page.locator(".quick-dynamic-editor")).toHaveCount(0);
  await expect(page.locator("[data-dynamic-start], [data-dynamic-end]")).toHaveCount(0);
  const video = page.locator(".quick-dynamic-output video");
  await expect(video).toBeVisible({ timeout: 30_000 });
  const media = await inspectVoiceCardVideo(video);

  expect(media.type).toContain("video/mp4");
  expect(media.size).toBeGreaterThan(50_000);
  expect(media.width).toBe(720);
  expect(media.height).toBe(1280);
  expect(media.duration).toBeGreaterThanOrEqual(3.5);
  expect(media.duration).toBeLessThanOrEqual(4.5);
  expect(media.videoTracks).toBe(1);
  expect(media.audioTracks).toBe(1);
  if (process.env.VPA_DYNAMIC_CAPTURE) {
    await video.evaluate(async (element) => {
      const target = Math.max(0, Math.min(element.duration - 0.25, 7));
      if (Math.abs(element.currentTime - target) < 0.05) return;
      await new Promise((resolvePromise, reject) => {
        element.addEventListener("seeked", resolvePromise, { once: true });
        element.addEventListener("error", reject, { once: true });
        element.currentTime = target;
      });
    });
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_DYNAMIC_CAPTURE),
    });
  }

  await page.locator("[data-dynamic-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(1);
  expect(await page.evaluate(() => window.__vpaLastShare.files[0])).toMatchObject({
    name: "vpa-voice-card.mp4",
  });
  expect(await page.evaluate(() => window.__vpaLastShare.files[0].type))
    .toBe("video/mp4");
  expect(await page.evaluate(() => window.__vpaLastShare.hasUrl)).toBe(false);
  expect(await page.evaluate(() => window.__vpaLastShare.text))
    .toContain("https://share.example/r/abcdefghijklmnop");
  const analytics = await page.evaluate(() => {
    return (window.dataLayer || [])
      .map((entry) => Array.from(entry))
      .filter((entry) => entry[0] === "event")
      .map((entry) => ({
        name: entry[1],
        params: entry[2],
      }));
  });
  expect(analytics).toEqual(expect.arrayContaining([
    expect.objectContaining({
      name: "audio_share_opt_in",
      params: expect.objectContaining({
        enabled: true,
        source: "dynamic_card",
      }),
    }),
    expect.objectContaining({
      name: "share_success",
      params: expect.objectContaining({
        media: "mp4",
        score_band: expectedScoreBand,
      }),
    }),
  ]));
  expect(analytics.some(({ params }) => (
    Object.hasOwn(params || {}, "audio")
    || Object.hasOwn(params || {}, "score")
  ))).toBe(false);

  await page.locator("[data-dynamic-close]").click();
  await page.evaluate(() => {
    const nativeSupport = MediaRecorder.isTypeSupported.bind(MediaRecorder);
    MediaRecorder.isTypeSupported = (type) => (
      type.startsWith("video/webm") && nativeSupport(type)
    );
    window.__vpaLastShare = null;
  });
  await page.locator("[data-dynamic-open]").click();
  await expect(video).toBeVisible({ timeout: 30_000 });
  const webmMedia = await inspectVoiceCardVideo(video);
  expect(webmMedia.type).toContain("video/webm");
  expect(webmMedia.duration).toBeGreaterThanOrEqual(7.5);
  expect(webmMedia.videoTracks).toBe(1);
  expect(webmMedia.audioTracks).toBe(1);
  await page.locator("[data-dynamic-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(1);
  expect(await page.evaluate(() => window.__vpaLastShare.files[0].name))
    .toBe("vpa-voice-card.webm");
  expect(await page.evaluate(() => window.__vpaLastShare.files[0].type))
    .toBe("video/webm");

  await page.locator("[data-dynamic-close]").click();
  await page.evaluate(() => {
    Object.defineProperty(HTMLCanvasElement.prototype, "captureStream", {
      configurable: true,
      value: undefined,
    });
    window.__vpaLastShare = null;
  });
  await page.locator("[data-dynamic-open]").click();
  await expect(page.locator(".quick-dynamic-output img")).toBeVisible();
  await expect(page.locator(".quick-dynamic-output audio")).toBeVisible();

  const fallbackAudio = await page.locator(".quick-dynamic-output audio").evaluate(async (element) => {
    const blob = await fetch(element.currentSrc || element.src).then((response) => response.blob());
    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      return {
        duration: decoded.duration,
        size: blob.size,
        type: blob.type,
      };
    } finally {
      await context.close();
    }
  });
  expect(fallbackAudio.type).toBe("audio/wav");
  expect(fallbackAudio.size).toBeGreaterThan(100_000);
  expect(fallbackAudio.duration).toBeCloseTo(4, 1);

  await page.locator("[data-dynamic-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(2);
  expect(await page.evaluate(() => window.__vpaLastShare.files.map((file) => file.name))).toEqual([
    "vpa-result.png",
    "vpa-voice-clip.wav",
  ]);
  expect(await page.evaluate(() => window.__vpaLastShare.text))
    .toContain("https://share.example/r/abcdefghijklmnop");
  expect(runtimeErrors).toEqual([]);
});
test.describe("mobile quick experience", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
  });

  test("quick landing and result remain inside the viewport", async ({ page }) => {
    const runtimeErrors = captureRuntimeErrors(page);
    await openDevelopmentPage(page);
    await page.evaluate(async (analysis) => {
      const { recorderCtl } = await import("/assets/app.js");
      recorderCtl.getLastRecordingUrl = () => "/tests/.generated-media/tone.wav";
      window.vpaAdvancedExperience.renderAnalysis(analysis);
    }, fixture);

    await page.locator("[data-quick-share]").click();
    await page.locator("[data-quick-audio]").check();
    await expect(page.locator(".quick-dynamic-card")).toBeVisible();
    await expect(page.locator("[data-dynamic-start], [data-dynamic-end]")).toHaveCount(0);
    const overflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth - document.documentElement.clientWidth;
    });
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator("#experienceNav")).toBeVisible();
    await expect(page.locator(".quick-result")).toBeVisible();
    if (process.env.VPA_QUICK_MOBILE_CAPTURE) {
      await page.screenshot({
        fullPage: true,
        path: resolve(process.env.VPA_QUICK_MOBILE_CAPTURE),
      });
    }
    expect(runtimeErrors).toEqual([]);
  });
});
