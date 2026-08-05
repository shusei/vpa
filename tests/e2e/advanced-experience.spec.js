import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import {
  captureRuntimeErrors,
  installDeterministicRuntime,
  waitForAnalysis,
} from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

async function openDevelopmentPage(page) {
  await installDeterministicRuntime(page);
  await page.goto("/");
  await expect(page.locator("#playBtn")).toBeAttached();
  await expect(page.locator(".player")).toBeHidden();
  await expect.poll(() => page.evaluate(() => Boolean(window.vpaAdvancedExperience))).toBe(true);
  await expect.poll(() => page.evaluate(() => Boolean(window.vpaExperience))).toBe(true);
}

test("professional experience renders advanced result and creates a local share card", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.locator('#experienceNav [data-experience-target="professional"]').click();
  const result = await page.evaluate((analysis) => {
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  expect(result.ready).toBe(true);
  expect(result.score).toBeGreaterThanOrEqual(60);
  expect(result.score).toBeLessThan(100);
  await expect(page.locator("#advancedExperience")).toBeVisible();
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  await expect(page.locator(".advanced-experience__pitch")).toContainText("代表音高");
  await expect(page.locator(".advanced-experience__pitch strong")).toContainText("232.8");
  await expect(page.locator("#advancedExperience").getByText("聲音年齡印象", { exact: true }))
    .toBeVisible();
  await expect(page.getByRole("heading", { name: "聲音年齡 2.0 證據" })).toBeVisible();
  await expect(page.locator(".voice-age-evidence__metrics")).toContainText("Jitter");
  await expect(page.locator(".voice-age-evidence__metrics")).toContainText("Shimmer");
  await expect(page.locator(".voice-age-evidence__metrics")).toContainText("HNR");
  await expect(page.locator(".voice-age-evidence__metrics")).toContainText("CPP");
  await expect(page.locator(".voice-age-evidence > code"))
    .toHaveText("voice-age-impression-2.0.0-research");
  const ageEvents = await page.evaluate(() => (
    (window.dataLayer || [])
      .map((entry) => Array.from(entry))
      .filter((entry) => entry[0] === "event" && entry[1] === "voice_age_evaluated")
      .map((entry) => entry[2])
  ));
  expect(ageEvents).toHaveLength(1);
  expect(ageEvents[0]).toMatchObject({
    confidence: "medium",
    ready: true,
    sample_type: "connectedSpeech",
    version: "voice-age-impression-2.0.0-research",
  });
  const privateKeys = [
    "age",
    "age_band",
    "archetype",
    "cpp",
    "hnr",
    "jitter",
    "shimmer",
  ];
  expect(Object.keys(ageEvents[0]).filter((key) => privateKeys.includes(key))).toEqual([]);
  await expect(page.getByRole("button", { name: "分享圖片＋文字（推薦）" })).toBeEnabled();
  await expect(page.locator(".advanced-share__hint")).toContainText(
    "X／Threads／LINE 會直接開啟發文頁",
  );

  const blob = await page.evaluate(async () => {
    const card = await window.vpaAdvancedExperience.createCard();
    return {
      size: card.size,
      type: card.type,
    };
  });
  expect(blob.type).toBe("image/png");
  expect(blob.size).toBeGreaterThan(10_000);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "下載分享卡" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("vpa-advanced-result.png");
  await expect(page.getByText("分享卡已下載。")).toBeVisible();
  if (process.env.VPA_CARD_CAPTURE) {
    await download.saveAs(resolve(process.env.VPA_CARD_CAPTURE));
  }
  if (process.env.VPA_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_CAPTURE),
    });
  }
  expect(runtimeErrors).toEqual([]);
});

test("professional X sharing publishes the personalized public result", async ({ page }) => {
  let releasePublicShare;
  const publicShareGate = new Promise((resolvePromise) => {
    releasePublicShare = resolvePromise;
  });
  await page.addInitScript(() => {
    window.VPA_SHARE_SERVICE_ORIGIN = "https://share.example";
  });
  await page.route("https://share.example/api/shares", async (route) => {
    await publicShareGate;
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
  await openDevelopmentPage(page);
  await page.evaluate((analysis) => {
    window.vpaExperience.setExperience("professional");
    window.vpaAdvancedExperience.renderAnalysis(analysis);
    window.open = (url) => {
      if (String(url) !== "about:blank") window.__vpaOpenedShareUrl = String(url);
      return null;
    };
  }, fixture);

  await expect(page.locator("[data-share-platform]")).toHaveCount(3);
  await expect(page.locator('[data-share-platform="facebook"], [data-share-instagram]')).toHaveCount(0);

  await expect(page.locator('[data-share-platform="x"]')).toBeDisabled();
  releasePublicShare();
  await expect(page.locator('[data-share-platform="x"]')).toBeEnabled();
  await page.locator('[data-share-platform="x"]').click();
  await expect.poll(() => page.evaluate(() => window.__vpaOpenedShareUrl || ""))
    .toContain("https://twitter.com/intent/tweet?");
  const opened = new URL(await page.evaluate(() => window.__vpaOpenedShareUrl));
  expect(`${opened.origin}${opened.pathname}`).toBe("https://twitter.com/intent/tweet");
  expect(opened.searchParams.get("url")).toBe("https://share.example/r/abcdefghijklmnop");
  expect(opened.searchParams.get("hashtags")).toBe("VoicePresentationAnalyzer");
  expect(opened.toString()).not.toContain("#vpa-challenge=");
  expect(runtimeErrors).toEqual([]);
  await page.evaluate(() => { window.__vpaOpenedShareUrl = ""; });
  await page.locator('[data-share-platform="threads"]').click();
  await expect.poll(() => page.evaluate(() => window.__vpaOpenedShareUrl || ""))
    .toContain("https://www.threads.com/intent/post?text=");
  const threadsOpened = decodeURIComponent(await page.evaluate(() => window.__vpaOpenedShareUrl));
  expect(threadsOpened).toContain("https://share.example/r/abcdefghijklmnop");
  expect(threadsOpened).not.toContain("#vpa-challenge=");

});

test("age refusal keeps the strict presentation result and sharing available", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);
  const withoutAgeMetrics = structuredClone(fixture);
  delete withoutAgeMetrics.offlineSamples.extensions["voice-age-v2"];

  const result = await page.evaluate((analysis) => {
    window.vpaExperience.setExperience("professional");
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, withoutAgeMetrics);

  expect(result.ready).toBe(true);
  expect(result.voiceAge.ready).toBe(false);
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  await expect(page.locator("#advancedExperience").getByText("本次不推估", { exact: true }))
    .toBeVisible();
  await expect(page.locator(".voice-age-evidence")).toContainText(
    "本次保留男女聲結果，但不輸出聲音年齡",
  );
  await expect(page.getByRole("button", { name: "分享圖片＋文字（推薦）" })).toBeEnabled();
  const caption = await page.evaluate(() => (
    window.vpaAdvancedExperience.formatResult(window.vpaAdvancedExperience.getLastResult()).caption
  ));
  expect(caption).toContain("未可靠推估聲音年齡");
  expect(runtimeErrors).toEqual([]);
});

test("decoded dev audio captures local Voice Age 2.0 acoustic summaries", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);
  await page.locator('#experienceNav [data-experience-target="professional"]').click();
  await page.locator("#fileInput").setInputFiles(resolve("tests/.generated-media/tone.wav"));
  const analysis = await waitForAnalysis(page);
  const voiceQuality = analysis.offlineSamples.extensions["voice-age-v2"];

  expect(voiceQuality.version).toBe("voice-quality-1.0.0");
  expect(voiceQuality.sampleType).toBe("sustainedVowel");
  expect(voiceQuality.quality.ready).toBe(true);
  expect(voiceQuality.metrics.jitterLocal.reliable).toBe(true);
  expect(voiceQuality.metrics.shimmerLocal.reliable).toBe(true);
  expect(voiceQuality.metrics.hnr.valueDb).toBeGreaterThan(0);
  expect(voiceQuality.metrics.cpp.valueDb).toBeGreaterThan(0);
  expect(runtimeErrors).toEqual([]);
});

test("advanced experiment text keeps accessible contrast across every theme", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);
  await page.evaluate((analysis) => {
    window.vpaExperience.setExperience("professional");
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  const themes = await page.locator(".theme-item").evaluateAll((buttons) => {
    return buttons.map((button) => button.dataset.theme);
  });
  const failures = [];

  for (const theme of themes) {
    await page.locator("#settingsBtn").click();
    await page.locator(`.theme-item[data-theme="${theme}"]`).click();
    const audit = await page.evaluate(() => {
      const host = document.querySelector("#advancedExperience");
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

      function composite(foreground, background) {
        const alpha = foreground[3] / 255;
        return [
          (foreground[0] * alpha) + (background[0] * (1 - alpha)),
          (foreground[1] * alpha) + (background[1] * (1 - alpha)),
          (foreground[2] * alpha) + (background[2] * (1 - alpha)),
          255,
        ];
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
        const backgroundRgba = rgba(background);
        const foregroundRgba = composite(rgba(foreground), backgroundRgba);
        const foregroundLuminance = luminance(foregroundRgba);
        const backgroundLuminance = luminance(backgroundRgba);
        return (Math.max(foregroundLuminance, backgroundLuminance) + 0.05)
          / (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
      }

      function resolvedPair(foregroundToken, backgroundToken) {
        const probe = document.createElement("span");
        probe.style.color = `var(${foregroundToken})`;
        probe.style.backgroundColor = `var(${backgroundToken})`;
        host.append(probe);
        const style = getComputedStyle(probe);
        const pair = {
          background: style.backgroundColor,
          foreground: style.color,
        };
        probe.remove();
        return {
          ...pair,
          ratio: contrast(pair.foreground, pair.background),
        };
      }

      const pairs = [
        ["--experiment-ink", "--experiment-bg-a"],
        ["--experiment-ink", "--experiment-bg-b"],
        ["--experiment-ink", "--experiment-card"],
        ["--experiment-muted", "--experiment-bg-a"],
        ["--experiment-muted", "--experiment-card"],
        ["--experiment-active-ink", "--experiment-active-a"],
        ["--experiment-active-ink", "--experiment-active-b"],
        ["--experiment-emphasis", "--experiment-bg-a"],
        ["--experiment-emphasis", "--experiment-card"],
        ["--stream-ink", "--band-gray"],
        ["--stream-axis", "--band-gray"],
      ].map(([foregroundToken, backgroundToken]) => ({
        backgroundToken,
        foregroundToken,
        ...resolvedPair(foregroundToken, backgroundToken),
      }));
      return {
        faction: document.documentElement.dataset.faction,
        pairs,
      };
    });
    if (process.env.VPA_THEME_CAPTURE && ["day", "pony2026", "warm", "contrast"].includes(theme)) {
      await page.locator("#advancedExperience").screenshot({
        path: resolve(`test-results/advanced-panel-${theme}.png`),
      });
    }

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

test.describe("mobile advanced experience", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
  });

  test("experience selector and result panel stay within the viewport", async ({ page }) => {
    const runtimeErrors = captureRuntimeErrors(page);
    await openDevelopmentPage(page);
    await page.locator('#experienceNav [data-experience-target="professional"]').click();
    await page.evaluate((analysis) => {
      window.vpaAdvancedExperience.renderAnalysis(analysis);
    }, fixture);

    const overflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth - document.documentElement.clientWidth;
    });
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator("#advancedExperience")).toBeVisible();
    await expect(page.locator(".advanced-share__platforms")).toBeVisible();
    expect(runtimeErrors).toEqual([]);
  });
});
