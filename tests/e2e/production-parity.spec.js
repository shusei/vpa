import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime, openProductionPage } from "./helpers.js";

const fixture = JSON.parse(readFileSync(resolve("fixtures/analysis/sweet_feminine.json"), "utf8"));

test("production advanced analysis preserves every acoustic fixture value", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openProductionPage(page);

  const actual = await page.evaluate(async (input) => {
    const metrics = await import("/assets/js/advanced-metrics.js");
    const pitch = await import("/assets/js/pitch-shared.js");
    const { computeAdvancedSummary } = await import("/assets/js/advanced-summary-core.js");
    const summary = computeAdvancedSummary({
      analyzeConnectedSpeech: metrics.analyzeConnectedSpeech,
      analyzeIntonation: metrics.analyzeIntonation,
      analyzeSpeechRate: metrics.analyzeSpeechRate,
      analyzeVowelFocus: metrics.analyzeVowelFocus,
      averageEnergy: metrics.averageEnergy,
      averageFinite: metrics.averageFinite,
      buildEligibleFrameMask: metrics.buildEligibleFrameMask,
      categorizeBreathiness: metrics.categorizeBreathiness,
      categorizeBrightness: metrics.categorizeBrightness,
      categorizeTilt: metrics.categorizeTilt,
      describeResonanceFromEnergy: metrics.describeResonanceFromEnergy,
      detectVoiceLeaning: metrics.detectVoiceLeaning,
      FORMANT_CONFIDENCE_THRESHOLD: pitch.CONFIDENCE_INCLUDE_THRESHOLD,
      FORMANT_MAX_GAP_FRAMES: 8,
      lastPf: input.probabilities.feminine,
      lastPm: input.probabilities.masculine,
      makeStats: pitch.makeStats,
      offlineFeatureStore: input.offlineSamples,
      percentileSorted: pitch.percentileSorted,
      PS_INTERVAL_MS: pitch.PS_INTERVAL_MS,
      summarizeBreathiness: metrics.summarizeBreathiness,
      summarizeFormantTrends: metrics.summarizeFormantTrends,
    });
    return JSON.parse(JSON.stringify(summary, (_key, value) => {
      if (typeof value === "number" && !Number.isFinite(value)) return null;
      return value;
    }));
  }, fixture);

  const clean = (obj) => {
    const cloned = JSON.parse(JSON.stringify(obj));
    if (cloned.intonation) {
      delete cloned.intonation.rangeKey;
      delete cloned.intonation.slopeKey;
    }
    return JSON.parse(JSON.stringify(cloned).replaceAll(" (", "（").replaceAll(")", "）"));
  };

  const cleanData = (obj) => {
    const preserveDataAndShape = (value) => {
      if (Array.isArray(value)) return value.map(preserveDataAndShape);
      if (value && typeof value === "object") {
        return Object.fromEntries(
          Object.entries(value).map(([key, entry]) => [key, preserveDataAndShape(entry)]),
        );
      }
      return typeof value === "string" ? "<localized-text>" : value;
    };
    return preserveDataAndShape(clean(obj));
  };

  expect(cleanData(actual)).toEqual(cleanData(fixture.advanced));
  expect(runtimeErrors).toEqual([]);
});

test("all themes apply and survive a reload", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openProductionPage(page);
  const themes = await page.locator(".theme-item").evaluateAll((buttons) => buttons.map((button) => button.dataset.theme));
  const failures = [];

  for (const theme of themes) {
    await page.locator("#settingsBtn").click();
    await page.locator(`.theme-item[data-theme="${theme}"]`).click();
    const applied = await page.locator("html").getAttribute("data-theme");
    if (applied !== theme) failures.push(`${theme}: applied=${applied}`);

    await page.reload();
    await expect(page.locator("#playBtn")).toBeAttached();
    const restored = await page.locator("html").getAttribute("data-theme");
    const checked = await page.locator(`.theme-item[data-theme="${theme}"]`).getAttribute("aria-checked");
    if (restored !== theme || checked !== "true") {
      const storage = await page.evaluate(() => ({
        mode: localStorage.getItem("vpa::theme.mode"),
        light: localStorage.getItem("vpa::theme.light"),
        dark: localStorage.getItem("vpa::theme.dark"),
      }));
      failures.push(`${theme}: restored=${restored}, checked=${checked}, storage=${JSON.stringify(storage)}`);
    }
  }

  expect(themes.length).toBeGreaterThanOrEqual(30);
  expect(failures).toEqual([]);
  expect(runtimeErrors).toEqual([]);
});

test.describe("mobile production interactions", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
    isMobile: true,
  });

  test("keeps primary tabs, menus and overlays usable without horizontal overflow", async ({ page }) => {
    const runtimeErrors = captureRuntimeErrors(page);
    await openProductionPage(page);

    await page.locator("#helpBtn").click();
    await expect(page.locator("#helpOverlay")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.locator("#helpOverlay")).toBeHidden();

    await page.locator("#guideBtn").click();
    await expect(page.locator("#guideOverlay")).toBeVisible();
    await page.locator(".guide-close").click();

    await page.locator("#practiceToggle").click();
    await expect(page.locator("#practicePanel")).toBeVisible();
    await page.locator("#settingsBtn").click();
    await expect(page.locator("#themeMenu")).toBeVisible();

    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator("#recordBtn")).toBeVisible();
    await expect(page.locator("#uploadFab")).toBeVisible();
    expect(runtimeErrors).toEqual([]);
  });
});

test("GA4 tag submits page_view and production analysis_completed", async ({ page }) => {
  await installDeterministicRuntime(page, { mockAnalytics: false });
  const acceptedEvents = [];
  await page.route("https://www.googletagmanager.com/gtag/js**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/javascript",
      body: `
        setTimeout(() => {
          const queuedGtag = window.gtag;
          window.gtag = function(command, name, parameters) {
            queuedGtag?.apply(this, arguments);
            if (command === "event") {
              fetch("https://www.google-analytics.com/g/collect?en=" + encodeURIComponent(name));
            }
          };
          fetch("https://www.google-analytics.com/g/collect?en=page_view");
        }, 0);
      `,
    });
  });
  await page.route(/https:\/\/(?:www\.)?google-analytics\.com\/g\/collect.*/, async (route) => {
    await route.fulfill({ status: 204, body: "" });
  });
  page.on("response", (response) => {
    const url = response.url();
    if (url.includes("google-analytics.com/g/collect")) {
      const eventName = new URL(url).searchParams.get("en");
      acceptedEvents.push({ eventName, status: response.status() });
    }
  });

  await page.goto("/");
  await expect(page.locator("#playBtn")).toBeAttached();
  await expect.poll(() => acceptedEvents.some((event) => event.eventName === "page_view"), {
    timeout: 30_000,
  }).toBe(true);
  expect(acceptedEvents.find((event) => event.eventName === "page_view")?.status).toBe(204);

  await page.evaluate(async () => {
    const { createAnalysisTelemetryController } = await import("/assets/js/analysis-telemetry.js");
    createAnalysisTelemetryController().setLatestAnalysisExport({
      source: "playwright_validation",
      pitch: { stats: { med: 220 }, spreadHz: 40 },
      summary: { voicedRatio: 0.9 },
    });
  });

  await expect.poll(() => acceptedEvents.some((event) => event.eventName === "analysis_completed"), {
    timeout: 30_000,
  }).toBe(true);
  expect(acceptedEvents.find((event) => event.eventName === "analysis_completed")?.status).toBe(204);
  await expect.poll(() => page.evaluate(() => window.__vpaLastGAEvent?.name || "")).toBe("analysis_completed");
  console.log(`[ga4] accepted ${acceptedEvents.map((event) => `${event.eventName}:${event.status}`).join(", ")}`);
});
