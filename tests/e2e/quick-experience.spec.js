import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

async function openDevelopmentPage(page, options = {}) {
  await installDeterministicRuntime(page, options);
  await page.goto("/dev.html");
  await expect.poll(() => page.evaluate(() => Boolean(
    window.vpaAdvancedExperience && window.vpaExperience
  ))).toBe(true);
}

test("quick and professional experiences share one analysis result", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await expect(page.locator("html")).toHaveAttribute("data-experience", "quick");
  await expect(page.locator("#quickExperience")).toBeVisible();
  await expect(page.locator("body > .hero")).toBeHidden();

  const result = await page.evaluate((analysis) => {
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);
  await expect(page.locator(".quick-result__score strong")).toHaveText(String(result.score));

  await page.locator('.quick-result [data-experience-target="professional"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-experience", "professional");
  await expect(page.locator("body > .hero")).toBeVisible();
  await expect(page.locator("#advancedExperience")).toBeVisible();
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0)).toBe(0);

  await page.locator('#experienceNav [data-experience-target="quick"]').click();
  await expect(page.locator(".quick-result__score strong")).toHaveText(String(result.score));
  if (process.env.VPA_QUICK_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_QUICK_CAPTURE),
    });
  }
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
    const result = window.vpaAdvancedExperience.renderAnalysis(analysis);
    const { createChallengeUrl } = await import("/assets/experiments/challenge-link.js");
    return createChallengeUrl(result);
  }, fixture);

  expect(challenge.url).not.toContain("audio");
  expect(Object.keys(challenge.payload).sort()).toEqual([
    "ageMax",
    "ageMin",
    "archetype",
    "id",
    "schema",
    "score",
    "scoreVersion",
  ]);

  await page.goto(challenge.url);
  await page.reload();
  await expect(page.locator(".quick-challenge-invite")).toContainText(`${challenge.payload.score}`);
  await page.evaluate((analysis) => {
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);
  await expect(page.locator(".quick-comparison")).toContainText("同分");
  expect(runtimeErrors).toEqual([]);
});

test("audio sharing is opt-in and uses the system file share path", async ({ page }) => {
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
          url: payload.url,
        };
      },
    });
  });
  await openDevelopmentPage(page);

  await page.evaluate(async (analysis) => {
    const { recorderCtl } = await import("/assets/app.js");
    const audioUrl = URL.createObjectURL(new Blob(["voice"], { type: "audio/webm" }));
    recorderCtl.getLastRecordingUrl = () => audioUrl;
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  await page.locator("[data-quick-share]").click();
  await expect(page.locator("[data-quick-audio]")).not.toBeChecked();
  await page.locator("[data-quick-system-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(1);
  expect(await page.evaluate(() => window.__vpaLastShare.files[0].name)).toBe("vpa-result.png");

  await page.locator("[data-quick-audio]").check();
  await expect(page.locator(".quick-audio-preview audio")).toBeVisible();
  await page.locator("[data-quick-system-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(2);
  expect(await page.evaluate(() => window.__vpaLastShare.files.map((file) => file.name))).toEqual([
    "vpa-result.png",
    "vpa-voice.weba",
  ]);
  expect(await page.evaluate(() => Object.keys(localStorage).some((key) => key.includes("audio"))))
    .toBe(false);
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
    await page.evaluate((analysis) => {
      window.vpaAdvancedExperience.renderAnalysis(analysis);
    }, fixture);

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
