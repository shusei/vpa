import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

const expected = {
  en: { quick: "Comfort matters more than the score", advanced: "Comfort comes first—do not train to the score" },
  ja: { quick: "点数より快適さが大切", advanced: "快適さを最優先し、点数を目標に練習しない" },
  "zh-Hans": { quick: "舒服比高分重要", advanced: "先以舒适为准，不要追着分数练" },
  "zh-Hant": { quick: "舒服比高分重要", advanced: "先以舒適為準，不要追著分數練" },
};

async function switchLocale(page, locale) {
  if (await page.locator("html").getAttribute("lang") === locale) return;
  await page.locator("[data-quick-locale-toggle]").click();
  await page.locator(`[data-quick-locale="${locale}"]`).click();
  await expect(page.locator("html")).toHaveAttribute("lang", locale);
}

test("Quick, Advanced, Help, and Manual expose safety guidance in all locales", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await installDeterministicRuntime(page, { storedLocale: "zh-Hant" });
  await page.addInitScript(() => {
    window.VPA_SHARE_SERVICE_ORIGIN = "";
  });
  await page.goto("/");
  await expect.poll(() => page.evaluate(() => Boolean(window.vpaAdvancedExperience))).toBe(true);

  for (const [locale, copy] of Object.entries(expected)) {
    await switchLocale(page, locale);
    await page.evaluate((analysis) => window.vpaAdvancedExperience.renderAnalysis(analysis), fixture);

    await expect(page.locator(".quick-result__safety")).toBeVisible();
    await expect(page.locator(".quick-result__safety")).toContainText(copy.quick);

    await page.locator('.quick-result [data-experience-target="professional"]').click();
    await expect(page.locator(".advanced-experience__safety")).toBeVisible();
    await expect(page.locator(".advanced-experience__safety")).toContainText(copy.advanced);
    await page.locator("#helpBtn").click();
    await expect(page.locator("#helpOverlay .help-safety-panel")).toBeVisible();
    await expect(page.locator("#helpOverlay .help-evidence-links a")).toHaveCount(4);
    await expect(page.locator('#helpOverlay a[href*="asha.org"]')).toHaveAttribute("rel", "noopener");
    await page.locator("#helpOverlay .help-close").click();

    await page.locator("#guideBtn").click();
    await expect(page.locator("#guideOverlay .manual-stop-panel")).toBeVisible();
    await expect(page.locator("#guideOverlay .manual-source-panel a")).toHaveCount(4);
    await expect(page.locator('#guideOverlay a[href*="nidcd.nih.gov/health/hoarseness"]')).toHaveAttribute("rel", "noopener");
    await page.locator("#guideOverlay .guide-close").click();

    await page.locator('#experienceNav [data-experience-target="quick"]').click();
  }

  expect(runtimeErrors).toEqual([]);
});

test("safety content remains readable without horizontal overflow on a phone", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installDeterministicRuntime(page, { storedLocale: "en" });
  await page.addInitScript(() => {
    window.VPA_SHARE_SERVICE_ORIGIN = "";
  });
  await page.goto("/");
  await expect.poll(() => page.evaluate(() => Boolean(window.vpaAdvancedExperience))).toBe(true);
  await page.evaluate((analysis) => window.vpaAdvancedExperience.renderAnalysis(analysis), fixture);
  await expect(page.locator(".quick-result__safety")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);

  await page.locator('.quick-result [data-experience-target="professional"]').click();
  await expect(page.locator(".advanced-experience__safety")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);

  await page.locator("#guideBtn").click();
  await expect(page.locator("#guideOverlay .manual-stop-panel")).toBeVisible();
  expect(await page.locator("#guideOverlay .help-dialog").evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
});
