import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

async function openDevelopmentPage(page) {
  await installDeterministicRuntime(page);
  await page.goto("/dev.html");
  await expect(page.locator("#playBtn")).toBeAttached();
  await expect.poll(() => page.evaluate(() => Boolean(window.vpaAdvancedExperience))).toBe(true);
}

test("development page renders advanced result and creates a local share card", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.getByRole("button", { name: "進階嚴格" }).click();
  const result = await page.evaluate((analysis) => {
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  expect(result.ready).toBe(true);
  expect(result.score).toBeGreaterThanOrEqual(60);
  expect(result.score).toBeLessThan(100);
  await expect(page.locator("#advancedExperience")).toBeVisible();
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  await expect(page.getByText("聲音年齡印象", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "產生分享卡" })).toBeEnabled();

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

test.describe("mobile advanced experience", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
  });

  test("mode selector and result panel stay within the viewport", async ({ page }) => {
    const runtimeErrors = captureRuntimeErrors(page);
    await openDevelopmentPage(page);
    await page.getByRole("button", { name: "進階嚴格" }).click();
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
