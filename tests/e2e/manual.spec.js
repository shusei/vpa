import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, openProductionPage } from "./helpers.js";

test.describe("Voice Manual Overlay", () => {
  let runtimeErrors;

  test.beforeEach(async ({ page }) => {
    runtimeErrors = captureRuntimeErrors(page);
    await openProductionPage(page);
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  test("opens the manual from the book button", async ({ page }) => {
    const guideBtn = page.locator("#guideBtn");
    const overlay = page.locator("#guideOverlay");

    await expect(overlay).toBeHidden();
    await guideBtn.click();
    await expect(overlay).toBeVisible();
    await expect(page.locator("#guideTitle")).toHaveText("女聲訓練手冊");
    await expect(page.locator("#guideContent")).toContainText("0) 你每天照做什麼");
  });

  test("closes the manual and restores focus", async ({ page }) => {
    const guideBtn = page.locator("#guideBtn");
    const overlay = page.locator("#guideOverlay");

    await guideBtn.click();
    await expect(overlay).toBeVisible();
    await page.locator(".guide-close").click();
    await expect(overlay).toBeHidden();
    await expect(guideBtn).toBeFocused();
  });
});
