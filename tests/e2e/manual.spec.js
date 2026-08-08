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
    await expect(page.locator("#guideTitle")).toHaveText("女性化聲音手冊");
    await expect(page.locator("#guideContent")).toContainText("一次安全的自我比較流程");
    await expect(page.locator("#guideContent .manual-stop-panel")).toBeVisible();
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

  test("footer author shortcut opens the author card and restores focus on close", async ({ page }) => {
    const shortcut = page.locator("[data-author-shortcut]:visible");
    const overlay = page.locator("#helpOverlay");

    await shortcut.scrollIntoViewIfNeeded();
    await shortcut.click();
    await expect(overlay).toBeVisible();
    await expect(page.locator(".help-author")).toBeInViewport();
    await expect(page.locator("#helpOverlay .help-close")).toBeVisible();

    await page.locator("#helpOverlay .help-close").click();
    await expect(overlay).toBeHidden();
    await expect(shortcut).toBeFocused();
  });
});
