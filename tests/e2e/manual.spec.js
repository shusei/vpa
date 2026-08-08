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

  test("mobile manual and help close controls stay above the mode banner", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });

    for (const surface of [
      { open: "#guideBtn", overlay: "#guideOverlay", close: ".guide-close" },
      { open: "#helpBtn", overlay: "#helpOverlay", close: ".help-close" },
    ]) {
      await page.locator(surface.open).click();
      const overlay = page.locator(surface.overlay);
      const closeButton = overlay.locator(surface.close);
      await expect(overlay).toBeVisible();
      await expect(closeButton).toBeVisible();

      const layers = await page.evaluate(({ overlaySelector, closeSelector }) => {
        const overlayElement = document.querySelector(overlaySelector);
        const nav = document.querySelector(".experience-nav");
        const closeElement = overlayElement?.querySelector(closeSelector);
        if (!overlayElement || !nav || !closeElement) return null;
        const rect = closeElement.getBoundingClientRect();
        const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
        return {
          closeIsTopmost: hit === closeElement || closeElement.contains(hit),
          nav: Number.parseInt(getComputedStyle(nav).zIndex, 10),
          overlay: Number.parseInt(getComputedStyle(overlayElement).zIndex, 10),
        };
      }, { overlaySelector: surface.overlay, closeSelector: surface.close });

      expect(layers).not.toBeNull();
      expect(layers.overlay).toBeGreaterThan(layers.nav);
      expect(layers.closeIsTopmost).toBe(true);
      await closeButton.click();
      await expect(overlay).toBeHidden();
    }
  });
});
