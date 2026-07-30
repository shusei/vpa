import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, openProductionPage } from "./helpers.js";

test.describe("Landing Page", () => {
  let runtimeErrors;

  test.beforeEach(async ({ page }) => {
    runtimeErrors = captureRuntimeErrors(page);
    await openProductionPage(page);
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  test("has correct metadata", async ({ page }) => {
    await expect(page).toHaveTitle(/Voice Presentation Analyzer/);
    await expect(page.locator("h1.hero-title")).toHaveText("Voice Presentation Analyzer");
  });

  test("shows critical UI elements", async ({ page }) => {
    await expect(page.locator("#helpBtn")).toBeVisible();
    await expect(page.locator("#settingsBtn")).toBeVisible();
    await expect(page.locator("#recordBtn")).toBeVisible();
    await expect(page.locator("#uploadFab")).toBeVisible();
    await expect(page.locator("footer.footer")).toBeVisible();
  });

  test("starts in the ready state", async ({ page }) => {
    await expect(page.locator("#recordBtn")).toBeEnabled();
    await expect(page.locator("#statusLabel")).toHaveText(/準備就緒/);
  });
});
