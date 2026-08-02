import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

test.describe("Public landing page", () => {
  let runtimeErrors;

  test.beforeEach(async ({ page }) => {
    runtimeErrors = captureRuntimeErrors(page);
    await installDeterministicRuntime(page);
    await page.goto("/");
    await expect.poll(() => page.evaluate(() => Boolean(window.vpaExperience))).toBe(true);
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  test("uses the promoted quick experience as the homepage", async ({ page }) => {
    await expect(page).toHaveTitle(/Voice Presentation Analyzer/);
    await expect(page.locator("html")).toHaveAttribute("data-experience", "quick");
    await expect(page.locator(".quick-landing h1")).toBeVisible();
    await expect(page.locator("body > .hero")).toBeHidden();
    const deployment = await page.evaluate(() => ({
      publicAppUrl: window.VPA_PUBLIC_APP_URL,
      workerOrigin: window.VPA_SHARE_SERVICE_ORIGIN,
    }));
    expect(deployment.publicAppUrl).toBe("https://shusei.github.io/vpa/");
    expect(deployment.workerOrigin).toBe("https://vpa-share.evelynjoellelin.workers.dev");
  });

  test("shows the primary quick actions", async ({ page }) => {
    await expect(page.locator("#experienceNav")).toBeVisible();
    await expect(page.locator("[data-quick-record]")).toBeVisible();
    await expect(page.locator(".quick-standard-cta")).toBeVisible();
    await expect(page.locator("[data-quick-locale-toggle]")).toBeVisible();
    await expect(page.locator("[data-quick-locale-toggle]")).toHaveText("繁");
    await expect(page.locator(".quick-footer")).toBeVisible();
  });

  test("starts ready to record", async ({ page }) => {
    await expect(page.locator("[data-quick-record]")).toBeEnabled();
    await expect(page.locator('[data-quick-stage="idle"]')).toBeVisible();
  });
});
