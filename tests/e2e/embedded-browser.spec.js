import { expect, test } from "@playwright/test";
import { installDeterministicRuntime } from "./helpers.js";

test.describe("embedded mobile browser guard", () => {
  test.use({
    userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/UQ1A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/124.0 Mobile Safari/537.36 Line/14.9.0",
    viewport: { height: 844, width: 390 },
  });

  test("warns before recording and enables constrained local inference", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/");

    const guard = page.locator("[data-embedded-browser-guard]");
    await expect(guard).toBeVisible();
    await expect(guard).toContainText("Safari");
    await expect(guard).toContainText("Chrome");
    await expect(page.getByRole("button", { name: "複製連結" })).toBeVisible();
    await expect.poll(() => page.evaluate(() => window.vpaEmbeddedBrowser)).toMatchObject({
      app: "line",
      embedded: true,
      platform: "android",
    });
  });
});
