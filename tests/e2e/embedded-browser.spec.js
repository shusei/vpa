import { expect, test } from "@playwright/test";
import { installDeterministicRuntime } from "./helpers.js";

test.describe("embedded mobile browser guard", () => {
  test.use({
    userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/UQ1A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/124.0 Mobile Safari/537.36 Line/14.9.0",
    viewport: { height: 844, width: 390 },
  });

  test("opens LINE links with the external-browser directive", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/dev.html?source=line#record");

    const guard = page.locator("[data-embedded-browser-guard]");
    await expect(guard).toBeVisible();
    await page.locator("[data-embedded-browser-open]").click();
    await expect(page).toHaveURL(/openExternalBrowser=1/);
    const externalUrl = new URL(page.url());
    expect(externalUrl.searchParams.get("source")).toBe("line");
    expect(externalUrl.hash).toBe("#record");
    await expect(guard).toBeVisible();
  });

  test("warns before recording and enables constrained local inference", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/dev.html");

    const guard = page.locator("[data-embedded-browser-guard]");
    await expect(guard).toBeVisible();
    await expect(guard).toContainText("Safari");
    await expect(guard).toContainText("Chrome");
    await expect(page.getByRole("button", { name: "用瀏覽器開啟" })).toBeVisible();
    await expect(page.getByRole("button", { name: "複製連結" })).toBeVisible();
    await expect.poll(() => page.evaluate(() => window.vpaEmbeddedBrowser)).toMatchObject({
      app: "line",
      embedded: true,
      platform: "android",
    });

    await page.getByRole("button", { name: "繼續在這裡使用" }).click();
    await page.locator("[data-quick-record]").click();
    await expect.poll(() => page.evaluate(() => window.__vpaPipelineCalls?.length || 0)).toBe(1);
    await page.locator("[data-quick-record]").click();
  });
});
