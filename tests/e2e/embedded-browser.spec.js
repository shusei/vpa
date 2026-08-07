import { expect, test } from "@playwright/test";
import { installDeterministicRuntime } from "./helpers.js";

test.describe("embedded mobile browser guard", () => {
  test.use({
    userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/UQ1A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/124.0 Mobile Safari/537.36 Line/14.9.0",
    viewport: { height: 844, width: 390 },
  });

  test("opens LINE links with the external-browser directive", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/?source=line#record");

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
    await page.goto("/");

    const guard = page.locator("[data-embedded-browser-guard]");
    await expect(guard).toBeVisible();
    await expect(guard).toContainText(/Open in your browser|常用的瀏覽器/);
    await expect(page.getByRole("button", { name: /Open in browser|用瀏覽器開啟/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Copy link|複製連結/ })).toBeVisible();
    await expect.poll(() => page.evaluate(() => window.vpaEmbeddedBrowser)).toMatchObject({
      app: "line",
      embedded: true,
      platform: "android",
    });

    await page.getByRole("button", { name: /Continue here|繼續在這裡使用/ }).click();
    await page.locator("[data-quick-record]").click();
    await expect.poll(() => page.evaluate(() => window.__vpaPipelineCalls?.length || 0)).toBe(1);
    await page.locator("[data-quick-record]").click();
  });
});

test.describe("iOS social app challenge browser", () => {
  test.use({
    userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Twitter for iPhone",
    viewport: { height: 844, width: 390 },
  });

  test("opens a challenge immediately without an unusable external-browser prompt", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/#vpa-challenge=abc");

    await expect(page.locator("[data-embedded-browser-guard]")).toHaveCount(0);
    await expect(page.locator("[data-quick-record]")).toBeVisible();
    await expect(page.locator("[data-embedded-fast-notice]")).toContainText("社群快速挑戰");
    await expect(page.locator("[data-embedded-fast-notice]")).toContainText("下方中間附近");
    await expect(page.locator("[data-embedded-fast-notice]")).toContainText("Safari");
    await expect.poll(() => page.evaluate(() => window.vpaEmbeddedBrowser)).toMatchObject({
      app: "x",
      embedded: true,
      platform: "ios",
    });
  });

  test("finishes a challenge locally without downloading the large model", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/#vpa-challenge=abc");

    await page.locator("[data-quick-record]").click();
    await expect(page.locator('[data-quick-stage="recording"]')).toBeVisible({
      timeout: 30_000,
    });
    await page.waitForTimeout(1200);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator('[data-quick-stage="result"]')).toBeVisible({
      timeout: 60_000,
    });

    expect(await page.evaluate(() => window.__vpaPipelineCalls?.length || 0)).toBe(0);
    expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0)).toBe(0);
    expect(await page.evaluate(() => window.vpaLatestAnalysis?.device)).toBe("acoustic-social-1");
  });

  test("does not show a fake one-tap Safari button on ordinary pages", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/");

    const guard = page.locator("[data-embedded-browser-guard]");
    await expect(guard).toBeVisible();
    await expect(guard.locator("[data-embedded-browser-open]")).toHaveCount(0);
    await expect(guard.locator("[data-embedded-browser-copy]")).toBeVisible();
    await expect(guard.locator("[data-embedded-browser-close]")).toBeVisible();
  });
});
test.describe("iOS Threads challenge browser", () => {
  test.use({
    userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Barcelona 335.0.0",
    viewport: { height: 844, width: 390 },
  });

  test("uses the same local fast challenge path", async ({ page }) => {
    await installDeterministicRuntime(page);
    await page.goto("/#vpa-challenge=abc");

    await expect(page.locator("[data-embedded-browser-guard]")).toHaveCount(0);
    await expect(page.locator("[data-embedded-fast-notice]")).toBeVisible();
    await expect.poll(() => page.evaluate(() => window.vpaEmbeddedBrowser)).toMatchObject({
      app: "threads",
      embedded: true,
      platform: "ios",
    });

    await page.locator("[data-quick-record]").click();
    await expect(page.locator('[data-quick-stage="recording"]')).toBeVisible({
      timeout: 30_000,
    });
    await page.waitForTimeout(1200);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator('[data-quick-stage="result"]')).toBeVisible({
      timeout: 60_000,
    });
    expect(await page.evaluate(() => window.__vpaPipelineCalls?.length || 0)).toBe(0);
    expect(await page.evaluate(() => window.vpaLatestAnalysis?.device)).toBe("acoustic-social-1");
  });
});
