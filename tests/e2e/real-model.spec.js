import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { waitForAnalysis } from "./helpers.js";

test("real remote model completes a production WASM inference", async ({ page }) => {
  test.skip(process.env.VPA_REAL_MODEL !== "1", "Run explicitly because the model download is large.");
  test.setTimeout(600_000);

  await page.addInitScript(() => {
    try {
      localStorage.setItem("vpa.locale", "zh-Hant");
      localStorage.setItem("vpa.onboardTipDone", "1");
      localStorage.setItem("vpa.themeTipDone", "1");
      localStorage.setItem("vpa::experiment.experience", "professional");
    } catch { }
    Object.defineProperty(Navigator.prototype, "gpu", {
      configurable: true,
      get: () => undefined,
    });
  });
  await page.route("https://www.googletagmanager.com/gtag/js**", async (route) => {
    await route.fulfill({ status: 200, contentType: "text/javascript", body: "" });
  });

  await page.goto("/");
  await expect(page.locator("#playBtn")).toBeAttached();
  await page.locator("#fileInput").setInputFiles(resolve("tests/.generated-media/tone.mp3"));
  const analysis = await waitForAnalysis(page, 0, { timeout: 300_000 });

  expect(analysis.device).toBe("wasm");
  expect(analysis.probabilities.feminine).toBeGreaterThanOrEqual(0);
  expect(analysis.probabilities.masculine).toBeGreaterThanOrEqual(0);
  expect(analysis.probabilities.feminine + analysis.probabilities.masculine).toBeCloseTo(1, 5);
});

test("real remote model completes a production WebGPU inference when available", async ({ page }) => {
  test.skip(process.env.VPA_REAL_MODEL !== "1", "Run explicitly because the model download is large.");
  test.setTimeout(600_000);

  await page.addInitScript(() => {
    try {
      localStorage.setItem("vpa.locale", "zh-Hant");
      localStorage.setItem("vpa.onboardTipDone", "1");
      localStorage.setItem("vpa.themeTipDone", "1");
      localStorage.setItem("vpa::experiment.experience", "professional");
    } catch { }
  });
  await page.route("https://www.googletagmanager.com/gtag/js**", async (route) => {
    await route.fulfill({ status: 200, contentType: "text/javascript", body: "" });
  });

  await page.goto("/");
  await expect(page.locator("#playBtn")).toBeAttached();
  const hasWebGPU = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!hasWebGPU, "The current Chromium environment does not expose WebGPU.");

  await page.locator("#fileInput").setInputFiles(resolve("tests/.generated-media/tone.mp3"));
  const analysis = await waitForAnalysis(page, 0, { timeout: 300_000 });

  expect(analysis.device).toBe("webgpu");
  expect(analysis.probabilities.feminine + analysis.probabilities.masculine).toBeCloseTo(1, 5);
});
