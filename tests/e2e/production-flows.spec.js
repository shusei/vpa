import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import {
  captureRuntimeErrors,
  openProductionPage,
  waitForAnalysis,
} from "./helpers.js";

const mediaDir = resolve("tests/.generated-media");

test.describe("Production audio flows", () => {
  let runtimeErrors;

  test.beforeEach(async ({ page }) => {
    runtimeErrors = captureRuntimeErrors(page);
    await openProductionPage(page);
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  test("records from an authorized microphone and completes analysis", async ({ page }) => {
    const recordButton = page.locator("#recordBtn");

    await recordButton.click();
    await expect(page.locator("body")).toHaveClass(/recording/);
    await expect(page.locator("#statusLabel")).toContainText("錄音中");
    await page.waitForTimeout(1_200);

    await recordButton.click();
    await expect(page.locator("body")).not.toHaveClass(/recording/);
    const analysis = await waitForAnalysis(page);

    expect(analysis.probabilities).toEqual({ feminine: 0.64, masculine: 0.36 });
    await expect(page.locator("#playBtn")).toBeEnabled();
    await expect(page.locator("#streamStats")).not.toBeEmpty();
    expect(await page.locator("#playback").getAttribute("src")).toMatch(/^blob:/);
  });

  for (const extension of ["mp3", "m4a", "mp4"]) {
    test(`uploads and analyzes a real ${extension.toUpperCase()} file`, async ({ page }) => {
      await page.locator("#fileInput").setInputFiles(resolve(mediaDir, `tone.${extension}`));
      const analysis = await waitForAnalysis(page);

      expect(analysis.probabilities).toEqual({ feminine: 0.64, masculine: 0.36 });
      expect(analysis.realtimeStream.volumeDb.length).toBeGreaterThan(0);
      await expect(page.locator("#playBtn")).toBeEnabled();
      await expect(page.locator("#streamStats")).not.toBeEmpty();
      await expect(page.locator("#femaleVal")).toHaveText("64.0%");
      await expect(page.locator("#maleVal")).toHaveText("36.0%");
    });
  }

  for (const expectedDevice of ["wasm", "webgpu"]) {
    test(`passes ${expectedDevice} to the model pipeline for the matching browser capability`, async ({ page }) => {
      await page.evaluate((device) => {
        Object.defineProperty(Navigator.prototype, "gpu", {
          configurable: true,
          get: () => device === "webgpu" ? {} : undefined,
        });
      }, expectedDevice);
      await page.locator("#fileInput").setInputFiles(resolve(mediaDir, "tone.mp3"));
      await waitForAnalysis(page);

      const devices = await page.evaluate(() => (window.__vpaPipelineCalls || []).map((call) => call.device));
      expect(devices).toEqual([expectedDevice]);
    });
  }
});
