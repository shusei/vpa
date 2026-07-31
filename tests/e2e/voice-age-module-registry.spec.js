import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import {
  captureRuntimeErrors,
  installDeterministicRuntime,
  waitForAnalysis,
} from "./helpers.js";

test("dev analysis keeps the Voice Age analyzer connected after cache busting", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await installDeterministicRuntime(page);
  await page.goto("/dev.html");
  await page.locator('#experienceNav [data-experience-target="professional"]').click();
  await page.locator("#fileInput").setInputFiles(resolve("tests/.generated-media/tone.wav"));

  const analysis = await waitForAnalysis(page);
  expect(analysis.offlineSamples.extensions["voice-age-v2"]?.quality?.ready).toBe(true);
  await expect.poll(() => page.evaluate(() => (
    window.vpaAdvancedExperience?.getLastResult()?.voiceAge?.ready
  ))).toBe(true);
  await expect(page.getByText("本次不推估", { exact: true })).toHaveCount(0);
  expect(runtimeErrors).toEqual([]);
});
