import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import {
  installDeterministicRuntime,
  openProductionPage,
  waitForAnalysis,
} from "./helpers.js";

const generatedMediaDir = resolve("tests/.generated-media");
const persistentMediaDir = resolve("fixtures/media");

async function openMediaPage(page) {
  await page.addInitScript(() => {
    window.__vpaStatusHistory = [];
    window.addEventListener("DOMContentLoaded", () => {
      const status = document.getElementById("statusLabel");
      if (!status) return;
      const capture = () => window.__vpaStatusHistory.push(status.textContent || "");
      new MutationObserver(capture).observe(status, { childList: true, subtree: true });
      capture();
    });
  });
  await installDeterministicRuntime(page);
  await openProductionPage(page);
}

test.describe("@cross-browser video upload matrix", () => {
  test.beforeEach(async ({ page }) => {
    await openMediaPage(page);
  });

  test("analyzes MP4, HEVC/MOV, long, and large videos", async ({ page }) => {
    test.setTimeout(600_000);
    const uploads = [
      { path: resolve(generatedMediaDir, "tone.mp4"), long: false },
      { path: resolve(persistentMediaDir, "tone-hevc.mov"), long: false },
      { path: resolve(persistentMediaDir, "tone-long.mp4"), long: true },
      { path: resolve(generatedMediaDir, "tone-large.mp4"), long: false },
    ];
    let previousAnalysisId = 0;

    for (const upload of uploads) {
      await page.locator("#fileInput").setInputFiles(upload.path);
      const analysis = await waitForAnalysis(page, previousAnalysisId, { timeout: 180_000 });
      previousAnalysisId = analysis.analysisId;
      expect(analysis.offlineSamples.duration).toBeGreaterThan(3);
      expect(analysis.realtimeStream.volumeDb.length).toBeGreaterThan(0);
      await expect(page.locator("#playBtn")).toBeEnabled();
      if (upload.long) {
        expect(await page.evaluate(() => window.__vpaStatusHistory)).toEqual(
          expect.arrayContaining([expect.stringContaining("分析可能較久")]),
        );
      }
    }

    await expect(page.locator("#statusLabel")).not.toContainText("處理失敗");
  });

  test("rejects no-audio, corrupt, and empty videos then recovers", async ({ page }) => {
    test.setTimeout(300_000);
    const invalidUploads = [
      resolve(persistentMediaDir, "no-audio.mp4"),
      resolve(generatedMediaDir, "corrupt.mp4"),
      resolve(generatedMediaDir, "empty.mp4"),
    ];

    for (const upload of invalidUploads) {
      await page.locator("#fileInput").setInputFiles(upload);
      await expect(page.locator("#statusLabel")).toContainText("無法解碼或分析", {
        timeout: 120_000,
      });
      expect(await page.evaluate(() => window.vpaLatestAnalysis?.analysisId || 0)).toBe(0);
    }

    await page.locator("#fileInput").setInputFiles(resolve(generatedMediaDir, "tone.mp3"));
    const analysis = await waitForAnalysis(page, 0, { timeout: 120_000 });
    expect(analysis.probabilities).toEqual({ feminine: 0.64, masculine: 0.36 });
    await expect(page.locator("#playBtn")).toBeEnabled();
  });
});
