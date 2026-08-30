import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import zhHant from "../../assets/i18n/zh-Hant.js";
import zhHans from "../../assets/i18n/zh-Hans.js";
import en from "../../assets/i18n/en.js";
import ja from "../../assets/i18n/ja.js";
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
      const presentation = await page.evaluate(() => (
        window.vpaAdvancedExperience.evaluate(window.vpaLatestAnalysis)
      ));
      expect(presentation.ready).toBe(true);
      expect(presentation.score).not.toBe(64);
      await expect(page.locator("#femaleVal")).toHaveText(`${presentation.score.toFixed(1)}%`);
      await expect(page.locator("#maleVal")).toHaveText(`${(100 - presentation.score).toFixed(1)}%`);
    });
  }

  test("uses the vendored FFmpeg fallback when native MP4 decoding fails", async ({ page }) => {
    test.skip(
      !existsSync(resolve("assets/vendor/ffmpeg/ffmpeg-core.wasm")),
      "FFmpeg WASM is downloaded only for production and focused fallback tests.",
    );

    await page.evaluate(() => {
      const AudioContextLike = window.AudioContext || window.webkitAudioContext;
      AudioContextLike.prototype.decodeAudioData = function(_buffer, _success, failure) {
        const error = new DOMException("Forced native decoder failure", "EncodingError");
        if (typeof failure === "function") {
          queueMicrotask(() => failure(error));
          return undefined;
        }
        return Promise.reject(error);
      };
    });

    await page.locator("#fileInput").setInputFiles(resolve(mediaDir, "tone.mp4"));
    const analysis = await waitForAnalysis(page, 0, { timeout: 120_000 });

    expect(analysis.probabilities).toEqual({ feminine: 0.64, masculine: 0.36 });
    expect(analysis.realtimeStream.volumeDb.length).toBeGreaterThan(0);
    await expect(page.locator("#playBtn")).toBeEnabled();
  });

  test("practice cards use the latest integrated percentage and store it separately from legacy scores", async ({ page }) => {
    await page.locator("#practiceToggle").click();
    const card = page.locator("#practiceList .practice-card").first();
    const recordButton = card.locator('[data-act="toggle"]');
    const phraseId = await card.getAttribute("data-id");

    await recordButton.click();
    await expect(page.locator("body")).toHaveClass(/recording/);
    await page.waitForTimeout(1_200);
    await recordButton.click();
    await expect(page.locator("body")).not.toHaveClass(/recording/);
    await waitForAnalysis(page);

    const presentation = await page.evaluate(() => (
      window.vpaAdvancedExperience.evaluate(window.vpaLatestAnalysis)
    ));
    expect(presentation.ready).toBe(true);
    expect(presentation.score).not.toBe(64);
    await expect(card.locator(".practice-result .fem")).toContainText(`${presentation.score}%`);
    await expect(card.locator(".practice-result .masc")).toContainText(`${100 - presentation.score}%`);
    await expect(page.locator("#femaleVal")).toHaveText(`${presentation.score.toFixed(1)}%`);

    const stored = await page.evaluate(() => ({
      current: Object.fromEntries(JSON.parse(localStorage.getItem("vpa.practice.v2.history") || "[]")),
      legacy: localStorage.getItem("vpa.practice.v1.history"),
    }));
    const lastStored = stored.current[phraseId].at(-1);
    expect(lastStored.pf).toBe(presentation.score / 100);
    expect(lastStored.pm).toBe((100 - presentation.score) / 100);
    expect(stored.legacy).toBeNull();
  });

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
  test("keeps every information heading visible in all four locales", async ({ page }) => {
    const expectedQuickStartTitles = {
      "zh-Hant": zhHant.help.quickStart.title,
      "zh-Hans": zhHans.help.quickStart.title,
      en: en.help.quickStart.title,
      ja: ja.help.quickStart.title,
    };
    const summaries = page.locator("section.info > details > summary");

    for (const [locale, expectedTitle] of Object.entries(expectedQuickStartTitles)) {
      if (await page.locator("html").getAttribute("lang") !== locale) {
        await page.locator("[data-quick-locale-toggle]").click();
        await page.locator('[data-quick-locale="' + locale + '"]').click();
      }

      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await expect(summaries).toHaveCount(8);
      await expect(summaries.first()).toHaveText(expectedTitle);
      const summaryTexts = await summaries.allTextContents();
      expect(
        summaryTexts.every((text) => text.trim().length > 0),
        locale + " contains an empty information heading",
      ).toBe(true);
    }
  });
  test("one language switch updates the complete advanced analysis and supporting UI", async ({ page }) => {
    await page.locator("#fileInput").setInputFiles(resolve(mediaDir, "tone.mp3"));
    await waitForAnalysis(page);

    await page.locator("[data-quick-locale-toggle]").click();
    await page.locator('[data-quick-locale="en"]').click();
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.locator("#streamStats")).toContainText("Formant & spectral proxies");

    await page.locator("#practiceToggle").click();
    await expect(page.locator("#practiceList")).toBeVisible();
    await page.locator("#guideBtn").click();
    await expect(page.locator("#guideOverlay")).toBeVisible();
    await expect(page.locator("#guideTitle")).toHaveText("Feminine Voice Manual");
    await expect(page.locator("#guideBtn")).toHaveAttribute("aria-label", "Feminine Voice Manual");
    await expect(page.locator(".guide-close")).toHaveAttribute("aria-label", "Close manual");
    await expect(page.locator(".guide-top")).toHaveAttribute("aria-label", "Back to top");
    await expect(page.locator("#ver")).not.toHaveText("build");

    const englishSurfaces = await page.evaluate(() => ({
      advanced: document.querySelector("#streamStats")?.textContent || "",
      guide: document.querySelector("#guideOverlay")?.textContent || "",
      info: document.querySelector("section.info")?.textContent || "",
      practice: document.querySelector("#practiceList")?.textContent || "",
    }));
    for (const [surface, text] of Object.entries(englishSurfaces)) {
      expect(text, `${surface} still contains Chinese after switching to English`)
        .not.toMatch(/[\u3400-\u9fff]/u);
    }
  });
});
