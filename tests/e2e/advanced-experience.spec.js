import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

async function openDevelopmentPage(page) {
  await installDeterministicRuntime(page);
  await page.goto("/dev.html");
  await expect(page.locator("#playBtn")).toBeAttached();
  await expect.poll(() => page.evaluate(() => Boolean(window.vpaAdvancedExperience))).toBe(true);
}

test("development page renders advanced result and creates a local share card", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.getByRole("button", { name: "進階嚴格" }).click();
  const result = await page.evaluate((analysis) => {
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  expect(result.ready).toBe(true);
  expect(result.score).toBeGreaterThanOrEqual(60);
  expect(result.score).toBeLessThan(100);
  await expect(page.locator("#advancedExperience")).toBeVisible();
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  await expect(page.getByText("聲音年齡印象", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "產生分享卡" })).toBeEnabled();

  const blob = await page.evaluate(async () => {
    const card = await window.vpaAdvancedExperience.createCard();
    return {
      size: card.size,
      type: card.type,
    };
  });
  expect(blob.type).toBe("image/png");
  expect(blob.size).toBeGreaterThan(10_000);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "下載分享卡" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("vpa-advanced-result.png");
  await expect(page.getByText("分享卡已下載。")).toBeVisible();
  if (process.env.VPA_CARD_CAPTURE) {
    await download.saveAs(resolve(process.env.VPA_CARD_CAPTURE));
  }
  if (process.env.VPA_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_CAPTURE),
    });
  }
  expect(runtimeErrors).toEqual([]);
});

test("advanced experiment text keeps accessible contrast across every theme", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);
  await page.evaluate((analysis) => {
    window.vpaAdvancedExperience.setMode("advanced");
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  const themes = await page.locator(".theme-item").evaluateAll((buttons) => {
    return buttons.map((button) => button.dataset.theme);
  });
  const failures = [];

  for (const theme of themes) {
    await page.locator("#settingsBtn").click();
    await page.locator(`.theme-item[data-theme="${theme}"]`).click();
    const audit = await page.evaluate(() => {
      const host = document.querySelector(".analysis-mode-experiment");
      const canvas = document.createElement("canvas");
      canvas.width = 1;
      canvas.height = 1;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      function rgba(value) {
        ctx.clearRect(0, 0, 1, 1);
        ctx.fillStyle = "rgba(0, 0, 0, 0)";
        ctx.fillRect(0, 0, 1, 1);
        ctx.fillStyle = value;
        ctx.fillRect(0, 0, 1, 1);
        return Array.from(ctx.getImageData(0, 0, 1, 1).data);
      }

      function composite(foreground, background) {
        const alpha = foreground[3] / 255;
        return [
          (foreground[0] * alpha) + (background[0] * (1 - alpha)),
          (foreground[1] * alpha) + (background[1] * (1 - alpha)),
          (foreground[2] * alpha) + (background[2] * (1 - alpha)),
          255,
        ];
      }

      function luminance(color) {
        const channels = color.slice(0, 3).map((channel) => {
          const normalized = channel / 255;
          return normalized <= 0.04045
            ? normalized / 12.92
            : ((normalized + 0.055) / 1.055) ** 2.4;
        });
        return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2]);
      }

      function contrast(foreground, background) {
        const backgroundRgba = rgba(background);
        const foregroundRgba = composite(rgba(foreground), backgroundRgba);
        const foregroundLuminance = luminance(foregroundRgba);
        const backgroundLuminance = luminance(backgroundRgba);
        return (Math.max(foregroundLuminance, backgroundLuminance) + 0.05)
          / (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
      }

      function resolvedPair(foregroundToken, backgroundToken) {
        const probe = document.createElement("span");
        probe.style.color = `var(${foregroundToken})`;
        probe.style.backgroundColor = `var(${backgroundToken})`;
        host.append(probe);
        const style = getComputedStyle(probe);
        const pair = {
          background: style.backgroundColor,
          foreground: style.color,
        };
        probe.remove();
        return {
          ...pair,
          ratio: contrast(pair.foreground, pair.background),
        };
      }

      const pairs = [
        ["--experiment-ink", "--experiment-bg-a"],
        ["--experiment-ink", "--experiment-bg-b"],
        ["--experiment-ink", "--experiment-card"],
        ["--experiment-muted", "--experiment-bg-a"],
        ["--experiment-muted", "--experiment-card"],
        ["--experiment-active-ink", "--experiment-active-a"],
        ["--experiment-active-ink", "--experiment-active-b"],
        ["--experiment-emphasis", "--experiment-bg-a"],
        ["--experiment-emphasis", "--experiment-card"],
      ].map(([foregroundToken, backgroundToken]) => ({
        backgroundToken,
        foregroundToken,
        ...resolvedPair(foregroundToken, backgroundToken),
      }));
      return {
        faction: document.documentElement.dataset.faction,
        pairs,
      };
    });
    if (process.env.VPA_THEME_CAPTURE && ["day", "pony2026", "warm", "contrast"].includes(theme)) {
      await page.locator(".analysis-mode-experiment").screenshot({
        path: resolve(`test-results/advanced-mode-${theme}.png`),
      });
      await page.locator("#advancedExperience").screenshot({
        path: resolve(`test-results/advanced-panel-${theme}.png`),
      });
    }

    for (const pair of audit.pairs) {
      if (pair.ratio < 4.5) {
        failures.push({
          ...pair,
          faction: audit.faction,
          ratio: Number(pair.ratio.toFixed(2)),
          theme,
        });
      }
    }
  }

  expect(themes.length).toBeGreaterThanOrEqual(30);
  expect(failures).toEqual([]);
  expect(runtimeErrors).toEqual([]);
});

test.describe("mobile advanced experience", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
  });

  test("mode selector and result panel stay within the viewport", async ({ page }) => {
    const runtimeErrors = captureRuntimeErrors(page);
    await openDevelopmentPage(page);
    await page.getByRole("button", { name: "進階嚴格" }).click();
    await page.evaluate((analysis) => {
      window.vpaAdvancedExperience.renderAnalysis(analysis);
    }, fixture);

    const overflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth - document.documentElement.clientWidth;
    });
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator("#advancedExperience")).toBeVisible();
    await expect(page.locator(".advanced-share__platforms")).toBeVisible();
    expect(runtimeErrors).toEqual([]);
  });
});
