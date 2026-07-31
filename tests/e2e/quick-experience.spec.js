import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { captureRuntimeErrors, installDeterministicRuntime } from "./helpers.js";

const fixture = JSON.parse(readFileSync(
  resolve("fixtures/analysis/sweet_feminine.json"),
  "utf8",
));

async function openDevelopmentPage(page, options = {}) {
  await installDeterministicRuntime(page, options);
  await page.goto("/dev.html");
  await expect.poll(() => page.evaluate(() => Boolean(
    window.vpaAdvancedExperience && window.vpaExperience
  ))).toBe(true);
}

test("quick and professional experiences share one analysis result", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await expect(page.locator("html")).toHaveAttribute("data-experience", "quick");
  await expect(page.locator("#quickExperience")).toBeVisible();
  await expect(page.locator("body > .hero")).toBeHidden();
  await expect(page.locator(".quick-landing .quick-primary")).toHaveCount(1);
  await expect(page.locator(".quick-standard-cta")).toBeVisible();
  if (process.env.VPA_QUICK_LANDING_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_QUICK_LANDING_CAPTURE),
    });
  }

  const result = await page.evaluate((analysis) => {
    return window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);
  await expect(page.locator(".quick-result__feminine strong")).toHaveText(`${result.score}%`);
  await expect(page.locator(".quick-result__masculine strong")).toHaveText(`${100 - result.score}%`);

  await page.locator('.quick-result [data-experience-target="professional"]').click();
  await expect(page.locator("html")).toHaveAttribute("data-experience", "professional");
  await expect(page.locator("body > .hero")).toBeVisible();
  await expect(page.locator("#advancedExperience")).toBeVisible();
  await expect(page.locator(".advanced-experience__score strong")).toHaveText(`${result.score}%`);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0)).toBe(0);

  await page.locator('#experienceNav [data-experience-target="quick"]').click();
  await expect(page.locator(".quick-result__feminine strong")).toHaveText(`${result.score}%`);
  if (process.env.VPA_QUICK_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_QUICK_CAPTURE),
    });
  }
  expect(runtimeErrors).toEqual([]);
});

test("quick experience text keeps accessible contrast across every theme", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  const themes = await page.locator(".theme-item").evaluateAll((buttons) => {
    return buttons.map((button) => button.dataset.theme);
  });
  const failures = [];

  for (const theme of themes) {
    await page.locator(`.theme-item[data-theme="${theme}"]`).evaluate((button) => button.click());
    const audit = await page.evaluate(() => {
      const host = document.querySelector("#quickExperience");
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
        const foregroundLuminance = luminance(rgba(foreground));
        const backgroundLuminance = luminance(rgba(background));
        return (Math.max(foregroundLuminance, backgroundLuminance) + 0.05)
          / (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
      }

      function resolvedColor(value) {
        const probe = document.createElement("span");
        probe.style.color = value;
        host.append(probe);
        const color = getComputedStyle(probe).color;
        probe.remove();
        return color;
      }

      const pairs = [
        ["body", "var(--quick-ink)", "var(--quick-bg)"],
        ["body-gradient-end", "var(--quick-ink)", "var(--quick-bg-end)"],
        ["muted-body", "var(--quick-muted)", "var(--quick-bg)"],
        ["muted-body-gradient-end", "var(--quick-muted)", "var(--quick-bg-end)"],
        ["card", "var(--quick-ink)", "var(--quick-surface)"],
        ["muted-card", "var(--quick-muted)", "var(--quick-surface)"],
        ["strong-card", "var(--quick-ink)", "var(--quick-surface-strong)"],
        ["muted-strong-card", "var(--quick-muted)", "var(--quick-surface-strong)"],
        ["primary", "var(--quick-action-ink)", "var(--quick-action)"],
        ["primary-gradient-end", "var(--quick-action-ink)", "var(--quick-action-end)"],
        ["emphasis-body", "var(--quick-emphasis)", "var(--quick-bg)"],
        ["emphasis-card", "var(--quick-emphasis)", "var(--quick-surface)"],
        ["brand-accent", "var(--quick-accent-ink)", "var(--quick-accent)"],
        ["brand-accent-2", "var(--quick-accent-ink)", "var(--quick-accent-2)"],
        ["stop-start", "var(--quick-danger-ink)", "var(--quick-danger-a)"],
        ["stop-end", "var(--quick-danger-ink)", "var(--quick-danger-b)"],
      ].map(([label, foreground, background]) => {
        const resolvedForeground = resolvedColor(foreground);
        const resolvedBackground = resolvedColor(background);
        return {
          background: resolvedBackground,
          foreground: resolvedForeground,
          label,
          ratio: contrast(resolvedForeground, resolvedBackground),
        };
      });

      return {
        faction: document.documentElement.dataset.faction,
        pairs,
      };
    });

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

test("quick recording delegates to the production recording and analysis path", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.locator("[data-quick-record]").click();
  await expect(page.locator('[data-quick-stage="recording"]')).toBeVisible({ timeout: 30_000 });
  await page.waitForTimeout(1200);
  await page.locator("[data-quick-record]").click();
  await expect(page.locator('[data-quick-stage="result"]')).toBeVisible({ timeout: 60_000 });

  expect(await page.evaluate(() => window.vpaExperience.getLatestAnalysis()?.analysisId || 0))
    .toBeGreaterThan(0);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0))
    .toBeGreaterThan(0);
  expect(await page.evaluate(() => {
    return (window.dataLayer || [])
      .map((entry) => Array.from(entry))
      .filter((entry) => entry[0] === "event")
      .map((entry) => entry[1]);
  })).toEqual(expect.arrayContaining(["quick_test_started", "quick_test_completed"]));
  expect(runtimeErrors).toEqual([]);
});

test("three-line standard test runs the production path three times and uses the median", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  await page.locator("[data-quick-standard]").click();
  await expect(page.locator(".quick-standard h1")).toHaveText("三句標準測試");
  await expect(page.locator(".quick-prompt > span")).toContainText("1 / 3");
  if (process.env.VPA_STANDARD_LANDING_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_STANDARD_LANDING_CAPTURE),
    });
  }

  for (let step = 1; step <= 3; step += 1) {
    await page.locator("[data-quick-record]").click();
    await expect(page.locator('[data-quick-stage="recording"]')).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.locator(".quick-progress__prompt")).not.toBeEmpty();
    await page.waitForTimeout(1200);
    await page.locator("[data-quick-record]").click();

    if (step < 3) {
      await expect(page.locator('[data-quick-stage="standard-next"]')).toBeVisible({
        timeout: 60_000,
      });
      await expect(page.locator(".quick-standard-scores span")).toHaveCount(step);
      await page.locator("[data-quick-standard-next]").click();
      await expect(page.locator(".quick-prompt > span")).toContainText(`${step + 1} / 3`);
    }
  }

  await expect(page.locator('[data-quick-stage="result"]')).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.locator(".quick-standard-summary")).toBeVisible();
  const result = await page.evaluate(() => window.vpaExperience.getLatestResult());
  expect(result.quickTest).toEqual({
    mode: "standard",
    promptId: "standard-v1",
  });
  expect(result.standard.scores).toHaveLength(3);
  expect(result.score).toBe([...result.standard.scores].sort((a, b) => a - b)[1]);
  expect(await page.evaluate(() => window.__vpaInferenceCalls?.length || 0)).toBeGreaterThanOrEqual(3);
  const standardChallenge = await page.evaluate(async () => {
    const { createChallengeUrl } = await import("/assets/experiments/challenge-link.js");
    return createChallengeUrl(window.vpaExperience.getLatestResult());
  });
  expect(standardChallenge.payload).toMatchObject({
    promptId: "standard-v1",
    schema: 2,
    testMode: "standard",
  });
  if (process.env.VPA_STANDARD_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_STANDARD_CAPTURE),
    });
  }

  await page.goto(standardChallenge.url);
  await page.reload();
  await expect.poll(() => page.evaluate(() => window.vpaExperience?.getQuickTestMode()))
    .toBe("standard");
  await expect(page.locator(".quick-challenge-invite")).toContainText(`${result.score}%`);
  await expect(page.locator(".quick-standard h1")).toBeVisible();
  expect(runtimeErrors).toEqual([]);
});

test("Japanese browser language becomes the default locale and a saved choice wins", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page, {
    navigatorLanguages: ["ja-JP", "en-US"],
    storedLocale: null,
  });

  await expect(page.locator("html")).toHaveAttribute("lang", "ja");
  await expect(page.locator(".quick-landing h1")).toHaveText("あなたの声は、どんな印象？");
  await expect(page.locator("[data-quick-locale]")).toHaveValue("ja");
  if (process.env.VPA_JA_CAPTURE) {
    await page.screenshot({
      fullPage: true,
      path: resolve(process.env.VPA_JA_CAPTURE),
    });
  }

  await page.locator("[data-quick-locale]").selectOption("zh-Hant");
  await expect(page.locator(".quick-landing h1")).toHaveText("你的聲音，給人什麼印象？");
  expect(await page.evaluate(() => localStorage.getItem("vpa.locale"))).toBe("zh-Hant");

  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "zh-Hant");
  await expect(page.locator("[data-quick-locale]")).toHaveValue("zh-Hant");

  expect(await page.evaluate(async () => {
    const { i18nInternals } = await import("/assets/js/i18n.js");
    return ["zh-TW", "zh-CN", "en-US", "ja-JP", "fr-FR"]
      .map((locale) => i18nInternals.mapCandidateLocale(locale));
  })).toEqual(["zh-Hant", "zh-Hans", "en", "ja", null]);

  await page.locator("[data-quick-locale]").selectOption("ja");
  await page.locator('#experienceNav [data-experience-target="professional"]').click();
  await page.locator("#practiceToggle").click();
  await expect(page.locator("#practiceList")).toContainText("すみません、少しお時間をいただけますか。");
  await page.locator("#helpBtn").click();
  await expect(page.locator("#helpOverlay")).toContainText("使い方ガイド");
  await expect(page.locator("#helpOverlay")).toContainText("パネルの見方");
  expect(runtimeErrors).toEqual([]);
});

test("challenge link carries only summary data and compares the next result", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await openDevelopmentPage(page);

  const challenge = await page.evaluate(async (analysis) => {
    window.vpaAdvancedExperience.renderAnalysis(analysis);
    const result = window.vpaExperience.getLatestResult();
    const { createChallengeUrl } = await import("/assets/experiments/challenge-link.js");
    return createChallengeUrl(result);
  }, fixture);

  expect(challenge.url).not.toContain("audio");
  expect(Object.keys(challenge.payload).sort()).toEqual([
    "ageMax",
    "ageMin",
    "archetype",
    "id",
    "promptId",
    "schema",
    "score",
    "scoreVersion",
    "testMode",
  ]);
  expect(challenge.payload.schema).toBe(2);
  expect(challenge.payload.testMode).toBe("daily");

  await page.goto(challenge.url);
  await page.reload();
  await expect(page.locator(".quick-challenge-invite")).toContainText(`${challenge.payload.score}`);
  expect(await page.evaluate(() => window.vpaExperience.getLatestResult())).toBe(null);
  await page.evaluate((analysis) => {
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);
  await expect(page.locator(".quick-comparison")).toContainText("百分比相同");
  await expect(page.locator(".quick-comparison")).toContainText(
    `${challenge.payload.score}%`,
  );
  expect(runtimeErrors).toEqual([]);
});

test("audio sharing is opt-in and uses the system file share path", async ({ page }) => {
  const runtimeErrors = captureRuntimeErrors(page);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "canShare", {
      configurable: true,
      value: ({ files }) => Array.isArray(files) && files.length > 0,
    });
    Object.defineProperty(navigator, "share", {
      configurable: true,
      value: async (payload) => {
        window.__vpaLastShare = {
          files: (payload.files || []).map((file) => ({
            name: file.name,
            type: file.type,
          })),
          text: payload.text,
          url: payload.url,
        };
      },
    });
  });
  await openDevelopmentPage(page);

  await page.evaluate(async (analysis) => {
    const { recorderCtl } = await import("/assets/app.js");
    const audioUrl = URL.createObjectURL(new Blob(["voice"], { type: "audio/webm" }));
    recorderCtl.getLastRecordingUrl = () => audioUrl;
    window.vpaAdvancedExperience.renderAnalysis(analysis);
  }, fixture);

  await page.locator("[data-quick-share]").click();
  await expect(page.locator("[data-quick-audio]")).not.toBeChecked();
  await page.locator("[data-quick-system-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(1);
  expect(await page.evaluate(() => window.__vpaLastShare.files[0].name)).toBe("vpa-result.png");

  await page.locator("[data-quick-audio]").check();
  await expect(page.locator(".quick-audio-preview audio")).toBeVisible();
  await page.locator("[data-quick-system-share]").click();
  await expect.poll(() => page.evaluate(() => window.__vpaLastShare?.files.length)).toBe(2);
  expect(await page.evaluate(() => window.__vpaLastShare.files.map((file) => file.name))).toEqual([
    "vpa-result.png",
    "vpa-voice.weba",
  ]);
  expect(await page.evaluate(() => Object.keys(localStorage).some((key) => key.includes("audio"))))
    .toBe(false);
  expect(runtimeErrors).toEqual([]);
});

test.describe("mobile quick experience", () => {
  test.use({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
  });

  test("quick landing and result remain inside the viewport", async ({ page }) => {
    const runtimeErrors = captureRuntimeErrors(page);
    await openDevelopmentPage(page);
    await page.evaluate((analysis) => {
      window.vpaAdvancedExperience.renderAnalysis(analysis);
    }, fixture);

    const overflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth - document.documentElement.clientWidth;
    });
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator("#experienceNav")).toBeVisible();
    await expect(page.locator(".quick-result")).toBeVisible();
    if (process.env.VPA_QUICK_MOBILE_CAPTURE) {
      await page.screenshot({
        fullPage: true,
        path: resolve(process.env.VPA_QUICK_MOBILE_CAPTURE),
      });
    }
    expect(runtimeErrors).toEqual([]);
  });
});
