import { expect } from "@playwright/test";

const TRANSFORMERS_STUB = `
export const env = { backends: { onnx: { wasm: {} } } };
export async function pipeline(task, modelId, options = {}) {
  globalThis.__vpaPipelineCalls = globalThis.__vpaPipelineCalls || [];
  globalThis.__vpaPipelineCalls.push({ task, modelId, device: options.device });
  return async function classify(audio, inferenceOptions = {}) {
    globalThis.__vpaInferenceCalls = globalThis.__vpaInferenceCalls || [];
    globalThis.__vpaInferenceCalls.push({ samples: audio.length, inferenceOptions });
    return [
      { label: "female", score: 0.64 },
      { label: "male", score: 0.36 }
    ];
  };
}
`;

export async function installDeterministicRuntime(page, {
  mockAnalytics = true,
  navigatorLanguages = null,
  storedLocale = "zh-Hant",
} = {}) {
  await page.addInitScript(({ languages, locale }) => {
    if (!sessionStorage.getItem("__vpaTestSeeded")) {
      try {
        if (locale === null) {
          localStorage.removeItem("vpa.locale");
        } else {
          localStorage.setItem("vpa.locale", locale);
        }
        localStorage.setItem("vpa.onboardTipDone", "1");
        localStorage.setItem("vpa.themeTipDone", "1");
        sessionStorage.setItem("__vpaTestSeeded", "1");
      } catch { }
    }
    if (Array.isArray(languages) && languages.length) {
      Object.defineProperty(navigator, "languages", {
        configurable: true,
        get: () => [...languages],
      });
      Object.defineProperty(navigator, "language", {
        configurable: true,
        get: () => languages[0],
      });
    }
  }, {
    languages: navigatorLanguages,
    locale: storedLocale,
  });

  await page.route("https://cdn.jsdelivr.net/**/transformers.min.js", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/javascript",
      body: TRANSFORMERS_STUB,
    });
  });

  await page.route("https://cdn.buymeacoffee.com/**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "image/svg+xml",
      body: `<svg xmlns="http://www.w3.org/2000/svg" width="180" height="42" viewBox="0 0 180 42"><rect width="180" height="42" rx="21" fill="#ffdd00"/></svg>`,
    });
  });

  if (mockAnalytics) {
    await page.route("https://www.googletagmanager.com/gtag/js**", async (route) => {
      await route.fulfill({ status: 200, contentType: "text/javascript", body: "" });
    });
    await page.route(/https:\/\/(?:www\.)?google-analytics\.com\/g\/collect.*/, async (route) => {
      await route.fulfill({ status: 204, body: "" });
    });
  }
}

export function captureRuntimeErrors(page) {
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  return errors;
}

export async function openProductionPage(page, options) {
  await installDeterministicRuntime(page, options);
  await page.addInitScript(() => {
    localStorage.setItem("vpa::experiment.experience", "professional");
  });
  await page.goto("/");
  await expect(page.locator("#playBtn")).toBeAttached();
  await expect(page.locator(".player")).toBeHidden();
}

export async function waitForAnalysis(page, previousAnalysisId = 0, { timeout = 60_000 } = {}) {
  await expect.poll(async () => page.evaluate(() => window.vpaLatestAnalysis?.analysisId || 0), {
    timeout,
  }).toBeGreaterThan(previousAnalysisId);
  return page.evaluate(() => window.vpaLatestAnalysis);
}
