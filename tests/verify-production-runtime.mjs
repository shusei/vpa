import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const port = 4173;
const origin = `http://127.0.0.1:${port}`;
const appUrl = `${origin}/vpa/`;
const viteBin = resolve("node_modules/vite/bin/vite.js");
const server = spawn(process.execPath, [viteBin, "preview", "--host", "127.0.0.1", "--port", String(port)], {
  cwd: process.cwd(),
  stdio: ["ignore", "pipe", "pipe"],
});

async function waitForServer() {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(appUrl, { cache: "no-store" });
      if (response.ok) return;
    } catch { }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
  throw new Error("Vite preview did not become ready");
}

const fixture = JSON.parse(readFileSync(resolve("fixtures/analysis/sweet_feminine.json"), "utf8"));
const errors = [];
let browser;

try {
  await waitForServer();
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  await context.addInitScript(() => {
    window.VPA_SHARE_SERVICE_ORIGIN = "";
    localStorage.setItem("vpa.locale", "en");
    localStorage.setItem("vpa.onboardTipDone", "1");
    localStorage.setItem("vpa.themeTipDone", "1");
  });
  const page = await context.newPage();
  const localAssets = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("response", (response) => {
    const url = response.url();
    if (url.startsWith(origin) && /\.(?:js|css)(?:\?|$)/.test(url)) localAssets.push(url);
    if (url.startsWith(origin) && response.status() >= 400) errors.push(`${response.status()}: ${url}`);
  });
  await page.route("https://cdn.jsdelivr.net/**/transformers.min.js", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/javascript",
      body: "export const env={backends:{onnx:{wasm:{}}}};export async function pipeline(){return async()=>[{label:'female',score:.64},{label:'male',score:.36}]} ",
    });
  });
  await page.route("https://cdn.buymeacoffee.com/**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="180" height="42" />',
    });
  });
  await page.route("https://www.googletagmanager.com/**", (route) => route.fulfill({ status: 204, body: "" }));
  await page.goto(appUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.vpaAdvancedExperience && window.vpaExperience));

  assert.equal(await page.locator("html").getAttribute("data-experience"), "quick");
  assert.ok(await page.locator("#quickExperience").isVisible(), "production quick experience is not visible");
  assert.ok(localAssets.length >= 3, "production page did not request bundled JS/CSS");
  for (const url of localAssets) {
    assert.match(url, /\/vpa\/assets\/build\/[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{6,}\.(?:js|css)$/);
    assert.doesNotMatch(url, /\?v=/);
  }

  await page.evaluate((analysis) => window.vpaAdvancedExperience.renderAnalysis(analysis), fixture);
  assert.ok(await page.locator(".quick-result__insight").isVisible(), "production quick summary is missing");
  assert.equal(await page.locator(".quick-result .guidance-list").count(), 0, "production Quick still contains long guidance");
  assert.equal(await page.locator(".quick-result__safety").count(), 0, "production Quick still contains the safety panel");
  assert.ok(await page.locator(".quick-share-shortcuts").isVisible(), "production quick-share shortcuts are missing");
  await page.locator('.quick-result [data-experience-target="professional"]').click();
  assert.ok(await page.locator(".advanced-experience__safety").isVisible(), "production advanced safety card is missing");
  await page.locator("#helpBtn").click();
  assert.equal(await page.locator("#helpOverlay .help-evidence-links a").count(), 4);
  await page.locator("#helpOverlay .help-close").click();
  await page.locator("#guideBtn").click();
  assert.equal(await page.locator("#guideOverlay .manual-source-panel a").count(), 4);
  assert.deepEqual(errors, []);
  await context.close();
  console.log(`[PASS] Production runtime loaded ${new Set(localAssets).size} hashed JS/CSS assets with compact Quick results and Advanced safety UI intact.`);
} finally {
  await browser?.close();
  server.kill();
  await Promise.race([
    new Promise((resolvePromise) => server.once("exit", resolvePromise)),
    new Promise((resolvePromise) => setTimeout(resolvePromise, 2_000)),
  ]);
}
