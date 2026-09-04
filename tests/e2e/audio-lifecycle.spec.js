import { test, expect } from "@playwright/test";
import {
  captureRuntimeErrors,
  installDeterministicRuntime,
  installSyntheticMicrophone,
} from "./helpers.js";

// ----- Shared helpers -----

async function openQuickPage(page, path = "/") {
  await installDeterministicRuntime(page);
  await page.addInitScript(() => {
    window.VPA_SHARE_SERVICE_ORIGIN = "";
    window.VPA_PUBLIC_APP_URL = "";
  });
  await page.goto(path);
  try {
    await expect.poll(() => page.evaluate(() => Boolean(
      window.vpaAdvancedExperience && window.vpaExperience,
    ))).toBe(true);
  } catch (error) {
    const initErrors = await page.evaluate(() => window.__vpaTestCounters?.errors || []);
    throw new Error(`${error.message}\nInitialization errors: ${JSON.stringify(initErrors)}`);
  }
}

async function setupLifecycleCounters(page) {
  await page.addInitScript(() => {
    const counters = {
      blobUrlsCreated: 0,
      blobUrlsRevoked: 0,
      activeBlobUrls: new Set(),
      audioPlayResolved: 0,
      audioPlayRejected: 0,
      audioPlayPending: 0,
      audioEventListeners: { play: 0, pause: 0, ended: 0 },
      errors: [],
      // Pitch specific
      audioContextCount: 0,
      audioContextClosed: 0,
      processorCount: 0,
      processorDisconnected: 0,
      processorCallbacks: 0,
      rafStarted: 0,
      rafCancelled: 0,
      mediaTracksEnded: 0,
    };

    // --- Blob URL ---
    const origCreate = URL.createObjectURL.bind(URL);
    const origRevoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = function (obj) {
      const url = origCreate(obj);
      if (obj instanceof Blob && (obj.type.startsWith("audio/") || obj.type === "")) {
        counters.blobUrlsCreated++;
        counters.activeBlobUrls.add(url);
      }
      return url;
    };
    URL.revokeObjectURL = function (url) {
      if (counters.activeBlobUrls.has(url)) {
        counters.blobUrlsRevoked++;
        counters.activeBlobUrls.delete(url);
      }
      return origRevoke(url);
    };

    // --- AudioElement play() ---
    const origPlay = HTMLMediaElement.prototype.play;
    let delayNextPlay = false;
    let releaseDelayedPlay = null;
    window.__delayNextAudioPlay = () => { delayNextPlay = true; };
    window.__releaseDelayedAudioPlay = () => {
      releaseDelayedPlay?.();
      releaseDelayedPlay = null;
    };
    HTMLMediaElement.prototype.play = function () {
      counters.audioPlayPending++;
      const nativePromise = origPlay.call(this);
      const shouldDelay = delayNextPlay;
      delayNextPlay = false;
      const promise = shouldDelay
        ? Promise.resolve(nativePromise).then(() => new Promise((resolve) => {
            releaseDelayedPlay = resolve;
          }))
        : nativePromise;
      if (promise && typeof promise.then === "function") {
        promise.then(
          () => { counters.audioPlayPending--; counters.audioPlayResolved++; },
          () => { counters.audioPlayPending--; counters.audioPlayRejected++; },
        );
      } else {
        counters.audioPlayPending--;
        counters.audioPlayResolved++;
      }
      return promise;
    };

    // --- AudioElement event listener cleanup and once support ---
    const origAddEventListener = HTMLMediaElement.prototype.addEventListener;
    const origRemoveEventListener = HTMLMediaElement.prototype.removeEventListener;
    const trackedEvents = ["play", "pause", "ended"];
    const activeListeners = new Map();

    HTMLMediaElement.prototype.addEventListener = function (type, listener, options) {
      if (this.id === "playback" && trackedEvents.includes(type)) {
        const id = Symbol("listener");
        const entry = { type, listener, active: true };
        activeListeners.set(id, entry);
        counters.audioEventListeners[type]++;

        const isOnce = (typeof options === "object" && options?.once) || options === true;

        let wrappedListener = listener;
        if (isOnce) {
          wrappedListener = function(e) {
            if (entry.active) {
              entry.active = false;
              counters.audioEventListeners[type] = Math.max(0, counters.audioEventListeners[type] - 1);
            }
            if (typeof listener === "function") listener.call(this, e);
            else listener.handleEvent(e);
          };
        }

        const signal = typeof options === "object" ? options.signal : undefined;
        if (signal) {
          signal.addEventListener("abort", () => {
            if (entry.active) {
              entry.active = false;
              counters.audioEventListeners[type] = Math.max(0, counters.audioEventListeners[type] - 1);
            }
          }, { once: true });
        }

        entry.wrapped = wrappedListener;
        return origAddEventListener.call(this, type, wrappedListener, options);
      }
      return origAddEventListener.call(this, type, listener, options);
    };

    HTMLMediaElement.prototype.removeEventListener = function (type, listener, options) {
      if (this.id === "playback" && trackedEvents.includes(type)) {
        for (const [id, entry] of activeListeners.entries()) {
          if (entry.type === type && entry.listener === listener && entry.active) {
            entry.active = false;
            counters.audioEventListeners[type] = Math.max(0, counters.audioEventListeners[type] - 1);
            origRemoveEventListener.call(this, type, entry.wrapped, options);
            activeListeners.delete(id);
            return;
          }
        }
      }
      return origRemoveEventListener.call(this, type, listener, options);
    };

    // --- AudioContext & Pitch Stream ---
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) {
      const origCtx = Ctx;
      function MockCtx(...args) {
        const ctx = new origCtx(...args);
        counters.audioContextCount++;

        const origClose = ctx.close;
        ctx.close = function(...cArgs) {
          counters.audioContextClosed++;
          return origClose.apply(ctx, cArgs);
        };

        const origCreateScriptProcessor = ctx.createScriptProcessor;
        ctx.createScriptProcessor = function(...sArgs) {
          const proc = origCreateScriptProcessor.apply(ctx, sArgs);
          counters.processorCount++;
          origAddEventListener.call(proc, "audioprocess", () => {
            counters.processorCallbacks++;
          });

          const origDisconnect = proc.disconnect;
          proc.disconnect = function(...dArgs) {
            counters.processorDisconnected++;
            return origDisconnect.apply(proc, dArgs);
          };

          return proc;
        };

        return ctx;
      }
      MockCtx.prototype = origCtx.prototype;
      window.AudioContext = MockCtx;
      window.webkitAudioContext = MockCtx;
    }

    const origRAF = window.requestAnimationFrame;
    const origCAF = window.cancelAnimationFrame;
    window.requestAnimationFrame = function(cb) {
      counters.rafStarted++;
      return origRAF(cb);
    };
    window.cancelAnimationFrame = function(id) {
      counters.rafCancelled++;
      return origCAF(id);
    };

    const origStop = MediaStreamTrack.prototype.stop;
    MediaStreamTrack.prototype.stop = function() {
      counters.mediaTracksEnded++;
      return origStop.call(this);
    };

    // --- Error capture ---
    window.addEventListener("error", (e) => counters.errors.push(`error: ${e.message}`));
    window.addEventListener("unhandledrejection", (e) => counters.errors.push(`rejection: ${e.reason}`));

    window.__vpaTestCounters = counters;
  });
}

function getCounters(page) {
  return page.evaluate(() => {
    const c = window.__vpaTestCounters;
    return {
      blobUrlsCreated: c.blobUrlsCreated,
      blobUrlsRevoked: c.blobUrlsRevoked,
      activeBlobUrlCount: c.activeBlobUrls.size,
      audioPlayResolved: c.audioPlayResolved,
      audioPlayRejected: c.audioPlayRejected,
      audioPlayPending: c.audioPlayPending,
      audioEventListeners: { ...c.audioEventListeners },
      errors: [...c.errors],
      audioContextCount: c.audioContextCount,
      audioContextClosed: c.audioContextClosed,
      processorCount: c.processorCount,
      processorDisconnected: c.processorDisconnected,
      processorCallbacks: c.processorCallbacks,
      rafStarted: c.rafStarted,
      rafCancelled: c.rafCancelled,
      mediaTracksEnded: c.mediaTracksEnded,
    };
  });
}

async function getCanvasChecksum(page) {
  return page.evaluate(() => {
    const canvas = document.getElementById("pitchCanvas");
    if (!canvas) return 0;
    if (canvas.width === 0 || canvas.height === 0) return 0;
    const ctx = canvas.getContext("2d");
    if (!ctx) return 0;
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let hash = 2166136261;
    for (let i = 0; i < imgData.data.length; i += 4) {
      hash = Math.imul(hash ^ imgData.data[i], 16777619);
      hash = Math.imul(hash ^ imgData.data[i + 1], 16777619);
      hash = Math.imul(hash ^ imgData.data[i + 2], 16777619);
      hash = Math.imul(hash ^ imgData.data[i + 3], 16777619);
    }
    return hash >>> 0;
  });
}

async function getCanvasColorCount(page) {
  return page.evaluate(() => {
    const canvas = document.getElementById("pitchCanvas");
    if (!canvas || canvas.width === 0 || canvas.height === 0) return 0;
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    const pixelCount = canvas.width * canvas.height;
    const step = Math.max(1, Math.floor(pixelCount / 3600));
    const colors = new Set();
    for (let pixel = 0; pixel < pixelCount; pixel += step) {
      const offset = pixel * 4;
      colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]},${pixels[offset + 3]}`);
    }
    return colors.size;
  });
}

// ----- Test A: Quick → Professional realtime pitch -----
test.describe("Audio Lifecycle", () => {
  test("@cross-browser A: Quick → Professional pitch stream works without reload", async ({ page }) => {
    await installSyntheticMicrophone(page);
    await setupLifecycleCounters(page);
    const errors = captureRuntimeErrors(page);

    await openQuickPage(page);
    await expect(page.locator("#audioDebugDownload")).toBeHidden();
    expect(await page.evaluate(() => typeof window.vpaAudioDebug)).toBe("undefined");

    // 1. Quick 第一輪真的收到 audioprocess callbacks
    let counters = await getCounters(page);
    const initCallbacks = counters.processorCallbacks;

    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='recording']")).toBeVisible({ timeout: 30000 });

    // Wait for at least some callbacks
    await expect.poll(
      async () => (await getCounters(page)).processorCallbacks,
      { timeout: 10000 },
    ).toBeGreaterThan(initCallbacks + 5);

    // Stop recording
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='result']")).toBeVisible({ timeout: 90000 });

    // 2. Quick 停止後，第一輪 context/processor 已結束
    counters = await getCounters(page);
    const ctxRound1 = counters.audioContextCount;
    expect(ctxRound1).toBeGreaterThanOrEqual(1);
    expect(counters.audioContextClosed).toBe(ctxRound1 - 1);
    const procRound1 = counters.processorCount;
    expect(procRound1).toBeGreaterThanOrEqual(1);
    expect(counters.processorDisconnected).toBe(procRound1);

    // 3. 切換 Professional 不 reload
    await page.locator("[data-experience-target='professional']").first().click();
    await expect(page.locator("html[data-experience='professional']")).toBeAttached({ timeout: 5000 });

    const recordBtn = page.locator("#recordBtn");
    await expect(recordBtn).toBeVisible({ timeout: 10000 });
    await expect(recordBtn).toBeEnabled({ timeout: 5000 });

    // 4. Professional 建立的是第二個獨立 session
    counters = await getCounters(page);
    const callbacksBeforePro = counters.processorCallbacks;
    await recordBtn.click();
    await expect(page.locator("body.recording")).toBeAttached({ timeout: 10000 });
    await expect(page.locator("#pitchWrap")).toBeVisible();

    // 5. 第二輪 audioprocess callback count 明確增加
    await expect.poll(
      async () => (await getCounters(page)).processorCallbacks,
      { timeout: 10000 },
    ).toBeGreaterThan(callbacksBeforePro + 5);

    // 6. pitchNow 必須符合數值 Hz 格式
    const pitchText = await page.locator("#pitchNow").textContent();
    expect(pitchText).toMatch(/^\d+(\.\d+)?\s*Hz$/);

    // 7. pitchCanvas.width > 0、height > 0
    const dim = await page.evaluate(() => {
      const c = document.getElementById("pitchCanvas");
      const wrap = document.getElementById("pitchWrap");
      const dropZone = document.getElementById("dropZone");
      const style = getComputedStyle(wrap);
      const dropStyle = getComputedStyle(dropZone);
      const dropRect = dropZone.getBoundingClientRect();
      const rect = c.getBoundingClientRect();
      return {
        display: style.display,
        dropDisplay: dropStyle.display,
        dropHeight: dropRect.height,
        dropVisibility: dropStyle.visibility,
        dropWidth: dropRect.width,
        hidden: wrap.hasAttribute("hidden"),
        h: c.height,
        rectHeight: rect.height,
        rectWidth: rect.width,
        visibility: style.visibility,
        w: c.width,
      };
    });
    expect(dim.hidden).toBe(false);
    expect(dim.display).not.toBe("none");
    expect(dim.visibility).not.toBe("hidden");
    expect(dim.dropDisplay).not.toBe("none");
    expect(dim.dropVisibility).not.toBe("hidden");
    expect(dim.dropWidth).toBeGreaterThan(0);
    expect(dim.dropHeight).toBeGreaterThan(0);
    expect(dim.w).toBeGreaterThan(0);
    expect(dim.h).toBeGreaterThan(0);
    expect(dim.rectWidth).toBeGreaterThan(0);
    expect(dim.rectHeight).toBeGreaterThan(0);

    // 8. 第二輪錄音期間 canvas pixel checksum 確實改變
    const sumDuringPro = await getCanvasChecksum(page);
    await page.waitForTimeout(100);
    expect(await getCanvasChecksum(page)).not.toBe(sumDuringPro);

    await recordBtn.click();
    await expect.poll(
      async () => page.evaluate(() => window.vpaLatestAnalysis?.analysisId || 0),
      { timeout: 90000 },
    ).toBeGreaterThan(0);

    // 9. 停止後第二輪 closed, disconnected, RAF 停止, MediaStreamTrack ended
    counters = await getCounters(page);
    await expect.poll(
      async () => {
        const c = await getCounters(page);
        return c.audioContextClosed >= c.audioContextCount - 1 && c.processorDisconnected >= c.processorCount;
      },
      { timeout: 10000 }
    ).toBe(true);

    counters = await getCounters(page);
    const ctxRound2 = counters.audioContextCount;
    expect(ctxRound2 - counters.audioContextClosed).toBe(1);
    const procRound2 = counters.processorCount;
    expect(procRound2).toBeGreaterThanOrEqual(procRound1 + 1);
    expect(counters.processorDisconnected).toBe(procRound2);
    // Raf cancelled count should increase when stopped
    expect(counters.rafCancelled).toBeGreaterThan(0);
    expect(counters.mediaTracksEnded).toBeGreaterThanOrEqual(2);

    // 10. Errors empty
    const relevantErrors = errors.filter(
      (e) => !e.includes("net::") && !e.includes("favicon") && !e.includes("model-preload"),
    );
    expect(relevantErrors).toEqual([]);
    expect(counters.errors).toEqual([]);
  });

  test("@cross-browser Quick → Professional resumes reusable pitch context inside the user gesture", async ({ page }) => {
    await installSyntheticMicrophone(page, {
      forceMockAudio: true,
      gestureBoundResume: true,
    });
    await setupLifecycleCounters(page);
    const errors = captureRuntimeErrors(page);

    await openQuickPage(page);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='recording']")).toBeVisible();
    await page.waitForTimeout(250);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='result']")).toBeVisible({ timeout: 90000 });

    await page.evaluate(() => {
      window.__vpaTestRequireGestureResume = true;
    });
    await page.locator("[data-experience-target='professional']").first().click();
    await expect(page.locator("html[data-experience='professional']")).toBeAttached();

    const callbacksBefore = (await getCounters(page)).processorCallbacks;
    const recordBtn = page.locator("#recordBtn");
    await recordBtn.click();
    await expect(page.locator("body.recording")).toBeAttached();
    await expect(page.locator("#pitchWrap")).toBeVisible();

    const panel = await page.locator("#pitchWrap").evaluate((element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      const canvas = element.querySelector("#pitchCanvas");
      const canvasRect = canvas.getBoundingClientRect();
      return {
        canvasHeight: canvasRect.height,
        canvasWidth: canvasRect.width,
        display: style.display,
        height: rect.height,
        hidden: element.hasAttribute("hidden"),
        visibility: style.visibility,
        width: rect.width,
      };
    });
    expect(panel.hidden).toBe(false);
    expect(panel.display).not.toBe("none");
    expect(panel.visibility).not.toBe("hidden");
    expect(panel.width).toBeGreaterThan(0);
    expect(panel.height).toBeGreaterThan(0);
    expect(panel.canvasWidth).toBeGreaterThan(0);
    expect(panel.canvasHeight).toBeGreaterThan(0);

    await expect.poll(
      async () => (await getCounters(page)).processorCallbacks,
      { timeout: 10000 },
    ).toBeGreaterThan(callbacksBefore + 3);
    await expect(page.locator("#pitchNow")).toHaveText(/^\d+(?:\.\d+)?Hz$/);
    const firstChecksum = await getCanvasChecksum(page);
    await page.waitForTimeout(100);
    expect(await getCanvasChecksum(page)).not.toBe(firstChecksum);

    await recordBtn.click();
    await expect(page.locator("#pitchWrap")).toBeHidden();
    await expect.poll(
      async () => (await getCounters(page)).processorDisconnected,
      { timeout: 10000 },
    ).toBe((await getCounters(page)).processorCount);

    const relevantErrors = errors.filter(
      (error) => !error.includes("net::") && !error.includes("favicon") && !error.includes("model-preload"),
    );
    expect(relevantErrors).toEqual([]);
    expect((await getCounters(page)).errors).toEqual([]);
  });

  test("audio diagnostics stay opt-in, bounded, and export metadata without audio", async ({ page }) => {
    await installSyntheticMicrophone(page);
    await setupLifecycleCounters(page);
    await openQuickPage(page, "/?vpaAudioDebug=1");

    await expect(page.locator("#audioDebugDownload")).toBeVisible();
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='recording']")).toBeVisible();
    await expect.poll(
      async () => page.evaluate(() => window.vpaAudioDebug.getReport().events.some(
        (event) => event.type === "pitch.processor.callback",
      )),
    ).toBe(true);
    await page.waitForTimeout(500);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='result']")).toBeVisible({ timeout: 90000 });

    const report = await page.evaluate(() => window.vpaAudioDebug.getReport());
    expect(report.schemaVersion).toBe(1);
    expect(report.events.length).toBeLessThanOrEqual(600);
    expect(report.events.some((event) => event.type === "recording.session.begin")).toBe(true);
    expect(report.events.some((event) => event.type === "recording.microphone.ready")).toBe(true);
    expect(report.events.some((event) => event.type === "pitch.context.suspend.after")).toBe(true);
    expect(JSON.stringify(report)).not.toContain("audioSamples");
    expect(JSON.stringify(report)).not.toContain("deviceId");

    const downloadPromise = page.waitForEvent("download");
    await page.locator("#audioDebugDownload").click();
    await downloadPromise;
  });

  test("mobile Professional pitch canvas remains readable in light and dark themes without overflow", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await installSyntheticMicrophone(page);
    const errors = captureRuntimeErrors(page);
    await openQuickPage(page);

    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    await page.locator("[data-experience-target='professional']").first().click();
    const recordBtn = page.locator("#recordBtn");
    await recordBtn.click();
    await expect(page.locator("#pitchWrap")).toBeVisible();
    await expect(page.locator("#pitchNow")).toHaveText(/^\d+(?:\.\d+)?Hz$/);

    await page.locator('.theme-item[data-theme="day"]').evaluate((button) => button.click());
    await expect(page.locator("html[data-faction='light']")).toBeAttached();
    await page.waitForTimeout(100);
    const lightChecksum = await getCanvasChecksum(page);
    expect(await getCanvasColorCount(page)).toBeGreaterThan(4);

    await page.locator('.theme-item[data-theme="night"]').evaluate((button) => button.click());
    await expect(page.locator("html[data-faction='dark']")).toBeAttached();
    await page.waitForTimeout(100);
    const darkChecksum = await getCanvasChecksum(page);
    expect(await getCanvasColorCount(page)).toBeGreaterThan(4);
    expect(darkChecksum).not.toBe(lightChecksum);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);

    await recordBtn.click();
    await expect(page.locator("#pitchWrap")).toBeHidden();
    const relevantErrors = errors.filter(
      (error) => !error.includes("net::") && !error.includes("favicon") && !error.includes("model-preload"),
    );
    expect(relevantErrors).toEqual([]);
  });

  test("@cross-browser Professional 30-round record/play soak keeps realtime pitch visible", async ({ page }) => {
    test.setTimeout(600_000);
    await installSyntheticMicrophone(page);
    await setupLifecycleCounters(page);
    const errors = captureRuntimeErrors(page);

    await openQuickPage(page);
    await page.locator("[data-experience-target='professional']").first().click();
    await expect(page.locator("html[data-experience='professional']")).toBeAttached();

    const recordBtn = page.locator("#recordBtn");
    const playBtn = page.locator("#playBtn");
    const heapSamples = [];
    let previousAnalysisId = 0;
    let previousCallbacks = 0;

    for (let round = 1; round <= 30; round += 1) {
      await recordBtn.click();
      await expect(page.locator("body.recording")).toBeAttached();
      await expect(page.locator("#pitchWrap")).toBeVisible();

      await expect.poll(
        async () => (await getCounters(page)).processorCallbacks,
        { timeout: 10000 },
      ).toBeGreaterThan(previousCallbacks + 3);
      previousCallbacks = (await getCounters(page)).processorCallbacks;

      const dimensions = await page.locator("#pitchCanvas").evaluate((canvas) => ({
        height: canvas.height,
        width: canvas.width,
      }));
      expect(dimensions.width).toBeGreaterThan(0);
      expect(dimensions.height).toBeGreaterThan(0);
      await expect(page.locator("#pitchNow")).toHaveText(/^\d+(?:\.\d+)?Hz$/);

      await recordBtn.click();
      await expect.poll(
        async () => page.evaluate(() => window.vpaLatestAnalysis?.analysisId || 0),
        { timeout: 90000 },
      ).toBeGreaterThan(previousAnalysisId);
      previousAnalysisId = await page.evaluate(() => window.vpaLatestAnalysis.analysisId);

      await expect(playBtn).toBeVisible();
      await expect(playBtn).toBeEnabled();
      await playBtn.click();
      await expect.poll(
        async () => page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0),
        { timeout: 5000 },
      ).toBeGreaterThan(0);
      await playBtn.click();
      await expect.poll(
        async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
        { timeout: 5000 },
      ).toBe(true);

      const heapSize = await page.evaluate(() => performance.memory?.usedJSHeapSize || null);
      if (Number.isFinite(heapSize)) heapSamples.push(heapSize);
    }

    const counters = await getCounters(page);
    expect(counters.audioContextCount - counters.audioContextClosed).toBe(1);
    expect(counters.processorCount).toBe(30);
    expect(counters.processorDisconnected).toBe(counters.processorCount);
    if (heapSamples.length > 1) {
      expect(Math.max(...heapSamples) - heapSamples[0]).toBeLessThan(128 * 1024 * 1024);
    }
    const relevantErrors = errors.filter(
      (error) => !error.includes("net::") && !error.includes("favicon") && !error.includes("model-preload"),
    );
    expect(relevantErrors).toEqual([]);
    expect(counters.errors).toEqual([]);
  });

  // ----- Test B: 15-round record/play stress test -----
  test("B: 15-round record/play stress test", async ({ page }) => {
    test.setTimeout(600_000);
    await setupLifecycleCounters(page);
    const errors = captureRuntimeErrors(page);

    await openQuickPage(page);
    const ROUNDS = 15;

    // We can use a short recorded audio to speed up tests by clicking stop early.
    for (let round = 1; round <= ROUNDS; round++) {
      // Record
      const isRecording = await page.locator("[data-quick-stage='recording']").isVisible().catch(() => false);
      if (!isRecording) {
        await page.locator("[data-quick-record]").click();
        await expect(page.locator("[data-quick-stage='recording']")).toBeVisible({ timeout: 30000 });
      }

      await page.waitForTimeout(500); // 0.5s recording is enough to produce some data
      await page.locator("[data-quick-record]").click();
      await expect(page.locator("[data-quick-stage='result']")).toBeVisible({ timeout: 90000 });

      // 1. expect(replayBtn).toBeVisible()
      const replayBtn = page.locator("[data-quick-replay]");
      await expect(replayBtn).toBeVisible();

      // 2. 點擊 play
      await replayBtn.click();

      // 3. expect.poll(audio.paused).toBe(false)
      await expect.poll(
        async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
        { timeout: 5000 }
      ).toBe(false);

      // 4. expect(audio.error).toBeNull()
      // 5. expect(audio.readyState).toBeGreaterThanOrEqual(2)
      const audioInfo = await page.evaluate(() => {
        const a = document.querySelector("audio#playback");
        return {
          err: a.error ? a.error.code : null,
          rs: a.readyState,
          src: a.src
        };
      });
      expect(audioInfo.err).toBeNull();
      expect(audioInfo.rs).toBeGreaterThanOrEqual(2);

      // 6. poll 至 currentTime > 0 (表示有在播放)
      await expect.poll(
        async () => page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0),
        { timeout: 5000 }
      ).toBeGreaterThan(0);

      // 7. play() pending count 必須在時限內回到 0
      await expect.poll(
        async () => (await getCounters(page)).audioPlayPending,
        { timeout: 5000 }
      ).toBe(0);

      if (round % 2 === 0) {
        // 8. 偶數輪 pause
        await replayBtn.click(); // Pause
        await expect.poll(
          async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
          { timeout: 5000 }
        ).toBe(true);

        const tPaused = await page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0);

        // 再 play
        await replayBtn.click();
        await expect.poll(
          async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
          { timeout: 5000 }
        ).toBe(false);

        // 由接近原位置繼續，不得回到 0
        const tResumed = await page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0);
        expect(tResumed).toBeGreaterThanOrEqual(tPaused);
      } else {
        // 9. 奇數輪等待 ended
        await page.evaluate(() => new Promise((resolve, reject) => {
          const a = document.querySelector("audio#playback");
          if (!a || a.ended) { resolve(); return; }
          const timer = setTimeout(() => reject(new Error("Timeout waiting for ended")), 10000);
          a.addEventListener("ended", () => {
            clearTimeout(timer);
            resolve();
          }, { once: true });
        }));
      }

      // Check counters per round BEFORE clicking retry
      const c = await getCounters(page);

      // 11/12. 每輪結束 URL 控制 (因 test B 還在 loop，目前應有 current URL 活著)
      // Revoked counts should roughly match created counts - 1 (since one is active for recording state)
      // We will strictly assert this at the very end.

      // 13. 每輪結束不得有 pending play promise
      expect(c.audioPlayPending).toBe(0);
      // Decode contexts close; the one reusable realtime context remains suspended.
      expect(c.audioContextClosed).toBe(c.audioContextCount - 1);
      expect(c.processorDisconnected).toBe(c.processorCount);

      // Listener counts stable
      expect(c.audioEventListeners.play).toBeLessThanOrEqual(1);
      expect(c.audioEventListeners.pause).toBeLessThanOrEqual(1);
      expect(c.audioEventListeners.ended).toBeLessThanOrEqual(1);

      if (round < ROUNDS) {
        // 10. retry 按鈕必須存在，點擊後必須真的進入下一輪 recording
        const retryBtn = page.locator("[data-quick-retry]");
        await expect(retryBtn).toBeVisible();
        await retryBtn.click();
        await expect(page.locator("[data-quick-stage='recording']")).toBeVisible({ timeout: 30000 });
      }
    }

    // After loop, the final recording is already stopped (stop clicked at top of loop, retry not clicked at end)
    const finalC = await getCounters(page);
    expect(finalC.audioPlayPending).toBe(0);

    // URL instrumentation check
    // 每次新 playback source 建立一個新 URL。
    // 被替換的舊 URL恰好 revoke 一次。
    // 測試結束時允許目前播放器持有一個 active URL，其餘全部已撤銷。
    expect(finalC.blobUrlsCreated).toBe(ROUNDS); // 15 rounds
    expect(finalC.activeBlobUrlCount).toBe(1);
    expect(finalC.blobUrlsRevoked).toBe(finalC.blobUrlsCreated - 1);

    const relevantErrors = errors.filter(
      (e) => !e.includes("net::") && !e.includes("favicon") && !e.includes("model-preload"),
    );
    expect(relevantErrors).toEqual([]);
    expect(finalC.errors).toEqual([]);
  });

  // ----- Test C: 播放中替換來源 & Focused Race Test -----
  test("C: setPlaybackSource during active playback is safe + Race Test", async ({ page }) => {
    await setupLifecycleCounters(page);
    const errors = captureRuntimeErrors(page);
    await openQuickPage(page);

    // 1. 第一個 replay 按鈕存在
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='recording']")).toBeVisible();
    await page.waitForTimeout(500);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='result']")).toBeVisible();

    const replayBtn = page.locator("[data-quick-replay]");
    await expect(replayBtn).toBeVisible();

    // 2. 第一段音訊真的開始播放，currentTime 真的前進
    await replayBtn.click();
    await expect.poll(
      async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
      { timeout: 5000 }
    ).toBe(false);

    await expect.poll(
      async () => page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0),
      { timeout: 5000 }
    ).toBeGreaterThan(0);

    let counters = await getCounters(page);
    const firstUrlCount = counters.blobUrlsCreated;
    expect(counters.activeBlobUrlCount).toBe(1);

    // 3. 播放中開始新錄音
    const retryBtn = page.locator("[data-quick-retry]");
    await expect(retryBtn).toBeVisible();
    await retryBtn.click();
    await expect(page.locator("[data-quick-stage='recording']")).toBeVisible();

    // 4. 開始新錄音
    await page.waitForTimeout(500);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='result']")).toBeVisible();

    // 5. 第一段來源被安全卸載並 revoke，產生不同的新 URL
    await expect.poll(
      async () => (await getCounters(page)).blobUrlsRevoked,
      { timeout: 5000 }
    ).toBe(firstUrlCount);

    counters = await getCounters(page);
    expect(counters.blobUrlsCreated).toBe(firstUrlCount + 1);
    expect(counters.activeBlobUrlCount).toBe(1);

    // 6. 第二段音訊可播放，currentTime 確實前進
    await replayBtn.click();
    await expect.poll(
      async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
      { timeout: 5000 }
    ).toBe(false);

    const t2 = await page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0);
    await expect.poll(
      async () => page.evaluate(() => document.querySelector("audio#playback")?.currentTime || 0),
      { timeout: 5000 }
    ).toBeGreaterThan(t2 + 0.05);

    // 7. 第二段播放完成後不得留下 pending play promise
    counters = await getCounters(page);
    expect(counters.audioPlayPending).toBe(0);

    // --- Focused race test ---
    // pause -> delayed play -> retry/stop -> set new source -> resolve old play promise
    await replayBtn.click();
    await expect.poll(
      async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
      { timeout: 5000 },
    ).toBe(true);

    await page.evaluate(() => window.__delayNextAudioPlay());
    await replayBtn.click();
    await expect.poll(
      async () => (await getCounters(page)).audioPlayPending,
      { timeout: 5000 },
    ).toBe(1);

    await retryBtn.click();
    await expect(page.locator("[data-quick-stage='recording']")).toBeVisible();
    await page.evaluate(() => window.__releaseDelayedAudioPlay());
    await expect.poll(
      async () => (await getCounters(page)).audioPlayPending,
      { timeout: 5000 },
    ).toBe(0);

    // Stop recording
    await page.waitForTimeout(500);
    await page.locator("[data-quick-record]").click();
    await expect(page.locator("[data-quick-stage='result']")).toBeVisible();
    await expect(page.locator("[data-quick-replay] .quick-result__refine-icon")).toHaveText("▶");
    await expect.poll(
      async () => page.evaluate(() => document.querySelector("audio#playback")?.paused),
      { timeout: 5000 },
    ).toBe(true);

    // 8. 無 media error、pending promise 或 runtime error
    counters = await getCounters(page);
    expect(counters.audioPlayPending).toBe(0);
    const audioError = await page.evaluate(() => document.querySelector("audio#playback")?.error);
    expect(audioError).toBeNull();

    const relevantErrors = errors.filter(
      (e) => !e.includes("net::") && !e.includes("favicon") && !e.includes("model-preload"),
    );
    expect(relevantErrors).toEqual([]);
    expect(counters.errors).toEqual([]);
  });
});
