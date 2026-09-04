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

export async function installSyntheticMicrophone(page, {
  forceMockAudio = false,
  gestureBoundResume = false,
} = {}) {
  await page.addInitScript(({ forceMock, requireGesture }) => {
    let NativeAudioContext = window.AudioContext || window.webkitAudioContext;

    if (requireGesture) {
      window.__vpaTestGestureActive = false;
      window.__vpaTestRequireGestureResume = false;
      document.addEventListener("click", () => {
        window.__vpaTestGestureActive = true;
      }, true);
      document.addEventListener("click", () => {
        window.__vpaTestGestureActive = false;
      });
    }

    if (forceMock || !NativeAudioContext || typeof window.MediaRecorder !== "function") {
      class MockAudioBuffer {
        constructor(length = 16_000, sampleRate = 16_000) {
          this.duration = length / sampleRate;
          this.length = length;
          this.numberOfChannels = 1;
          this.sampleRate = sampleRate;
          this.samples = new Float32Array(length);
          for (let i = 0; i < length; i += 1) {
            this.samples[i] = Math.sin((2 * Math.PI * 220 * i) / sampleRate) * 0.15;
          }
        }

        getChannelData() {
          return this.samples;
        }
      }

      class MockAudioNode extends EventTarget {
        connect() {
          return this;
        }

        disconnect() { }
      }

      class MockScriptProcessor extends MockAudioNode {
        constructor(sampleRate) {
          super();
          this.onaudioprocess = null;
          this.sampleRate = sampleRate;
          this.timer = null;
        }

        connect() {
          if (this.timer === null) {
            this.timer = setInterval(() => {
              const event = new Event("audioprocess");
              Object.defineProperty(event, "inputBuffer", {
                value: new MockAudioBuffer(2048, this.sampleRate),
              });
              this.dispatchEvent(event);
              this.onaudioprocess?.(event);
            }, 25);
          }
          return this;
        }

        disconnect() {
          if (this.timer !== null) clearInterval(this.timer);
          this.timer = null;
        }
      }

      class MockAudioContext {
        constructor() {
          this.destination = {};
          this.sampleRate = 16_000;
          this.state = window.__vpaTestRequireGestureResume ? "suspended" : "running";
        }

        close() {
          this.state = "closed";
          return Promise.resolve();
        }

        createBuffer(_channels, length, sampleRate) {
          return new MockAudioBuffer(length, sampleRate);
        }

        createMediaStreamSource() {
          return new MockAudioNode();
        }

        createScriptProcessor() {
          return new MockScriptProcessor(this.sampleRate);
        }

        decodeAudioData() {
          return Promise.resolve(new MockAudioBuffer());
        }

        resume() {
          if (window.__vpaTestRequireGestureResume && !window.__vpaTestGestureActive) {
            const error = new DOMException(
              "AudioContext.resume() requires a user gesture.",
              "NotAllowedError",
            );
            return Promise.reject(error);
          }
          this.state = "running";
          return Promise.resolve();
        }

        suspend() {
          this.state = "suspended";
          return Promise.resolve();
        }
      }

      class MockMediaStreamTrack extends EventTarget {
        constructor() {
          super();
          this.kind = "audio";
          this.readyState = "live";
        }

        getSettings() {
          return { channelCount: 1, sampleRate: 16_000 };
        }

        stop() {
          this.readyState = "ended";
        }
      }

      class MockMediaStream {
        constructor() {
          this.track = new MockMediaStreamTrack();
        }

        getAudioTracks() {
          return [this.track];
        }

        getTracks() {
          return [this.track];
        }
      }

      class MockMediaRecorder extends EventTarget {
        static isTypeSupported() {
          return true;
        }

        constructor(stream, options = {}) {
          super();
          this.mimeType = options.mimeType || "audio/webm";
          this.ondataavailable = null;
          this.onerror = null;
          this.onstop = null;
          this.state = "inactive";
          this.stream = stream;
        }

        start() {
          this.state = "recording";
        }

        stop() {
          this.state = "inactive";
          queueMicrotask(() => {
            const dataEvent = new Event("dataavailable");
            Object.defineProperty(dataEvent, "data", {
              value: new Blob([new Uint8Array([1, 2, 3, 4])], { type: this.mimeType }),
            });
            this.ondataavailable?.(dataEvent);
            this.onstop?.(new Event("stop"));
          });
        }
      }

      window.AudioContext = MockAudioContext;
      window.webkitAudioContext = MockAudioContext;
      window.MediaRecorder = MockMediaRecorder;
      window.MediaStream = MockMediaStream;
      window.MediaStreamTrack = MockMediaStreamTrack;
      NativeAudioContext = MockAudioContext;

      const mediaState = new WeakMap();
      const mediaPrototype = window.HTMLMediaElement.prototype;
      const nativePaused = Object.getOwnPropertyDescriptor(mediaPrototype, "paused");
      const nativeCurrentTime = Object.getOwnPropertyDescriptor(mediaPrototype, "currentTime");
      Object.defineProperty(mediaPrototype, "paused", {
        configurable: true,
        get() {
          return mediaState.get(this)?.paused ?? nativePaused?.get?.call(this) ?? true;
        },
      });
      Object.defineProperty(mediaPrototype, "currentTime", {
        configurable: true,
        get() {
          return mediaState.get(this)?.currentTime ?? nativeCurrentTime?.get?.call(this) ?? 0;
        },
        set(value) {
          const state = mediaState.get(this) || { paused: true, currentTime: 0 };
          state.currentTime = Number(value) || 0;
          mediaState.set(this, state);
        },
      });
      mediaPrototype.play = function() {
        const state = mediaState.get(this) || { paused: true, currentTime: 0 };
        state.paused = false;
        state.currentTime = Math.max(0.1, state.currentTime);
        mediaState.set(this, state);
        this.dispatchEvent(new Event("play"));
        return Promise.resolve();
      };
      mediaPrototype.pause = function() {
        const state = mediaState.get(this) || { paused: true, currentTime: 0 };
        state.paused = true;
        mediaState.set(this, state);
        this.dispatchEvent(new Event("pause"));
      };
    }

    if (!navigator.mediaDevices) {
      Object.defineProperty(navigator, "mediaDevices", {
        configurable: true,
        value: {},
      });
    }

    let microphoneContext = null;
    navigator.mediaDevices.getUserMedia = async () => {
      if (window.__vpaTestRequireGestureResume) {
        window.__vpaTestGestureActive = false;
      }
      if (typeof window.MediaStream === "function" && window.MediaStream.name === "MockMediaStream") {
        return new window.MediaStream();
      }
      if (!microphoneContext || microphoneContext.state === "closed") {
        microphoneContext = new NativeAudioContext();
      }
      if (microphoneContext.state === "suspended") {
        await microphoneContext.resume();
      }

      const destination = microphoneContext.createMediaStreamDestination();
      const oscillator = microphoneContext.createOscillator();
      const gain = microphoneContext.createGain();
      oscillator.frequency.value = 220;
      gain.gain.value = 0.15;
      oscillator.connect(gain);
      gain.connect(destination);
      oscillator.start();

      const [track] = destination.stream.getAudioTracks();
      const nativeStop = track.stop.bind(track);
      track.stop = () => {
        try { oscillator.stop(); } catch { }
        try { oscillator.disconnect(); } catch { }
        try { gain.disconnect(); } catch { }
        nativeStop();
      };
      return destination.stream;
    };
  }, {
    forceMock: forceMockAudio,
    requireGesture: gestureBoundResume,
  });
}

export async function openProductionPage(page, options) {
  await installDeterministicRuntime(page, options);
  await page.addInitScript(() => {
    localStorage.setItem("vpa::experiment.experience", "professional");
  });
  await page.goto("/");
  await page.evaluate(() => {
    window.VPA_SHARE_SERVICE_ORIGIN = "";
  });
  await expect(page.locator("#playBtn")).toBeAttached();
  await expect(page.locator(".player")).toBeHidden();
}

export async function waitForAnalysis(page, previousAnalysisId = 0, { timeout = 60_000 } = {}) {
  await expect.poll(async () => page.evaluate(() => window.vpaLatestAnalysis?.analysisId || 0), {
    timeout,
  }).toBeGreaterThan(previousAnalysisId);
  return page.evaluate(() => window.vpaLatestAnalysis);
}
