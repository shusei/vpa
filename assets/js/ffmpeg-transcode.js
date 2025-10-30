import { fetchFile } from "https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.15/dist/umd/index.js";

const FFMPEG_SCRIPT_URL = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/umd/ffmpeg.min.js";
const FFMPEG_MODULE_URL = `${FFMPEG_SCRIPT_URL}?module`;

const CORE_BASE_URL = "https://cdn.jsdelivr.net/npm/@ffmpeg/core-mt@0.12.15/dist/umd";
const CORE_JS_URL = `${CORE_BASE_URL}/ffmpeg-core.js`;
const CORE_WORKER_URL = `${CORE_BASE_URL}/ffmpeg-core.worker.js`;
const CORE_WASM_URL = `${CORE_BASE_URL}/ffmpeg-core.wasm`;

let ffmpegInstance = null;
let ffmpegLoadingPromise = null;
let ffmpegModulePromise = null;
const progressSubscribers = new Set();

async function loadFFmpegModule() {
  if (!ffmpegModulePromise) {
    ffmpegModulePromise = (async () => {
      const namespace = await (async () => {
        const globalNamespace = globalThis?.FFmpeg;
        if (globalNamespace && typeof globalNamespace.createFFmpeg === "function") {
          return globalNamespace;
        }

        if (typeof document !== "undefined" && document) {
          await new Promise((resolve, reject) => {
            const existingScript = Array.from(document.getElementsByTagName("script")).find(
              (script) => script.src === FFMPEG_SCRIPT_URL,
            );

            if (existingScript) {
              if (existingScript.getAttribute("data-ffmpeg-loaded") === "true") {
                resolve();
                return;
              }
              existingScript.addEventListener(
                "load",
                () => {
                  existingScript.setAttribute("data-ffmpeg-loaded", "true");
                  resolve();
                },
                { once: true },
              );
              existingScript.addEventListener(
                "error",
                () => reject(new Error(`Failed to load FFmpeg script at ${FFMPEG_SCRIPT_URL}`)),
                { once: true },
              );
              return;
            }

            const script = document.createElement("script");
            script.src = FFMPEG_SCRIPT_URL;
            script.async = true;
            script.onload = () => {
              script.setAttribute("data-ffmpeg-loaded", "true");
              resolve();
            };
            script.onerror = () => {
              reject(new Error(`Failed to load FFmpeg script at ${FFMPEG_SCRIPT_URL}`));
            };
            (document.head || document.body || document.documentElement).appendChild(script);
          });

          const loadedNamespace = globalThis?.FFmpeg;
          if (loadedNamespace && typeof loadedNamespace.createFFmpeg === "function") {
            return loadedNamespace;
          }
          throw new Error("FFmpeg UMD module did not expose the expected API");
        }

        try {
          const module = await import(/* webpackIgnore: true */ FFMPEG_MODULE_URL);
          const candidate = module?.default ?? module;
          if (candidate && typeof candidate.createFFmpeg === "function") {
            return candidate;
          }
        } catch (error) {
          throw new Error(
            `Failed to dynamically import FFmpeg module from ${FFMPEG_MODULE_URL}: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
        throw new Error("FFmpeg module loading failed");
      })();

      return namespace;
    })();
  }

  return ffmpegModulePromise;
}

function notifyStatus(callback, event) {
  if (typeof callback === "function") {
    callback(event);
  }
}

function subscribeProgress(callback) {
  progressSubscribers.add(callback);
  return () => progressSubscribers.delete(callback);
}

async function verifyCoreReachable() {
  if (typeof fetch !== "function") {
    return;
  }

  try {
    const response = await fetch(CORE_WASM_URL, { method: "HEAD", cache: "no-store" });
    if (!response || !response.ok) {
      const status = response ? `${response.status} ${response.statusText}` : "unknown";
      throw new Error(`HEAD request failed: ${status}`);
    }

    const allowOrigin = response.headers.get("access-control-allow-origin");
    if (!allowOrigin || allowOrigin.trim() === "") {
      throw new Error("Access-Control-Allow-Origin header missing");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error ?? "unknown error");
    throw new Error(`Unable to reach ffmpeg-core.wasm at ${CORE_WASM_URL}: ${message}`);
  }
}

async function ensureFFmpegLoaded(statusCallback) {
  if (ffmpegInstance) {
    return ffmpegInstance;
  }

  if (!ffmpegLoadingPromise) {
    ffmpegLoadingPromise = (async () => {
      notifyStatus(statusCallback, { type: "load-start" });
      try {
        await verifyCoreReachable();
      } catch (error) {
        ffmpegLoadingPromise = null;
        throw error;
      }
      const module = await loadFFmpegModule();
      const { createFFmpeg } = module;
      if (typeof createFFmpeg !== "function") {
        ffmpegLoadingPromise = null;
        throw new Error("FFmpeg module did not provide createFFmpeg");
      }
      const instance = createFFmpeg({
        corePath: CORE_JS_URL,
        wasmPath: CORE_WASM_URL,
        workerPath: CORE_WORKER_URL,
      });
      instance.on("progress", (event) => {
        if (!event || typeof event.progress !== "number") return;
        for (const callback of progressSubscribers) {
          try {
            callback(event.progress);
          } catch (_err) {
            // ignore subscriber errors
          }
        }
      });

      try {
        await instance.load();
      } catch (error) {
        ffmpegLoadingPromise = null;
        throw error;
      }
      ffmpegInstance = instance;
      return instance;
    })();
  }

  return ffmpegLoadingPromise;
}

export async function transcodeToMonoFloat32(input, targetSampleRate, statusCallback) {
  const ffmpeg = await ensureFFmpegLoaded(statusCallback);
  const unsubscribe = subscribeProgress((progress) => {
    notifyStatus(statusCallback, { type: "transcode-progress", progress });
  });

  try {
    const inputName = `input-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const outputName = `output-${Date.now()}-${Math.random().toString(36).slice(2)}.f32`;
    const data = await fetchFile(input);
    ffmpeg.FS("writeFile", inputName, data);

    notifyStatus(statusCallback, { type: "transcode-start" });
    try {
      await ffmpeg.run(
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inputName,
        "-vn",
        "-ac",
        "1",
        "-ar",
        `${targetSampleRate}`,
        "-f",
        "f32le",
        outputName,
      );
    } finally {
      try {
        ffmpeg.FS("unlink", inputName);
      } catch (_err) {
        // ignore clean-up failure
      }
    }

    notifyStatus(statusCallback, { type: "transcode-complete" });
    const rawOutput = ffmpeg.FS("readFile", outputName);
    try {
      ffmpeg.FS("unlink", outputName);
    } catch (_err) {
      // ignore clean-up failure
    }

    const bytes = rawOutput instanceof Uint8Array ? rawOutput : new Uint8Array(rawOutput);
    if (bytes.byteLength % 4 !== 0) {
      throw new Error(`Unexpected ffmpeg output length: ${bytes.byteLength}`);
    }
    const frameCount = bytes.byteLength / 4;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const float32 = new Float32Array(frameCount);
    for (let i = 0; i < frameCount; i += 1) {
      float32[i] = view.getFloat32(i * 4, true);
    }
    return { float32, sr: targetSampleRate, durationSec: frameCount / targetSampleRate };
  } finally {
    unsubscribe();
  }
}
