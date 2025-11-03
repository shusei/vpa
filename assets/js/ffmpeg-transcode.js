import * as FF from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";

const FFMPEG_VER = "0.12.15";
const CORE_VER = "0.12.10";
const LOCAL_VENDOR_BASE = new URL("../vendor/ffmpeg/", import.meta.url).href;

const LOCAL_WORKER_SRC = `${LOCAL_VENDOR_BASE}worker.js`;
const LOCAL_CORE_JS_SRC = `${LOCAL_VENDOR_BASE}ffmpeg-core.js`;
const LOCAL_CORE_WASM_SRC = `${LOCAL_VENDOR_BASE}ffmpeg-core.wasm`;
const CDN_WORKER_SRC = `https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@${FFMPEG_VER}/dist/esm/worker.js`;
const CDN_CORE_JS_SRC = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm/ffmpeg-core.js`;
const CDN_CORE_WASM_SRC = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm/ffmpeg-core.wasm`;

let ffmpegInstance = null;
let ffmpegLoadingPromise = null;
const progressSubscribers = new Set();

function notifyStatus(callback, event) {
  if (typeof callback === "function") {
    callback(event);
  }
}

function subscribeProgress(callback) {
  progressSubscribers.add(callback);
  return () => progressSubscribers.delete(callback);
}

function attachProgressHandler(instance) {
  if (typeof instance?.on === "function") {
    instance.on("progress", (event) => {
      const progress = Math.max(
        0,
        Math.min(1, Number(event?.progress ?? 0) || 0),
      );
      for (const callback of progressSubscribers) {
        try {
          callback(progress);
        } catch (_err) {
          // ignore subscriber errors
        }
      }
    });
    return;
  }

  if (typeof instance?.setProgress === "function") {
    instance.setProgress(({ ratio }) => {
      const progress = Math.max(0, Math.min(1, Number(ratio ?? 0) || 0));
      for (const callback of progressSubscribers) {
        try {
          callback(progress);
        } catch (_err) {
          // ignore subscriber errors
        }
      }
    });
  }
}

async function resolveLocalFFmpegURLs(mirrored) {
  const sources = [
    {
      label: "worker",
      mime: "text/javascript",
      urls: [LOCAL_WORKER_SRC, CDN_WORKER_SRC],
    },
    {
      label: "core",
      mime: "text/javascript",
      urls: [LOCAL_CORE_JS_SRC, CDN_CORE_JS_SRC],
    },
    {
      label: "wasm",
      mime: "application/wasm",
      urls: [LOCAL_CORE_WASM_SRC, CDN_CORE_WASM_SRC],
    },
  ];

  const attempts = [];

  async function fetchWithFallback({ label, mime, urls }) {
    for (const src of urls) {
      try {
        const blobUrl = await toBlobURL(src, mime);
        mirrored?.push(blobUrl);
        return blobUrl;
      } catch (error) {
        const original = error instanceof Error ? error.message : String(error ?? "");
        attempts.push(`- ${label} (${src}): ${original}`);
      }
    }

    throw new Error(`Unable to mirror ffmpeg ${label} asset.`);
  }

  try {
    const [workerURL, coreURL, wasmURL] = await Promise.all(
      sources.map((entry) => fetchWithFallback(entry)),
    );

    const resolved = { workerURL, coreURL, wasmURL };
    console.info("[ffmpeg] using", resolved);
    return resolved;
  } catch (error) {
    const attemptDetails = attempts.length ? `\nTried sources:\n${attempts.join("\n")}` : "";
    const original = error instanceof Error ? error.message : String(error ?? "");
    throw new Error(`FFmpeg assets missing or blocked. ${original}${attemptDetails}`);
  }
}

function getFFmpeg() {
  if (typeof FF.FFmpeg !== "function") {
    throw new Error("@ffmpeg/ffmpeg is missing FFmpeg class constructor.");
  }

  return new FF.FFmpeg({ log: false });
}

function isModernInstance(instance) {
  return typeof instance?.exec === "function";
}

async function ensureFFmpegLoaded(statusCallback) {
  if (typeof location !== "undefined" && location.protocol === "file:") {
    throw new Error(
      "ffmpeg 回退需要透過 HTTP 伺服器載入資產；請用本機伺服器啟動專案（例如：npx serve .）。",
    );
  }

  if (ffmpegInstance) {
    return ffmpegInstance;
  }

  if (!ffmpegLoadingPromise) {
    ffmpegLoadingPromise = (async () => {
      notifyStatus(statusCallback, {
        type: "load-start",
        ffmpegVersion: FFMPEG_VER,
        coreVersion: CORE_VER,
      });
      const mirrored = [];
      const urls = await resolveLocalFFmpegURLs(mirrored);
      const instance = getFFmpeg();
      attachProgressHandler(instance);

      try {
        await instance.load({
          workerURL: urls.workerURL,
          coreURL: urls.coreURL,
          wasmURL: urls.wasmURL,
        });
        setTimeout(() => {
          mirrored.forEach((url) => {
            try {
              URL.revokeObjectURL(url);
            } catch (_err) {
              // ignore revoke failure
            }
          });
        }, 30_000);
      } catch (error) {
        ffmpegLoadingPromise = null;
        mirrored.forEach((url) => {
          try {
            URL.revokeObjectURL(url);
          } catch (_err) {
            // ignore revoke failure
          }
        });
        throw error;
      }
      ffmpegInstance = instance;
      return instance;
    })();
    ffmpegLoadingPromise.catch(() => {
      ffmpegLoadingPromise = null;
    });
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
    if (isModernInstance(ffmpeg)) {
      await ffmpeg.writeFile(inputName, data);
    } else {
      ffmpeg.FS("writeFile", inputName, data);
    }

    notifyStatus(statusCallback, { type: "transcode-start" });
    try {
      const args = [
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
      ];
      if (isModernInstance(ffmpeg)) {
        await ffmpeg.exec(args);
      } else {
        await ffmpeg.run(...args);
      }
    } finally {
      try {
        if (isModernInstance(ffmpeg)) {
          if (typeof ffmpeg.deleteFile === "function") {
            await ffmpeg.deleteFile(inputName);
          }
        } else {
          ffmpeg.FS("unlink", inputName);
        }
      } catch (_err) {
        // ignore clean-up failure
      }
    }

    notifyStatus(statusCallback, { type: "transcode-complete" });
    const rawOutput = isModernInstance(ffmpeg)
      ? await ffmpeg.readFile(outputName)
      : ffmpeg.FS("readFile", outputName);
    try {
      if (isModernInstance(ffmpeg)) {
        if (typeof ffmpeg.deleteFile === "function") {
          await ffmpeg.deleteFile(outputName);
        }
      } else {
        ffmpeg.FS("unlink", outputName);
      }
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

export const transcodeToFloat32 = transcodeToMonoFloat32;
