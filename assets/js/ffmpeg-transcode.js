import { FFmpeg } from "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/ffmpeg.min.js";
import { fetchFile } from "https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.10/dist/umd/index.js";

const CORE_BASE_URL = "https://cdn.jsdelivr.net/npm/@ffmpeg/core-mt@0.12.10/dist/umd";
const CORE_JS_URL = `${CORE_BASE_URL}/ffmpeg-core.js`;
const CORE_WORKER_URL = `${CORE_BASE_URL}/ffmpeg-core.worker.js`;
const CORE_WASM_URL = `${CORE_BASE_URL}/ffmpeg-core.wasm`;

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
      const instance = new FFmpeg();
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
        await instance.load({ coreURL: CORE_JS_URL, wasmURL: CORE_WASM_URL, workerURL: CORE_WORKER_URL });
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
    await ffmpeg.writeFile(inputName, data);

    notifyStatus(statusCallback, { type: "transcode-start" });
    try {
      await ffmpeg.exec([
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
      ]);
    } finally {
      try {
        await ffmpeg.deleteFile(inputName);
      } catch (_err) {
        // ignore clean-up failure
      }
    }

    notifyStatus(statusCallback, { type: "transcode-complete" });
    const rawOutput = await ffmpeg.readFile(outputName);
    try {
      await ffmpeg.deleteFile(outputName);
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
