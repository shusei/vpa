import * as FF from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";

const FFMPEG_VER = "0.12.15";
const CORE_VER = "0.12.10";
const CORE_BASE = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm`;
const CORE_JS_URL = `${CORE_BASE}/ffmpeg-core.js`;
const CORE_WORKER_URL = `${CORE_BASE}/ffmpeg-core.worker.js`;
const CORE_WASM_URL = `${CORE_BASE}/ffmpeg-core.wasm`;

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
    if (!allowOrigin) {
      console.warn(
        "[ffmpeg] HEAD ok but ACAO missing; will rely on load() for CORS check",
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error ?? "unknown error");
    throw new Error(`Unable to reach ffmpeg-core.wasm at ${CORE_WASM_URL}: ${message}`);
  }
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

async function getFFmpeg() {
  if (typeof FF.createFFmpeg === "function") {
    return FF.createFFmpeg({
      corePath: CORE_JS_URL,
      wasmPath: CORE_WASM_URL,
      workerPath: CORE_WORKER_URL,
    });
  }

  const instance = new FF.FFmpeg({
    log: false,
  });
  return instance;
}

function isModernInstance(instance) {
  return typeof instance?.exec === "function";
}

async function ensureFFmpegLoaded(statusCallback) {
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
      try {
        await verifyCoreReachable();
      } catch (error) {
        ffmpegLoadingPromise = null;
        throw error;
      }
      const instance = await getFFmpeg();
      attachProgressHandler(instance);

      try {
        if (isModernInstance(instance)) {
          await instance.load({
            coreURL: CORE_JS_URL,
            wasmURL: CORE_WASM_URL,
            workerURL: CORE_WORKER_URL,
          });
        } else if (typeof instance.load === "function") {
          await instance.load();
        }
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
