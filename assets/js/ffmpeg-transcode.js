import * as FF from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";

const FFMPEG_VER = "0.12.15";
const CORE_VER = "0.12.15";
const CORE_BASE_OLD = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist/esm`;
const CORE_BASE_NEW = `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${CORE_VER}/dist`;
const FFMPEG_BASE = `https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@${FFMPEG_VER}/dist/esm`;

// fallback: core script file may be .mjs instead of .js
const CORE_SCRIPT_CANDIDATES = [
  `${CORE_BASE_NEW}/ffmpeg-core.js`,
  `${CORE_BASE_NEW}/ffmpeg-core.js?module`,
  `${CORE_BASE_NEW}/ffmpeg-core.mjs`,
  `${CORE_BASE_NEW}/ffmpeg-core.mjs?module`,
  `${CORE_BASE_NEW}/js/ffmpeg-core.js`,
  `${CORE_BASE_NEW}/js/ffmpeg-core.mjs`,
  `${CORE_BASE_NEW}/esm/ffmpeg-core.js`,
  `${CORE_BASE_NEW}/esm/ffmpeg-core.mjs`,
  `${CORE_BASE_NEW}/wasm/ffmpeg-core.js`,
  `${CORE_BASE_NEW}/wasm/ffmpeg-core.mjs`,
  `${CORE_BASE_OLD}/ffmpeg-core.js`,
  `${CORE_BASE_OLD}/ffmpeg-core.js?module`,
  `${CORE_BASE_OLD}/ffmpeg-core.mjs`,
  `${CORE_BASE_OLD}/ffmpeg-core.mjs?module`,
  `${CORE_BASE_OLD}/js/ffmpeg-core.js`,
  `${CORE_BASE_OLD}/js/ffmpeg-core.mjs`,
  `${CORE_BASE_OLD}/wasm/ffmpeg-core.js`,
  `${CORE_BASE_OLD}/wasm/ffmpeg-core.mjs`,
];

const CORE_WASM_CANDIDATES = [
  `${CORE_BASE_NEW}/ffmpeg-core.wasm`,
  `${CORE_BASE_NEW}/wasm/ffmpeg-core.wasm`,
  `${CORE_BASE_NEW}/esm/ffmpeg-core.wasm`,
  `${CORE_BASE_NEW}/js/ffmpeg-core.wasm`,
  `${CORE_BASE_OLD}/ffmpeg-core.wasm`,
  `${CORE_BASE_OLD}/wasm/ffmpeg-core.wasm`,
  `${CORE_BASE_OLD}/esm/ffmpeg-core.wasm`,
];

const CORE_WORKER_CANDIDATES = [
  `${CORE_BASE_NEW}/ffmpeg-core.worker.js`,
  `${CORE_BASE_NEW}/ffmpeg-core.worker.js?module`,
  `${CORE_BASE_NEW}/esm/ffmpeg-core.worker.js`,
  `${CORE_BASE_NEW}/js/ffmpeg-core.worker.js`,
  `${CORE_BASE_NEW}/wasm/ffmpeg-core.worker.js`,
  `${CORE_BASE_OLD}/ffmpeg-core.worker.js`,
  `${CORE_BASE_OLD}/ffmpeg-core.worker.js?module`,
  `${CORE_BASE_OLD}/esm/ffmpeg-core.worker.js`,
  `${CORE_BASE_OLD}/js/ffmpeg-core.worker.js`,
  `${CORE_BASE_OLD}/wasm/ffmpeg-core.worker.js`,
];

let CORE_JS_URL = `${CORE_BASE_NEW}/ffmpeg-core.js`;
let CORE_WASM_URL = `${CORE_BASE_NEW}/ffmpeg-core.wasm`;
let CORE_WORKER_URL = `${CORE_BASE_NEW}/ffmpeg-core.worker.js`;
const FFMPEG_WRAPPER_URL = `${FFMPEG_BASE}/worker.js`;

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

async function tryPick(urls, expectMime, { batch = 4, timeoutMs = 3000 } = {}) {
  if (typeof fetch !== "function") {
    return urls[0];
  }

  const groups = [];
  for (let i = 0; i < urls.length; i += batch) {
    groups.push(urls.slice(i, i + batch));
  }

  const makeController = () => {
    if (typeof AbortController !== "function") {
      return { signal: undefined, done: () => {} };
    }
    const ac = new AbortController();
    const id = setTimeout(() => ac.abort(), timeoutMs);
    return {
      signal: ac.signal,
      done: () => clearTimeout(id),
    };
  };

  const raceAny =
    typeof Promise.any === "function"
      ? Promise.any.bind(Promise)
      : (promises) =>
          new Promise((resolve, reject) => {
            let rejected = 0;
            const total = promises.length;
            if (total === 0) {
              reject(new Error("No promises provided"));
              return;
            }
            promises.forEach((promise) => {
              Promise.resolve(promise).then(resolve, () => {
                rejected += 1;
                if (rejected === total) {
                  reject(new Error("All promises rejected"));
                }
              });
            });
          });

  const probeOne = async (url) => {
    const { signal, done } = makeController();
    try {
      let response = await fetch(url, { method: "HEAD", cache: "no-store", signal }).catch(() => null);
      if (!response || !(response.ok || response.status === 206)) {
        response = await fetch(url, {
          method: "GET",
          headers: { Range: "bytes=0-16" },
          cache: "no-store",
          signal,
        }).catch(() => null);
      }

      if (response && (response.ok || response.status === 206)) {
        const contentType = (response.headers.get("content-type") || "").toLowerCase();
        if (!expectMime || (contentType && contentType.includes(expectMime))) {
          return url;
        }
      }

      throw new Error("probe-failed");
    } finally {
      done();
    }
  };

  for (const group of groups) {
    try {
      const picked = await raceAny(group.map((url) => probeOne(url)));
      if (picked) {
        return picked;
      }
    } catch (_err) {
      // fall through to next group
    }
  }

  return urls[0];
}

async function verifyCoreReachable() {
  if (typeof fetch !== "function") {
    return;
  }

  // fallback: 若新路徑 404 就改用舊路徑
  CORE_JS_URL = await tryPick(CORE_SCRIPT_CANDIDATES, "javascript");
  CORE_WASM_URL = await tryPick(CORE_WASM_CANDIDATES, "application/wasm");
  CORE_WORKER_URL = await tryPick(CORE_WORKER_CANDIDATES, "javascript");

  console.debug("[ffmpeg] picked core:", {
    js: CORE_JS_URL,
    wasm: CORE_WASM_URL,
    worker: CORE_WORKER_URL,
  });

  try {
    const resources = [
      {
        url: CORE_WASM_URL,
        label: "@ffmpeg/core wasm",
        expect: "application/wasm",
      },
      {
        url: FFMPEG_WRAPPER_URL,
        label: "@ffmpeg/ffmpeg worker",
        expect: "javascript",
      },
    ];

    for (const { url, label, expect } of resources) {
      let response = await fetch(url, { method: "HEAD", cache: "no-store" });
      if (!response || !response.ok) {
        response = await fetch(url, {
          method: "GET",
          headers: { Range: "bytes=0-16" },
          cache: "no-store",
        });
        if (!response || !response.ok) {
          const status = response ? `${response.status} ${response.statusText}` : "unknown";
          throw new Error(`[${label}] probe failed: ${status}`);
        }
      }

      const contentType = (response.headers.get("content-type") || "").toLowerCase();
      if (expect && contentType && !contentType.includes(expect)) {
        console.warn(
          `[ffmpeg] Unexpected Content-Type for ${label}: ${contentType}`,
        );
      }

      const allowOrigin = response.headers.get("access-control-allow-origin");
      if (!allowOrigin) {
        console.warn(
          `[ffmpeg] HEAD ok but ACAO missing for ${label}; will rely on load() for CORS check`,
        );
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error ?? "unknown error");
    throw new Error(`Unable to reach ffmpeg asset: ${message}`);
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

async function getFFmpeg(mirrored) {
  if (typeof FF.FFmpeg === "function") {
    return new FF.FFmpeg({ log: false });
  }

  if (typeof FF.createFFmpeg === "function") {
    const [corePath, wasmPath, workerPath] = await Promise.all([
      toBlobURL(CORE_JS_URL, "text/javascript").then((url) => {
        mirrored.push(url);
        return url;
      }),
      toBlobURL(CORE_WASM_URL, "application/wasm").then((url) => {
        mirrored.push(url);
        return url;
      }),
      toBlobURL(CORE_WORKER_URL, "text/javascript")
        .then((url) => {
          if (url) {
            mirrored.push(url);
          }
          return url;
        })
        .catch(() => undefined),
    ]);

    return FF.createFFmpeg({ log: false, corePath, wasmPath, workerPath });
  }

  throw new Error("@ffmpeg/ffmpeg: no compatible constructor available");
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
      const mirrored = [];
      const instance = await getFFmpeg(mirrored);
      attachProgressHandler(instance);

      try {
        if (isModernInstance(instance)) {
          const [classWorkerURL, coreURL, wasmURL, workerURL] = await Promise.all([
            toBlobURL(FFMPEG_WRAPPER_URL, "text/javascript"),
            toBlobURL(CORE_JS_URL, "text/javascript"),
            toBlobURL(CORE_WASM_URL, "application/wasm"),
            toBlobURL(CORE_WORKER_URL, "text/javascript").catch(() => undefined),
          ]);
          [classWorkerURL, coreURL, wasmURL, workerURL]
            .filter(Boolean)
            .forEach((url) => mirrored.push(url));
          await instance.load({ coreURL, wasmURL, workerURL, classWorkerURL });
        } else if (typeof instance.load === "function") {
          await instance.load();
        }
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
