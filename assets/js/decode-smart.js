function isSafariLikeBrowser() {
  if (typeof navigator === "undefined") return false;
  const ua = String(navigator.userAgent || "").toLowerCase();
  if (!ua.includes("safari")) return false;
  const vendor = String(navigator.vendor || "").toLowerCase();
  const blockers = [
    "chrome",
    "crios",
    "crmo",
    "android",
    "edge",
    "edg",
    "opr",
    "opera",
    "firefox",
    "fxios",
  ];
  return !blockers.some((token) => ua.includes(token)) && !vendor.includes("google");
}

function isLikelyM4A(blobOrFile) {
  if (!blobOrFile) return false;
  const type = typeof blobOrFile.type === "string" ? blobOrFile.type.toLowerCase() : "";
  if (type.includes("mp4") || type.includes("m4a")) return true;
  if (typeof blobOrFile.name === "string") {
    const lower = blobOrFile.name.toLowerCase();
    return lower.endsWith(".m4a") || lower.endsWith(".mp4") || lower.endsWith(".mp3");
  }
  return false;
}

const WEB_AUDIO_DECODE_TIMEOUT_MS = 15_000;

function withDecodeTimeout(promise, timeoutMs) {
  let timeoutId = null;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error("WebAudio decode timed out"));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timeoutId !== null) clearTimeout(timeoutId);
  });
}

export async function decodeViaWebAudio(
  blobOrFile,
  targetSR = 16000,
  mixChannelDataToMono,
  decodeTimeoutMs = WEB_AUDIO_DECODE_TIMEOUT_MS,
) {
  const arrayBuf = await blobOrFile.arrayBuffer();
  const legacyArrayBuf = arrayBuf.slice(0);
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let offline = null;
  try {
    let audioBuf;
    try {
      audioBuf = await withDecodeTimeout(ctx.decodeAudioData(arrayBuf), decodeTimeoutMs);
    } catch (err) {
      audioBuf = await withDecodeTimeout(new Promise((resolve, reject) => {
        try {
          ctx.decodeAudioData(legacyArrayBuf, resolve, reject);
        } catch (legacyErr) {
          reject(legacyErr);
        }
      }), decodeTimeoutMs);
    }
    const mono = ctx.createBuffer(1, audioBuf.length, audioBuf.sampleRate);
    const outCh = mono.getChannelData(0);
    const channels = [];
    for (let i = 0; i < audioBuf.numberOfChannels; i++) {
      const chData = audioBuf.getChannelData(i);
      if (chData) channels.push(chData);
    }
    const mixed = mixChannelDataToMono(channels, outCh);
    if (mixed === 0 && channels[0]) {
      outCh.set(channels[0]);
    }

    let out;
    if (audioBuf.sampleRate === targetSR) {
      out = outCh.slice(0);
    } else {
      offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
      const src = offline.createBufferSource();
      src.buffer = mono; src.connect(offline.destination); src.start(0);
      const rendered = await offline.startRendering();
      out = rendered.getChannelData(0).slice(0);
    }
    return { float32: out, sr: targetSR, durationSec: out.length / targetSR };
  } finally {
    try { await ctx.close(); } catch { }
    offline = null;
  }
}

export async function decodeSmartToFloat32(blobOrFile, targetSR, deps = {}) {
  const { setStatus, t, log, mixChannelDataToMono } = deps;
  setStatus(t("status.webaudioDecode"), true);
  try {
    return await decodeViaWebAudio(blobOrFile, targetSR, mixChannelDataToMono);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err ?? "");
    log("[decode] WebAudio decode failed:", message);
    const safariHint = isSafariLikeBrowser() && isLikelyM4A(blobOrFile);
    setStatus(safariHint ? t("status.ffmpegPrepareSafari") : t("status.ffmpegPrepare"), true);
    try {
      const { transcodeToMonoFloat32 } = await import("./ffmpeg-transcode.js");
      return await transcodeToMonoFloat32(blobOrFile, targetSR, (event) => {
        if (!event) return;
        if (event.type === "load-start") {
          setStatus(t("status.ffmpegLoading"), true);
          return;
        }
        if (event.type === "transcode-progress") {
          const pct = Math.min(99, Math.max(0, Math.round((event.progress || 0) * 100)));
          setStatus(t("status.ffmpegTranscode", { progress: pct }), true);
          return;
        }
        if (event.type === "transcode-complete") {
          setStatus(t("status.ffmpegTranscode", { progress: 100 }), true);
        }
      });
    } catch (ffmpegError) {
      const fallbackMessage = ffmpegError instanceof Error ? ffmpegError.message : String(ffmpegError ?? "");
      log("[decode] ffmpeg fallback failed:", fallbackMessage);
      setStatus(t("errors.ffmpegModuleLoadFailed", { message: fallbackMessage }), true);
      throw new Error(t("status.decodeFailure"));
    }
  }
}
