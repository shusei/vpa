const FRAME_RATE = 30;

const VIDEO_PROFILES = [
  {
    extension: "mp4",
    mimeType: "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
  },
  {
    extension: "mp4",
    mimeType: "video/mp4",
  },
  {
    extension: "webm",
    mimeType: "video/webm;codecs=vp9,opus",
  },
  {
    extension: "webm",
    mimeType: "video/webm;codecs=vp8,opus",
  },
  {
    extension: "webm",
    mimeType: "video/webm",
  },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function easeOutCubic(value) {
  const normalized = clamp(value, 0, 1);
  return 1 - ((1 - normalized) ** 3);
}

function roundedRect(ctx, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, width, height, safeRadius);
    return;
  }
  ctx.moveTo(x + safeRadius, y);
  ctx.arcTo(x + width, y, x + width, y + height, safeRadius);
  ctx.arcTo(x + width, y + height, x, y + height, safeRadius);
  ctx.arcTo(x, y + height, x, y, safeRadius);
  ctx.arcTo(x, y, x + width, y, safeRadius);
  ctx.closePath();
}

function fillRoundedRect(ctx, x, y, width, height, radius, fillStyle) {
  roundedRect(ctx, x, y, width, height, radius);
  ctx.fillStyle = fillStyle;
  ctx.fill();
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight, maxLines = 2) {
  const characters = Array.from(String(text || ""));
  const lines = [];
  let line = "";
  characters.forEach((character) => {
    const candidate = `${line}${character}`;
    if (line && ctx.measureText(candidate).width > maxWidth) {
      lines.push(line);
      line = character;
      return;
    }
    line = candidate;
  });
  if (line) lines.push(line);
  lines.slice(0, maxLines).forEach((value, index) => {
    const suffix = index === maxLines - 1 && lines.length > maxLines ? "…" : "";
    ctx.fillText(`${value}${suffix}`, x, y + (index * lineHeight));
  });
}

export function getSupportedVideoProfiles(MediaRecorderLike = globalThis.MediaRecorder) {
  if (typeof MediaRecorderLike !== "function") return [];
  if (typeof MediaRecorderLike.isTypeSupported !== "function") {
    return [{ extension: "webm", mimeType: "" }];
  }
  return VIDEO_PROFILES.filter(({ mimeType }) => {
    try {
      return MediaRecorderLike.isTypeSupported(mimeType);
    } catch {
      return false;
    }
  });
}

export function defaultClipRange(duration) {
  const safeDuration = Math.max(0, Number(duration) || 0);
  return {
    duration: safeDuration,
    end: safeDuration,
    outputDuration: safeDuration,
    start: 0,
  };
}

export function extractWaveform(audioBuffer, clip, bucketCount = 96) {
  const buckets = Math.max(16, Math.min(256, Math.round(bucketCount)));
  if (!audioBuffer || !(clip?.duration > 0)) {
    return Array.from({ length: buckets }, () => 0.04);
  }
  const sampleRate = Number(audioBuffer.sampleRate) || 1;
  const startSample = Math.max(0, Math.floor(clip.start * sampleRate));
  const endSample = Math.min(
    audioBuffer.length,
    Math.ceil(clip.end * sampleRate),
  );
  const samplesPerBucket = Math.max(1, Math.floor(
    Math.max(1, endSample - startSample) / buckets,
  ));
  const channelCount = Math.max(1, Number(audioBuffer.numberOfChannels) || 1);
  const peaks = [];
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const from = startSample + (bucket * samplesPerBucket);
    const to = bucket === buckets - 1
      ? endSample
      : Math.min(endSample, from + samplesPerBucket);
    let peak = 0;
    for (let channel = 0; channel < channelCount; channel += 1) {
      const data = audioBuffer.getChannelData(channel);
      for (let index = from; index < to; index += 1) {
        peak = Math.max(peak, Math.abs(data[index] || 0));
      }
    }
    peaks.push(peak);
  }
  const maximum = Math.max(0.001, ...peaks);
  return peaks.map((peak) => Math.max(0.04, peak / maximum));
}

function drawWaveform(ctx, waveform, {
  activeColor,
  mutedColor,
  progress,
  width,
  x,
  y,
}) {
  const gap = 3;
  const barWidth = Math.max(2, (width / waveform.length) - gap);
  waveform.forEach((peak, index) => {
    const ratio = index / Math.max(1, waveform.length - 1);
    const height = 18 + (peak * 94);
    const barX = x + (index * (barWidth + gap));
    fillRoundedRect(
      ctx,
      barX,
      y - (height / 2),
      barWidth,
      height,
      barWidth / 2,
      ratio <= progress ? activeColor : mutedColor,
    );
  });
}

function drawFrame(ctx, {
  labels,
  outputDuration,
  progress,
  result,
  theme,
  waveform,
}) {
  const dark = theme !== "light";
  const ink = dark ? "#f7f3ec" : "#17272c";
  const muted = dark ? "#c5d4d7" : "#52676c";
  const surface = dark ? "rgba(13,32,38,.82)" : "rgba(255,255,255,.82)";
  const accent = dark ? "#f0a3ba" : "#b54e73";
  const accent2 = dark ? "#83c4dd" : "#347f9d";
  const background = ctx.createLinearGradient(0, 0, 720, 1280);
  background.addColorStop(0, dark ? "#10242c" : "#e6f3f5");
  background.addColorStop(0.52, dark ? "#27444d" : "#fff8eb");
  background.addColorStop(1, dark ? "#6b3349" : "#f4c7d2");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, 720, 1280);

  const drift = Math.sin(progress * Math.PI * 2) * 34;
  ctx.fillStyle = dark ? "rgba(131,196,221,.15)" : "rgba(52,127,157,.12)";
  ctx.beginPath();
  ctx.arc(90 + drift, 150, 180, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = dark ? "rgba(240,163,186,.16)" : "rgba(181,78,115,.12)";
  ctx.beginPath();
  ctx.arc(650 - drift, 1080, 220, 0, Math.PI * 2);
  ctx.fill();

  fillRoundedRect(ctx, 38, 38, 644, 1204, 42, surface);
  ctx.fillStyle = accent;
  ctx.font = "800 23px 'Noto Sans TC', sans-serif";
  ctx.letterSpacing = "2px";
  ctx.fillText(labels.brand, 78, 100);
  ctx.letterSpacing = "0px";
  ctx.fillStyle = muted;
  ctx.font = "700 22px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.title, 78, 150);

  const reveal = easeOutCubic((progress - 0.04) / 0.3);
  const feminine = Math.round(result.score * reveal);
  const masculine = 100 - feminine;
  ctx.fillStyle = accent;
  ctx.font = "900 150px Georgia, serif";
  ctx.fillText(`${feminine}`, 74, 342);
  ctx.font = "800 42px 'Noto Sans TC', sans-serif";
  ctx.fillText("%", 315, 333);
  ctx.fillStyle = ink;
  ctx.font = "800 28px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.feminine, 82, 394);
  ctx.textAlign = "right";
  ctx.fillText(`${labels.masculine} ${masculine}%`, 638, 394);
  ctx.textAlign = "left";

  fillRoundedRect(ctx, 80, 422, 560, 20, 10, dark ? "#29454f" : "#dce8ea");
  const scoreWidth = Math.max(10, 560 * (feminine / 100));
  const scoreGradient = ctx.createLinearGradient(80, 0, 640, 0);
  scoreGradient.addColorStop(0, accent);
  scoreGradient.addColorStop(1, accent2);
  fillRoundedRect(ctx, 80, 422, scoreWidth, 20, 10, scoreGradient);

  const detailsReveal = easeOutCubic((progress - 0.2) / 0.25);
  ctx.globalAlpha = detailsReveal;
  fillRoundedRect(
    ctx,
    78,
    492,
    272,
    160,
    24,
    dark ? "rgba(255,255,255,.07)" : "rgba(23,39,44,.06)",
  );
  fillRoundedRect(
    ctx,
    370,
    492,
    272,
    160,
    24,
    dark ? "rgba(255,255,255,.07)" : "rgba(23,39,44,.06)",
  );
  ctx.fillStyle = muted;
  ctx.font = "700 17px 'Noto Sans TC', sans-serif";
  wrapText(ctx, labels.age, 104, 528, 220, 22, 2);
  wrapText(ctx, labels.archetype, 396, 528, 220, 22, 2);
  ctx.fillStyle = ink;
  ctx.font = "800 31px 'Noto Sans TC', sans-serif";
  wrapText(ctx, labels.ageValue, 104, 600, 220, 38, 2);
  wrapText(ctx, labels.archetypeValue, 396, 600, 220, 38, 2);
  ctx.globalAlpha = 1;

  ctx.fillStyle = muted;
  ctx.font = "700 20px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.waveform, 82, 724);
  drawWaveform(ctx, waveform, {
    activeColor: accent2,
    mutedColor: dark ? "rgba(255,255,255,.13)" : "rgba(23,39,44,.12)",
    progress: clamp(progress * outputDuration / Math.max(0.001, outputDuration), 0, 1),
    width: 554,
    x: 82,
    y: 816,
  });

  const challengeReveal = easeOutCubic((progress - 0.58) / 0.24);
  ctx.globalAlpha = challengeReveal;
  fillRoundedRect(
    ctx,
    78,
    932,
    564,
    170,
    26,
    dark ? "rgba(7,20,25,.55)" : "rgba(255,255,255,.68)",
  );
  ctx.fillStyle = accent;
  ctx.font = "800 20px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.challengeLabel, 108, 974);
  ctx.fillStyle = ink;
  ctx.font = "800 28px 'Noto Sans TC', sans-serif";
  wrapText(ctx, labels.challenge, 108, 1022, 500, 39, 2);
  ctx.globalAlpha = 1;

  ctx.fillStyle = muted;
  ctx.font = "500 17px 'Noto Sans TC', sans-serif";
  wrapText(ctx, labels.disclaimer, 82, 1160, 550, 25, 2);
  fillRoundedRect(ctx, 78, 1202, 564, 8, 4, dark ? "#29454f" : "#dce8ea");
  fillRoundedRect(ctx, 78, 1202, 564 * progress, 8, 4, accent);
}

async function decodeAudio(audioUrl, AudioContextLike) {
  const response = await fetch(audioUrl);
  if (!response.ok) throw new Error(`dynamic card audio fetch failed: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const audioContext = new AudioContextLike();
  try {
    const audioBuffer = await audioContext.decodeAudioData(bytes.slice(0));
    return { audioBuffer, audioContext };
  } catch (error) {
    try {
      await audioContext.close();
    } catch {
      // Ignore close failures.
    }
    throw error;
  }
}

function encodeWaveClip(audioBuffer, clip) {
  const channelCount = Math.max(1, Math.min(2, audioBuffer.numberOfChannels));
  const sampleRate = Math.round(audioBuffer.sampleRate);
  const startFrame = Math.max(0, Math.floor(clip.start * sampleRate));
  const endFrame = Math.min(audioBuffer.length, Math.ceil(clip.end * sampleRate));
  const frameCount = Math.max(0, endFrame - startFrame);
  const bytesPerSample = 2;
  const blockAlign = channelCount * bytesPerSample;
  const dataLength = frameCount * blockAlign;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataLength, true);

  const channels = Array.from({ length: channelCount }, (_, index) => (
    audioBuffer.getChannelData(index)
  ));
  let offset = 44;
  for (let frame = startFrame; frame < endFrame; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = clamp(channels[channel][frame], -1, 1);
      const encoded = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, Math.round(encoded), true);
      offset += bytesPerSample;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function recordProfile({
  audioBuffer,
  audioContext,
  canvas,
  clip,
  draw,
  MediaRecorderLike,
  onProgress,
  profile,
}) {
  const destination = audioContext.createMediaStreamDestination();
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(destination);
  const canvasStream = canvas.captureStream(FRAME_RATE);
  const outputStream = new MediaStream([
    ...canvasStream.getVideoTracks(),
    ...destination.stream.getAudioTracks(),
  ]);
  const options = {
    audioBitsPerSecond: 128000,
    videoBitsPerSecond: 4500000,
  };
  if (profile.mimeType) options.mimeType = profile.mimeType;
  let recorder;
  try {
    recorder = new MediaRecorderLike(outputStream, options);
  } catch (error) {
    outputStream.getTracks().forEach((track) => track.stop());
    try {
      source.disconnect();
    } catch {
      // Ignore disconnect failures.
    }
    throw error;
  }
  const chunks = [];
  let animationFrame = 0;
  let settled = false;

  return new Promise((resolve, reject) => {
    const startedAt = performance.now();

    function cleanup() {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      outputStream.getTracks().forEach((track) => track.stop());
      try {
        source.disconnect();
      } catch {
        // Ignore disconnect failures.
      }
    }

    function fail(error) {
      if (settled) return;
      settled = true;
      try {
        if (recorder.state !== "inactive") recorder.stop();
      } catch {
        // Ignore stop failures.
      }
      cleanup();
      reject(error instanceof Error ? error : new Error("dynamic card recording failed"));
    }

    recorder.ondataavailable = (event) => {
      if (event.data?.size) chunks.push(event.data);
    };
    recorder.onerror = (event) => {
      fail(event.error || new Error("dynamic card recorder error"));
    };
    recorder.onstop = () => {
      if (settled) return;
      settled = true;
      cleanup();
      const mimeType = recorder.mimeType || profile.mimeType || "video/webm";
      const blob = new Blob(chunks, { type: mimeType });
      if (!blob.size) {
        reject(new Error("dynamic card video is empty"));
        return;
      }
      resolve({
        blob,
        extension: profile.extension,
        mimeType,
      });
    };

    function frame(now) {
      if (settled) return;
      const elapsed = Math.max(0, (now - startedAt) / 1000);
      const progress = clamp(elapsed / clip.outputDuration, 0, 1);
      draw(progress);
      onProgress?.({
        format: profile.extension,
        phase: "encoding",
        progress,
      });
      if (progress >= 1) {
        try {
          if (recorder.state !== "inactive") recorder.stop();
        } catch (error) {
          fail(error);
        }
        return;
      }
      animationFrame = requestAnimationFrame(frame);
    }

    try {
      draw(0);
      recorder.start(250);
      source.start(
        audioContext.currentTime + 0.05,
        clip.start,
        clip.duration,
      );
      animationFrame = requestAnimationFrame(frame);
    } catch (error) {
      fail(error);
    }
  });
}

export async function readAudioDuration(audioUrl) {
  if (!audioUrl) throw new TypeError("audio URL is required");
  const AudioContextLike = window.AudioContext || window.webkitAudioContext;
  if (typeof AudioContextLike !== "function") {
    throw new Error("Web Audio is unavailable");
  }
  const { audioBuffer, audioContext } = await decodeAudio(audioUrl, AudioContextLike);
  try {
    return audioBuffer.duration;
  } finally {
    try {
      await audioContext.close();
    } catch {
      // Ignore close failures.
    }
  }
}

export async function createSelectedAudioFile({
  audioUrl,
}) {
  if (!audioUrl) throw new TypeError("audio URL is required");
  const AudioContextLike = window.AudioContext || window.webkitAudioContext;
  if (typeof AudioContextLike !== "function") {
    throw new Error("Web Audio is unavailable");
  }
  const { audioBuffer, audioContext } = await decodeAudio(audioUrl, AudioContextLike);
  try {
    const clip = defaultClipRange(audioBuffer.duration);
    if (!(clip.duration > 0)) throw new Error("Selected audio clip is empty");
    const blob = encodeWaveClip(audioBuffer, clip);
    return new File([blob], "vpa-voice-clip.wav", { type: "audio/wav" });
  } finally {
    try {
      await audioContext.close();
    } catch {
      // Ignore close failures.
    }
  }
}

export async function generateDynamicVoiceCard({
  audioUrl,
  labels,
  onProgress,
  result,
  theme = "dark",
}) {
  if (!audioUrl) throw new TypeError("audio URL is required");
  if (!result?.ready) throw new TypeError("a ready result is required");
  const AudioContextLike = window.AudioContext || window.webkitAudioContext;
  if (typeof AudioContextLike !== "function") throw new Error("Web Audio is unavailable");
  if (typeof HTMLCanvasElement.prototype.captureStream !== "function") {
    throw new Error("Canvas stream capture is unavailable");
  }
  const profiles = getSupportedVideoProfiles();
  if (!profiles.length) throw new Error("No supported video recorder format");

  onProgress?.({ phase: "decoding", progress: 0 });
  const { audioBuffer, audioContext } = await decodeAudio(audioUrl, AudioContextLike);
  try {
    const clip = defaultClipRange(audioBuffer.duration);
    if (!(clip.duration > 0)) throw new Error("Selected audio clip is empty");
    const waveform = extractWaveform(audioBuffer, clip);
    const canvas = document.createElement("canvas");
    canvas.width = 720;
    canvas.height = 1280;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) throw new Error("Canvas is unavailable");
    if (audioContext.state === "suspended") await audioContext.resume();
    const draw = (progress) => drawFrame(ctx, {
      labels,
      outputDuration: clip.outputDuration,
      progress,
      result,
      theme,
      waveform,
    });
    const attempts = [];
    for (const profile of profiles) {
      try {
        const output = await recordProfile({
          audioBuffer,
          audioContext,
          canvas,
          clip,
          draw,
          MediaRecorderLike: MediaRecorder,
          onProgress,
          profile,
        });
        onProgress?.({
          format: output.extension,
          phase: "complete",
          progress: 1,
        });
        return {
          ...output,
          clip,
          duration: clip.outputDuration,
        };
      } catch (error) {
        attempts.push(`${profile.mimeType || "default"}: ${error.message}`);
      }
    }
    throw new Error(`Dynamic card encoding failed. ${attempts.join(" | ")}`);
  } finally {
    try {
      await audioContext.close();
    } catch {
      // Ignore close failures.
    }
  }
}

export const dynamicVoiceCardInternals = {
  FRAME_RATE,
  VIDEO_PROFILES,
  clamp,
  encodeWaveClip,
  easeOutCubic,
};
