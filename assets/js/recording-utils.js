export function pickSupportedMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  try {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported) {
      for (const t of cands) if (MediaRecorder.isTypeSupported(t)) return t;
    }
  } catch { }
  return "";
}

export const PREFERRED_RECORDING_BITRATE = 128000;

const captureInfoByStream = new WeakMap();
const AUDIO_PREFERENCES = Object.freeze({
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: Object.freeze({ ideal: 1 }),
  sampleRate: Object.freeze({ ideal: 48000 }),
});

export function buildNeutralAudioConstraints(mediaDevices = globalThis.navigator?.mediaDevices) {
  let supported = null;
  try {
    supported = mediaDevices?.getSupportedConstraints?.() || null;
  } catch { }

  const audio = {};
  for (const [key, value] of Object.entries(AUDIO_PREFERENCES)) {
    // Older browsers without getSupportedConstraints safely ignore unknown
    // processing keys, but sample-rate/channel constraints have caused capture
    // failures in older WebKit builds. Only add those when support is reported.
    const isFormatPreference = key === "sampleRate" || key === "channelCount";
    if ((supported && supported[key]) || (!supported && !isFormatPreference)) {
      audio[key] = value;
    }
  }
  return audio;
}

export function describeMicStream(stream, { usedFallback = false } = {}) {
  const track = stream?.getAudioTracks?.()?.[0] || null;
  let settings = {};
  try {
    settings = track?.getSettings?.() || {};
  } catch { }
  const echoCancellation = typeof settings.echoCancellation === "boolean"
    ? settings.echoCancellation
    : (typeof settings.echoCancellation === "string" ? true : null);
  const processingValues = [echoCancellation, settings.noiseSuppression, settings.autoGainControl];
  const reportedProcessing = processingValues.filter((value) => typeof value === "boolean");
  return {
    usedFallback,
    verified: reportedProcessing.length === processingValues.length,
    processingActive: reportedProcessing.some((value) => value === true),
    settings: {
      sampleRate: Number.isFinite(settings.sampleRate) ? settings.sampleRate : null,
      channelCount: Number.isFinite(settings.channelCount) ? settings.channelCount : null,
      echoCancellation,
      noiseSuppression: typeof settings.noiseSuppression === "boolean" ? settings.noiseSuppression : null,
      autoGainControl: typeof settings.autoGainControl === "boolean" ? settings.autoGainControl : null,
    },
  };
}

export function getMicCaptureInfo(stream) {
  return captureInfoByStream.get(stream) || describeMicStream(stream);
}

export function buildMediaRecorderOptions(mimeType, audioBitsPerSecond = PREFERRED_RECORDING_BITRATE) {
  const options = { audioBitsPerSecond };
  if (mimeType) options.mimeType = mimeType;
  return options;
}

export function createAudioMediaRecorder(
  stream,
  mimeType,
  { MediaRecorderClass = globalThis.MediaRecorder, logger = console } = {},
) {
  if (!MediaRecorderClass) throw new Error("record-unsupported");
  try {
    return new MediaRecorderClass(stream, buildMediaRecorderOptions(mimeType));
  } catch (error) {
    logger?.warn?.("[MediaRecorder] preferred bitrate unavailable", error);
  }
  if (mimeType) {
    try {
      return new MediaRecorderClass(stream, { mimeType });
    } catch (error) {
      logger?.warn?.("[MediaRecorder] preferred MIME unavailable", error);
    }
  }
  return new MediaRecorderClass(stream);
}

export async function requestMicStream({
  mediaDevices = globalThis.navigator?.mediaDevices,
  logger = console,
} = {}) {
  const preferredAudio = buildNeutralAudioConstraints(mediaDevices);
  const base = { audio: preferredAudio };
  const fallback = { audio: true };
  const getUserMedia = mediaDevices?.getUserMedia?.bind(mediaDevices);
  if (!getUserMedia) {
    throw new Error("record-unsupported");
  }
  const disableTrackProcessing = async (stream) => {
    try {
      const tracks = stream?.getAudioTracks?.() || [];
      await Promise.all(tracks.map(async (track) => {
        if (!track?.applyConstraints) return;
        try {
          await track.applyConstraints(preferredAudio);
        } catch (err) {
          logger?.warn?.("[audio track] neutral constraints unavailable", err);
        }
      }));
    } catch (err) {
      logger?.warn?.("[audio track] constraints traversal failed", err);
    }
  };
  try {
    const stream = await getUserMedia(base);
    await disableTrackProcessing(stream);
    const info = describeMicStream(stream);
    captureInfoByStream.set(stream, info);
    logger?.info?.("[audio capture]", info);
    return stream;
  } catch (err) {
    logger?.warn?.("[getUserMedia] preferred constraints failed", err);
  }
  const fallbackStream = await getUserMedia(fallback);
  await disableTrackProcessing(fallbackStream);
  const info = describeMicStream(fallbackStream, { usedFallback: true });
  captureInfoByStream.set(fallbackStream, info);
  logger?.info?.("[audio capture fallback]", info);
  return fallbackStream;
}
