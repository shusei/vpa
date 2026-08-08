import assert from "node:assert/strict";
import {
  PREFERRED_RECORDING_BITRATE,
  buildMediaRecorderOptions,
  buildNeutralAudioConstraints,
  createAudioMediaRecorder,
  getMicCaptureInfo,
  requestMicStream,
} from "../assets/js/recording-utils.js";

const silentLogger = { info() {}, warn() {} };

{
  const constraints = buildNeutralAudioConstraints({
    getSupportedConstraints: () => ({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: true,
      sampleRate: true,
    }),
  });
  assert.deepEqual(constraints, {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: { ideal: 1 },
    sampleRate: { ideal: 48000 },
  });
}

{
  const constraints = buildNeutralAudioConstraints({ getSupportedConstraints: () => ({}) });
  assert.deepEqual(constraints, {}, "unsupported constraints must not break older iPhone/Android browsers");
}

{
  const constraints = buildNeutralAudioConstraints({
    getSupportedConstraints: () => ({ echoCancellation: true, noiseSuppression: true }),
  });
  assert.deepEqual(constraints, {
    echoCancellation: false,
    noiseSuppression: false,
  }, "partially capable iPhone/Android browsers should receive only supported preferences");
}

{
  let requested = null;
  let applied = null;
  const track = {
    applyConstraints: async (value) => { applied = value; },
    getSettings: () => ({
      sampleRate: 48000,
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    }),
  };
  const stream = { getAudioTracks: () => [track] };
  const mediaDevices = {
    getSupportedConstraints: () => ({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: true,
      sampleRate: true,
    }),
    getUserMedia: async (value) => { requested = value; return stream; },
  };
  const result = await requestMicStream({ mediaDevices, logger: silentLogger });
  assert.equal(result, stream);
  assert.equal(requested.audio.channelCount.ideal, 1);
  assert.equal(requested.audio.sampleRate.ideal, 48000);
  assert.equal(applied.echoCancellation, false);
  assert.deepEqual(getMicCaptureInfo(stream), {
    usedFallback: false,
    verified: true,
    processingActive: false,
    settings: {
      sampleRate: 48000,
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
}

{
  let attempts = 0;
  const track = { applyConstraints: async () => {}, getSettings: () => ({}) };
  const stream = { getAudioTracks: () => [track] };
  const mediaDevices = {
    getSupportedConstraints: () => ({ echoCancellation: true }),
    getUserMedia: async () => {
      attempts += 1;
      if (attempts === 1) throw new Error("preferred constraints rejected");
      return stream;
    },
  };
  await requestMicStream({ mediaDevices, logger: silentLogger });
  assert.equal(attempts, 2);
  assert.equal(getMicCaptureInfo(stream).usedFallback, true);
  assert.equal(getMicCaptureInfo(stream).verified, false);
}

{
  const track = {
    getSettings: () => ({
      echoCancellation: "all",
      noiseSuppression: false,
      autoGainControl: false,
    }),
  };
  const stream = { getAudioTracks: () => [track] };
  const info = getMicCaptureInfo(stream);
  assert.equal(info.verified, true);
  assert.equal(info.processingActive, true, "string echo-cancellation modes are active processing");
  assert.equal(info.settings.echoCancellation, true);
}

{
  assert.deepEqual(buildMediaRecorderOptions("audio/mp4"), {
    mimeType: "audio/mp4",
    audioBitsPerSecond: PREFERRED_RECORDING_BITRATE,
  });
  const calls = [];
  class RecorderWithFallback {
    constructor(stream, options) {
      calls.push(options);
      if (options?.audioBitsPerSecond) throw new Error("bitrate unsupported");
      this.stream = stream;
      this.options = options;
    }
  }
  const stream = {};
  const recorder = createAudioMediaRecorder(stream, "audio/mp4", {
    MediaRecorderClass: RecorderWithFallback,
    logger: silentLogger,
  });
  assert.equal(calls.length, 2);
  assert.deepEqual(recorder.options, { mimeType: "audio/mp4" });
}

{
  class FullRecorder {
    constructor(stream, options) {
      this.stream = stream;
      this.options = options;
    }
  }
  const recorder = createAudioMediaRecorder({}, "audio/webm;codecs=opus", {
    MediaRecorderClass: FullRecorder,
    logger: silentLogger,
  });
  assert.equal(recorder.options.audioBitsPerSecond, 128000);
  assert.equal(recorder.options.mimeType, "audio/webm;codecs=opus");
}

console.log("[PASS] Neutral mobile recording constraints and recorder fallbacks passed.");
