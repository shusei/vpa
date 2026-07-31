export const EMBEDDED_INFERENCE_MAX_SEC = 8;

export function selectRepresentativeSamples(
  samples,
  sampleRate,
  { maxDurationSec = EMBEDDED_INFERENCE_MAX_SEC, segmentCount = 3 } = {},
) {
  const rate = Math.floor(Number(sampleRate));
  const maxSeconds = Number(maxDurationSec);
  const count = Math.max(1, Math.floor(Number(segmentCount) || 1));
  if (!(samples instanceof Float32Array) || rate < 1 || !Number.isFinite(maxSeconds) || maxSeconds <= 0) {
    return {
      durationSec: samples?.length && rate > 0 ? samples.length / rate : 0,
      samples,
      used: false,
    };
  }

  const maxSamples = Math.max(1, Math.floor(rate * maxSeconds));
  if (samples.length <= maxSamples) {
    return {
      durationSec: samples.length / rate,
      samples,
      used: false,
    };
  }

  const output = new Float32Array(maxSamples);
  const segmentLength = Math.floor(maxSamples / count);
  let outputOffset = 0;
  for (let index = 0; index < count; index += 1) {
    const length = index === count - 1 ? maxSamples - outputOffset : segmentLength;
    const progress = count === 1 ? 0.5 : index / (count - 1);
    const start = Math.min(
      samples.length - length,
      Math.max(0, Math.round((samples.length - length) * progress)),
    );
    output.set(samples.subarray(start, start + length), outputOffset);
    outputOffset += length;
  }

  return {
    durationSec: output.length / rate,
    samples: output,
    sourceDurationSec: samples.length / rate,
    used: true,
  };
}
