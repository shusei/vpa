const VERSION = "voice-quality-1.0.0";
const ANALYSIS_RATE = 8000;
const FRAME_SEC = 0.08;
const STEP_SEC = 0.02;
const MAX_ANALYSIS_SEC = 12;
const WINDOW_SEC = 4;
const MIN_PITCH_HZ = 70;
const MAX_PITCH_HZ = 500;
const MIN_PERIODICITY = 0.56;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clamp01(value) {
  return clamp(Number(value) || 0, 0, 1);
}

function finite(value) {
  return Number.isFinite(Number(value)) ? Number(value) : NaN;
}

function interpolate(points, rawValue) {
  const value = finite(rawValue);
  if (!Number.isFinite(value)) return 0;
  if (value <= points[0][0]) return points[0][1];
  for (let index = 1; index < points.length; index += 1) {
    const [nextX, nextY] = points[index];
    const [prevX, prevY] = points[index - 1];
    if (value <= nextX) {
      const ratio = (value - prevX) / Math.max(1e-9, nextX - prevX);
      return prevY + ((nextY - prevY) * ratio);
    }
  }
  return points[points.length - 1][1];
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function medianAbsoluteDeviation(values, center = median([...values])) {
  if (!Number.isFinite(center)) return NaN;
  return median(values.filter(Number.isFinite).map((value) => Math.abs(value - center)));
}

function chooseWindows(samples, sampleRate) {
  const maxSamples = Math.max(1, Math.floor(MAX_ANALYSIS_SEC * sampleRate));
  if (samples.length <= maxSamples) return [samples];
  const windowSamples = Math.max(1, Math.floor(WINDOW_SEC * sampleRate));
  const maxStart = Math.max(0, samples.length - windowSamples);
  return [0, 0.5, 1].map((ratio) => {
    const start = Math.round(maxStart * ratio);
    return samples.subarray(start, Math.min(samples.length, start + windowSamples));
  });
}

function resampleWindow(input, sourceRate) {
  if (sourceRate === ANALYSIS_RATE) return Float32Array.from(input);
  const outputLength = Math.max(1, Math.floor(input.length * ANALYSIS_RATE / sourceRate));
  const output = new Float32Array(outputLength);
  if (sourceRate > ANALYSIS_RATE) {
    const ratio = sourceRate / ANALYSIS_RATE;
    for (let index = 0; index < outputLength; index += 1) {
      const start = Math.floor(index * ratio);
      const end = Math.max(start + 1, Math.min(input.length, Math.floor((index + 1) * ratio)));
      let sum = 0;
      for (let sourceIndex = start; sourceIndex < end; sourceIndex += 1) {
        sum += input[sourceIndex];
      }
      output[index] = sum / (end - start);
    }
    return output;
  }
  const ratio = sourceRate / ANALYSIS_RATE;
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(input.length - 1, left + 1);
    const fraction = position - left;
    output[index] = input[left] + ((input[right] - input[left]) * fraction);
  }
  return output;
}

function removeDc(input) {
  let mean = 0;
  for (let index = 0; index < input.length; index += 1) mean += input[index];
  mean /= Math.max(1, input.length);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = input[index] - mean;
  }
  return output;
}

function lowPass(input, cutoffHz = 650) {
  const output = new Float32Array(input.length);
  const smoothing = Math.exp((-2 * Math.PI * cutoffHz) / ANALYSIS_RATE);
  let previous = 0;
  for (let index = 0; index < input.length; index += 1) {
    previous = ((1 - smoothing) * input[index]) + (smoothing * previous);
    output[index] = previous;
  }
  return output;
}

function frameLevel(input, start, length) {
  let sum = 0;
  let sumSquares = 0;
  let peak = 0;
  let clipped = 0;
  const end = Math.min(input.length, start + length);
  for (let index = start; index < end; index += 1) {
    const value = input[index];
    const absolute = Math.abs(value);
    sum += value;
    sumSquares += value * value;
    if (absolute > peak) peak = absolute;
    if (absolute >= 0.995) clipped += 1;
  }
  const count = Math.max(1, end - start);
  const mean = sum / count;
  return {
    clippedRatio: clipped / count,
    mean,
    peak,
    rms: Math.sqrt(Math.max(0, (sumSquares / count) - (mean * mean))),
  };
}

function estimatePeriod(input, start, length, mean) {
  const minLag = Math.max(2, Math.floor(ANALYSIS_RATE / MAX_PITCH_HZ));
  const maxLag = Math.min(length - 3, Math.ceil(ANALYSIS_RATE / MIN_PITCH_HZ));
  const correlations = new Float64Array(maxLag + 1);
  let bestLag = 0;
  let bestCorrelation = -1;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let cross = 0;
    let leftEnergy = 0;
    let rightEnergy = 0;
    const count = length - lag;
    for (let offset = 0; offset < count; offset += 1) {
      const left = input[start + offset] - mean;
      const right = input[start + offset + lag] - mean;
      cross += left * right;
      leftEnergy += left * left;
      rightEnergy += right * right;
    }
    const denominator = Math.sqrt(leftEnergy * rightEnergy);
    const correlation = denominator > 1e-12 ? cross / denominator : 0;
    correlations[lag] = correlation;
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }
  if (!bestLag) return { correlation: 0, periodSamples: NaN };
  const left = correlations[bestLag - 1] || bestCorrelation;
  const center = bestCorrelation;
  const right = correlations[bestLag + 1] || bestCorrelation;
  const denominator = left - (2 * center) + right;
  const offset = Math.abs(denominator) > 1e-9
    ? clamp(0.5 * (left - right) / denominator, -0.5, 0.5)
    : 0;
  return {
    correlation: clamp(bestCorrelation, -1, 1),
    periodSamples: bestLag + offset,
  };
}

function collectFrames(windows) {
  const frameLength = Math.max(8, Math.round(FRAME_SEC * ANALYSIS_RATE));
  const step = Math.max(1, Math.round(STEP_SEC * ANALYSIS_RATE));
  const frames = [];
  windows.forEach((window, windowIndex) => {
    for (let start = 0; start + frameLength <= window.length; start += step) {
      const level = frameLevel(window, start, frameLength);
      const period = estimatePeriod(window, start, frameLength, level.mean);
      frames.push({
        ...level,
        ...period,
        start,
        step,
        windowIndex,
      });
    }
  });
  return frames;
}

function longestVoicedDuration(frames) {
  let best = 0;
  let current = 0;
  let previous = null;
  frames.forEach((frame) => {
    const contiguous = previous
      && previous.windowIndex === frame.windowIndex
      && frame.start - previous.start === frame.step;
    if (frame.accepted) {
      current = contiguous && previous.accepted ? current + 1 : 1;
      if (current > best) best = current;
    } else {
      current = 0;
    }
    previous = frame;
  });
  return best * STEP_SEC;
}

function classifySampleType(frames, explicitType) {
  if (explicitType === "connectedSpeech" || explicitType === "sustainedVowel") {
    return { source: "explicit", value: explicitType };
  }
  const accepted = frames.filter((frame) => frame.accepted);
  const pitches = accepted.map((frame) => ANALYSIS_RATE / frame.periodSamples);
  const levels = accepted.map((frame) => frame.rms);
  const pitchMedian = median([...pitches]);
  const levelMedian = median([...levels]);
  const pitchVariation = medianAbsoluteDeviation(pitches, pitchMedian) / Math.max(1e-9, pitchMedian);
  const levelVariation = medianAbsoluteDeviation(levels, levelMedian) / Math.max(1e-9, levelMedian);
  const coverage = accepted.length / Math.max(1, frames.length);
  const sustained = coverage >= 0.72
    && pitchVariation <= 0.045
    && levelVariation <= 0.22
    && longestVoicedDuration(frames) >= 2.2;
  return {
    source: "estimated",
    value: sustained ? "sustainedVowel" : "connectedSpeech",
  };
}

function positiveCrossings(input) {
  const crossings = [];
  for (let index = 1; index < input.length; index += 1) {
    const left = input[index - 1];
    const right = input[index];
    if (left <= 0 && right > 0) {
      const fraction = Math.abs(right - left) > 1e-12 ? -left / (right - left) : 0;
      crossings.push((index - 1) + fraction);
    }
  }
  return crossings;
}

function selectPulseCrossings(crossings, expectedPeriod) {
  if (!crossings.length || !Number.isFinite(expectedPeriod)) return [];
  const selected = [crossings[0]];
  let candidateIndex = 1;
  while (candidateIndex < crossings.length) {
    const previous = selected[selected.length - 1];
    const minNext = previous + (expectedPeriod * 0.58);
    const maxNext = previous + (expectedPeriod * 1.42);
    while (candidateIndex < crossings.length && crossings[candidateIndex] < minNext) {
      candidateIndex += 1;
    }
    let best = -1;
    let bestDistance = Infinity;
    while (candidateIndex < crossings.length && crossings[candidateIndex] <= maxNext) {
      const distance = Math.abs(crossings[candidateIndex] - (previous + expectedPeriod));
      if (distance < bestDistance) {
        best = candidateIndex;
        bestDistance = distance;
      }
      candidateIndex += 1;
    }
    if (best >= 0) {
      selected.push(crossings[best]);
      candidateIndex = best + 1;
    } else if (candidateIndex < crossings.length) {
      selected.push(crossings[candidateIndex]);
      candidateIndex += 1;
    }
  }
  return selected;
}

function cycleAmplitude(input, start, end) {
  const from = clamp(Math.floor(start), 0, input.length - 1);
  const to = clamp(Math.ceil(end), from + 1, input.length);
  let min = Infinity;
  let max = -Infinity;
  for (let index = from; index < to; index += 1) {
    if (input[index] < min) min = input[index];
    if (input[index] > max) max = input[index];
  }
  return Number.isFinite(min) && Number.isFinite(max) ? max - min : NaN;
}

function collectCycles(windows, acceptedFrames) {
  const periods = [];
  const amplitudes = [];
  windows.forEach((window, windowIndex) => {
    const localFrames = acceptedFrames.filter((frame) => frame.windowIndex === windowIndex);
    const expectedPeriod = median(localFrames.map((frame) => frame.periodSamples));
    if (!Number.isFinite(expectedPeriod)) return;
    const crossings = selectPulseCrossings(positiveCrossings(lowPass(window)), expectedPeriod);
    for (let index = 1; index < crossings.length; index += 1) {
      const period = crossings[index] - crossings[index - 1];
      const pitch = ANALYSIS_RATE / period;
      if (pitch < MIN_PITCH_HZ || pitch > MAX_PITCH_HZ) continue;
      periods.push(period / ANALYSIS_RATE);
      amplitudes.push(cycleAmplitude(window, crossings[index - 1], crossings[index]));
    }
  });
  return { amplitudes, periods };
}

function localJitter(periods) {
  const differences = [];
  const acceptedPeriods = [];
  for (let index = 1; index < periods.length; index += 1) {
    const left = periods[index - 1];
    const right = periods[index];
    if (!(left > 0) || !(right > 0)) continue;
    if (Math.max(left, right) / Math.min(left, right) > 1.3) continue;
    differences.push(Math.abs(right - left));
    acceptedPeriods.push(left, right);
  }
  const periodMean = acceptedPeriods.length
    ? acceptedPeriods.reduce((sum, value) => sum + value, 0) / acceptedPeriods.length
    : NaN;
  return {
    pairs: differences.length,
    valuePct: Number.isFinite(periodMean) && periodMean > 0
      ? (differences.reduce((sum, value) => sum + value, 0) / differences.length) / periodMean * 100
      : NaN,
  };
}

function localShimmer(periods, amplitudes) {
  const differences = [];
  const acceptedAmplitudes = [];
  for (let index = 1; index < Math.min(periods.length, amplitudes.length); index += 1) {
    const leftPeriod = periods[index - 1];
    const rightPeriod = periods[index];
    const left = amplitudes[index - 1];
    const right = amplitudes[index];
    if (!(left > 0) || !(right > 0)) continue;
    if (Math.max(leftPeriod, rightPeriod) / Math.min(leftPeriod, rightPeriod) > 1.3) continue;
    if (Math.max(left, right) / Math.min(left, right) > 1.6) continue;
    differences.push(Math.abs(right - left));
    acceptedAmplitudes.push(left, right);
  }
  const amplitudeMean = acceptedAmplitudes.length
    ? acceptedAmplitudes.reduce((sum, value) => sum + value, 0) / acceptedAmplitudes.length
    : NaN;
  return {
    pairs: differences.length,
    valuePct: Number.isFinite(amplitudeMean) && amplitudeMean > 0
      ? (differences.reduce((sum, value) => sum + value, 0) / differences.length) / amplitudeMean * 100
      : NaN,
  };
}

function fft(real, imaginary, inverse = false) {
  const length = real.length;
  for (let index = 1, reversed = 0; index < length; index += 1) {
    let bit = length >> 1;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
      [imaginary[index], imaginary[reversed]] = [imaginary[reversed], imaginary[index]];
    }
  }
  for (let size = 2; size <= length; size <<= 1) {
    const angle = (inverse ? 2 : -2) * Math.PI / size;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    for (let offset = 0; offset < length; offset += size) {
      let weightReal = 1;
      let weightImaginary = 0;
      for (let index = 0; index < size / 2; index += 1) {
        const even = offset + index;
        const odd = even + (size / 2);
        const oddReal = (real[odd] * weightReal) - (imaginary[odd] * weightImaginary);
        const oddImaginary = (real[odd] * weightImaginary) + (imaginary[odd] * weightReal);
        real[odd] = real[even] - oddReal;
        imaginary[odd] = imaginary[even] - oddImaginary;
        real[even] += oddReal;
        imaginary[even] += oddImaginary;
        const nextWeightReal = (weightReal * stepReal) - (weightImaginary * stepImaginary);
        weightImaginary = (weightReal * stepImaginary) + (weightImaginary * stepReal);
        weightReal = nextWeightReal;
      }
    }
  }
  if (inverse) {
    for (let index = 0; index < length; index += 1) {
      real[index] /= length;
      imaginary[index] /= length;
    }
  }
}

function cppForFrame(input, start, length) {
  let fftSize = 1;
  while (fftSize < length) fftSize <<= 1;
  const real = new Float64Array(fftSize);
  const imaginary = new Float64Array(fftSize);
  for (let index = 0; index < length; index += 1) {
    const window = 0.5 - (0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, length - 1)));
    real[index] = input[start + index] * window;
  }
  fft(real, imaginary);
  for (let index = 0; index < fftSize; index += 1) {
    real[index] = Math.log(Math.max(1e-18, (real[index] * real[index]) + (imaginary[index] * imaginary[index])));
    imaginary[index] = 0;
  }
  fft(real, imaginary, true);
  const minQuefrency = Math.max(1, Math.floor(ANALYSIS_RATE / MAX_PITCH_HZ));
  const maxQuefrency = Math.min(
    Math.floor(fftSize / 2) - 1,
    Math.ceil(ANALYSIS_RATE / MIN_PITCH_HZ),
  );
  const cepstrumDb = new Float64Array(Math.floor(fftSize / 2));
  for (let index = 1; index < cepstrumDb.length; index += 1) {
    cepstrumDb[index] = 10 * Math.log10(Math.max(1e-18, real[index] * real[index]));
  }
  let peakIndex = minQuefrency;
  for (let index = minQuefrency + 1; index <= maxQuefrency; index += 1) {
    if (cepstrumDb[index] > cepstrumDb[peakIndex]) peakIndex = index;
  }
  const trendStart = Math.max(1, Math.round(0.001 * ANALYSIS_RATE));
  const trendEnd = Math.min(cepstrumDb.length - 1, Math.round(0.03 * ANALYSIS_RATE));
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  for (let index = trendStart; index <= trendEnd; index += 1) {
    if (Math.abs(index - peakIndex) <= 2) continue;
    const x = index / ANALYSIS_RATE;
    const y = cepstrumDb[index];
    if (!Number.isFinite(y)) continue;
    count += 1;
    sumX += x;
    sumY += y;
    sumXX += x * x;
    sumXY += x * y;
  }
  const denominator = (count * sumXX) - (sumX * sumX);
  if (count < 3 || Math.abs(denominator) < 1e-12) return NaN;
  const slope = ((count * sumXY) - (sumX * sumY)) / denominator;
  const intercept = (sumY - (slope * sumX)) / count;
  const background = intercept + (slope * (peakIndex / ANALYSIS_RATE));
  return clamp(cepstrumDb[peakIndex] - background, 0, 45);
}

function sampleFrames(frames, limit) {
  if (frames.length <= limit) return frames;
  const selected = [];
  for (let index = 0; index < limit; index += 1) {
    selected.push(frames[Math.round(index * (frames.length - 1) / Math.max(1, limit - 1))]);
  }
  return selected;
}

function confidenceKey(score) {
  if (score >= 0.78) return "high";
  if (score >= 0.52) return "medium";
  return "low";
}

export function analyzeVoiceQuality(samples, sampleRate, options = {}) {
  if (!(samples instanceof Float32Array)) throw new TypeError("voice samples must be Float32Array");
  if (!Number.isFinite(sampleRate) || sampleRate < 4000) throw new TypeError("sample rate is invalid");

  const durationSec = samples.length / sampleRate;
  const windows = chooseWindows(samples, sampleRate)
    .map((window) => removeDc(resampleWindow(window, sampleRate)));
  const analyzedDurationSec = windows.reduce((sum, window) => sum + (window.length / ANALYSIS_RATE), 0);
  let peak = 0;
  let sumSquares = 0;
  let sampleCount = 0;
  let clipped = 0;
  windows.forEach((window) => {
    for (let index = 0; index < window.length; index += 1) {
      const absolute = Math.abs(window[index]);
      if (absolute > peak) peak = absolute;
      sumSquares += window[index] * window[index];
      sampleCount += 1;
      if (absolute >= 0.995) clipped += 1;
    }
  });
  const globalRms = Math.sqrt(sumSquares / Math.max(1, sampleCount));
  const clippedRatio = clipped / Math.max(1, sampleCount);
  const frames = collectFrames(windows);
  const maxFrameRms = Math.max(0, ...frames.map((frame) => frame.rms));
  const energyFloor = Math.max(0.0008, globalRms * 0.22, maxFrameRms * 0.07);
  frames.forEach((frame) => {
    const pitch = ANALYSIS_RATE / frame.periodSamples;
    frame.accepted = frame.rms >= energyFloor
      && frame.clippedRatio <= 0.08
      && frame.correlation >= MIN_PERIODICITY
      && pitch >= MIN_PITCH_HZ
      && pitch <= MAX_PITCH_HZ;
  });

  const acceptedFrames = frames.filter((frame) => frame.accepted);
  const sampleType = classifySampleType(frames, options.sampleType);
  const voicedCoverage = acceptedFrames.length / Math.max(1, frames.length);
  const voicedDurationSec = acceptedFrames.length * STEP_SEC;
  const longestVoicedSec = longestVoicedDuration(frames);
  const pitchMedianHz = median(acceptedFrames.map((frame) => ANALYSIS_RATE / frame.periodSamples));
  const periodicity = median(acceptedFrames.map((frame) => frame.correlation));
  const hnrValues = acceptedFrames.map((frame) => {
    const correlation = clamp(frame.correlation, 1e-5, 1 - 1e-5);
    return clamp(10 * Math.log10(correlation / (1 - correlation)), -20, 60);
  });
  const hnrDb = median(hnrValues);
  const cppFrames = sampleFrames(acceptedFrames, 180);
  const cppValues = cppFrames
    .map((frame) => cppForFrame(
      windows[frame.windowIndex],
      frame.start,
      Math.round(FRAME_SEC * ANALYSIS_RATE),
    ))
    .filter(Number.isFinite);
  const cppDb = median(cppValues);
  const cycles = collectCycles(windows, acceptedFrames);
  const jitter = localJitter(cycles.periods);
  const shimmer = localShimmer(cycles.periods, cycles.amplitudes);
  const perturbationReliable = sampleType.value === "sustainedVowel"
    && jitter.pairs >= 80
    && shimmer.pairs >= 80
    && longestVoicedSec >= 2.2;

  const minimumDuration = sampleType.value === "sustainedVowel" ? 2.5 : 3.5;
  const reasons = [];
  if (durationSec < minimumDuration) reasons.push("duration");
  if (peak < 0.008 || globalRms < 0.002) reasons.push("level");
  if (clippedRatio > 0.02) reasons.push("clipping");
  if (voicedDurationSec < 1.2 || voicedCoverage < 0.18) reasons.push("voicing");
  if (!Number.isFinite(hnrDb) || !Number.isFinite(cppDb)) reasons.push("periodicity");
  if (sampleType.value === "sustainedVowel" && !perturbationReliable) reasons.push("sustainedStability");

  const durationScore = interpolate([
    [0, 0.02],
    [minimumDuration, 0.58],
    [minimumDuration + 3, 1],
  ], durationSec);
  const levelScore = interpolate([
    [0.001, 0.05],
    [0.006, 0.35],
    [0.02, 0.85],
    [0.05, 1],
  ], globalRms);
  const voicedScore = interpolate([
    [0.1, 0.05],
    [0.25, 0.45],
    [0.5, 0.82],
    [0.75, 1],
  ], voicedCoverage);
  const periodicityScore = interpolate([
    [0.5, 0.05],
    [0.62, 0.45],
    [0.78, 0.82],
    [0.9, 1],
  ], periodicity);
  const clippingScore = 1 - clamp01(clippedRatio / 0.025);
  const qualityScore = Math.pow(
    Math.max(0.01, durationScore)
      * Math.max(0.01, levelScore)
      * Math.max(0.01, voicedScore)
      * Math.max(0.01, periodicityScore)
      * Math.max(0.01, clippingScore),
    1 / 5,
  );

  return {
    analyzedDurationSec,
    durationSec,
    metrics: {
      cpp: {
        frames: cppValues.length,
        reliable: cppValues.length >= 12,
        valueDb: cppDb,
      },
      hnr: {
        frames: hnrValues.length,
        reliable: hnrValues.length >= 12,
        valueDb: hnrDb,
      },
      jitterLocal: {
        pairs: jitter.pairs,
        reliable: perturbationReliable,
        valuePct: jitter.valuePct,
      },
      shimmerLocal: {
        pairs: shimmer.pairs,
        reliable: perturbationReliable,
        valuePct: shimmer.valuePct,
      },
    },
    quality: {
      confidenceKey: confidenceKey(qualityScore),
      ready: reasons.length === 0,
      reasons,
      score: clamp01(qualityScore),
    },
    sampleType: sampleType.value,
    sampleTypeSource: sampleType.source,
    signal: {
      clippedRatio,
      globalRms,
      peak,
    },
    version: VERSION,
    voicing: {
      coverage: voicedCoverage,
      frameCount: frames.length,
      longestVoicedSec,
      medianPitchHz: pitchMedianHz,
      periodicity,
      voicedDurationSec,
      voicedFrameCount: acceptedFrames.length,
    },
  };
}

export const voiceQualityInternals = {
  ANALYSIS_RATE,
  MAX_ANALYSIS_SEC,
  VERSION,
  cppForFrame,
  localJitter,
  localShimmer,
  median,
};
