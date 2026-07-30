export function trimYinBuffers(yinBuffers) {
  yinBuffers.x = new Float32Array(0);
  yinBuffers.diff = new Float32Array(0);
  yinBuffers.cmndf = new Float32Array(0);
}

function ensureYinCapacity(yinBuffers, n) {
  if (yinBuffers.x.length < n) {
    yinBuffers.x = new Float32Array(n);
  }
  if (yinBuffers.diff.length < n + 1) {
    yinBuffers.diff = new Float32Array(n + 1);
  }
  if (yinBuffers.cmndf.length < n + 1) {
    yinBuffers.cmndf = new Float32Array(n + 1);
  }
}

export function detectPitchACF(acfBuffers, input, sr, range) {
  // 簡化自相關（ACF）+ 降採樣到 ~16k；限制 50–600 Hz
  const ds = Math.max(1, Math.floor(sr / 16000));
  const N = Math.floor(input.length / ds);
  if (N < 128) return null;
  if (acfBuffers.x.length < N) {
    acfBuffers.x = new Float32Array(N);
  }
  const x = acfBuffers.x;
  let mean = 0; for (let i = 0; i < N; i++) { mean += input[i * ds]; }
  mean /= N;
  let energy = 0; for (let i = 0; i < N; i++) { const v = input[i * ds] - mean; x[i] = v; energy += v * v; }
  if (energy <= 1e-8) return null;

  const srDS = sr / ds;
  const lagMin = Math.floor(srDS / range.max);
  const lagMax = Math.floor(srDS / range.min);

  let bestLag = -1, bestR = 0;
  for (let lag = lagMin; lag <= lagMax; lag++) {
    let num = 0, den0 = 0, den1 = 0;
    for (let i = 0; i < N - lag; i++) {
      const a = x[i], b = x[i + lag];
      num += a * b; den0 += a * a; den1 += b * b;
    }
    const r = num / Math.sqrt((den0 * den1) + 1e-10);
    if (r > bestR) { bestR = r; bestLag = lag; }
  }
  if (bestLag < 0 || bestR < 0.6) return null;
  const freq = srDS / bestLag;
  if (freq < range.min || freq > range.max) return null;
  return freq;
}

export function detectPitchYinLite(yinBuffers, input, sr, range, clamp01) {
  const ds = Math.max(1, Math.floor(sr / 16000));
  const N = Math.floor(input.length / ds);
  if (N < 128) return null;

  ensureYinCapacity(yinBuffers, N);
  const x = yinBuffers.x;

  let mean = 0;
  for (let i = 0; i < N; i++) { mean += input[i * ds]; }
  mean /= N;

  let energy = 0;
  let peak = 0;
  let absSum = 0;
  for (let i = 0; i < N; i++) {
    const v = input[i * ds] - mean;
    x[i] = v;
    energy += v * v;
    const abs = Math.abs(v);
    absSum += abs;
    if (abs > peak) peak = abs;
  }
  if (energy <= 1e-8) return null;

  const srDS = sr / ds;
  const tauMin = Math.max(1, Math.floor(srDS / range.max));
  const tauMax = Math.min(N - 3, Math.floor(srDS / range.min));
  if (tauMax <= tauMin) return null;

  const diff = yinBuffers.diff;
  const cmndf = yinBuffers.cmndf;
  diff[0] = 0;
  cmndf[0] = 1;

  const rms = Math.sqrt(energy / N);
  const avgAbs = absSum / N;
  const safeRms = Math.max(rms, 1e-8);
  const crest = peak > 1e-8 ? peak / safeRms : 0;
  const envelopeRatio = rms > 1e-8 ? avgAbs / safeRms : 0;
  const loudnessFactor = clamp01((rms - 0.0045) / 0.04);
  const crestFactor = clamp01((crest - 1.4) / 3.6);
  const envelopeFactor = clamp01((envelopeRatio - 0.72) / 0.28);
  const periodicity = Math.max(crestFactor, 0.6 * crestFactor + 0.4 * envelopeFactor);
  const threshold = Math.max(
    0.065,
    Math.min(0.19, 0.12 - 0.045 * periodicity + 0.03 * (1 - loudnessFactor))
  );
  let running = 0;
  let bestTau = -1;
  let bestVal = Infinity;
  let lastComputedTau = 0;
  let pendingExtraTau = -1;
  let foundBelowThreshold = false;

  for (let tau = 1; tau <= tauMax; tau++) {
    const limit = N - tau;
    let sum = 0;
    for (let i = 0; i < limit; i++) {
      const delta = x[i] - x[i + tau];
      sum += delta * delta;
    }
    diff[tau] = sum;
    running += sum;
    const val = running ? (sum * tau) / running : 1;
    cmndf[tau] = val;
    lastComputedTau = tau;

    if (tau < tauMin) continue;
    if (val < bestVal) {
      bestVal = val;
      bestTau = tau;
    }
    if (!foundBelowThreshold && val < threshold) {
      foundBelowThreshold = true;
      pendingExtraTau = tau + 1 <= tauMax ? tau + 1 : -1;
      if (pendingExtraTau === -1) {
        break;
      }
    } else if (foundBelowThreshold) {
      if (pendingExtraTau === tau) {
        pendingExtraTau = -1;
        break;
      }
      if (pendingExtraTau === -1) {
        break;
      }
    }
  }

  if (pendingExtraTau > lastComputedTau && pendingExtraTau <= tauMax) {
    const limit = N - pendingExtraTau;
    let sum = 0;
    for (let i = 0; i < limit; i++) {
      const delta = x[i] - x[i + pendingExtraTau];
      sum += delta * delta;
    }
    diff[pendingExtraTau] = sum;
    running += sum;
    const val = running ? (sum * pendingExtraTau) / running : 1;
    cmndf[pendingExtraTau] = val;
    lastComputedTau = pendingExtraTau;
    if (val < bestVal) {
      bestVal = val;
      bestTau = pendingExtraTau;
    }
  }

  if (bestTau <= 0) {
    for (let tau = tauMin; tau <= lastComputedTau; tau++) {
      const val = cmndf[tau];
      if (val < bestVal) {
        bestVal = val;
        bestTau = tau;
      }
    }
  }
  if (bestTau <= 0) return null;

  while (bestTau + 1 <= lastComputedTau && cmndf[bestTau + 1] <= cmndf[bestTau]) {
    bestTau += 1;
  }

  let refinedTau = bestTau;
  if (bestTau > 1 && bestTau < lastComputedTau) {
    const prev = cmndf[bestTau - 1];
    const curr = cmndf[bestTau];
    const next = cmndf[bestTau + 1];
    const denom = (next + prev - 2 * curr);
    if (Number.isFinite(denom) && Math.abs(denom) > 1e-6) {
      const offset = 0.5 * (prev - next) / denom;
      if (Number.isFinite(offset)) refinedTau = bestTau + Math.max(-1, Math.min(1, offset));
    }
  }

  const freq = srDS / refinedTau;
  if (freq < range.min || freq > range.max) return null;
  return freq;
}

export function estimateSpectralFeatures(frame, sr, range, detectPitchACF, acfBuffers, EPS) {
  if (!frame || !frame.length) return null;
  let energy = 0;
  for (let i = 0; i < frame.length; i++) { const v = frame[i]; energy += v * v; }
  if (energy <= 1e-8) return null;

  const n = frame.length;
  const windowed = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const pre = frame[i] - 0.97 * (i > 0 ? frame[i - 1] : 0);
    const win = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1));
    windowed[i] = pre * win;
  }
  const sizePow = Math.max(9, Math.ceil(Math.log2(n * 2)));
  const N = 1 << sizePow;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(windowed);
  fftRadix2(re, im);

  const half = Math.floor(N / 2);
  const mags = new Float32Array(half);
  for (let k = 0; k < half; k++) mags[k] = Math.hypot(re[k], im[k]);

  const smooth = new Float32Array(half);
  const smoothWin = 4;
  for (let k = 0; k < half; k++) {
    let sum = 0, count = 0;
    for (let j = -smoothWin; j <= smoothWin; j++) {
      const idx = k + j;
      if (idx >= 0 && idx < half) { sum += mags[idx]; count++; }
    }
    smooth[k] = count ? (sum / count) : mags[k];
  }

  const freqStep = sr / N;
  const peaks = [];
  for (let k = 3; k < half - 3; k++) {
    const freq = k * freqStep;
    if (freq < 90 || freq > 5000) continue;
    const val = smooth[k];
    if (val > smooth[k - 1] && val >= smooth[k + 1]) peaks.push({ freq, amp: val });
  }
  peaks.sort((a, b) => a.freq - b.freq);
  const compact = [];
  for (const p of peaks) {
    if (!compact.length) { compact.push(p); continue; }
    const last = compact[compact.length - 1];
    if (Math.abs(last.freq - p.freq) < 80) {
      if (p.amp > last.amp) compact[compact.length - 1] = p;
    } else {
      compact.push(p);
    }
  }

  const f0 = detectPitchACF(acfBuffers, frame, sr, range);
  const harmonicTolerance = f0 ? Math.max(40, f0 * 0.18) : 0;
  const isLikelyHarmonic = (freq) => {
    if (!f0 || !Number.isFinite(f0) || f0 < 50) return false;
    const harmonic = Math.round(freq / f0) * f0;
    if (harmonic <= 0) return false;
    return Math.abs(freq - harmonic) <= harmonicTolerance;
  };

  function selectPeak(low, high, { allowHarmonic = false, minGap = 0, previousFreq = null } = {}) {
    let best = null;
    for (const peak of compact) {
      if (peak.freq < low || peak.freq > high) continue;
      if (!allowHarmonic && isLikelyHarmonic(peak.freq)) continue;
      if (previousFreq && peak.freq - previousFreq < minGap) continue;
      if (!best || peak.amp > best.amp) best = peak;
    }
    if (!best && !allowHarmonic) {
      for (const peak of compact) {
        if (peak.freq < low || peak.freq > high) continue;
        if (previousFreq && peak.freq - previousFreq < minGap) continue;
        if (!best || peak.amp > best.amp) best = peak;
      }
    }
    return best;
  }

  const f1Peak = selectPeak(180, 900, { allowHarmonic: true });
  const f1 = f1Peak?.freq ?? NaN;
  const f2Peak = selectPeak(900, 3000, { previousFreq: f1, minGap: 150 });
  const f2 = f2Peak?.freq ?? NaN;
  const f3Peak = selectPeak(1500, 4500, { previousFreq: f2 || f1, minGap: 150 });
  const f3 = f3Peak?.freq ?? NaN;

  let low = 0, mid = 0, high = 0;
  for (let k = 0; k < half; k++) {
    const freq = k * freqStep;
    if (freq < 90 || freq > 5000) continue;
    const power = mags[k] * mags[k];
    if (freq < 1000) low += power;
    else if (freq < 3000) mid += power;
    else high += power;
  }
  const total = low + mid + high + EPS;
  const tilt = 10 * Math.log10((low + mid + EPS) / (high + EPS));
  const breathiness = Math.min(1, Math.max(0, high / total));
  const zcr = zeroCrossingRate(frame);

  return {
    f1: Number.isFinite(f1) ? f1 : NaN,
    f2: Number.isFinite(f2) ? f2 : NaN,
    f3: Number.isFinite(f3) ? f3 : NaN,
    energy: { low, mid, high, total },
    tilt,
    breathiness,
    zcr,
  };
}

function fftRadix2(re, im) {
  const n = re.length;
  if (n <= 1) return;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tmpRe = re[i]; re[i] = re[j]; re[j] = tmpRe;
      const tmpIm = im[i]; im[i] = im[j]; im[j] = tmpIm;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wLenRe = Math.cos(ang);
    const wLenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1, wIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextRe = wRe * wLenRe - wIm * wLenIm;
        const nextIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextRe; wIm = nextIm;
      }
    }
  }
}

function zeroCrossingRate(arr) {
  let count = 0;
  for (let i = 1; i < arr.length; i++) {
    const prev = arr[i - 1];
    const curr = arr[i];
    if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) count++;
  }
  return count / Math.max(1, arr.length - 1);
}
