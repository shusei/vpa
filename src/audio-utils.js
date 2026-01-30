export function mixChannelDataToMono(channels, outCh) {
  if (!outCh || typeof outCh.length !== "number") {
    throw new TypeError("outCh must be a Float32Array");
  }
  const frameCount = outCh.length;
  outCh.fill(0);
  if (!Array.isArray(channels) || frameCount === 0) return 0;

  let validChannels = 0;
  for (const chData of channels) {
    if (!(chData instanceof Float32Array) || chData.length !== frameCount) continue;
    let hasNaN = false;
    for (let i = 0; i < frameCount; i++) {
      if (Number.isNaN(chData[i])) {
        hasNaN = true;
        break;
      }
    }
    if (hasNaN) continue;
    for (let i = 0; i < frameCount; i++) {
      outCh[i] += chData[i];
    }
    validChannels++;
  }

  if (validChannels > 0) {
    const inv = 1 / validChannels;
    for (let i = 0; i < frameCount; i++) {
      outCh[i] *= inv;
    }
  }
  return validChannels;
}

export function rms(arr, a, b) {
  let s = 0;
  for (let i = a; i < b; i++) {
    const v = arr[i];
    s += v * v;
  }
  return Math.sqrt(s / Math.max(1, b - a));
}

export function zeroCrossingRate(arr) {
  let count = 0;
  for (let i = 1; i < arr.length; i++) {
    const prev = arr[i - 1];
    const curr = arr[i];
    if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) count++;
  }
  return count / Math.max(1, arr.length - 1);
}

export function fftRadix2(re, im) {
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
