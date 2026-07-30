import {
  VAD_MIN_APPLY_SEC,
  VAD_FRAME_MS,
  VAD_HOP_MS,
  VAD_PAD_MS,
  VAD_MIN_SEG_MS,
  VAD_MIN_VOICED_SEC,
  VAD_SILENCE_RATIO_TO_APPLY,
} from "./constants.js";

export function maybeApplyAdaptiveVAD(float32, sr) {
  const dur = float32.length / sr;
  if (dur < VAD_MIN_APPLY_SEC) return null;

  const frame = Math.max(1, Math.floor(sr * (VAD_FRAME_MS / 1000)));
  const hop = Math.max(1, Math.floor(sr * (VAD_HOP_MS / 1000)));
  const pad = Math.max(0, Math.floor(sr * (VAD_PAD_MS / 1000)));
  const minSeg = Math.max(1, Math.floor(sr * (VAD_MIN_SEG_MS / 1000)));

  const energies = [];
  for (let s = 0; s + frame <= float32.length; s += hop) {
    let acc = 0;
    for (let i = 0; i < frame; i++) {
      const v = float32[s + i];
      acc += v * v;
    }
    energies.push(acc / frame);
  }
  if (energies.length < 5) return null;

  const thr = Math.max(1e-7, percentile(energies, 20) * 1.5);
  const voicedMask = energies.map((e) => e > thr);
  smoothMask(voicedMask, 3);

  const segs = [];
  let i = 0;
  while (i < voicedMask.length) {
    while (i < voicedMask.length && !voicedMask[i]) i++;
    if (i >= voicedMask.length) break;
    let j = i;
    while (j < voicedMask.length && voicedMask[j]) j++;
    const s0 = Math.max(0, i * hop - pad);
    const s1 = Math.min(float32.length, j * hop + frame + pad);
    if ((s1 - s0) >= minSeg) segs.push([s0, s1]);
    i = j;
  }
  if (!segs.length) return null;

  const kept = segs.reduce((a, [s0, s1]) => a + (s1 - s0), 0);
  const keptSec = kept / sr;
  const silenceRatio = 1 - (keptSec / dur);
  if (silenceRatio < VAD_SILENCE_RATIO_TO_APPLY || keptSec < VAD_MIN_VOICED_SEC) return null;

  const out = new Float32Array(kept);
  let offset = 0;
  for (const [s0, s1] of segs) {
    out.set(float32.subarray(s0, s1), offset);
    offset += (s1 - s0);
  }
  return { used: true, arr: out, keptSec, segs };
}

function percentile(arr, p) {
  const a = arr.slice().sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
  return a[idx];
}

function smoothMask(mask, k = 3) {
  let count = 0;
  for (let i = 0; i <= mask.length; i++) {
    if (i < mask.length && !mask[i]) count++;
    else {
      if (count > 0 && count < k) {
        for (let j = i - count; j < i; j++) mask[j] = true;
      }
      count = 0;
    }
  }

  count = 0;
  for (let i = 0; i <= mask.length; i++) {
    if (i < mask.length && mask[i]) count++;
    else {
      if (count > 0 && count < k) {
        for (let j = i - count; j < i; j++) mask[j] = false;
      }
      count = 0;
    }
  }
}
