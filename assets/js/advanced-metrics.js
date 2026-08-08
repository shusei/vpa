import { EPS } from "./constants.js";
import { t } from "./i18n.js?v=1.4.17";
import {
  PS_INTERVAL_MS,
  CONFIDENCE_INCLUDE_THRESHOLD,
  CONFIDENCE_VOICED_THRESHOLD,
  makeStats,
  percentileSorted,
  computeIntonationMetrics,
} from "./pitch-shared.js";

export function averageFinite(arr, mask) {
  const values = Array.isArray(arr) ? arr : [];
  const maskArray = Array.isArray(mask?.mask) ? mask.mask : (Array.isArray(mask) ? mask : null);
  const limit = maskArray ? Math.min(values.length, maskArray.length) : values.length;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < limit; i++) {
    if (maskArray && !maskArray[i]) continue;
    const val = values[i];
    if (Number.isFinite(val)) {
      sum += val;
      count++;
    }
  }
  if (!count) return NaN;
  return sum / count;
}
export function averageEnergy(arr, info = {}) {
  if (!Array.isArray(arr) || !arr.length) return { low: 0, mid: 0, high: 0, total: 0, coverage: 0, validCount: 0 };
  const mask = Array.isArray(info.mask) ? info.mask : (Array.isArray(info.mask?.mask) ? info.mask.mask : null);
  const eligible = Number.isFinite(info.eligibleCount) ? info.eligibleCount : (mask ? mask.reduce((acc, flag) => acc + (flag ? 1 : 0), 0) : arr.length);
  const limit = mask ? Math.min(arr.length, mask.length) : arr.length;
  let low = 0, mid = 0, high = 0, valid = 0, considered = 0;
  for (let i = 0; i < limit; i++) {
    if (mask && !mask[i]) continue;
    considered++;
    const v = arr[i];
    if (!Array.isArray(v)) continue;
    const [l, m, h] = v;
    if (!Number.isFinite(l) && !Number.isFinite(m) && !Number.isFinite(h)) continue;
    low += Number.isFinite(l) ? l : 0;
    mid += Number.isFinite(m) ? m : 0;
    high += Number.isFinite(h) ? h : 0;
    valid++;
  }
  if (!valid) {
    const baseCoverage = eligible > 0 ? (considered / eligible) : 0;
    return { low: 0, mid: 0, high: 0, total: 0, coverage: Math.max(0, Math.min(1, baseCoverage)), validCount: 0 };
  }
  const avgLow = low / valid;
  const avgMid = mid / valid;
  const avgHigh = high / valid;
  const baseCoverage = eligible > 0 ? (valid / eligible) : 0;
  return {
    low: avgLow,
    mid: avgMid,
    high: avgHigh,
    total: avgLow + avgMid + avgHigh,
    coverage: Math.max(0, Math.min(1, baseCoverage)),
    validCount: valid,
  };
}

const RESONANCE_PRIOR_WEIGHT = 0.02;
const RESONANCE_PRIOR_MIN = 0.6;
const RESONANCE_DOMINANCE_DELTA = 0.12;
const RESONANCE_HEAD_MIN = 0.50;
const RESONANCE_MASK_MIN = 0.46;
const RESONANCE_CHEST_MIN = 0.54;
const RESONANCE_MIN_SAMPLES = 18;
const RESONANCE_MIN_COVERAGE = 0.2;
const RESONANCE_COVERAGE_GOOD = 0.35;

const FORMANT_MIN_SAMPLES = 24;
const FORMANT_MIN_COVERAGE = 0.18;
const FORMANT_GOOD_COVERAGE = 0.32;
const FORMANT_MIN_SPREAD = 70;
const FORMANT_CONFIDENCE_THRESHOLD = CONFIDENCE_INCLUDE_THRESHOLD;
const FORMANT_MAX_GAP_FRAMES = 8;

const BREATHINESS_EMA_TAU_SEC = 0.2;
const BRIGHTNESS_F3_LOW = 2400;
const BRIGHTNESS_F3_HIGH = 3400;
const BRIGHTNESS_WARM_Z = -0.7;
const BRIGHTNESS_SPARKLE_Z = 0.4;
const BRIGHTNESS_SWEET_Z = 1.0;
const BRIGHTNESS_TILT_SHARP = -1.5;
const BRIGHTNESS_BREATH_THRESHOLD = 0.45;

export function summarizeBreathiness(arr, info = {}, hopSec) {
  if (!Array.isArray(arr) || !arr.length) return { avg: NaN, count: 0 };
  const mask = Array.isArray(info.mask) ? info.mask : (Array.isArray(info.mask?.mask) ? info.mask.mask : null);
  const limit = mask ? Math.min(arr.length, mask.length) : arr.length;
  const step = Number.isFinite(hopSec) && hopSec > 0 ? hopSec : (PS_INTERVAL_MS / 1000);
  const tau = Math.max(0.08, BREATHINESS_EMA_TAU_SEC);
  const alpha = 1 - Math.exp(-step / tau);
  let ema = null;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < limit; i++) {
    if (mask && !mask[i]) continue;
    let val = arr[i];
    if (!Number.isFinite(val)) continue;
    val = Math.max(0, Math.min(1, val));
    if (ema == null) ema = val;
    else ema = ema + alpha * (val - ema);
    sum += ema;
    count++;
  }
  if (!count) return { avg: NaN, count: 0 };
  const avg = Math.max(0, Math.min(1, sum / count));
  return { avg, count };
}

export function buildEligibleFrameMask(store, { minConfidence = FORMANT_CONFIDENCE_THRESHOLD, maxGapFrames = FORMANT_MAX_GAP_FRAMES } = {}) {
  const voiced = Array.isArray(store.voiced) ? store.voiced : [];
  const confidence = Array.isArray(store.pitchConfidence) ? store.pitchConfidence : [];
  const n = Math.min(voiced.length, confidence.length);
  let mask = new Array(n).fill(false);
  for (let i = 0; i < n; i++) {
    const conf = confidence[i] ?? 0;
    mask[i] = Boolean(voiced[i]) && conf >= minConfidence;
  }
  if (maxGapFrames > 0 && mask.length) {
    let gapStart = -1;
    for (let i = 0; i <= n; i++) {
      const flag = i < n ? mask[i] : true;
      if (!flag) {
        if (gapStart < 0) gapStart = i;
      } else if (gapStart >= 0) {
        const gapLen = i - gapStart;
        const prev = gapStart > 0 ? mask[gapStart - 1] : false;
        const next = i < n ? mask[i] : false;
        if (prev && next && gapLen <= maxGapFrames) {
          for (let j = gapStart; j < i; j++) mask[j] = true;
        }
        gapStart = -1;
      }
    }
    const dilation = Math.max(1, Math.floor(maxGapFrames / 2));
    if (dilation > 0) {
      const expanded = mask.slice();
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        for (let d = 1; d <= dilation; d++) {
          if (i - d >= 0) expanded[i - d] = true;
          if (i + d < n) expanded[i + d] = true;
        }
      }
      mask = expanded;
    }
  }
  const count = mask.reduce((acc, flag) => acc + (flag ? 1 : 0), 0);
  return { mask, count, minConfidence, maxGapFrames };
}

export function categorizeBrightness({ f3Stats, tilt, breath, leaning } = {}) {
  if (!Number.isFinite(f3Stats?.med)) {
    return {
      label: t("analysis.brightness.insufficient.label"),
      hint: t("analysis.brightness.insufficient.hint"),
      key: "insufficient",
    };
  }
  const med = f3Stats.med;
  const center = (BRIGHTNESS_F3_LOW + BRIGHTNESS_F3_HIGH) / 2;
  const span = Math.max(1, (BRIGHTNESS_F3_HIGH - BRIGHTNESS_F3_LOW) / 2);
  const z = (med - center) / span;
  const tiltVal = Number.isFinite(tilt) ? tilt : NaN;
  const breathVal = Number.isFinite(breath) ? breath : NaN;
  let key = "balanced";
  if (z <= BRIGHTNESS_WARM_Z) key = "warm";
  else if (z >= BRIGHTNESS_SWEET_Z) {
    const needsRelax = (Number.isFinite(breathVal) && breathVal > BRIGHTNESS_BREATH_THRESHOLD)
      || (Number.isFinite(tiltVal) && tiltVal < BRIGHTNESS_TILT_SHARP);
    key = needsRelax ? "sharp" : "sweet";
  } else if (z >= BRIGHTNESS_SPARKLE_Z) {
    key = "sparkle";
  }
  let lookupKey = key;
  if (leaning === "masculine") {
    if (key === "sweet") lookupKey = "sweetMasculine";
    else if (key === "sparkle") lookupKey = "sparkleMasculine";
  }
  const label = t(`analysis.brightness.${lookupKey}.label`) || t(`analysis.brightness.${key}.label`);
  const hint = t(`analysis.brightness.${lookupKey}.hint`) || t(`analysis.brightness.${key}.hint`);
  return { label, hint, key, zScore: z };
}

export function detectVoiceLeaning(pf, pm) {
  const pfVal = Number.isFinite(pf) ? pf : 0;
  const pmVal = Number.isFinite(pm) ? pm : 0;
  const diff = Math.abs(pfVal - pmVal);
  if (diff < 0.08) return "neutral";
  return pfVal > pmVal ? "feminine" : "masculine";
}

export function normalizeResonanceBands(energy) {
  if (!energy) return { chest: NaN, mask: NaN, head: NaN, total: 0, coverage: 0 };
  const low = Math.max(0, Number.isFinite(energy.low) ? energy.low : 0);
  const mid = Math.max(0, Number.isFinite(energy.mid) ? energy.mid : 0);
  const high = Math.max(0, Number.isFinite(energy.high) ? energy.high : 0);
  const total = low + mid + high;
  if (!Number.isFinite(total) || total <= EPS) {
    return { chest: NaN, mask: NaN, head: NaN, total: 0, coverage: Number.isFinite(energy.coverage) ? energy.coverage : 0 };
  }
  const prior = Math.max(RESONANCE_PRIOR_MIN, total * RESONANCE_PRIOR_WEIGHT);
  const perBand = prior / 3;
  const denom = total + prior;
  return {
    chest: ((low + perBand) / denom),
    mask: ((mid + perBand) / denom),
    head: ((high + perBand) / denom),
    total,
    coverage: Number.isFinite(energy.coverage) ? energy.coverage : 0,
  };
}

export function describeResonanceFromEnergy(energy) {
  const pctFallback = { chest: 1 / 3, mask: 1 / 3, head: 1 / 3 };
  const normalized = normalizeResonanceBands(energy);
  const { chest, mask, head, total } = normalized;
  const hasAggregate = energy && (Number.isFinite(energy.coverage) || Number.isFinite(energy.validCount));
  const coverage = hasAggregate ? (Number.isFinite(energy.coverage) ? energy.coverage : 0) : NaN;
  const validCount = hasAggregate ? (Number.isFinite(energy.validCount) ? energy.validCount : 0) : 0;

  const insufficientEntry = () => {
    return {
      label: t("analysis.resonanceBalance.insufficient.label"),
      hint: t("analysis.resonanceBalance.insufficient.hint"),
      pct: pctFallback,
      total: 0,
      coverage: hasAggregate ? coverage : NaN,
      display: t("analysis.resonanceBalance.insufficient.label"),
    };
  };

  if (hasAggregate) {
    if (!Number.isFinite(coverage) || coverage < RESONANCE_MIN_COVERAGE || validCount < RESONANCE_MIN_SAMPLES) {
      const base = insufficientEntry();
      const coverageNote = t("analysis.resonanceBalance.coverageLowHint", { value: Math.round((coverage || 0) * 100) });
      if (coverageNote) base.hint = `${base.hint} ${coverageNote}`.trim();
      if (Number.isFinite(coverage)) {
        const suffix = t("analysis.resonanceBalance.coverageLowSuffix", { value: Math.round(coverage * 100) });
        base.display = suffix ? `${base.label}${suffix}` : base.label;
      }
      return base;
    }
  }

  if (!Number.isFinite(chest) || !Number.isFinite(mask) || !Number.isFinite(head)) {
    return insufficientEntry();
  }

  const chestLead = chest - Math.max(mask, head);
  const maskLead = mask - Math.max(chest, head);
  const headLead = head - Math.max(chest, mask);
  let key = "balanced";
  if (chestLead >= RESONANCE_DOMINANCE_DELTA && chest >= RESONANCE_CHEST_MIN) {
    key = "chestHeavy";
  } else if (maskLead >= RESONANCE_DOMINANCE_DELTA && mask >= RESONANCE_MASK_MIN) {
    key = "maskLead";
  } else if (headLead >= RESONANCE_DOMINANCE_DELTA && head >= RESONANCE_HEAD_MIN) {
    key = "headBright";
  }

  const label = t(`analysis.resonanceBalance.${key}.label`);
  let hint = t(`analysis.resonanceBalance.${key}.hint`);
  let display = label;
  if (hasAggregate && Number.isFinite(coverage)) {
    const coverageKey = coverage < RESONANCE_COVERAGE_GOOD ? "coverageLowHint" : "coverageHint";
    const hintNote = t(`analysis.resonanceBalance.${coverageKey}`, { value: Math.round(coverage * 100) });
    if (hintNote) hint = `${hint} ${hintNote}`.trim();
    let suffix = coverage < RESONANCE_COVERAGE_GOOD
      ? t("analysis.resonanceBalance.referenceSuffix", { value: Math.round(coverage * 100) })
      : t("analysis.resonanceBalance.coverageSuffix", { value: Math.round(coverage * 100) });
    if (!suffix && coverage < RESONANCE_COVERAGE_GOOD) {
      suffix = t("analysis.resonanceBalance.referenceOnly");
    }
    if (suffix) display = `${label}${suffix}`;
  }
  return {
    label,
    display,
    hint,
    pct: { chest, mask, head },
    total,
    coverage: hasAggregate ? coverage : NaN,
  };
}

export function categorizeTilt(tilt) {
  if (!Number.isFinite(tilt)) {
    return {
      label: t("analysis.tilt.insufficient.label"),
      hint: t("analysis.tilt.insufficient.hint"),
    };
  }
  let key = "bright";
  if (tilt >= 7.5) key = "warm";
  else if (tilt >= 4.5) key = "gentleWarm";
  else if (tilt >= -1) key = "balanced";
  return {
    label: t(`analysis.tilt.${key}.label`),
    hint: t(`analysis.tilt.${key}.hint`),
  };
}

export function categorizeBreathiness(val, ctx = {}) {
  if (!Number.isFinite(val)) {
    return {
      label: t("analysis.breathiness.insufficient.label"),
      hint: t("analysis.breathiness.insufficient.hint"),
    };
  }
  const snr = Number.isFinite(ctx.snr) ? ctx.snr : NaN;
  const brightnessKey = ctx.brightnessKey;
  const tilt = Number.isFinite(ctx.tilt) ? ctx.tilt : NaN;
  const styleEligible = Number.isFinite(snr) ? snr > 20 : false;
  const needsRelax = brightnessKey === "sharp" || (Number.isFinite(tilt) && tilt < BRIGHTNESS_TILT_SHARP);

  let key = "airy";
  if (val < 0.08) key = "dense";
  else if (val <= 0.18) key = "balanced";
  else if (val <= 0.28) key = "airy";
  else if (val <= 0.45) key = styleEligible ? "style" : "airy";
  else {
    if (needsRelax) key = "tooAiry";
    else key = styleEligible ? "style" : "airy";
  }

  return {
    label: t(`analysis.breathiness.${key}.label`),
    hint: t(`analysis.breathiness.${key}.hint`),
    key,
  };
}

export function makeFormantHint(label, value, low, high) {
  const labelName = t(`analysis.formant.rangeLabels.${label}`);
  const labelWithName = labelName ? `${label} (${labelName})` : label;
  if (!Number.isFinite(value)) {
    return t("analysis.formant.insufficient", { label: labelWithName });
  }
  const lowHint = t(`analysis.formant.low.${label}`);
  const highHint = t(`analysis.formant.high.${label}`);
  if (value < low) {
    const msg = t("analysis.formant.lowMessage", { label: labelWithName, hint: lowHint });
    return msg || `${labelWithName} ${lowHint}`;
  }
  if (value > high) {
    const msg = t("analysis.formant.highMessage", { label: labelWithName, hint: highHint });
    return msg || `${labelWithName} ${highHint}`;
  }
  return t("analysis.formant.inRange", { label: labelWithName });
}

export function summarizeFormantTrends(store, statsBundle, options = {}) {
  let eligibleCount = Number.isFinite(options?.eligibleCount) ? options.eligibleCount : null;
  if ((eligibleCount == null || eligibleCount <= 0) && Array.isArray(store.voiced)) {
    eligibleCount = store.voiced.reduce((acc, flag) => acc + (flag ? 1 : 0), 0);
  }
  const hasAggregate = Number.isFinite(eligibleCount) && eligibleCount > 0;

  const makeEntry = (label, stats, values, low, high) => {
    const sampleCount = values.length;
    const coverageRaw = hasAggregate ? (sampleCount / eligibleCount) : 0;
    const coverage = hasAggregate ? Math.max(0, Math.min(1, coverageRaw)) : NaN;
    const spread = stats ? (stats.p95 - stats.p05) : NaN;
    const reliable = hasAggregate
      && sampleCount >= FORMANT_MIN_SAMPLES
      && coverageRaw >= FORMANT_MIN_COVERAGE
      && Number.isFinite(spread)
      && spread >= FORMANT_MIN_SPREAD
      && Number.isFinite(stats?.med);

    const trendKey = reliable
      ? (stats.med < low ? "low" : (stats.med > high ? "high" : "inRange"))
      : "insufficient";

    const baseHint = reliable
      ? makeFormantHint(label, stats.med, low, high)
      : makeFormantHint(label, NaN, low, high);

    const extraHints = [];
    if (hasAggregate && sampleCount < FORMANT_MIN_SAMPLES) {
      const msg = t("analysis.formant.moreSamplesHint");
      if (msg) extraHints.push(msg);
    }
    if (hasAggregate && coverageRaw < FORMANT_MIN_COVERAGE) {
      const msg = t("analysis.formant.coverageLowHint", { value: Math.round(Math.max(0, Math.min(1, coverageRaw)) * 100) });
      if (msg) extraHints.push(msg);
    }

    const hint = [baseHint, ...extraHints].filter(Boolean).join(" ").trim();
    const display = buildFormantTrendDisplay(trendKey, coverage, hasAggregate);

    return {
      trend: trendKey,
      median: reliable ? stats.med : NaN,
      display,
      hint,
      coverage: hasAggregate ? coverage : NaN,
      samples: sampleCount,
    };
  };

  return {
    f1: makeEntry("F1", statsBundle.f1, statsBundle.f1Vals || [], 170, 420),
    f2: makeEntry("F2", statsBundle.f2, statsBundle.f2Vals || [], 1450, 2750),
    f3: makeEntry("F3", statsBundle.f3, statsBundle.f3Vals || [], 2400, 3400),
  };
}

export function buildFormantTrendDisplay(trendKey, coverage, hasAggregate) {
  const baseRaw = t(`analysis.formant.trendLabels.${trendKey}`);
  const base = baseRaw || "—";
  if (!hasAggregate || !Number.isFinite(coverage) || coverage <= 0) {
    return base;
  }
  const clamped = Math.max(0, Math.min(1, coverage));
  const key = clamped < FORMANT_GOOD_COVERAGE ? "coverageLowSuffix" : "coverageSuffix";
  const suffix = t(`analysis.formant.${key}`, { value: Math.round(clamped * 100) });
  if (!suffix) return base;
  return `${base}${suffix}`;
}

export function analyzeVowelFocus(store, maskInfo) {
  const formants = Array.isArray(store.formants) ? store.formants : [];
  let mask = null;
  if (Array.isArray(maskInfo)) mask = maskInfo;
  else if (maskInfo && Array.isArray(maskInfo.mask)) mask = maskInfo.mask;
  if (!mask || !mask.length) {
    const built = buildEligibleFrameMask(store, {
      minConfidence: FORMANT_CONFIDENCE_THRESHOLD,
      maxGapFrames: FORMANT_MAX_GAP_FRAMES,
    });
    if (Array.isArray(built?.mask) && built.mask.length) mask = built.mask;
  }

  if (!mask || !mask.length) {
    return {
      ratio: NaN,
      label: t("analysis.vowelFocus.insufficient.label"),
      hint: t("analysis.vowelFocus.insufficient.hint"),
    };
  }

  let voiced = 0, focus = 0;
  const limit = Math.min(formants.length, mask.length);
  for (let i = 0; i < limit; i++) {
    if (!mask[i]) continue;
    const form = formants[i];
    if (!form) continue;
    const f1 = form[0], f2 = form[1];
    if (!Number.isFinite(f1) || !Number.isFinite(f2)) continue;
    voiced++;
    if (f1 >= 170 && f1 <= 480 && f2 >= 1400 && f2 <= 3000) focus++;
  }
  const ratio = voiced ? focus / voiced : NaN;
  if (!Number.isFinite(ratio)) {
    return {
      ratio: NaN,
      label: t("analysis.vowelFocus.insufficient.label"),
      hint: t("analysis.vowelFocus.insufficient.hint"),
    };
  }
  let key = "weak";
  if (ratio >= 0.5) key = "strong";
  else if (ratio >= 0.3) key = "medium";
  return {
    ratio,
    label: t(`analysis.vowelFocus.${key}.label`),
    hint: t(`analysis.vowelFocus.${key}.hint`),
  };
}

export function analyzeSpeechRate(store) {
  const hopSec = store.frameSec || (PS_INTERVAL_MS / 1000);
  const n = store.db.length;
  if (!n) {
    return {
      syllPerSec: NaN,
      wordsPerMin: NaN,
      label: t("analysis.speechRate.insufficient.label"),
      hint: t("analysis.speechRate.insufficient.hint"),
      key: "insufficient",
    };
  }
  const duration = hopSec * n;
  let peaks = 0;
  let lastPeak = -Infinity;
  for (let i = 1; i < n - 1; i++) {
    if (!store.voiced[i]) continue;
    const prev = store.db[i - 1] ?? store.db[i];
    const curr = store.db[i];
    const next = store.db[i + 1] ?? store.db[i];
    if ((curr - prev) > 1.2 && curr >= next - 0.5) {
      const t = i * hopSec;
      if (t - lastPeak >= 0.18) { peaks++; lastPeak = t; }
    }
  }
  if (!peaks) {
    const voicedFrames = store.voiced.filter(Boolean).length;
    if (voicedFrames) { peaks = Math.max(1, Math.round((voicedFrames * hopSec) / 0.22)); }
  }
  const syllPerSec = peaks / Math.max(duration, EPS);
  const wordsPerMin = syllPerSec > 0 ? (syllPerSec / 1.5) * 60 : NaN;
  if (!Number.isFinite(syllPerSec) || syllPerSec <= 0) {
    return {
      syllPerSec: NaN,
      wordsPerMin: NaN,
      label: t("analysis.speechRate.insufficient.label"),
      hint: t("analysis.speechRate.insufficient.hint"),
      key: "insufficient",
    };
  }
  let key = "fast";
  if (syllPerSec < 2.2) key = "tooSlow";
  else if (syllPerSec <= 4.2) key = "balanced";
  return {
    syllPerSec,
    wordsPerMin,
    label: t(`analysis.speechRate.${key}.label`),
    hint: t(`analysis.speechRate.${key}.hint`),
    key,
  };
}

export function analyzeConnectedSpeech(voicedArr, hopSec) {
  if (!Array.isArray(voicedArr) || !voicedArr.length) {
    return {
      ratio: NaN,
      label: t("analysis.liaison.insufficient.label"),
      hint: t("analysis.liaison.insufficient.hint"),
    };
  }
  let segments = 0;
  let inVoiced = false;
  let gapDur = 0;
  const gaps = [];
  for (let i = 0; i < voicedArr.length; i++) {
    if (voicedArr[i]) {
      if (!inVoiced) {
        segments++;
        if (gapDur > 0) { gaps.push(gapDur); gapDur = 0; }
      }
      inVoiced = true;
    } else {
      if (inVoiced) {
        inVoiced = false;
        gapDur = hopSec;
      } else if (gapDur > 0) {
        gapDur += hopSec;
      } else {
        gapDur = hopSec;
      }
    }
  }
  const totalBreaks = Math.max(0, segments - 1);
  const shortGaps = gaps.filter(g => g <= 0.16).length;
  const ratio = totalBreaks ? shortGaps / totalBreaks : (segments > 0 ? 1 : NaN);
  if (!Number.isFinite(ratio)) {
    return {
      ratio: NaN,
      label: t("analysis.liaison.insufficient.label"),
      hint: t("analysis.liaison.insufficient.hint"),
    };
  }
  let key = "weak";
  if (ratio >= 0.7) key = "strong";
  else if (ratio >= 0.4) key = "medium";
  return {
    ratio,
    label: t(`analysis.liaison.${key}.label`),
    hint: t(`analysis.liaison.${key}.hint`),
  };
}

export function analyzeIntonation(data, hopSec) {
  const metrics = computeIntonationMetrics(data, hopSec, {
    confidenceThreshold: CONFIDENCE_INCLUDE_THRESHOLD,
    voicedThreshold: CONFIDENCE_VOICED_THRESHOLD,
    eps: EPS,
  }) || {};
  const points = Array.isArray(metrics.points) ? metrics.points : [];
  const rawPoints = Array.isArray(metrics.rawPoints) ? metrics.rawPoints : [];
  const shadedRanges = Array.isArray(metrics.shadedRanges) ? metrics.shadedRanges : [];
  const slopeCount = Number(metrics.slopeSampleCount) || 0;

  if (slopeCount < 3) {
    return {
      points: [],
      rawPoints,
      shadedRanges,
      slope: NaN,
      slopeLabel: t("analysis.intonation.insufficient.slopeLabel"),
      hint: t("analysis.intonation.insufficient.slopeHint"),
      range: NaN,
      rangeHint: "",
      minHz: NaN,
      maxHz: NaN,
    };
  }

  const slope = Number.isFinite(metrics.slope) ? metrics.slope : NaN;
  const validRange = Number.isFinite(metrics.range) ? metrics.range : NaN;
  let slopeKey = "flat";
  if (slope > 12) slopeKey = "rising";
  else if (slope < -12) slopeKey = "falling";
  const slopeLabel = t(`analysis.intonation.slope.${slopeKey}.label`);
  const slopeHint = t(`analysis.intonation.slope.${slopeKey}.hint`);

  let rangeKey = "narrow";
  if (validRange >= 90) rangeKey = "rich";
  else if (validRange >= 50) rangeKey = "medium";
  const rangeHint = t(`analysis.intonation.range.${rangeKey}.hint`);
  const rangeLabel = t(`analysis.intonation.range.${rangeKey}.label`);
  const hint = slopeHint.trim() === rangeHint.trim()
    ? slopeHint.trim()
    : `${slopeHint} ${rangeHint}`.trim();

  return {
    points,
    rawPoints,
    shadedRanges,
    slope,
    slopeKey,
    slopeLabel,
    range: validRange,
    rangeKey,
    rangeLabel,
    hint,
    rangeHint,
    minHz: Number.isFinite(metrics.minHz) ? metrics.minHz : NaN,
    maxHz: Number.isFinite(metrics.maxHz) ? metrics.maxHz : NaN,
  };
}
