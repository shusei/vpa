const EPSILON = 1e-9;

export const PS_INTERVAL_MS = 50;
export const PS_MIN_HZ = 50;
export const PS_MAX_HZ = 450;
export const PS_SMOOTH_BASE_ALPHA = 0.08;
export const PS_SMOOTH_FAST_ALPHA = 0.45;
export const PS_SMOOTH_FAST_THRESHOLD_SEMITONES = 1.5;
export const PS_SMOOTH_MAX_STEP_SEMITONES = 2.4;
export const PS_SMOOTH_MEDIAN_WINDOW = 5;
export const PS_MIN_DB_FOR_PITCH = 42;
export const PS_NOISE_BUFFER_MS = 10000;
export const PS_NOISE_MIN_SAMPLES = 8;
export const PS_NOISE_CAPTURE_RANGE_DB = 18;
export const PS_MIN_DB_ABOVE_NOISE = 5;
export const PS_NOISE_GATE_MAX_BOOST_DB = 14;
export const OCTAVE_REJECT_SEMITONES = 9.5;
export const OCTAVE_WARN_SEMITONES = 6.5;
export const FLOOR_GUARD_RATIO = 0.8;
export const FLOOR_GUARD_ABSOLUTE = 120;
export const BREATH_ZCR_MUTE = 0.28;
export const BREATH_SCORE_MUTE = 0.5;
export const BREATH_ZCR_SOFT = 0.24;
export const BREATH_SCORE_SOFT = 0.4;
export const CONFIDENCE_VOICED_THRESHOLD = 0.5;
export const CONFIDENCE_INCLUDE_THRESHOLD = 0.6;
export const CONFIDENCE_LOW_CLAMP = 0.55;
export const VAD_ON_CONFIDENCE = 0.6;
export const VAD_OFF_CONFIDENCE = 0.5;
export const VAD_HOLD_OFF_MS = 120;
export const VOICE_PRESETS = {
  auto: null,
  masculine: { min: 80, max: 220 },
  neutral: { min: 120, max: 300 },
  feminine: { min: 150, max: 400 },
};
export const PITCH_RANGE_HARD = { min: PS_MIN_HZ, max: PS_MAX_HZ };
export const PITCH_PROFILE_DEFAULT = "auto";
export const DEFAULT_PITCH_RANGE = { ...VOICE_PRESETS.neutral };
export const PITCH_COUNTER_KEYS = [
  "hardMute",
  "lowConfidence",
  "octaveCorrected",
  "octaveRejected",
  "floorGuard",
];

export function clampPitchRange(range){
  const source = range || DEFAULT_PITCH_RANGE;
  const baseMin = Number(source?.min);
  const baseMax = Number(source?.max);
  const min = Math.max(
    PITCH_RANGE_HARD.min,
    Math.min(PITCH_RANGE_HARD.max - 20, Number.isFinite(baseMin) ? baseMin : DEFAULT_PITCH_RANGE.min)
  );
  const max = Math.max(
    min + 20,
    Math.min(PITCH_RANGE_HARD.max, Number.isFinite(baseMax) ? baseMax : DEFAULT_PITCH_RANGE.max)
  );
  return { min, max };
}

function createPitchCounters(){
  const counters = {};
  for (const key of PITCH_COUNTER_KEYS){
    counters[key] = 0;
  }
  return counters;
}

function resetPitchCounters(counters){
  if (!counters) return;
  for (const key of PITCH_COUNTER_KEYS){
    counters[key] = 0;
  }
}

function incrementPitchCounter(state, key){
  if (!state?.counters) return;
  if (!(key in state.counters)) return;
  state.counters[key] += 1;
}

export function createPitchPostState(){
  return {
    volumeHistory: [],
    snrHistory: [],
    medianWindow: [],
    avgWindow: [],
    lastSmooth: null,
    lastAccepted: null,
    silentStreak: 0,
    lastConfidence: 0,
    vad: { isVoiced: false, pendingOffMs: 0 },
    lastOctaveMultiplier: 1,
    octaveStableFrames: 0,
    counters: createPitchCounters(),
  };
}

export function resetPitchPostState(state){
  if (!state) return;
  state.volumeHistory.length = 0;
  state.snrHistory.length = 0;
  state.medianWindow.length = 0;
  state.avgWindow.length = 0;
  state.lastSmooth = null;
  state.lastAccepted = null;
  state.silentStreak = 0;
  state.lastConfidence = 0;
  state.lastOctaveMultiplier = 1;
  state.octaveStableFrames = 0;
  if (state.vad){
    state.vad.isVoiced = false;
    state.vad.pendingOffMs = 0;
  } else {
    state.vad = { isVoiced: false, pendingOffMs: 0 };
  }
  resetPitchCounters(state.counters);
}

function handleSilentFrame(state){
  state.silentStreak += 1;
  if (state.silentStreak >= 6){
    state.medianWindow.length = 0;
    state.avgWindow.length = 0;
    state.lastSmooth = null;
    state.lastAccepted = null;
    state.lastOctaveMultiplier = 1;
    state.octaveStableFrames = 0;
  }
}

function pushHistory(arr, value, maxLen){
  if (!Number.isFinite(value)) return;
  arr.push(value);
  if (arr.length > maxLen){
    arr.splice(0, arr.length - maxLen);
  }
}

function computeHistoryStats(arr){
  const n = Array.isArray(arr) ? arr.length : 0;
  if (!n) return { count:0, mean:NaN, sd:NaN };
  const mean = arr.reduce((a,b)=>a+b,0) / n;
  const variance = arr.reduce((a,v)=> a + (v-mean)*(v-mean), 0) / n;
  return { count:n, mean, sd: Math.sqrt(Math.max(0, variance)) };
}

function smoothPitchCandidate(candidate, range, state, frameMs){
  const stepMs = Number.isFinite(frameMs) && frameMs > 0 ? frameMs : PS_INTERVAL_MS;
  const clamped = Math.min(range.max, Math.max(range.min, candidate));
  const medianWindowSize = 5;
  state.medianWindow.push(clamped);
  if (state.medianWindow.length > medianWindowSize){
    state.medianWindow.shift();
  }
  const sorted = state.medianWindow.slice().sort((a,b)=>a-b);
  const midIdx = Math.floor(sorted.length / 2);
  const median = sorted[midIdx] ?? clamped;
  const q1 = percentileSorted(sorted, 25);
  const q3 = percentileSorted(sorted, 75);
  const iqr = Number.isFinite(q3) && Number.isFinite(q1) ? Math.max(1, q3 - q1) : null;
  let target = median;
  if (Number.isFinite(iqr)){
    const minClamp = q1 - 1.5 * iqr;
    const maxClamp = q3 + 1.5 * iqr;
    target = Math.max(minClamp, Math.min(maxClamp, target));
  }

  const prev = state.lastSmooth;
  if (Number.isFinite(prev)){
    const prevLog = Math.log2(prev);
    const targetLog = Math.log2(target);
    let delta = targetLog - prevLog;
    const baseStep = 150 / 1200; // 150 cents
    const maxLogStep = baseStep * (stepMs / 20);
    if (Number.isFinite(maxLogStep) && maxLogStep > 0){
      delta = Math.max(-maxLogStep, Math.min(maxLogStep, delta));
    }
    let next = 2 ** (prevLog + delta);
    const maxHzStep = 60 * (stepMs / 20);
    if (Number.isFinite(maxHzStep) && maxHzStep > 0){
      next = Math.max(prev - maxHzStep, Math.min(prev + maxHzStep, next));
    }
    target = next;
  }

  const smoothAlpha = 0.35;
  if (Number.isFinite(state.lastSmooth)){
    target = state.lastSmooth + smoothAlpha * (target - state.lastSmooth);
  }

  const bounded = Math.min(range.max, Math.max(range.min, target));
  state.lastSmooth = bounded;
  return bounded;
}

function correctPitchOctave(candidate, range, state, ctx = {}){
  const ref = Number.isFinite(state.lastSmooth)
    ? state.lastSmooth
    : (Number.isFinite(state.lastAccepted) ? state.lastAccepted : null);
  if (!Number.isFinite(ref)){
    return Math.min(range.max, Math.max(range.min, candidate));
  }
  const snr = Number.isFinite(ctx.snr) ? ctx.snr : (Number.isFinite(ctx.db) && Number.isFinite(ctx.ambientDb)
    ? ctx.db - ctx.ambientDb
    : NaN);
  const energyWeight = Number.isFinite(snr)
    ? Math.max(0, Math.min(1, (snr - 6) / 18))
    : 0.5;
  const voicingProb = Math.max(0, Math.min(1, Number.isFinite(state.lastConfidence) ? state.lastConfidence : 0.5));
  const multipliers = [0.5, 1, 2];
  let best = candidate;
  let bestScore = Infinity;
  let bestMultiplier = 1;
  const refLog = Math.log2(ref);
  for (const k of multipliers){
    const adjusted = candidate * k;
    if (adjusted < range.min || adjusted > range.max) continue;
    const diff = Math.abs(Math.log2(adjusted) - refLog);
    const energyFactor = 1 / (0.55 + 0.45 * energyWeight);
    const voicingFactor = 1 / (0.6 + 0.4 * voicingProb);
    const score = diff * energyFactor * voicingFactor;
    if (score < bestScore){
      bestScore = score;
      best = adjusted;
      bestMultiplier = k;
    }
  }
  const clamped = Math.min(range.max, Math.max(range.min, best));
  const changed = Math.abs(Math.log2(clamped / Math.max(candidate, 1e-9))) >= 0.5;
  if (state){
    const prevMult = Number.isFinite(state.lastOctaveMultiplier) ? state.lastOctaveMultiplier : 1;
    if (changed){
      const prevFrames = Number.isFinite(state.octaveStableFrames) ? state.octaveStableFrames : 0;
      if (!Number.isFinite(prevFrames) || prevFrames < 0) state.octaveStableFrames = 0;
      if (bestMultiplier !== prevMult || prevFrames === 0){
        incrementPitchCounter(state, "octaveCorrected");
      }
      state.octaveStableFrames = (Number.isFinite(state.octaveStableFrames) ? state.octaveStableFrames : 0) + 1;
    } else {
      state.octaveStableFrames = 0;
    }
    state.lastOctaveMultiplier = bestMultiplier;
  }
  return clamped;
}

function updateVoicedState(state, confidence, hasPitch, frameMs){
  const stepMs = Number.isFinite(frameMs) && frameMs > 0 ? frameMs : PS_INTERVAL_MS;
  const vadState = state.vad || (state.vad = { isVoiced: false, pendingOffMs: 0 });
  if (hasPitch && confidence >= VAD_ON_CONFIDENCE){
    vadState.isVoiced = true;
    vadState.pendingOffMs = 0;
  } else if (!hasPitch || confidence < VAD_OFF_CONFIDENCE){
    if (vadState.isVoiced){
      vadState.pendingOffMs += stepMs;
      if (vadState.pendingOffMs >= VAD_HOLD_OFF_MS){
        vadState.isVoiced = false;
        vadState.pendingOffMs = 0;
      }
    } else {
      vadState.pendingOffMs = Math.min(VAD_HOLD_OFF_MS, vadState.pendingOffMs + stepMs);
    }
  }
  return vadState.isVoiced;
}

export function appendPitchSample(rawHz, { db = NaN, ambientDb = NaN, spectral = null } = {}, ctx = {}){
  const state = ctx.state || createPitchPostState();
  const arrays = ctx.arrays || {};
  const rawArr = arrays.raw;
  const smoothArr = arrays.smooth;
  const voicedArr = arrays.voiced;
  const confidenceArr = arrays.confidence;
  const getRange = typeof ctx.getRange === "function" ? ctx.getRange : null;
  const staticRange = ctx.range;
  const range = clampPitchRange(getRange ? getRange() : staticRange);
  const frameMs = Number.isFinite(ctx.frameMs) ? ctx.frameMs : PS_INTERVAL_MS;

  const rawFinite = Number.isFinite(rawHz) ? rawHz : null;
  if (rawArr){ rawArr.push(rawFinite); }

  let processed = null;
  let voiced = false;
  let confidence = 0;
  const snr = Number.isFinite(db) && Number.isFinite(ambientDb) ? (db - ambientDb) : NaN;

  const debug = typeof ctx.debug === "function" ? ctx.debug : null;
  const debugInfo = debug ? {
    raw: Number.isFinite(rawHz) ? rawHz : null,
    db,
    ambientDb,
    snr: Number.isFinite(db) && Number.isFinite(ambientDb) ? (db - ambientDb) : NaN,
    candidate: null,
    corrected: null,
    processed: null,
    confidence: 0,
    hardMute: false,
    suppressed: false,
    reasons: [],
  } : null;

  if (!Number.isFinite(rawHz)){
    handleSilentFrame(state);
  } else {
    let candidate = Math.min(range.max, Math.max(range.min, rawHz));
    if (debugInfo) debugInfo.candidate = candidate;
    candidate = correctPitchOctave(candidate, range, state, { snr, db, ambientDb });
    if (debugInfo) debugInfo.corrected = candidate;
    let conf = 1;
    let hardMute = false;
    let floorMute = false;
    let octaveMute = false;
    let suppressed = false;
    const reasons = debugInfo ? debugInfo.reasons : null;

    const volStats = computeHistoryStats(state.volumeHistory);
    if (Number.isFinite(db) && volStats.count >= 6){
      const delta = db - volStats.mean;
      const sigma = Math.max(volStats.sd, 1);
      if (delta < -2 * sigma){
        suppressed = true;
        if (reasons) reasons.push("volumeLow");
      } else if (delta < -1.4 * sigma){
        conf = Math.min(conf, CONFIDENCE_LOW_CLAMP);
        if (reasons) reasons.push("volumeSoft");
      }
    }

    if (Number.isFinite(snr)){
      if (snr < 10){
        hardMute = true;
        if (reasons) reasons.push("snrLow");
      } else if (snr < 12){
        conf = Math.min(conf, CONFIDENCE_LOW_CLAMP);
        if (reasons) reasons.push("snrBorderline");
      } else if (snr < 14){
        conf = Math.min(conf, 0.7);
        if (reasons) reasons.push("snrOk");
      }
    }

    if (spectral){
      const zcr = Number(spectral.zcr);
      const breath = Number(spectral.breathiness);
      if (Number.isFinite(zcr) && Number.isFinite(breath)){
        if (zcr > BREATH_ZCR_MUTE && breath > BREATH_SCORE_MUTE){
          suppressed = true;
          if (reasons) reasons.push("breathMute");
        } else if (zcr > BREATH_ZCR_SOFT && breath > BREATH_SCORE_SOFT){
          conf = Math.min(conf, CONFIDENCE_LOW_CLAMP);
          if (reasons) reasons.push("breathSoft");
        }
      }
    }

    const prevAccepted = state.lastAccepted;
    if (Number.isFinite(prevAccepted)){
      const prevSafe = Math.max(prevAccepted, 1e-6);
      const diffSemitones = Math.abs(Math.log2(candidate / prevSafe)) * 12;
      if (diffSemitones > OCTAVE_REJECT_SEMITONES){
        hardMute = true;
        octaveMute = true;
        if (reasons) reasons.push("octaveReject");
      } else if (diffSemitones > OCTAVE_WARN_SEMITONES){
        conf = Math.min(conf, 0.6);
        if (reasons) reasons.push("octaveWarn");
      }
    }

    const floorGuardHz = Math.max(FLOOR_GUARD_ABSOLUTE, range.min * FLOOR_GUARD_RATIO);
    const refPitch = Number.isFinite(prevAccepted) ? prevAccepted : state.lastSmooth;
    if (candidate < floorGuardHz){
      if (Number.isFinite(refPitch) && (refPitch / Math.max(candidate, 1)) >= 1.6){
        hardMute = true;
        floorMute = true;
        if (reasons) reasons.push("floorGuard");
      } else {
        conf = Math.min(conf, CONFIDENCE_LOW_CLAMP);
        if (reasons) reasons.push("floorWarn");
      }
    } else if (candidate < range.min * 0.9){
      conf = Math.min(conf, CONFIDENCE_LOW_CLAMP);
      if (reasons) reasons.push("rangeLow");
    }

    if (candidate > range.max * 1.18){
      hardMute = true;
      if (reasons) reasons.push("rangeHigh");
    } else if (candidate > range.max * 1.05){
      conf = Math.min(conf, 0.6);
      if (reasons) reasons.push("rangeWarn");
    }

    if (!hardMute){
      confidence = Math.max(0, Math.min(1, conf));
      processed = smoothPitchCandidate(candidate, range, state, frameMs);
      state.lastAccepted = processed;
      state.silentStreak = 0;
      if (suppressed){
        confidence = 0;
      }
      if (confidence >= CONFIDENCE_INCLUDE_THRESHOLD){
        pushHistory(state.volumeHistory, db, 220);
        pushHistory(state.snrHistory, snr, 220);
      }
      if (confidence > 0 && confidence < 0.75){
        incrementPitchCounter(state, "lowConfidence");
      }
    } else {
      handleSilentFrame(state);
      incrementPitchCounter(state, "hardMute");
      if (octaveMute) incrementPitchCounter(state, "octaveRejected");
      if (floorMute) incrementPitchCounter(state, "floorGuard");
    }

    if (debugInfo){
      debugInfo.hardMute = hardMute;
      debugInfo.confidence = confidence;
      if (suppressed) debugInfo.suppressed = true;
    }
  }

  const finalValue = Number.isFinite(processed) ? processed : null;
  if (smoothArr){ smoothArr.push(finalValue); }
  state.lastConfidence = confidence;
  const hasPitch = Number.isFinite(finalValue);
  const voicedState = updateVoicedState(state, confidence, hasPitch, frameMs);
  voiced = voicedState;
  if (voicedArr){ voicedArr.push(voiced); }
  if (confidenceArr){ confidenceArr.push(confidence); }

  if (debugInfo){
    debugInfo.processed = finalValue;
    debugInfo.voiced = voiced;
    debugInfo.finalConfidence = confidence;
    debug(debugInfo);
  }
  return { processed: finalValue, voiced, confidence };
}

export function makeNoiseTracker(){
  const buf = [];
  const maxSamples = Math.max(1, Math.round(PS_NOISE_BUFFER_MS / PS_INTERVAL_MS));
  return {
    reset(){ buf.length = 0; },
    capture(db){
      if (!Number.isFinite(db)) return;
      const capped = Math.min(db, PS_MIN_DB_FOR_PITCH + PS_NOISE_CAPTURE_RANGE_DB);
      buf.push(capped);
      if (buf.length > maxSamples) buf.shift();
    },
    getAmbient(){
      if (buf.length < PS_NOISE_MIN_SAMPLES) return null;
      const sorted = buf.slice().sort((a,b)=>a-b);
      const val = percentileSorted(sorted, 20);
      return Number.isFinite(val) ? val : null;
    },
    getThreshold(){
      let threshold = PS_MIN_DB_FOR_PITCH;
      const ambient = this.getAmbient();
      if (ambient != null){
        const dynamic = ambient + PS_MIN_DB_ABOVE_NOISE;
        const maxBoosted = PS_MIN_DB_FOR_PITCH + PS_NOISE_GATE_MAX_BOOST_DB;
        threshold = Math.max(threshold, Math.min(maxBoosted, dynamic));
      }
      return { threshold, ambient };
    },
    shouldDetect(db, wasVoiced){
      if (!Number.isFinite(db)) return { detect:false, threshold:PS_MIN_DB_FOR_PITCH, ambient:null };
      const { threshold, ambient } = this.getThreshold();
      if (db >= threshold) return { detect:true, threshold, ambient };
      if (wasVoiced && db >= PS_MIN_DB_FOR_PITCH){
        return { detect:true, threshold, ambient, hysteresis:true };
      }
      return { detect:false, threshold, ambient };
    },
  };
}

export function filterPitchForStats(samples){
  if (!Array.isArray(samples)) return [];
  const finite = samples.filter(Number.isFinite);
  if (finite.length < 3) return finite.slice();

  const logVals = finite.map((v)=>Math.log2(Math.max(v, PS_MIN_HZ)));
  const sortedLog = logVals.slice().sort((a,b)=>a-b);
  const medianLog = percentileSorted(sortedLog, 50);
  const diffSemitones = logVals.map((v)=>Math.abs(v - medianLog) * 12);
  const sortedDiffs = diffSemitones.slice().sort((a,b)=>a-b);
  const mad = percentileSorted(sortedDiffs, 50);
  const tolerance = Math.max(0.5, (mad || 0) * 3);
  const filtered = finite.filter((_, idx)=> diffSemitones[idx] <= tolerance);

  const minKeep = Math.max(3, Math.ceil(finite.length * 0.6));
  if (filtered.length < minKeep) return finite;
  return filtered;
}

export function makeStats(arr){
  if (!arr.length) return { avg:NaN, med:NaN, p95:NaN, p05:NaN, sd:NaN };
  const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
  const sorted = arr.slice().sort((x,y)=>x-y);
  const med  = percentileSorted(sorted, 50);
  const p95  = percentileSorted(sorted, 95);
  const p05  = percentileSorted(sorted, 5);
  const sd   = Math.sqrt(arr.reduce((a,v)=> a + (v-mean)*(v-mean), 0) / arr.length);
  return { avg:mean, med, p95, p05, sd };
}

export function percentileSorted(sorted, p){
  if (!sorted.length) return NaN;
  const i = (p/100) * (sorted.length - 1);
  const i0 = Math.floor(i);
  const i1 = Math.min(sorted.length-1, i0+1);
  const t = i - i0;
  return sorted[i0]*(1-t) + sorted[i1]*t;
}

export function computeIntonationMetrics(data, hopSec, {
  confidenceThreshold = CONFIDENCE_INCLUDE_THRESHOLD,
  voicedThreshold = CONFIDENCE_VOICED_THRESHOLD,
  eps = EPSILON,
} = {}){
  const processed = Array.isArray(data?.processed) ? data.processed : [];
  const raw = Array.isArray(data?.raw) ? data.raw : [];
  const confidence = Array.isArray(data?.confidence) ? data.confidence : [];
  const voicedArr = Array.isArray(data?.voiced) ? data.voiced : [];
  const step = Number.isFinite(hopSec) ? hopSec : (PS_INTERVAL_MS/1000);

  const points=[];
  const rawPoints=[];
  const shadedRanges=[];
  let minHz=Infinity, maxHz=-Infinity;

  const pushRange=(type, start, end)=>{
    if (end <= start) return;
    const last = shadedRanges.length ? shadedRanges[shadedRanges.length-1] : null;
    if (last && last.type === type && Math.abs(last.end - start) <= step*1.5){
      last.end = end;
    } else {
      shadedRanges.push({ type, start, end });
    }
  };

  for (let i=0;i<processed.length;i++){
    const t = i * step;
    const hz = processed[i];
    const conf = confidence[i] ?? 0;
    if (Number.isFinite(hz)){
      points.push({ t, hz, confidence: conf });
      if (conf >= confidenceThreshold){
        if (hz < minHz) minHz = hz;
        if (hz > maxHz) maxHz = hz;
      }
    }
    const rawHz = raw[i];
    if (Number.isFinite(rawHz)) rawPoints.push({ t, hz: rawHz });
    const isVoiced = voicedArr[i] ?? (Number.isFinite(hz) && conf >= voicedThreshold);
    if (!isVoiced){
      pushRange("mute", t, t + step);
    } else if (conf < confidenceThreshold){
      pushRange("soft", t, t + step);
    }
  }

  const slopePoints = points.filter(p=> Number.isFinite(p.hz) && (p.confidence ?? 0) >= confidenceThreshold);
  let slope = NaN;
  if (slopePoints.length >= 3){
    const n = slopePoints.length;
    let sumT=0, sumH=0, sumTT=0, sumTH=0;
    for (const {t,hz} of slopePoints){
      sumT += t; sumH += hz; sumTT += t*t; sumTH += t*hz;
    }
    const denom = (n*sumTT - sumT*sumT);
    slope = denom === 0 ? NaN : (n*sumTH - sumT*sumH) / Math.max(denom, eps);
  }
  const validMin = Number.isFinite(minHz) ? minHz : NaN;
  const validMax = Number.isFinite(maxHz) ? maxHz : NaN;
  const range = (Number.isFinite(validMin) && Number.isFinite(validMax)) ? (validMax - validMin) : NaN;

  return {
    points,
    rawPoints,
    shadedRanges,
    minHz: validMin,
    maxHz: validMax,
    range,
    slope,
    slopeSampleCount: slopePoints.length,
    step,
    confidenceThreshold,
  };
}

export function fmt1(x){
  return Number.isFinite(x) ? (Math.round(x*10)/10).toFixed(1) : "—";
}

export function logPostProcessingDiagnostics(state, { spread, intonationRange }){
  try{
    const counters = state?.counters || {};
    console.info(`[音高後處理] 音高散布（p95−p05）：${fmt1(spread)} Hz`);
    console.info(`[音高後處理] 音高動態（處理後曲線 max−min）：${fmt1(intonationRange)} Hz`);
    console.info(`[音高後處理] 被靜音幀數（音量 < 平均−2σ / SNR < 10 dB / 高 ZCR＋氣聲 / 半頻疑慮）：${counters.hardMute ?? 0}`);
    console.info(`[音高後處理] 低信心幀數（SNR 10–14 dB 或能量邊界）：${counters.lowConfidence ?? 0}`);
    console.info(`[音高後處理] 八度修正次數（>|9 半音| 跳躍插補）：${counters.octaveCorrected ?? 0}`);
    console.info(`[音高後處理] 低頻防呆剔除幀數（< 地板 × 0.8 視為外點）：${counters.floorGuard ?? 0}`);
  }catch(err){
    console.warn("[logPostProcessingDiagnostics]", err);
  }
}

export { EPSILON };
