import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";
import { t } from "./i18n.js";
import {
    MODEL_ID, TARGET_SR, WARN_LONG_SEC, MAX_WHOLE_SEC, STREAM_WIN_CAND, STREAM_HOP_S,
    VAD_MIN_APPLY_SEC, VAD_FRAME_MS, VAD_HOP_MS, VAD_PAD_MS, VAD_MIN_SEG_MS, VAD_MIN_VOICED_SEC, VAD_SILENCE_RATIO_TO_APPLY,
    MEDIA_RECORDER_DATA_TIMEOUT_MS, THREAD_STORAGE_KEY, EPS,
    PITCH_RANGE_KEY, PITCH_PROFILE_KEY,
    AUTO_WIDE_RANGE, AUTO_SAMPLE_MS, AUTO_MIN_VALID_FRAMES, AUTO_REEVAL_WINDOW_MS, AUTO_INVALID_RATIO_LIMIT, AUTO_OCTAVE_SPIKE_LIMIT,
    AUTO_HYSTERESIS_UP, AUTO_HYSTERESIS_DOWN,
    PITCH_RUNTIME_BASE_BUDGET_MS, PITCH_RUNTIME_OVER_BUDGET_LIMIT, PITCH_RUNTIME_RECOVERY_MS, PITCH_RUNTIME_OFFLINE_MULTIPLIER,
    PITCH_RUNTIME_MIN_BUDGET_MS, PITCH_RUNTIME_MAX_BUDGET_MS, PITCH_RETRY_MIN_INTERVAL_MS, PITCH_RETRY_COOLDOWN_MS, PITCH_RETRY_ERROR_COOLDOWN_MS,
    PITCH_RETRY_ERROR_GUARD_MS, PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS
} from "./config.js";
import {
    fmtSec, clamp01, toMap, microYield, percentile, smoothMask, startHeartbeat, stopHeartbeat, isOOMError
} from "./utils.js";
import { mixChannelDataToMono } from "./audio-utils.js";
import { detectPitchACF, detectPitchYinLite, estimateSpectralFeatures, trimYinBuffers } from "./pitch.js";
import {
    appendPitchSample as sharedAppendPitchSample,
    createPitchPostState,
    resetPitchPostState,
    PS_INTERVAL_MS, PS_MIN_HZ, PS_MAX_HZ, CONFIDENCE_INCLUDE_THRESHOLD,
    DEFAULT_PITCH_RANGE, VOICE_PRESETS
} from "./pitch-shared.js";
import { computeAdvancedSummary } from "./analysis/advanced-summary.js";

// Callbacks interface
let callbacks = {
    onStatus: () => { },
    onRender: () => { },
    onUpdateRealtimeMonitor: () => { },
    onFinishStreamStats: () => { },
    onStartDrawLoop: () => { },
};

export function setCallbacks(cbs) {
    callbacks = { ...callbacks, ...cbs };
}

// Global State
let clf = null;
let currentDevice = 'wasm';
let mediaRecorder = null;
let chunks = [];
let lastRecordingUrl = null;

export function hasLastRecording() { return !!lastRecordingUrl; }
export function getLastRecordingUrl() { return lastRecordingUrl; }
export let isRecording = false;
export let busy = false;

// Pitch & Analysis State
export const pitchPostState = createPitchPostState();

export const autoRangeState = {
    stage: "idle",
    sampleValues: [],
    sampleDurationMs: 0,
    timelineMs: 0,
    lastUpdateMs: 0,
    currentRange: null,
    lastMedian: null,
    windowFrames: [],
    windowMs: 0,
    windowInvalidMs: 0,
    octaveEvents: [],
    prevOctaveCount: 0,
};

export const pitchStrategyState = {
    activeKey: null,
    lockUntil: 0,
    lockReason: null,
    lockReasonDetail: null,
    overBudgetStreak: 0,
    lastSwitch: 0,
    lastOverBudget: 0,
    runtimeEwma: 0,
    lastEnableAttempt: 0,
    lockedAt: 0,
    lockDuration: 0,
    lockContext: null,
    autoRetryUntil: 0,
};

// Pitch Stream State
let psCtx = null;
let psSrc = null;
let psProc = null;
export let psRunning = false;
let psRAF = null;

// Data Buffers
export const psHz = [];
export const psHzSmooth = [];
export const psDb = [];
export const psVoiced = [];
export const psConfidence = [];

// Offline Feature Store
export const offlineFeatureStore = {
    frameSec: 0,
    pitchRaw: [],
    pitchProcessed: [],
    pitchConfidence: [],
    db: [],
    voiced: [],
    formants: [],
    tilt: [],
    breathiness: [],
    energy: [],
    zcr: [],
    advSummary: null, // Store computed summary here
};

// Noise Trackers
class NoiseTracker {
    constructor() { this.reset(); }
    reset() { this.minDb = 100; this.sum = 0; this.count = 0; }
    capture(db) {
        if (!Number.isFinite(db)) return;
        if (db < this.minDb) this.minDb = db;
        this.sum += db; this.count++;
    }
    get ambient() { return this.count > 10 ? this.minDb : 30; }
    shouldDetect(db, wasVoiced) {
        const amb = this.ambient;
        const thr = Math.max(amb + 12, 40); // Gate
        if (db < thr && !wasVoiced) return { detect: false, ambient: amb };
        return { detect: true, ambient: amb };
    }
}
const psRealtimeNoiseTracker = new NoiseTracker();
const psOfflineNoiseTracker = new NoiseTracker();

// Config
env.allowLocalModels = false;
env.useBrowserCache = true;

// --- Helpers ---
function setStatus(msg, showLoading) { callbacks.onStatus(msg, showLoading); }
function render(pf, pm) { callbacks.onRender(pf, pm); }

// --- Model ---
export async function ensurePipeline() {
    if (clf) return clf;
    setStatus(t("status.modelLoading"), true);
    const progress_callback = (p) => {
        if (!p) return;
        let pct = null;
        if (typeof p.loadedBytes === 'number' && typeof p.totalBytes === 'number' && p.totalBytes > 0) pct = p.loadedBytes / p.totalBytes;
        else if (typeof p.progress === 'number' && isFinite(p.progress)) pct = p.progress;
        const label = p.status || t("status.modelDownloading");
        if (pct == null) setStatus(`${label}…`, true);
        else setStatus(`${label} ${Math.min(99, Math.max(0, Math.floor(pct * 100)))}% …`, true);
    };

    let threads = 1;
    try {
        const saved = localStorage.getItem(THREAD_STORAGE_KEY);
        if (saved) threads = parseInt(saved, 10) || 1;
    } catch { }
    env.backends.onnx.wasm.numThreads = threads;

    const device = (typeof navigator !== 'undefined' && navigator.gpu) ? 'webgpu' : 'wasm';
    clf = await pipeline("audio-classification", MODEL_ID, { progress_callback, device });
    currentDevice = device;
    setStatus(t("status.modelReady", { device }));
    return clf;
}

import { transcodeToFloat32 } from "./ffmpeg-transcode.js";

// --- Audio Decoding ---
export async function decodeSmartToFloat32(blobOrFile, targetSR) {
    setStatus(t("status.webaudioDecode"), true);
    try {
        return await decodeViaWebAudio(blobOrFile, targetSR);
    } catch (err) {
        console.warn("[decode] WebAudio failed, trying FFmpeg...", err);
        try {
            setStatus(t("status.ffmpegDecode"), true);
            return await transcodeToFloat32(blobOrFile, targetSR, (ev) => {
                if (ev.type === "transcode-progress") {
                    setStatus(t("status.ffmpegProgress", { progress: Math.round(ev.progress * 100) }), true);
                }
            });
        } catch (ffmpegErr) {
            console.error("[decode] FFmpeg failed", ffmpegErr);
            throw err; // Throw original WebAudio error or FFmpeg error? Usually better to throw the last one or a combined one.
        }
    }
}

async function decodeViaWebAudio(blobOrFile, targetSR = 16000) {
    const arrayBuf = await blobOrFile.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    let offline = null;
    try {
        let audioBuf;
        try {
            audioBuf = await ctx.decodeAudioData(arrayBuf);
        } catch (err) {
            audioBuf = await new Promise((resolve, reject) => {
                try { ctx.decodeAudioData(arrayBuf.slice(0), resolve, reject); } catch (e) { reject(e); }
            });
        }
        const mono = ctx.createBuffer(1, audioBuf.length, audioBuf.sampleRate);
        const outCh = mono.getChannelData(0);
        const channels = [];
        for (let i = 0; i < audioBuf.numberOfChannels; i++) {
            const chData = audioBuf.getChannelData(i);
            if (chData) channels.push(chData);
        }
        mixChannelDataToMono(channels, outCh);

        let out;
        if (audioBuf.sampleRate === targetSR) {
            out = outCh.slice(0);
        } else {
            offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
            const src = offline.createBufferSource();
            src.buffer = mono; src.connect(offline.destination); src.start(0);
            const rendered = await offline.startRendering();
            out = rendered.getChannelData(0).slice(0);
        }
        return { float32: out, sr: targetSR, durationSec: out.length / targetSR };
    } finally {
        try { await ctx.close(); } catch { }
    }
}

// --- Analysis ---
let analysisSeq = 0;
let activeAnalysisToken = 0;

function startAnalysisRun(source) {
    analysisSeq++;
    activeAnalysisToken = analysisSeq;
    return activeAnalysisToken;
}
function isAnalysisActive(token) { return token === activeAnalysisToken; }
function finishAnalysisRun(token) { if (isAnalysisActive(token)) activeAnalysisToken = 0; }

export async function handleFileOrBlob(fileOrBlob, source = "upload") {
    const token = startAnalysisRun(source);
    let decoded = null;
    try {
        setStatus(t("status.decoding"), true);
        decoded = await decodeSmartToFloat32(fileOrBlob, TARGET_SR);
        if (!isAnalysisActive(token)) return;
        let { float32, sr, durationSec } = decoded;

        offlineExtractStreamMetrics(float32, sr, false);

        if (durationSec > WARN_LONG_SEC) {
            setStatus(t("status.warnLong", { duration: fmtSec(durationSec) }), true);
            await microYield();
            if (!isAnalysisActive(token)) return;
        }

        const vad = maybeApplyAdaptiveVAD(float32, sr);
        if (vad && vad.used) {
            float32 = vad.arr; durationSec = vad.keptSec;
            setStatus(t("status.vadApplied", { ratio: Math.round((1 - vad.keptSec / decoded.durationSec) * 100), duration: fmtSec(durationSec) }), true);
            offlineExtractStreamMetrics(float32, sr, true);
            await microYield();
            if (!isAnalysisActive(token)) return;
        }

        if (!isAnalysisActive(token)) return;

        if (durationSec <= MAX_WHOLE_SEC) {
            await analyzeWhole(float32, sr, durationSec, token);
        } else {
            await analyzeStreamed(float32, sr, durationSec, t("status.streamingSwitch", { limit: MAX_WHOLE_SEC }), token);
        }

        if (!isAnalysisActive(token)) return;
        callbacks.onFinishStreamStats();
    } catch (e) {
        console.error("[handleFileOrBlob]", e);
        if (isAnalysisActive(token)) {
            setStatus(t("status.errorPrefix", { message: e?.message || t("status.decodeFailure") }));
        }
    } finally {
        finishAnalysisRun(token);
    }
}

async function analyzeWhole(float32, sr, durationSec, token) {
    if (!isAnalysisActive(token)) return;
    const model = await ensurePipeline();
    if (!isAnalysisActive(token)) return;

    const started = performance.now();
    startHeartbeat(() => {
        if (!isAnalysisActive(token)) return;
        const elapsed = (performance.now() - started) / 1000;
        setStatus(t("status.analyzeWhole", { duration: fmtSec(durationSec), elapsed: fmtSec(elapsed) }), true);
    });

    try {
        const res = await model(float32, { sampling_rate: sr, topk: 2 });
        if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
        const map = toMap(res);
        render(map.female || 0, map.male || 0);
        setStatus(t("status.analyzeWholeDone"));
    } catch (err) {
        if (isOOMError(err)) {
            console.warn("[analyzeWhole] OOM → switch to streamed mode…");
            stopHeartbeat();
            if (isAnalysisActive(token)) {
                await analyzeStreamed(float32, sr, durationSec, t("status.analyzeWholeOOM"), token);
            }
            return;
        }
        console.error("[analyzeWhole]", err);
        setStatus(t("status.analyzeWholeFailed"));
    } finally { stopHeartbeat(); }
}

async function analyzeStreamed(float32, sr, durationSec, reason = t("status.streamingDefaultReason"), token) {
    if (!isAnalysisActive(token)) return;
    const model = await ensurePipeline();
    if (!isAnalysisActive(token)) return;

    const strategy = pickStreamStrategy(durationSec);
    const reasonBits = [reason, strategy.label].filter(Boolean);
    const reasonLabel = reasonBits.join("｜");

    let lastErr = null;
    for (const winSec of strategy.wins) {
        try {
            await runStreamedWithWindow(model, float32, sr, durationSec, winSec, strategy.hop, reasonLabel || reason, token);
            if (!isAnalysisActive(token)) return;
            return;
        } catch (e) {
            lastErr = e;
            if (isOOMError(e)) { console.warn(`[streamed] OOM at win=${winSec}s → downshift`); continue; }
            else { console.error(`[streamed] error at win=${winSec}s`, e); break; }
        }
    }
    console.error("[analyzeStreamed] failed", lastErr);
    if (isAnalysisActive(token)) {
        setStatus(t("status.analyzeStreamFailed"));
    }
}

async function runStreamedWithWindow(model, float32, sr, durationSec, WIN_S, HOP_S, reason, token) {
    if (!isAnalysisActive(token)) return;
    const win = Math.max(1, Math.floor(WIN_S * sr));
    const hop = Math.max(1, Math.floor(HOP_S * sr));

    const chunks = [];
    for (let s = 0; s < float32.length; s += hop) {
        const e = Math.min(s + win, float32.length);
        if (e - s < Math.floor(0.5 * sr)) break;
        chunks.push([s, e]);
        if (e === float32.length) break;
    }
    if (!chunks.length) chunks.push([0, Math.min(win, float32.length)]);

    let avgMs = 0, processedSec = 0;
    let logitSum = 0, wSum = 0;

    const started = performance.now();
    startHeartbeat(() => {
        if (!isAnalysisActive(token)) return;
        const elapsed = (performance.now() - started) / 1000;
        const pct = processedSec > 0 ? Math.min(99, Math.round((processedSec / durationSec) * 100)) : 0;
        setStatus(t("status.analyzeStream", { win: WIN_S, step: HOP_S, reason, progress: pct, elapsed: fmtSec(elapsed) }), true);
    });

    try {
        for (let i = 0; i < chunks.length; i++) {
            if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
            const [s0, s1] = chunks[i];
            const seg = float32.subarray(s0, s1);
            const dur = (s1 - s0) / sr;

            const t0 = performance.now();
            const out = await model(seg, { sampling_rate: sr, topk: 2 });
            if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
            const dt = performance.now() - t0;
            avgMs = avgMs === 0 ? dt : (avgMs * 0.65 + dt * 0.35);

            const map = toMap(out);
            const pf = clamp01(map.female || EPS);
            const pm = clamp01(map.male || EPS);
            const logit = Math.log(pf) - Math.log(pm);

            logitSum += logit * dur; wSum += dur;

            const logitAvg = logitSum / Math.max(wSum, EPS);
            const pf_now = 1 / (1 + Math.exp(-logitAvg));
            const pm_now = 1 - pf_now;
            render(pf_now, pm_now);

            processedSec = Math.min(durationSec, (s1 / sr));
            const remain = chunks.length - i - 1;
            const etaSec = (remain * (avgMs / 1000));
            const pct = Math.round(((i + 1) / chunks.length) * 100);
            setStatus(t("status.analyzeStreamChunk", { win: WIN_S, current: i + 1, total: chunks.length, progress: pct, done: fmtSec(processedSec), totalDuration: fmtSec(durationSec), eta: fmtSec(etaSec) }), true);
            await microYield();
            if (!isAnalysisActive(token)) { stopHeartbeat(); return; }
        }
        const logitAvg = logitSum / Math.max(wSum, EPS);
        const pf = 1 / (1 + Math.exp(-logitAvg));
        const pm = 1 - pf;
        render(pf, pm);
        if (isAnalysisActive(token)) {
            setStatus(t("status.analyzeStreamDone"));
        }
    } finally { stopHeartbeat(); }
}

const STREAM_STRATEGY_DEFAULT = Object.freeze({
    hop: STREAM_HOP_S,
    wins: [...STREAM_WIN_CAND],
    label: ""
});

function pickStreamStrategy(durationSec) {
    if (!Number.isFinite(durationSec) || durationSec <= MAX_WHOLE_SEC) {
        return STREAM_STRATEGY_DEFAULT;
    }
    const dedupeWins = (wins) => {
        const seen = new Set();
        const out = [];
        for (const w of wins) {
            const key = w.toFixed(2);
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(w);
        }
        return out;
    };
    const gpuWins = dedupeWins([18, 12, ...STREAM_WIN_CAND, 4]);
    const gpuWinsLong = dedupeWins([24, 18, 12, ...STREAM_WIN_CAND, 4]);
    const wasmWins = dedupeWins([12, ...STREAM_WIN_CAND, 4]);

    if (currentDevice === "webgpu") {
        if (durationSec >= 600) {
            return { hop: 6, wins: gpuWinsLong, label: t("status.strategyGpu6") };
        }
        return { hop: 4, wins: gpuWins, label: t("status.strategyGpu4") };
    }
    if (durationSec >= 420) {
        return { hop: 4, wins: wasmWins, label: t("status.strategyCpu4") };
    }
    if (durationSec >= 240) {
        return { hop: 3.5, wins: wasmWins, label: t("status.strategyCpu35") };
    }
    return STREAM_STRATEGY_DEFAULT;
}

function maybeApplyAdaptiveVAD(float32, sr) {
    const dur = float32.length / sr;
    if (dur < VAD_MIN_APPLY_SEC) return null;

    const frame = Math.max(1, Math.floor(sr * (VAD_FRAME_MS / 1000)));
    const hop = Math.max(1, Math.floor(sr * (VAD_HOP_MS / 1000)));
    const pad = Math.max(0, Math.floor(sr * (VAD_PAD_MS / 1000)));
    const minSeg = Math.max(1, Math.floor(sr * (VAD_MIN_SEG_MS / 1000)));

    const energies = [];
    for (let s = 0; s + frame <= float32.length; s += hop) {
        let acc = 0; for (let i = 0; i < frame; i++) { const v = float32[s + i]; acc += v * v; }
        energies.push(acc / frame);
    }
    if (energies.length < 5) return null;

    const thr = Math.max(1e-7, percentile(energies, 20) * 1.5);
    const voicedMask = energies.map(e => e > thr);
    smoothMask(voicedMask, 3);

    const segs = [];
    let i = 0;
    while (i < voicedMask.length) {
        while (i < voicedMask.length && !voicedMask[i]) i++;
        if (i >= voicedMask.length) break;
        let j = i; while (j < voicedMask.length && voicedMask[j]) j++;
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
    for (const [s0, s1] of segs) { out.set(float32.subarray(s0, s1), offset); offset += (s1 - s0); }
    return { used: true, arr: out, keptSec, segs };
}

// --- Pitch Stream ---
function appendPitchSample(rawHz, meta = {}, opts = {}) {
    const frameMs = Number.isFinite(opts?.dtMs) ? opts.dtMs : PS_INTERVAL_MS;
    const result = sharedAppendPitchSample(rawHz, meta, {
        state: pitchPostState,
        arrays: {
            raw: psHz,
            smooth: psHzSmooth,
            voiced: psVoiced,
            confidence: psConfidence,
        },
        getRange: getPitchDetectorRange,
        frameMs,
    });
    handleAutoRangeFrame(result, { dtMs: frameMs });
    return result;
}

function runPitchDetection(input, sampleRate, opts) {
    const range = getPitchDetectorRange();
    return detectPitchYinLite(input, sampleRate, range);
}

function applyDbCalibration(rawDb) {
    return { value: rawDb };
}

function maybeEnableAdvancedPitch(context, opts) {
    // Placeholder
}

function bandLabel(hz) {
    if (!hz) return "—";
    if (hz < 85) return t("pitchBands.bandLow");
    if (hz < 165) return t("pitchBands.bandBlue");
    if (hz < 180) return t("pitchBands.bandNeutral");
    if (hz < 310) return t("pitchBands.bandPink");
    if (hz < 450) return t("pitchBands.bandHigh");
    if (hz <= PS_MAX_HZ) return t("pitchBands.bandFalsetto");
    return t("pitchBands.bandUnknown");
}

export async function startPitchStream(userMediaStream) {
    try {
        isRecording = true; // Set immediately to prevent re-entry
        psHz.length = 0; psHzSmooth.length = 0; psDb.length = 0; psVoiced.length = 0; psConfidence.length = 0;
        resetPitchPostState(pitchPostState);
        psRealtimeNoiseTracker.reset();
        startAutoRangeSession({ preserveRange: false });

        maybeEnableAdvancedPitch("realtime", { allowRetry: true });

        const Ctx = window.AudioContext || window.webkitAudioContext;
        psCtx = new Ctx();
        psSrc = psCtx.createMediaStreamSource(userMediaStream);

        psSrc = psCtx.createMediaStreamSource(userMediaStream);

        try {
            mediaRecorder = new MediaRecorder(userMediaStream);
            chunks = [];
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunks.push(e.data);
            };
            mediaRecorder.start();
        } catch (mrErr) {
            console.warn("[startPitchStream] MediaRecorder failed, continuing without recording support", mrErr);
            mediaRecorder = null;
        }

        psProc = psCtx.createScriptProcessor(2048, 1, 1);
        const sampleRate = psCtx.sampleRate;

        let lastTick = 0;
        psProc.onaudioprocess = (ev) => {
            const input = ev.inputBuffer.getChannelData(0);
            const rms = Math.sqrt(input.reduce((a, v) => a + v * v, 0) / Math.max(1, input.length));
            const rawDb = 20 * Math.log10(Math.max(rms, 1e-6)) + 100;
            const { value: db } = applyDbCalibration(rawDb);
            const wasVoiced = psVoiced.length ? psVoiced[psVoiced.length - 1] : false;
            let hz = null;
            let spectral = null;
            const gate = psRealtimeNoiseTracker.shouldDetect(db, wasVoiced);
            if (gate.detect) {
                const candHz = runPitchDetection(input, sampleRate, { context: "realtime" });
                if (candHz != null) {
                    hz = candHz;
                    const range = getPitchDetectorRange();
                    spectral = estimateSpectralFeatures(input, sampleRate, range);
                } else {
                    psRealtimeNoiseTracker.capture(db);
                }
            } else {
                psRealtimeNoiseTracker.capture(db);
            }
            const now = performance.now();
            if (now - lastTick >= PS_INTERVAL_MS) {
                psDb.push(db);
                const { processed } = appendPitchSample(
                    hz ?? null,
                    { db, ambientDb: gate.ambient, spectral },
                    { dtMs: PS_INTERVAL_MS }
                );
                const displayHz = Number.isFinite(processed)
                    ? processed
                    : (Number.isFinite(hz) ? hz : null);
                const maxN = Math.round(15000 / PS_INTERVAL_MS);
                if (psDb.length > maxN) {
                    psDb.shift(); psHz.shift(); psHzSmooth.shift(); psVoiced.shift(); psConfidence.shift();
                }
                lastTick = now;

                callbacks.onUpdateRealtimeMonitor({
                    hz: displayHz,
                    db,
                    band: bandLabel(displayHz),
                    spectral
                });
            }
        };

        psSrc.connect(psProc); psProc.connect(psCtx.destination);
        psRunning = true;
        // isRecording is already set to true at the beginning
        callbacks.onStartDrawLoop();
    } catch (e) {
        console.error("[startPitchStream]", e);
        isRecording = false; // Reset on error
        psRunning = false;
        throw e; // Re-throw for main.js to catch
    }
}

export async function stopPitchStream() {
    psRunning = false;
    isRecording = false;
    if (psRAF) { cancelAnimationFrame(psRAF); psRAF = null; }

    const stopPromise = new Promise(resolve => {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: "audio/webm" });
                if (lastRecordingUrl) URL.revokeObjectURL(lastRecordingUrl);
                lastRecordingUrl = URL.createObjectURL(blob);
                chunks = [];
                resolve();
                // Trigger analysis
                handleFileOrBlob(blob, "record");
            };
            mediaRecorder.stop();
        } else {
            resolve();
        }
    });

    try {
        psProc?.disconnect(); psSrc?.disconnect();
        psCtx?.close();
    } catch { } finally {
        psProc = null; psSrc = null; psCtx = null;
        mediaRecorder = null;
    }
    return stopPromise;
}

function offlineExtractStreamMetrics(float32, sr, append = false) {
    try {
        if (!append) {
            psHz.length = 0; psHzSmooth.length = 0; psDb.length = 0; psVoiced.length = 0; psConfidence.length = 0;
            offlineFeatureStore.frameSec = 0;
            offlineFeatureStore.pitchRaw.length = 0;
            offlineFeatureStore.pitchProcessed.length = 0;
            offlineFeatureStore.pitchConfidence.length = 0;
            offlineFeatureStore.db.length = 0;
            offlineFeatureStore.voiced.length = 0;
            offlineFeatureStore.formants.length = 0;
            offlineFeatureStore.tilt.length = 0;
            offlineFeatureStore.breathiness.length = 0;
            offlineFeatureStore.energy.length = 0;
            offlineFeatureStore.zcr.length = 0;
            offlineFeatureStore.advSummary = null;

            resetPitchPostState(pitchPostState);
            psOfflineNoiseTracker.reset();
            startAutoRangeSession({ preserveRange: false });
        }
        if (!append) maybeEnableAdvancedPitch("offline", { allowRetry: true });
        const step = Math.max(1, Math.floor((PS_INTERVAL_MS / 1000) * sr));
        const frame = Math.min(Math.floor(0.08 * sr), 8192);
        offlineFeatureStore.frameSec = step / sr;
        for (let i = 0; i + frame <= float32.length; i += step) {
            const seg = float32.subarray(i, i + frame);
            const rawDb = 20 * Math.log10(Math.max(rms(seg, 0, seg.length), 1e-6)) + 100;
            const { value: db } = applyDbCalibration(rawDb);
            const wasVoiced = psVoiced.length ? psVoiced[psVoiced.length - 1] : false;
            let hz = null;
            let spectral = null;
            const gate = psOfflineNoiseTracker.shouldDetect(db, wasVoiced);
            if (gate.detect) {
                const candHz = runPitchDetection(seg, sr, { context: "offline" });
                if (candHz != null) {
                    hz = candHz;
                    const range = getPitchDetectorRange();
                    spectral = estimateSpectralFeatures(seg, sr, range);
                } else {
                    psOfflineNoiseTracker.capture(db);
                }
            } else {
                psOfflineNoiseTracker.capture(db);
            }
            psDb.push(db);
            const { processed, voiced, confidence } = appendPitchSample(
                hz ?? null,
                { db, ambientDb: gate.ambient, spectral },
                { dtMs: Math.max(1, Math.round((offlineFeatureStore.frameSec || (PS_INTERVAL_MS / 1000)) * 1000)) }
            );
            offlineFeatureStore.pitchRaw.push(Number.isFinite(hz) ? hz : NaN);
            offlineFeatureStore.pitchProcessed.push(Number.isFinite(processed) ? processed : NaN);
            offlineFeatureStore.pitchConfidence.push(confidence);
            offlineFeatureStore.voiced.push(Boolean(voiced));
            offlineFeatureStore.db.push(db);
            if (spectral) {
                offlineFeatureStore.formants.push([spectral.f1 ?? NaN, spectral.f2 ?? NaN, spectral.f3 ?? NaN]);
                offlineFeatureStore.tilt.push(spectral.tilt ?? NaN);
                offlineFeatureStore.breathiness.push(spectral.breathiness ?? NaN);
                offlineFeatureStore.energy.push([spectral.energy?.low ?? NaN, spectral.energy?.mid ?? NaN, spectral.energy?.high ?? NaN]);
                offlineFeatureStore.zcr.push(spectral.zcr ?? NaN);
            } else {
                offlineFeatureStore.formants.push([NaN, NaN, NaN]);
                offlineFeatureStore.tilt.push(NaN);
                offlineFeatureStore.breathiness.push(NaN);
                offlineFeatureStore.energy.push([NaN, NaN, NaN]);
                offlineFeatureStore.zcr.push(NaN);
                offlineFeatureStore.zcr.push(NaN);
            }
        }
        // Compute and store summary immediately after extraction
        offlineFeatureStore.advSummary = computeAdvancedSummary(offlineFeatureStore);
    } catch (e) { console.error("[offlineExtractStreamMetrics]", e); }
}

function rms(arr, a, b) { let s = 0; for (let i = a; i < b; i++) { const v = arr[i]; s += v * v; } return Math.sqrt(s / Math.max(1, b - a)); }

export function startAutoRangeSession({ preserveRange = false } = {}) {
    autoRangeState.stage = "bootstrap";
    autoRangeState.sampleValues = [];
    autoRangeState.sampleDurationMs = 0;
    autoRangeState.timelineMs = 0;
    autoRangeState.lastUpdateMs = 0;
    if (!preserveRange) {
        autoRangeState.currentRange = null;
    }
    autoRangeState.lastMedian = null;
    autoRangeState.windowFrames = [];
    autoRangeState.windowMs = 0;
    autoRangeState.windowInvalidMs = 0;
    autoRangeState.octaveEvents = [];
    autoRangeState.prevOctaveCount = 0;
}

// Re-export or use the settings-aware range getter
import { getPitchDetectorRange as getSettingsRange } from "./state/pitch-settings.js";

// ... (existing code)

export function getPitchDetectorRange() {
    return getSettingsRange(autoRangeState);
}

// Keep handleAutoRangeFrame as is, it updates autoRangeState which is passed to getSettingsRange
export function handleAutoRangeFrame(result, { dtMs }) {
    if (!result) return;
    const hz = result.processed;
    const hasPitch = Number.isFinite(hz);

    if (hasPitch) {
        autoRangeState.sampleValues.push(hz);
        autoRangeState.sampleDurationMs += dtMs;
    }

    if (autoRangeState.sampleDurationMs > AUTO_SAMPLE_MS && autoRangeState.sampleValues.length > AUTO_MIN_VALID_FRAMES) {
        const sorted = autoRangeState.sampleValues.sort((a, b) => a - b);
        const p05 = percentile(sorted, 5);
        const p95 = percentile(sorted, 95);
        // Add hysteresis / padding logic from original app if needed.
        // For now, this baseline logic is functional.
        const min = Math.max(50, p05 * 0.8);
        const max = Math.min(600, p95 * 1.2);
        autoRangeState.currentRange = { min, max };

        autoRangeState.sampleValues = [];
        autoRangeState.sampleDurationMs = 0;
    }
}
