/**
 * Advanced Summary Computation
 * Extracted from legacy app.js (lines 3482-3584)
 * Main aggregator that combines all analysis modules
 */

import { makeStats, percentileSorted, PS_INTERVAL_MS } from "../pitch-shared.js";
import { buildEligibleFrameMask, averageFinite, averageEnergy, FORMANT_CONFIDENCE_THRESHOLD, FORMANT_MAX_GAP_FRAMES } from "./helpers.js";
import { summarizeFormantTrends } from "./formant-analysis.js";
import { describeResonanceFromEnergy, detectVoiceLeaning } from "./resonance.js";
import { categorizeTilt, categorizeBrightness, categorizeBreathiness } from "./brightness.js";
import { summarizeBreathiness } from "./helpers.js";
import { analyzeVowelFocus } from "./vowel-focus.js";
import { analyzeSpeechRate } from "./speech-rate.js";
import { analyzeConnectedSpeech } from "./liaison.js";
import { analyzeIntonation } from "./intonation.js";

/**
 * Compute complete advanced summary from offline feature store
 * @param {Object} offlineFeatureStore - Feature data from service
 * @param {number} lastPf - Last female probability
 * @param {number} lastPm - Last male probability
 * @returns {Object|null} Advanced analysis summary
 */
export function computeAdvancedSummary(offlineFeatureStore, lastPf, lastPm) {
    const store = offlineFeatureStore;
    const processedPitch = Array.isArray(store.pitchProcessed) ? store.pitchProcessed : (store.pitch || []);
    const rawPitch = Array.isArray(store.pitchRaw) ? store.pitchRaw : processedPitch;
    const pitchConfidence = Array.isArray(store.pitchConfidence) ? store.pitchConfidence : [];
    const n = processedPitch.length;
    const hopSec = store.frameSec || (PS_INTERVAL_MS / 1000);
    const duration = hopSec * n;
    if (!n || duration < 0.5) return null;

    const maskInfo = buildEligibleFrameMask(store, {
        minConfidence: FORMANT_CONFIDENCE_THRESHOLD,
        maxGapFrames: FORMANT_MAX_GAP_FRAMES,
    });
    let mask = Array.isArray(maskInfo?.mask) && maskInfo.mask.length ? maskInfo.mask : null;
    let eligibleCount = Number.isFinite(maskInfo?.count) ? maskInfo.count : 0;
    if ((!mask || !eligibleCount) && Array.isArray(store.voiced) && store.voiced.length) {
        mask = store.voiced.map(Boolean);
        eligibleCount = mask.reduce((acc, flag) => acc + (flag ? 1 : 0), 0);
    }

    const formantArr = Array.isArray(store.formants) ? store.formants : [];
    const limit = mask ? Math.min(formantArr.length, mask.length) : formantArr.length;
    const f1Vals = [], f2Vals = [], f3Vals = [];
    for (let i = 0; i < limit; i++) {
        if (mask && !mask[i]) continue;
        const form = formantArr[i];
        if (!form) continue;
        const [f1, f2, f3] = form;
        if (Number.isFinite(f1)) f1Vals.push(f1);
        if (Number.isFinite(f2)) f2Vals.push(f2);
        if (Number.isFinite(f3)) f3Vals.push(f3);
    }
    const f1Stats = f1Vals.length ? makeStats(f1Vals) : null;
    const f2Stats = f2Vals.length ? makeStats(f2Vals) : null;
    const f3Stats = f3Vals.length ? makeStats(f3Vals) : null;
    const formantSummary = summarizeFormantTrends(store, {
        f1: f1Stats,
        f2: f2Stats,
        f3: f3Stats,
        f1Vals,
        f2Vals,
        f3Vals,
    }, {
        mask,
        eligibleCount,
    });

    const energyAvg = averageEnergy(store.energy, { mask, eligibleCount });
    const resonanceDesc = describeResonanceFromEnergy(energyAvg);
    const tiltAvg = averageFinite(store.tilt, mask);
    const tiltInfo = categorizeTilt(tiltAvg);
    const breathSummary = summarizeBreathiness(store.breathiness, { mask }, hopSec);
    const breathAvg = breathSummary.avg;
    const vols = Array.isArray(store.db) ? store.db.filter(Number.isFinite) : [];
    const volStats = vols.length ? makeStats(vols) : null;
    const envDb = vols.length ? percentileSorted(vols.slice().sort((a, b) => a - b), 10) : NaN;
    const snrEstimate = Number.isFinite(volStats?.med) && Number.isFinite(envDb) ? (volStats.med - envDb) : NaN;
    const leaning = detectVoiceLeaning(lastPf, lastPm);
    const brightnessInfo = categorizeBrightness({ f3Stats, tilt: tiltAvg, breath: breathAvg, leaning });
    const breathInfo = categorizeBreathiness(breathAvg, {
        snr: snrEstimate,
        brightnessKey: brightnessInfo.key,
        tilt: tiltAvg,
    });
    const vowelInfo = analyzeVowelFocus(store, { mask, count: eligibleCount });
    const speech = analyzeSpeechRate(store);
    const liaison = analyzeConnectedSpeech(store.voiced, hopSec);
    const intonation = analyzeIntonation({
        processed: processedPitch,
        raw: rawPitch,
        confidence: pitchConfidence,
        voiced: store.voiced,
    }, hopSec);

    return {
        formants: formantSummary,
        resonanceLabel: resonanceDesc.label,
        resonanceDisplay: resonanceDesc.display || resonanceDesc.label,
        resonanceHint: resonanceDesc.hint,
        energyPct: resonanceDesc.pct,
        tiltAvg,
        tiltLabel: tiltInfo.label,
        tiltHint: tiltInfo.hint,
        brightnessLabel: brightnessInfo.label,
        brightnessHint: brightnessInfo.hint,
        brightnessKey: brightnessInfo.key,
        breathinessAvg: breathAvg,
        breathinessLabel: breathInfo.label,
        breathinessHint: breathInfo.hint,
        breathinessKey: breathInfo.key,
        speechRate: speech,
        speechRateLabel: speech.label,
        speechRateHint: speech.hint,
        vowelFocusRatio: vowelInfo.ratio,
        vowelLabel: vowelInfo.label,
        vowelHint: vowelInfo.hint,
        liaisonRatio: liaison.ratio,
        liaisonLabel: liaison.label,
        liaisonHint: liaison.hint,
        intonation,
        snrEstimate,
    };
}
