import { strict as assert } from 'node:assert';
import {
    clampPitchRange,
    createPitchPostState,
    correctPitchOctave,
    smoothPitchCandidate,
    makeStats,
    PS_MIN_HZ,
    PS_MAX_HZ
} from '../assets/js/pitch-shared.js';

console.log("Running unit tests for pitch-shared.js...");

// --- Test: clampPitchRange ---
{
    const range = clampPitchRange({ min: 100, max: 300 });
    assert.equal(range.min, 100, "min should be 100");
    assert.equal(range.max, 300, "max should be 300");

    const hardClamped = clampPitchRange({ min: 10, max: 1000 });
    assert.ok(hardClamped.min >= PS_MIN_HZ, "min should be clamped to hard limit");
    assert.ok(hardClamped.max <= PS_MAX_HZ, "max should be clamped to hard limit");
    console.log("[OK] clampPitchRange");
}

// --- Test: createPitchPostState ---
{
    const state = createPitchPostState();
    assert.ok(Array.isArray(state.volumeHistory), "volumeHistory should be array");
    assert.equal(state.silentStreak, 0, "silentStreak should be 0");
    assert.equal(typeof state.counters, 'object', "counters should be object");
    console.log("[OK] createPitchPostState");
}

// --- Test: correctPitchOctave ---
{
    const state = createPitchPostState();
    const range = { min: 80, max: 300 };

    // Case 1: No history, should return candidate as is
    const initial = correctPitchOctave(150, range, state);
    assert.equal(initial, 150, "Should return candidate if no history");

    // Case 2: History exists (e.g. 150Hz). Input is 300Hz (octave up).
    // Should prefer 150Hz if it's closer to history (150Hz).
    state.lastAccepted = 150;
    const correctedDown = correctPitchOctave(300, range, state);
    // 300Hz is exactly one octave up. The logic penalizes jumps.
    // Ideally it should correct 300 -> 150 to match history.
    assert.ok(Math.abs(correctedDown - 150) < 1, "Should correct 300Hz down to ~150Hz to match history");

    // Case 3: Input is 75Hz (octave down).
    const correctedUp = correctPitchOctave(75, range, state);
    assert.ok(Math.abs(correctedUp - 150) < 1, "Should correct 75Hz up to ~150Hz to match history");

    console.log("[OK] correctPitchOctave");
}

// --- Test: smoothPitchCandidate ---
{
    const state = createPitchPostState();
    const range = { min: 50, max: 600 };

    // Feed a sequence: 100, 100, 200 (spike), 100, 100
    // Median filter window size is 5.
    smoothPitchCandidate(100, range, state);
    smoothPitchCandidate(100, range, state);
    const spiked = smoothPitchCandidate(200, range, state);
    // The spike might be smoothed but not fully removed immediately depending on window fill,
    // but let's check if it's dampened or if median works after more samples.

    smoothPitchCandidate(100, range, state);
    const final = smoothPitchCandidate(100, range, state);

    // With [100, 100, 200, 100, 100], median is 100.
    // Alpha smoothing will drag it towards 100.
    assert.ok(Math.abs(final - 100) < 10, "Spike should be smoothed out by median filter");
    console.log("[OK] smoothPitchCandidate");
}

// --- Test: makeStats ---
{
    const data = [10, 20, 30, 40, 50];
    const stats = makeStats(data);
    assert.equal(stats.avg, 30, "Mean should be 30");
    assert.equal(stats.med, 30, "Median should be 30");
    assert.equal(stats.p05, 12, "p05 interpolation check"); // index 0.2 -> 0.8*10 + 0.2*20 = 12
    assert.equal(stats.p95, 48, "p95 interpolation check"); // index 3.8 -> 0.2*40 + 0.8*50 = 48
    console.log("[OK] makeStats");
}

console.log("ALL TESTS PASSED");
