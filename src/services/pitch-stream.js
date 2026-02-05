import { t } from "../i18n.js";
import { setStatus } from "../ui.js";
import {
    createPitchPostState, resetPitchPostState, pitchStrategies, selectPreferredPitchStrategy,
    getActivePitchStrategy, trackPitchRuntime, degradePitchStrategy,
    runPitchDetection
} from "../pitch-shared.js";

// --- State ---
// This state is shared/exported because `service.js` (facade) might need it,
// but ideally we encapsulate it. For now, we export getters.

let audioCtx = null;
let micStream = null;
let scriptNode = null;
let gainNode = null;
export let isRecording = false;
let recorder = null;
let recordedChunks = [];
let lastRecordingBlob = null;
let lastRecordingUrl = null;

let pitchLoopId = null;
// We'll keep these "live" variables here
let liveInputBuffer = null; // Float32Array from script processor

export function getIsRecording() { return isRecording; }
export function hasLastRecording() { return !!lastRecordingBlob; }
export function getLastRecordingUrl() { return lastRecordingUrl; }

// Allow external modules to subscribe to updates.
// The Facade will bridge these to the main callbacks.
let streamCallbacks = {
    onStatus: (msg) => { },
    onUpdateRealtimeMonitor: (data) => { },
};

export function setStreamCallbacks(cbs) {
    if (cbs.onStatus) streamCallbacks.onStatus = cbs.onStatus;
    if (cbs.onUpdateRealtimeMonitor) streamCallbacks.onUpdateRealtimeMonitor = cbs.onUpdateRealtimeMonitor;
}

// --- Audio Context & Stream ---

export async function startPitchStream(stream) {
    if (isRecording) return;
    try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!audioCtx) audioCtx = new Ctx();
        if (audioCtx.state === "suspended") await audioCtx.resume();

        micStream = stream;
        const source = audioCtx.createMediaStreamSource(stream);

        // Gain
        gainNode = audioCtx.createGain();
        gainNode.gain.value = 1.0;

        // ScriptProcessor (Legacy but reliable for simple pitch)
        // bufferSize=4096 => ~92ms @ 44.1k
        scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptNode.onaudioprocess = handleAudioProcess;

        source.connect(gainNode);
        gainNode.connect(scriptNode);
        scriptNode.connect(audioCtx.destination); // needed for chrome to fire

        // Recorder
        recordedChunks = [];
        // Try simple MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : "audio/webm";
        recorder = new MediaRecorder(stream, { mimeType });
        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };
        recorder.start();

        isRecording = true;
        resetPitchPostState(); // Reset smoothing

        // Start Loop
        if (pitchLoopId) clearInterval(pitchLoopId);
        pitchLoopId = setInterval(processPitchLoop, 30); // ~30fps

    } catch (err) {
        console.error("Start stream failed", err);
        streamCallbacks.onStatus(t("status.micError"));
        throw err;
    }
}

export async function stopPitchStream() {
    if (!isRecording) return;
    isRecording = false;

    if (pitchLoopId) clearInterval(pitchLoopId);
    pitchLoopId = null;

    if (scriptNode) { scriptNode.disconnect(); scriptNode.onaudioprocess = null; scriptNode = null; }
    if (gainNode) { gainNode.disconnect(); gainNode = null; }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }

    if (recorder && recorder.state !== "inactive") {
        await new Promise(r => {
            recorder.onstop = r;
            recorder.stop();
        });
        const blob = new Blob(recordedChunks, { type: "audio/webm" });
        lastRecordingBlob = blob;
        if (lastRecordingUrl) URL.revokeObjectURL(lastRecordingUrl);
        lastRecordingUrl = URL.createObjectURL(blob);
    }
    recorder = null;
}

function handleAudioProcess(e) {
    if (!isRecording) return;
    // Copy input buffer for the loop to consume
    const input = e.inputBuffer.getChannelData(0);
    // We just keep the latest chunk for "realtime" visualization
    // A more robust circular buffer could be used, but for simple Hz display this is okay-ish
    // However, YIN needs ~2048-4096 samples. ScriptProcessor gives 4096 usually.
    liveInputBuffer = new Float32Array(input);
}

function processPitchLoop() {
    if (!liveInputBuffer) return;
    if (!audioCtx) return;

    // Use current strategy
    const sr = audioCtx.sampleRate;
    const { hz, confidence, note, cents } = runPitchDetection(liveInputBuffer, sr);

    // Update Monitor
    // We might need to smoothing here if shared logic doesn't cover live stream
    // Actually runPitchDetection already returns raw values,
    // usually we want smoothed for UI.
    // The previous implementation did smoothing inside the loop.
    // Let's rely on the caller/UI to visualize or smooth.
    // Wait, the original `analyzeWhole` does full pass.
    // The original `startPitchStream` loop did:
    //   const { hz, confidence } = detectPitch(input, sr)
    //   updateRealtimeMonitor({ hz, confidence, ... })

    streamCallbacks.onUpdateRealtimeMonitor({
        hz,
        confidence,
        note,
        cents,
        buffer: liveInputBuffer // Pass buffer for visualizer
    });

    liveInputBuffer = null; // consume
}
