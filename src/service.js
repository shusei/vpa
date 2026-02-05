import { t } from "./i18n.js";
import { setStatus } from "./ui.js";

// Re-export constants & core logic
export {
    pitchStrategies, resetPitchPostState, trackPitchRuntime, maybeEnableAdvancedPitch
} from "./pitch-shared.js";

// Import modules
import * as Decoder from "./services/audio-decoder.js";
import * as PitchStream from "./services/pitch-stream.js";
import * as Analysis from "./services/analysis-runner.js";

// --- Global Facade State (compatible with main.js) ---

// UI Callbacks (injected by main.js)
let callbacks = {
    onStatus: (msg, loading) => { },
    onRender: (pf, pm) => { },
    onUpdateRealtimeMonitor: (data) => { },
    onStartDrawLoop: () => { },
    onFinishStreamStats: (data) => { }
};

export function setCallbacks(cbs) {
    callbacks = { ...callbacks, ...cbs };
    // Propagate to sub-modules
    PitchStream.setStreamCallbacks({
        onStatus: callbacks.onStatus,
        onUpdateRealtimeMonitor: callbacks.onUpdateRealtimeMonitor
    });
}

// --- Facade API ---

export const decodeSmartToFloat32 = Decoder.decodeSmartToFloat32;

export function startPitchStream(stream) {
    callbacks.onStartDrawLoop();
    return PitchStream.startPitchStream(stream);
}

export const stopPitchStream = PitchStream.stopPitchStream;

// Getters for state found in main.js
// export function getIsRecording() { return PitchStream.getIsRecording(); }
// export const isRecording = PitchStream.getIsRecording(); 

// We rely on the re-export below for live bindings.

// Re-export live bindings mechanism
export { hasLastRecording, getLastRecordingUrl, getIsRecording, isRecording } from "./services/pitch-stream.js";

// Wrapper for `handleFileOrBlob` to inject callbacks and manage busy state
export let busy = false;

export async function handleFileOrBlob(fileOrBlob, context = "upload") {
    if (busy) return;
    busy = true;
    try {
        return await Analysis.handleFileOrBlob(fileOrBlob, context, {
            onRender: callbacks.onRender,
            onFinishStreamStats: callbacks.onFinishStreamStats
        });
    } finally {
        busy = false;
    }
}
