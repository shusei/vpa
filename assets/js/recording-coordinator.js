const RECORDING_STATES = new Set([
  "analyzing",
  "error",
  "idle",
  "recording",
  "requesting",
  "stopping",
]);

const RECORDING_SOURCES = new Set(["practice", "professional", "quick"]);
const STATE_TRANSITIONS = {
  analyzing: new Set(["error", "idle"]),
  error: new Set(["requesting"]),
  idle: new Set(["requesting"]),
  recording: new Set(["error", "stopping"]),
  requesting: new Set(["error", "recording"]),
  stopping: new Set(["analyzing", "error"]),
};

function initialSnapshot() {
  return {
    error: null,
    pitchState: "inactive",
    sessionId: 0,
    source: null,
    state: "idle",
  };
}

export function createRecordingCoordinator({
  diagnostics = null,
  isExternalBusy = () => false,
  onStateApplied = () => { },
  startRecording,
  stopRecording,
} = {}) {
  if (typeof startRecording !== "function" || typeof stopRecording !== "function") {
    throw new TypeError("RecordingCoordinator requires startRecording and stopRecording functions.");
  }

  const listeners = new Set();
  let sessionSequence = 0;
  let snapshot = initialSnapshot();

  function publish(next) {
    snapshot = Object.freeze({ ...snapshot, ...next });
    onStateApplied(snapshot);
    listeners.forEach((listener) => {
      try {
        listener(snapshot);
      } catch (error) {
        console.warn("[recording-coordinator] subscriber failed", error);
      }
    });
  }

  function transition(state, detail = {}) {
    if (!RECORDING_STATES.has(state)) {
      throw new TypeError(`Unknown recording state: ${state}`);
    }
    if (state !== snapshot.state && !STATE_TRANSITIONS[snapshot.state]?.has(state)) {
      diagnostics?.record("recording.state.invalid", {
        from: snapshot.state,
        sessionId: snapshot.sessionId,
        to: state,
      });
      return false;
    }
    publish({
      error: state === "error" ? (detail.error || snapshot.error || new Error("Recording failed.")) : null,
      pitchState: detail.pitchState || snapshot.pitchState,
      state,
    });
    diagnostics?.record("recording.state", {
      error: snapshot.error ? {
        message: String(snapshot.error.message || snapshot.error),
        name: String(snapshot.error.name || "Error"),
      } : null,
      pitchState: snapshot.pitchState,
      sessionId: snapshot.sessionId,
      source: snapshot.source,
      state: snapshot.state,
    });
    return true;
  }

  async function start({ source = "professional" } = {}) {
    if (!RECORDING_SOURCES.has(source)) {
      throw new TypeError(`Unknown recording source: ${source}`);
    }
    if (!["idle", "error"].includes(snapshot.state) || isExternalBusy()) return false;

    const sessionId = ++sessionSequence;
    publish({
      error: null,
      pitchState: "preparing",
      sessionId,
      source,
      state: "requesting",
    });
    diagnostics?.record("recording.session.begin", { sessionId, source });

    try {
      const started = await startRecording({ sessionId, source });
      if (!started && snapshot.sessionId === sessionId && snapshot.state === "requesting") {
        transition("error", { error: new Error("Recording did not start.") });
      }
      if (started && snapshot.sessionId === sessionId && snapshot.state === "requesting") {
        transition("error", { error: new Error("Recording started without entering the recording state.") });
        return false;
      }
      return Boolean(started);
    } catch (error) {
      if (snapshot.sessionId === sessionId) transition("error", { error });
      throw error;
    }
  }

  async function stop() {
    if (snapshot.state !== "recording") return false;
    const sessionId = snapshot.sessionId;
    transition("stopping");
    try {
      const stopped = Boolean(await stopRecording({ sessionId }));
      if (!stopped && snapshot.sessionId === sessionId && snapshot.state === "stopping") {
        transition("error", { error: new Error("Recording did not stop.") });
      }
      return stopped;
    } catch (error) {
      if (snapshot.sessionId === sessionId) transition("error", { error });
      throw error;
    }
  }

  function handleFlowState(state, detail = {}) {
    if (detail.sessionId !== snapshot.sessionId) {
      diagnostics?.record("recording.state.stale", {
        activeSessionId: snapshot.sessionId,
        ignoredSessionId: detail.sessionId,
        state,
      });
      return false;
    }
    return transition(state, detail);
  }

  function updatePitchState(pitchState, detail = {}) {
    if (detail.sessionId !== snapshot.sessionId) return false;
    publish({ pitchState });
    diagnostics?.record("recording.pitch-state", {
      pitchState,
      sessionId: snapshot.sessionId,
      source: snapshot.source,
    });
    return true;
  }

  function getSnapshot() {
    return snapshot;
  }

  function subscribe(listener) {
    if (typeof listener !== "function") return () => { };
    listeners.add(listener);
    listener(snapshot);
    return () => listeners.delete(listener);
  }

  return {
    get busy() {
      return ["analyzing", "requesting", "stopping"].includes(snapshot.state) || isExternalBusy();
    },
    get isRecording() {
      return snapshot.state === "recording";
    },
    getSnapshot,
    handleFlowState,
    start,
    stop,
    subscribe,
    updatePitchState,
  };
}

export const recordingCoordinatorInternals = {
  RECORDING_SOURCES,
  RECORDING_STATES,
  STATE_TRANSITIONS,
  initialSnapshot,
};
