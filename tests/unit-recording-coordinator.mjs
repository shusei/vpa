import assert from "node:assert/strict";

import { createRecordingCoordinator } from "../assets/js/recording-coordinator.js";

function deferred() {
  let resolve;
  const promise = new Promise((resolver) => { resolve = resolver; });
  return { promise, resolve };
}

// ----- Nominal lifecycle and stale-session protection -----

{
  const snapshots = [];
  const stopGate = deferred();
  let coordinator;
  let stoppedSession = null;
  coordinator = createRecordingCoordinator({
    onStateApplied: (snapshot) => snapshots.push({ ...snapshot }),
    startRecording: async (meta) => {
      coordinator.handleFlowState("recording", meta);
      return true;
    },
    stopRecording: async (meta) => {
      stoppedSession = meta.sessionId;
      await stopGate.promise;
      return true;
    },
  });

  assert.equal(await coordinator.start({ source: "quick" }), true);
  assert.equal(coordinator.isRecording, true);
  assert.equal(coordinator.getSnapshot().source, "quick");
  assert.equal(coordinator.handleFlowState("idle", {
    sessionId: coordinator.getSnapshot().sessionId,
  }), false, "recording cannot skip stopping and analyzing");
  assert.equal(await coordinator.start({ source: "professional" }), false, "recording start must not re-enter");

  const pendingStop = coordinator.stop();
  assert.equal(coordinator.getSnapshot().state, "stopping");
  assert.equal(await coordinator.start({ source: "professional" }), false, "start must wait for an in-flight stop");
  stopGate.resolve();
  assert.equal(await pendingStop, true);
  assert.equal(await coordinator.start({ source: "professional" }), false, "start must wait for analysis");

  assert.equal(coordinator.handleFlowState("analyzing", { sessionId: stoppedSession }), true);
  assert.equal(coordinator.getSnapshot().state, "analyzing");
  assert.equal(coordinator.handleFlowState("idle", { sessionId: stoppedSession, pitchState: "inactive" }), true);
  assert.equal(coordinator.getSnapshot().state, "idle");

  assert.equal(coordinator.handleFlowState("error", {
    error: new Error("stale"),
    sessionId: stoppedSession - 1,
  }), false);
  assert.equal(coordinator.getSnapshot().state, "idle", "an old session must not overwrite the active state");
  assert(snapshots.some((snapshot) => snapshot.state === "requesting"));
  assert(snapshots.some((snapshot) => snapshot.state === "recording"));
  assert(snapshots.some((snapshot) => snapshot.state === "stopping"));
  assert(snapshots.some((snapshot) => snapshot.state === "analyzing"));
}

// ----- Delayed start, error recovery, and source validation -----

{
  const startGate = deferred();
  let coordinator;
  let shouldFail = false;
  coordinator = createRecordingCoordinator({
    startRecording: async (meta) => {
      await startGate.promise;
      if (shouldFail) throw new Error("microphone failed");
      coordinator.handleFlowState("recording", meta);
      return true;
    },
    stopRecording: async () => true,
  });

  const firstStart = coordinator.start({ source: "practice" });
  assert.equal(coordinator.getSnapshot().state, "requesting");
  assert.equal(await coordinator.start({ source: "quick" }), false, "a delayed start must stay single-flight");
  startGate.resolve();
  assert.equal(await firstStart, true);
  assert.equal(await coordinator.stop(), true);
  coordinator.handleFlowState("analyzing", {
    sessionId: coordinator.getSnapshot().sessionId,
  });
  coordinator.handleFlowState("idle", {
    pitchState: "inactive",
    sessionId: coordinator.getSnapshot().sessionId,
  });

  shouldFail = true;
  await assert.rejects(coordinator.start({ source: "professional" }), /microphone failed/);
  assert.equal(coordinator.getSnapshot().state, "error");
  shouldFail = false;
  assert.equal(await coordinator.start({ source: "professional" }), true, "error state must allow a new session");

  await assert.rejects(
    coordinator.start({ source: "unknown" }),
    /Unknown recording source/,
  );
}

console.log("[PASS] Recording coordinator lifecycle guards passed.");
