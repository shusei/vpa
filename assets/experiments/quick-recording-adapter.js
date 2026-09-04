export function createQuickRecordingAdapter(recorder) {
  if (!recorder || typeof recorder.start !== "function" || typeof recorder.stop !== "function") {
    throw new TypeError("Quick recording requires a recording coordinator.");
  }

  return {
    get busy() {
      return recorder.busy;
    },
    get isRecording() {
      return recorder.isRecording;
    },
    getSnapshot() {
      return recorder.getSnapshot();
    },
    start() {
      return recorder.start({ source: "quick" });
    },
    stop() {
      return recorder.stop();
    },
    subscribe(listener) {
      return recorder.subscribe((snapshot) => {
        if (snapshot.source === "quick") listener(snapshot);
      });
    },
  };
}
