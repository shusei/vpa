export function pickSupportedMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  try {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported) {
      for (const t of cands) if (MediaRecorder.isTypeSupported(t)) return t;
    }
  } catch { }
  return "";
}

export async function requestMicStream() {
  const base = { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } };
  const fallback = { audio: true };
  const getUserMedia = navigator?.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices);
  if (!getUserMedia) {
    throw new Error("record-unsupported");
  }
  const disableTrackProcessing = async (stream) => {
    try {
      const tracks = stream?.getAudioTracks?.() || [];
      await Promise.all(tracks.map(async (track) => {
        if (!track?.applyConstraints) return;
        try {
          await track.applyConstraints({ echoCancellation: false, noiseSuppression: false, autoGainControl: false });
        } catch (err) {
          console.warn("[audio track] disable processing failed", err);
        }
      }));
    } catch (err) {
      console.warn("[audio track] constraints traversal failed", err);
    }
  };
  try {
    const stream = await getUserMedia(base);
    await disableTrackProcessing(stream);
    return stream;
  } catch (err) {
    console.warn("[getUserMedia] preferred constraints failed", err);
  }
  const fallbackStream = await getUserMedia(fallback);
  await disableTrackProcessing(fallbackStream);
  return fallbackStream;
}
