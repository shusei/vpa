const AUDIO_EXTENSIONS = new Map([
  ["audio/mp4", "m4a"],
  ["audio/mpeg", "mp3"],
  ["audio/ogg", "ogg"],
  ["audio/wav", "wav"],
  ["audio/webm", "weba"],
  ["video/mp4", "m4a"],
]);

function extensionFor(type) {
  const normalized = String(type || "").split(";")[0].trim().toLowerCase();
  return AUDIO_EXTENSIONS.get(normalized) || "weba";
}

export async function audioFileFromUrl(url, fetchLike = fetch) {
  if (!url) return null;
  const response = await fetchLike(url);
  if (!response?.ok) throw new Error(`audio fetch failed: ${response?.status || "unknown"}`);
  const blob = await response.blob();
  if (!blob.size) throw new Error("audio blob is empty");
  const extension = extensionFor(blob.type);
  return new File([blob], `vpa-voice.${extension}`, {
    type: blob.type || "audio/webm",
  });
}

export async function shareResultFiles({
  audioFile = null,
  cardBlob,
  caption,
  navigatorLike = navigator,
  title,
  url,
}) {
  if (!cardBlob) throw new TypeError("cardBlob is required");
  if (typeof navigatorLike?.share !== "function") {
    return { method: "unsupported" };
  }
  const cardFile = new File([cardBlob], "vpa-result.png", { type: "image/png" });
  const files = audioFile ? [cardFile, audioFile] : [cardFile];
  if (
    typeof navigatorLike.canShare === "function"
    && navigatorLike.canShare({ files })
  ) {
    await navigatorLike.share({
      files,
      text: caption,
      title,
      url,
    });
    return {
      includesAudio: Boolean(audioFile),
      method: "files",
    };
  }
  if (audioFile) {
    return {
      audioFile,
      cardBlob,
      method: "unsupported-files",
    };
  }
  await navigatorLike.share({ text: caption, title, url });
  return { method: "url" };
}

export const audioShareInternals = {
  extensionFor,
};
