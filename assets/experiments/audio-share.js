export function buildShareText(caption, url) {
  return [caption, url]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join("\n");
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
      text: buildShareText(caption, url),
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
