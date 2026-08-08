import { buildShareText } from "./audio-share.js?v=1.4.21";

function roundedRect(ctx, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, safeRadius);
}

function drawBar(ctx, { label, score, x, y, width, color, muted, textColor }) {
  ctx.fillStyle = muted;
  roundedRect(ctx, x, y, width, 18, 9);
  ctx.fill();
  const filled = Math.max(12, width * Math.max(0, Math.min(1, score)));
  const gradient = ctx.createLinearGradient(x, y, x + filled, y);
  gradient.addColorStop(0, color[0]);
  gradient.addColorStop(1, color[1]);
  ctx.fillStyle = gradient;
  roundedRect(ctx, x, y, filled, 18, 9);
  ctx.fill();
  ctx.fillStyle = textColor;
  ctx.font = "600 30px 'Noto Sans TC', sans-serif";
  ctx.fillText(label, x, y - 18);
  ctx.textAlign = "right";
  ctx.fillText(`${Math.round(score * 100)}`, x + width, y - 18);
  ctx.textAlign = "left";
}

function toBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("share card image generation failed"));
    }, "image/png");
  });
}

export function buildShareTargets({ caption, url }) {
  const combined = buildShareText(caption, url);
  const xParams = new URLSearchParams({
    hashtags: "VoicePresentationAnalyzer",
    text: caption,
    url,
  });
  return {
    line: `https://line.me/R/share?text=${encodeURIComponent(combined)}`,
    threads: `https://www.threads.com/intent/post?text=${encodeURIComponent(combined)}`,
    x: `https://twitter.com/intent/tweet?${xParams.toString()}`,
  };
}

export function buildShareUrl(locationLike = window.location) {
  const url = new URL(locationLike.href);
  url.hash = "";
  url.search = "";
  url.pathname = url.pathname.replace(/dev\.html$/, "");
  return url.toString();
}

export async function createShareCardBlob({ result, labels, shareUrl, theme = "dark" }) {
  const canvas = document.createElement("canvas");
  canvas.width = 1080;
  canvas.height = 1350;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas is unavailable");

  const dark = theme !== "light";
  const background = ctx.createLinearGradient(0, 0, 1080, 1350);
  background.addColorStop(0, dark ? "#0d2026" : "#fff8e9");
  background.addColorStop(0.55, dark ? "#173741" : "#f6dfbd");
  background.addColorStop(1, dark ? "#5b2c3d" : "#f4b9b7");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = dark ? "rgba(255,255,255,.075)" : "rgba(255,255,255,.58)";
  roundedRect(ctx, 70, 70, 940, 1210, 48);
  ctx.fill();

  ctx.fillStyle = dark ? "#f7f2e8" : "#2b2522";
  ctx.font = "800 48px 'Noto Sans TC', sans-serif";
  ctx.fillText("VPA / ADVANCED", 120, 150);
  ctx.fillStyle = dark ? "#f2c978" : "#9c5b22";
  ctx.font = "700 30px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.beta, 120, 200);

  ctx.fillStyle = dark ? "#f7f2e8" : "#2b2522";
  ctx.font = "900 180px 'Noto Sans TC', sans-serif";
  ctx.fillText(`${result.score}`, 112, 410);
  ctx.font = "700 46px 'Noto Sans TC', sans-serif";
  ctx.fillText("%", 390, 404);
  ctx.fillStyle = dark ? "#f2c978" : "#9c5b22";
  ctx.font = "700 38px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.strictScore, 120, 470);

  ctx.fillStyle = dark ? "rgba(7,20,25,.55)" : "rgba(255,255,255,.68)";
  roundedRect(ctx, 570, 260, 350, 240, 30);
  ctx.fill();
  ctx.fillStyle = dark ? "#f7f2e8" : "#2b2522";
  ctx.font = "700 30px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.voiceAge, 610, 320);
  ctx.font = "800 50px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.voiceAgeValue, 610, 385);
  ctx.fillStyle = dark ? "#f2c978" : "#9c5b22";
  ctx.font = "700 30px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.archetypeValue, 610, 450);

  const barColor = dark ? ["#63d4c8", "#f4bf6d"] : ["#187c78", "#c76f35"];
  const muted = dark ? "rgba(255,255,255,.12)" : "rgba(43,37,34,.12)";
  const barText = dark ? "#f7f2e8" : "#2b2522";
  const bars = [
    ["model", result.components.model],
    ["resonance", result.components.resonance],
    ["pitch", result.components.pitch],
    ["intonation", result.components.intonation],
  ];
  bars.forEach(([key, score], index) => {
    drawBar(ctx, {
      color: barColor,
      label: labels.components[key],
      muted,
      score,
      textColor: barText,
      width: 830,
      x: 125,
      y: 600 + (index * 105),
    });
  });

  ctx.fillStyle = dark ? "rgba(7,20,25,.55)" : "rgba(255,255,255,.68)";
  roundedRect(ctx, 120, 1025, 840, 125, 28);
  ctx.fill();
  ctx.fillStyle = dark ? "#f7f2e8" : "#2b2522";
  ctx.font = "600 31px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.challenge, 160, 1090);
  ctx.fillStyle = dark ? "#c9dbd8" : "#66594f";
  ctx.font = "500 24px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.disclaimer, 160, 1128);

  ctx.fillStyle = dark ? "#f2c978" : "#9c5b22";
  ctx.font = "700 29px 'Noto Sans TC', sans-serif";
  ctx.fillText(shareUrl.replace(/^https?:\/\//, "").replace(/\/$/, ""), 120, 1220);
  ctx.textAlign = "right";
  ctx.fillText(result.version, 960, 1220);
  ctx.textAlign = "left";
  return toBlob(canvas);
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function shareWithSystem({ blob, caption, title, url }) {
  if (typeof navigator.share !== "function") return { method: "unsupported" };
  const file = new File([blob], "vpa-advanced-result.png", { type: "image/png" });
  const filePayload = {
    files: [file],
    text: buildShareText(caption, url),
    title,
  };
  if (typeof navigator.canShare === "function" && navigator.canShare(filePayload)) {
    await navigator.share(filePayload);
    return { method: "files" };
  }
  await navigator.share({ text: buildShareText(caption, url), title });
  return { method: "url" };
}
