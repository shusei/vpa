function roundedRect(ctx, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, width, height, safeRadius);
    return;
  }
  ctx.moveTo(x + safeRadius, y);
  ctx.arcTo(x + width, y, x + width, y + height, safeRadius);
  ctx.arcTo(x + width, y + height, x, y + height, safeRadius);
  ctx.arcTo(x, y + height, x, y, safeRadius);
  ctx.arcTo(x, y, x + width, y, safeRadius);
  ctx.closePath();
}

function toBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("social preview generation failed"));
    }, "image/jpeg", 0.8);
  });
}

function fitFont(ctx, text, {
  maxWidth,
  minSize,
  size,
  weight,
}) {
  let current = size;
  while (current > minSize) {
    ctx.font = `${weight} ${current}px 'Noto Sans TC', sans-serif`;
    if (ctx.measureText(String(text || "")).width <= maxWidth) break;
    current -= 2;
  }
  ctx.font = `${weight} ${current}px 'Noto Sans TC', sans-serif`;
}

function drawWrappedText(ctx, text, {
  lineHeight,
  maxLines,
  maxWidth,
  x,
  y,
}) {
  const lines = [];
  let line = "";
  Array.from(String(text || "")).forEach((character) => {
    const candidate = `${line}${character}`;
    if (line && ctx.measureText(candidate).width > maxWidth) {
      lines.push(line);
      line = character;
    } else {
      line = candidate;
    }
  });
  if (line) lines.push(line);
  lines.slice(0, maxLines).forEach((value, index) => {
    const truncated = index === maxLines - 1 && lines.length > maxLines;
    ctx.fillText(`${value}${truncated ? "…" : ""}`, x, y + (index * lineHeight));
  });
}

export async function createSocialPreviewBlob({
  formatted,
  labels,
  pitchHz,
  result,
  theme = "dark",
}) {
  const canvas = document.createElement("canvas");
  canvas.width = 1200;
  canvas.height = 630;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas is unavailable");

  const dark = theme !== "light";
  const ink = dark ? "#f8f3ea" : "#243238";
  const muted = dark ? "#c8d8da" : "#5b6b70";
  const accent = dark ? "#f2c978" : "#9c5b22";
  const feminine = dark ? "#f1a5bd" : "#b54e73";
  const masculine = dark ? "#7cc7dd" : "#347f9d";
  const background = ctx.createLinearGradient(0, 0, 1200, 630);
  background.addColorStop(0, dark ? "#10252d" : "#eaf5f5");
  background.addColorStop(0.58, dark ? "#27434c" : "#fff7e8");
  background.addColorStop(1, dark ? "#69364b" : "#f3c4d0");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = dark ? "rgba(7,20,25,.56)" : "rgba(255,255,255,.72)";
  roundedRect(ctx, 42, 38, 1116, 554, 34);
  ctx.fill();

  ctx.fillStyle = accent;
  ctx.font = "800 27px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.brand, 82, 90);
  ctx.fillStyle = muted;
  ctx.font = "650 23px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.heading, 82, 129);

  ctx.fillStyle = ink;
  fitFont(ctx, formatted.archetype, {
    maxWidth: 650,
    minSize: 36,
    size: 66,
    weight: 900,
  });
  ctx.fillText(formatted.archetype, 82, 205);

  const score = Math.max(0, Math.min(100, Math.round(Number(result.score) || 0)));
  const masculineScore = 100 - score;
  ctx.fillStyle = feminine;
  ctx.font = "900 72px 'Noto Sans TC', sans-serif";
  ctx.fillText(`${score}%`, 82, 310);
  ctx.fillStyle = muted;
  ctx.font = "700 23px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.feminine, 86, 346);

  ctx.fillStyle = masculine;
  ctx.font = "900 52px 'Noto Sans TC', sans-serif";
  ctx.fillText(`${masculineScore}%`, 314, 307);
  ctx.fillStyle = muted;
  ctx.font = "700 23px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.masculine, 318, 346);

  ctx.fillStyle = "rgba(255,255,255,.13)";
  roundedRect(ctx, 82, 379, 650, 16, 8);
  ctx.fill();
  const scoreGradient = ctx.createLinearGradient(82, 0, 732, 0);
  scoreGradient.addColorStop(0, masculine);
  scoreGradient.addColorStop(1, feminine);
  ctx.fillStyle = scoreGradient;
  roundedRect(ctx, 82, 379, 650, 16, 8);
  ctx.fill();
  ctx.fillStyle = ink;
  ctx.beginPath();
  ctx.arc(82 + (650 * (score / 100)), 387, 12, 0, Math.PI * 2);
  ctx.fill();

  const stats = [
    [labels.pitch, Number.isFinite(Number(pitchHz)) ? `${Number(pitchHz).toFixed(1)} Hz` : "—"],
    [labels.voiceAge, formatted.age],
  ];
  stats.forEach(([label, value], index) => {
    const y = 438 + (index * 74);
    ctx.fillStyle = muted;
    ctx.font = "650 21px 'Noto Sans TC', sans-serif";
    ctx.fillText(label, 82, y);
    ctx.fillStyle = ink;
    fitFont(ctx, value, {
      maxWidth: 430,
      minSize: 25,
      size: 34,
      weight: 850,
    });
    ctx.fillText(value, 246, y);
  });

  ctx.fillStyle = dark ? "rgba(255,255,255,.075)" : "rgba(255,255,255,.58)";
  roundedRect(ctx, 782, 78, 324, 452, 28);
  ctx.fill();
  ctx.fillStyle = accent;
  ctx.font = "800 22px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.insight, 822, 129);
  ctx.fillStyle = ink;
  ctx.font = "700 29px 'Noto Sans TC', sans-serif";
  drawWrappedText(ctx, formatted.insight, {
    lineHeight: 43,
    maxLines: 5,
    maxWidth: 244,
    x: 822,
    y: 184,
  });

  ctx.fillStyle = muted;
  ctx.font = "600 18px 'Noto Sans TC', sans-serif";
  drawWrappedText(ctx, labels.disclaimer, {
    lineHeight: 29,
    maxLines: 3,
    maxWidth: 244,
    x: 822,
    y: 430,
  });

  ctx.fillStyle = accent;
  ctx.font = "800 22px 'Noto Sans TC', sans-serif";
  ctx.fillText(labels.cta, 822, 500);
  ctx.fillStyle = muted;
  ctx.font = "600 18px 'Noto Sans TC', sans-serif";
  ctx.fillText(result.version || "VPA", 82, 566);

  return toBlob(canvas);
}
