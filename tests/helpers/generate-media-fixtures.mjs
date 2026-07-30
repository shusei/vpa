import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const outputDir = resolve("tests/.generated-media");
mkdirSync(outputDir, { recursive: true });

function runFfmpeg(args, outputName) {
  const result = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], {
    encoding: "utf8",
  });
  if (result.error || result.status !== 0) {
    const detail = result.error?.message || result.stderr || `exit ${result.status}`;
    throw new Error(`Unable to generate ${outputName}: ${detail}`);
  }
}

const wavPath = resolve(outputDir, "tone.wav");
runFfmpeg([
  "-f", "lavfi",
  "-i", "sine=frequency=220:sample_rate=48000:duration=4",
  "-ac", "1",
  "-c:a", "pcm_s16le",
  wavPath,
], "tone.wav");

runFfmpeg([
  "-i", wavPath,
  "-c:a", "libmp3lame",
  "-b:a", "96k",
  resolve(outputDir, "tone.mp3"),
], "tone.mp3");

runFfmpeg([
  "-i", wavPath,
  "-c:a", "aac",
  "-b:a", "96k",
  resolve(outputDir, "tone.m4a"),
], "tone.m4a");

runFfmpeg([
  "-f", "lavfi",
  "-i", "color=c=black:s=320x240:r=15:d=4",
  "-i", wavPath,
  "-shortest",
  "-c:v", "libx264",
  "-preset", "ultrafast",
  "-pix_fmt", "yuv420p",
  "-c:a", "aac",
  "-b:a", "96k",
  resolve(outputDir, "tone.mp4"),
], "tone.mp4");

console.log(`[media] Generated deterministic fixtures in ${outputDir}`);
