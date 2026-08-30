import {
  closeSync,
  copyFileSync,
  mkdirSync,
  openSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { resolve } from "node:path";

const outputDir = resolve("tests/.generated-media");
const sourceDir = resolve("fixtures/media");
mkdirSync(outputDir, { recursive: true });

for (const name of ["tone.wav", "tone-30.wav", "tone.mp3", "tone.m4a", "tone.mp4"]) {
  copyFileSync(resolve(sourceDir, name), resolve(outputDir, name));
}

const largePath = resolve(outputDir, "tone-large.mp4");
copyFileSync(resolve(sourceDir, "tone.mp4"), largePath);
const freeBoxSize = 32 * 1024 * 1024;
const freeBoxHeader = Buffer.alloc(8);
freeBoxHeader.writeUInt32BE(freeBoxSize, 0);
freeBoxHeader.write("free", 4, "ascii");
const largeFile = openSync(largePath, "a");
writeSync(largeFile, freeBoxHeader);
const padding = Buffer.alloc(64 * 1024);
for (let remaining = freeBoxSize - freeBoxHeader.length; remaining > 0;) {
  const bytes = Math.min(remaining, padding.length);
  writeSync(largeFile, padding, 0, bytes);
  remaining -= bytes;
}
closeSync(largeFile);

writeFileSync(resolve(outputDir, "corrupt.mp4"), Buffer.from("not a playable media file"));
writeFileSync(resolve(outputDir, "empty.mp4"), Buffer.alloc(0));

console.log(`[media] Generated deterministic fixtures in ${outputDir}`);
