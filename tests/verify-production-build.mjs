import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const dist = resolve("dist");
const indexPath = resolve(dist, "index.html");
assert.ok(existsSync(indexPath), "dist/index.html is missing");

const html = readFileSync(indexPath, "utf8");
const buildDir = resolve(dist, "assets/build");
const buildFiles = readdirSync(buildDir);
const jsFiles = buildFiles.filter((name) => /-[A-Za-z0-9_-]{6,}\.js$/.test(name));
const cssFiles = buildFiles.filter((name) => /-[A-Za-z0-9_-]{6,}\.css$/.test(name));

assert.ok(jsFiles.length >= 2, "production build did not emit hashed JavaScript bundles");
assert.ok(cssFiles.length >= 1, "production build did not emit hashed CSS bundles");
assert.match(html, /assets\/build\/[A-Za-z0-9_-]+-[A-Za-z0-9_-]{6,}\.js/);
assert.match(html, /assets\/build\/[A-Za-z0-9_-]+-[A-Za-z0-9_-]{6,}\.css/);
assert.doesNotMatch(html, /assets\/(?:app|experiments\/experience-shell)\.js\?v=/);

for (const required of [
  "assets/avatar-evelyn.jpg",
  "assets/data",
  "assets/vendor/ffmpeg/ffmpeg-core.js",
  "assets/vendor/ffmpeg/worker.js",
  "favicon.ico",
  "ogp.png",
]) {
  assert.ok(existsSync(resolve(dist, required)), `runtime asset missing from production build: ${required}`);
}

console.log(`[PASS] Production build uses ${jsFiles.length} hashed JS and ${cssFiles.length} hashed CSS assets.`);
