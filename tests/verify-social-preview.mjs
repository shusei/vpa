import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const projectRoot = resolve(import.meta.dirname, "..");
const html = readFileSync(resolve(projectRoot, "index.html"), "utf8");
const expectedMeta = [
  '<link rel="canonical" href="https://shusei.github.io/vpa/"',
  '<meta property="og:image" content="https://shusei.github.io/vpa/ogp.png"',
  '<meta property="og:image:width" content="1200"',
  '<meta property="og:image:height" content="630"',
  '<meta name="twitter:card" content="summary_large_image"',
  '<meta name="twitter:image" content="https://shusei.github.io/vpa/ogp.png"',
  'assets/css/experiments.css?v=1.4.19',
  'assets/css/quick-experience.css?v=1.4.19',
  "window.VPA_EXPERIMENT_LOCALES = ['ja'];",
  'https://vpa-share.evelynjoellelin.workers.dev',
  'assets/experiments/experience-shell.js?v=1.4.19',
];
for (const token of expectedMeta) {
  assert.ok(html.includes(token), `Missing social metadata: ${token}`);
}

const png = readFileSync(resolve(projectRoot, "ogp.png"));
assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
assert.equal(png.readUInt32BE(16), 1200);
assert.equal(png.readUInt32BE(20), 630);
assert.ok(png.length < 5_000_000, "Social preview image must remain below 5 MB");
assert.ok(!html.includes("__APP_VERSION__"), "Production entry must not contain a cache placeholder");
assert.match(html, /document\.documentElement\.setAttribute\("data-experience", experience\)/);

console.log("production entry, social metadata, and 1200x630 PNG checks passed");
