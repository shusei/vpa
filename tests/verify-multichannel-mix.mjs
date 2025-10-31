import assert from "node:assert/strict";

import { mixChannelDataToMono } from "../assets/js/audio-utils.js";

function toArray(float32){
  return Array.from(float32, (value) => Number(value.toFixed(6)));
}

const frameCount = 8;
const left = Float32Array.from([0.5, 0.4, 0.3, 0.2, -0.1, -0.2, -0.3, -0.4]);
const right = Float32Array.from([-0.4, -0.3, -0.2, -0.1, 0.2, 0.3, 0.4, 0.5]);
const center = Float32Array.from([0.1, 0.1, 0.05, 0.05, 0, 0, -0.05, -0.1]);
const lfeNaN = Float32Array.from([0, NaN, 0, NaN, 0, NaN, 0, NaN]);

const out = new Float32Array(frameCount);
const valid = mixChannelDataToMono([left, right, center], out);

assert.equal(valid, 3, "should mix all three valid channels");
assert.ok(out.some((value) => value !== 0), "mixed output should not be silent");

const expected = new Float32Array(frameCount);
for (let i = 0; i < frameCount; i++){
  expected[i] = (left[i] + right[i] + center[i]) / 3;
}
assert.deepEqual(toArray(out), toArray(expected), "mono mix should average channel samples");

const outWithInvalid = new Float32Array(frameCount);
const validWithInvalid = mixChannelDataToMono([left, lfeNaN, right, center], outWithInvalid);

assert.equal(validWithInvalid, 3, "channels containing NaN should be skipped");
assert.deepEqual(toArray(outWithInvalid), toArray(expected), "invalid channels must not affect mix");

const outAllInvalid = new Float32Array(frameCount);
const allInvalid = mixChannelDataToMono([lfeNaN], outAllInvalid);
assert.equal(allInvalid, 0, "all-invalid input should report zero valid channels");
assert.deepEqual(toArray(outAllInvalid), toArray(new Float32Array(frameCount)), "output remains silence when all channels invalid");

console.info("Multi-channel mix verification passed.");
