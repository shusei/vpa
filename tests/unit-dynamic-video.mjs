import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  defaultClipRange,
} from "../assets/experiments/dynamic-voice-card.js";

const controllerSource = readFileSync(new URL("../assets/experiments/dynamic-card-controller.js", import.meta.url), "utf8");

test("the video duration exactly matches short recordings", () => {
  assert.deepEqual(defaultClipRange(4.25), {
    duration: 4.25,
    end: 4.25,
    outputDuration: 4.25,
    start: 0,
  });
});

test("the default video range preserves recordings longer than two minutes", () => {
  assert.deepEqual(defaultClipRange(135.4), {
    duration: 135.4,
    end: 135.4,
    outputDuration: 135.4,
    start: 0,
  });
});

test("the video UI has no trimming controls or second preview button", () => {
  assert.doesNotMatch(controllerSource, /data-dynamic-(?:start|end|generate|preview-play)/);
  assert.doesNotMatch(controllerSource, /requestedClip/);
  assert.match(controllerSource, /makePreviewAudible\(video\);/);
});
