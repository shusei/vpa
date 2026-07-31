import assert from "node:assert/strict";
import test from "node:test";
import {
  getPublicAppUrl,
  publishShareResult,
  shareServiceInternals,
} from "../assets/experiments/share-service.js";

test("normalizes the configured service origin and rejects insecure remote endpoints", () => {
  assert.equal(
    shareServiceInternals.normalizeOrigin("https://share.example/path"),
    "https://share.example",
  );
  assert.equal(shareServiceInternals.normalizeOrigin("http://share.example"), "");
  assert.equal(
    shareServiceInternals.normalizeOrigin("http://127.0.0.1:8787/path"),
    "http://127.0.0.1:8787",
  );
});

test("removes dev.html when deriving the public app URL", () => {
  assert.equal(getPublicAppUrl({
    href: "https://shusei.github.io/vpa/dev.html?test=1#result",
  }, {}), "https://shusei.github.io/vpa/");
});

test("publishes one compressed JPEG and validates the returned short URL", async () => {
  const imageBlob = new Blob([Uint8Array.from([0xff, 0xd8, 0xff])], { type: "image/jpeg" });
  let request;
  const result = await publishShareResult({
    fetchLike: async (url, options) => {
      request = { options, url };
      return new Response(JSON.stringify({
        id: "abcdefghijklmnop",
        imageUrl: "https://share.example/i/abcdefghijklmnop.jpg",
        url: "https://share.example/r/abcdefghijklmnop",
      }), {
        headers: { "Content-Type": "application/json" },
        status: 201,
      });
    },
    imageBlob,
    metadata: {
      description: "Result",
      targetUrl: "https://shusei.github.io/vpa/",
      title: "VPA",
    },
    serviceOrigin: "https://share.example",
  });
  assert.equal(request.url, "https://share.example/api/shares");
  assert.equal(request.options.method, "POST");
  assert.ok(request.options.body instanceof FormData);
  assert.equal(result.url, "https://share.example/r/abcdefghijklmnop");
});
