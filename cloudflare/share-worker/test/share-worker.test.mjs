import assert from "node:assert/strict";
import test from "node:test";
import worker from "../src/index.js";

class MemoryD1Statement {
  constructor(database, sql) {
    this.database = database;
    this.sql = sql;
    this.args = [];
  }

  bind(...args) {
    this.args = args;
    return this;
  }

  async run() {
    if (/INSERT INTO shares/i.test(this.sql)) {
      const [
        id,
        createdAt,
        expiresAt,
        locale,
        title,
        description,
        alt,
        targetUrl,
        image,
      ] = this.args;
      this.database.rows.set(id, {
        alt,
        created_at: createdAt,
        description,
        expires_at: expiresAt,
        id,
        image: Array.from(new Uint8Array(image)),
        locale,
        target_url: targetUrl,
        title,
      });
      return { success: true };
    }
    if (/DELETE FROM shares/i.test(this.sql)) {
      const [now] = this.args;
      let deleted = 0;
      for (const [id, row] of this.database.rows) {
        if (row.expires_at <= now && deleted < 5000) {
          this.database.rows.delete(id);
          deleted += 1;
        }
      }
      return { meta: { changes: deleted }, success: true };
    }
    throw new Error(`unsupported D1 run query: ${this.sql}`);
  }

  async first() {
    if (!/SELECT locale/i.test(this.sql)) {
      throw new Error(`unsupported D1 first query: ${this.sql}`);
    }
    const [id, now] = this.args;
    const row = this.database.rows.get(id);
    if (!row || row.expires_at <= now) return null;
    return structuredClone(row);
  }
}

class MemoryD1 {
  constructor() {
    this.rows = new Map();
  }

  prepare(sql) {
    return new MemoryD1Statement(this, sql);
  }
}

const env = {
  PUBLIC_APP_URL: "https://shusei.github.io/vpa/",
  SHARE_DB: new MemoryD1(),
  SHARE_TTL_DAYS: "365",
  SITE_ORIGINS: "https://shusei.github.io,http://127.0.0.1:8080",
};

function jpegBlob() {
  return new Blob([
    Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46]),
  ], { type: "image/jpeg" });
}

function shareForm(overrides = {}) {
  const form = new FormData();
  form.append("image", jpegBlob(), "vpa-result.jpg");
  form.append("metadata", JSON.stringify({
    alt: "女性化傾向 64%",
    description: "我的聲音分析結果是 64%。",
    locale: "zh-Hant",
    schema: 1,
    targetUrl: "https://shusei.github.io/vpa/#vpa-challenge=abc",
    title: "VPA｜我的聲音印象",
    ...overrides,
  }));
  return form;
}

test("creates one D1 share row and serves immediate OG metadata", async () => {
  const response = await worker.fetch(new Request("https://share.example/api/shares", {
    body: shareForm(),
    headers: { Origin: "https://shusei.github.io" },
    method: "POST",
  }), env);
  assert.equal(response.status, 201);
  assert.equal(response.headers.get("Access-Control-Allow-Origin"), "https://shusei.github.io");
  const created = await response.json();
  assert.match(created.url, /^https:\/\/share\.example\/r\/[A-Za-z0-9_-]{16}$/);
  assert.match(created.imageUrl, /^https:\/\/share\.example\/i\/[A-Za-z0-9_-]{16}\.jpg$/);
  assert.match(created.expiresAt, /^\d{4}-\d{2}-\d{2}T/);
  assert.equal(env.SHARE_DB.rows.size, 1);

  const result = await worker.fetch(new Request(created.url, {
    headers: { "User-Agent": "Twitterbot/1.0" },
  }), env);
  assert.equal(result.status, 200);
  assert.match(result.headers.get("Content-Type"), /^text\/html/);
  const html = await result.text();
  assert.match(html, /twitter:card" content="summary_large_image"/);
  assert.match(html, /og:image" content="https:\/\/share\.example\/i\//);
  assert.match(html, /og:image:type" content="image\/jpeg"/);
  assert.match(html, /女性化傾向 64%/);
  assert.match(html, /https:\/\/shusei\.github\.io\/vpa\/#vpa-challenge=abc/);
  assert.equal(result.headers.get("Vary"), "User-Agent");

  const human = await worker.fetch(new Request(created.url, {
    headers: { "User-Agent": "Twitter for iPhone" },
  }), env);
  assert.equal(human.status, 302);
  assert.equal(human.headers.get("Location"), "https://shusei.github.io/vpa/#vpa-challenge=abc");
  assert.equal(human.headers.get("Cache-Control"), "private, no-store");

  const image = await worker.fetch(new Request(created.imageUrl), env);
  assert.equal(image.status, 200);
  assert.equal(image.headers.get("Content-Type"), "image/jpeg");
  assert.match(image.headers.get("Cache-Control"), /max-age=/);
  assert.deepEqual(
    Array.from(new Uint8Array(await image.arrayBuffer()).slice(0, 3)),
    [0xff, 0xd8, 0xff],
  );
});

test("rejects uploads from an unconfigured browser origin", async () => {
  const response = await worker.fetch(new Request("https://share.example/api/shares", {
    body: new FormData(),
    headers: { Origin: "https://attacker.example" },
    method: "POST",
  }), env);
  assert.equal(response.status, 403);
});

test("rejects result targets outside the public app", async () => {
  const response = await worker.fetch(new Request("https://share.example/api/shares", {
    body: shareForm({ targetUrl: "https://attacker.example/" }),
    headers: { Origin: "https://shusei.github.io" },
    method: "POST",
  }), env);
  assert.equal(response.status, 400);
});

test("expired rows are removed in bounded scheduled batches", async () => {
  env.SHARE_DB.rows.set("expiredabcdefgh", {
    expires_at: Math.floor(Date.now() / 1000) - 1,
  });
  let cleanup;
  await worker.scheduled({}, env, {
    waitUntil(promise) {
      cleanup = promise;
    },
  });
  await cleanup;
  assert.equal(env.SHARE_DB.rows.has("expiredabcdefgh"), false);
});
