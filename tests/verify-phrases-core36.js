import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const filePath = resolve("assets/data/phrases/zh-Hant-core-36.json");
const pack = JSON.parse(await readFile(filePath, "utf8"));

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

assert(pack.id === "zh-Hant-core-36", "id 錯誤");
assert(/^\d+\.\d+\.\d+$/.test(pack.version), "version 非 semver");
assert(Array.isArray(pack.items) && pack.items.length === 36, "筆數需為 36");
assert(Array.isArray(pack.categories) && pack.categories.length > 0, "需要至少一個分類");

const catIds = new Set(pack.categories.map((c) => c.id));
const ids = new Set();
for (const it of pack.items) {
  assert(!ids.has(it.id), `id 重覆：${it.id}`);
  ids.add(it.id);
  assert(catIds.has(it.cat), `未知分類：${it.cat}`);
  assert(typeof it.text === "string" && it.text.trim().length > 0, `text 缺失：${it.id}`);
  if (it.alts) assert(Array.isArray(it.alts), `alts 必須為陣列：${it.id}`);
  if (it.tags) assert(Array.isArray(it.tags), `tags 必須為陣列：${it.id}`);
  if (it.difficulty) assert(["E", "M", "H"].includes(it.difficulty), `difficulty 非法：${it.id}`);
}

console.log("PASS: zh-Hant-core-36 結構正確");
