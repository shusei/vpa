const PACK_PATH_PREFIX = "assets/data/phrases";
export const USER_EDITS_KEY = "phrases:userEdits";
export const USER_EDITS_SCHEMA = 1;

const packCache = new Map();

const clone = (value) => (typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value)));

function resolvePackPath(id) {
  return `${PACK_PATH_PREFIX}/${id}.json`;
}

function isPlainObject(val) {
  return typeof val === "object" && val !== null && !Array.isArray(val);
}

export async function loadPack(id) {
  if (!id) throw new Error("pack id is required");
  if (packCache.has(id)) return clone(packCache.get(id));
  const res = await fetch(resolvePackPath(id), { cache: "no-cache" });
  if (!res.ok) throw new Error(`failed to load pack ${id}: ${res.status}`);
  const pack = await res.json();
  const { ok, errors } = validatePack(pack);
  if (!ok) {
    const detail = errors.join("; ");
    throw new Error(`invalid pack ${id}: ${detail}`);
  }
  packCache.set(id, pack);
  return clone(pack);
}

export function validatePack(pack) {
  const errors = [];
  if (!isPlainObject(pack)) errors.push("pack 必須為物件");
  if (!pack.id) errors.push("缺少 id");
  if (!pack.lang) errors.push("缺少 lang");
  if (!pack.version || !/^\d+\.\d+\.\d+$/.test(pack.version)) errors.push("version 必須為 semver x.y.z");
  if (!pack.title) errors.push("缺少 title");
  if (!pack.license) errors.push("缺少 license");
  if (!Array.isArray(pack.categories)) errors.push("categories 必須為陣列");
  if (!Array.isArray(pack.items)) errors.push("items 必須為陣列");
  const catIds = new Set();
  if (Array.isArray(pack.categories)) {
    for (const cat of pack.categories) {
      if (!cat || typeof cat.id !== "string" || !cat.id.trim()) {
        errors.push("category id 無效");
        continue;
      }
      if (catIds.has(cat.id)) errors.push(`category id 重覆: ${cat.id}`);
      catIds.add(cat.id);
      if (typeof cat.name !== "string" || !cat.name.trim()) errors.push(`category ${cat.id} name 無效`);
    }
  }
  const itemIds = new Set();
  if (Array.isArray(pack.items)) {
    for (const item of pack.items) {
      if (!item || typeof item.id !== "string" || !item.id.trim()) {
        errors.push("item id 無效");
        continue;
      }
      if (itemIds.has(item.id)) errors.push(`item id 重覆: ${item.id}`);
      itemIds.add(item.id);
      if (typeof item.cat !== "string" || !item.cat.trim()) errors.push(`item ${item.id} cat 無效`);
      if (!catIds.has(item.cat)) errors.push(`item ${item.id} cat ${item.cat} 不存在`);
      if (typeof item.text !== "string" || !item.text.trim()) errors.push(`item ${item.id} text 無效`);
      if (item.alts && !Array.isArray(item.alts)) errors.push(`item ${item.id} alts 必須為陣列`);
      if (item.tags && !Array.isArray(item.tags)) errors.push(`item ${item.id} tags 必須為陣列`);
      if (item.difficulty && !["E", "M", "H"].includes(item.difficulty)) errors.push(`item ${item.id} difficulty 無效`);
    }
  }
  return { ok: errors.length === 0, errors };
}

export function listCategories(pack) {
  return clone(pack.categories ?? []);
}

export function listItems(pack) {
  return clone(pack.items ?? []);
}

export function search(pack, query, options = {}) {
  const q = (query ?? "").trim().toLowerCase();
  const { category, tags = [], difficulty, favorites } = options;
  const tagSet = new Set(tags ?? []);
  const favSet = Array.isArray(favorites) ? new Set(favorites) : null;
  return pack.items.filter((item) => {
    if (category && item.cat !== category) return false;
    if (difficulty && item.difficulty !== difficulty) return false;
    if (favSet && !favSet.has(item.id)) return false;
    if (tagSet.size) {
      if (!item.tags || !item.tags.some((t) => tagSet.has(t))) return false;
    }
    if (!q) return true;
    const haystack = [item.text, ...(item.alts ?? []), ...(item.tags ?? []), item.notes ?? ""].join("\n").toLowerCase();
    return haystack.includes(q);
  });
}

function normaliseEdits(edits = {}) {
  const base = {
    schema: USER_EDITS_SCHEMA,
    baseId: "",
    baseVersion: "",
    categories: {
      added: [],
      updated: {},
      removed: [],
      order: null,
    },
    items: {
      added: [],
      updated: {},
      removed: [],
      order: {},
    },
    favorites: [],
    metadata: {
      updatedAt: new Date().toISOString(),
    },
  };
  if (!isPlainObject(edits)) return clone(base);
  const merged = clone(base);
  merged.baseId = typeof edits.baseId === "string" ? edits.baseId : base.baseId;
  merged.baseVersion = typeof edits.baseVersion === "string" ? edits.baseVersion : base.baseVersion;
  if (isPlainObject(edits.categories)) {
    const cats = edits.categories;
    if (Array.isArray(cats.added)) merged.categories.added = cats.added.map(clone);
    if (isPlainObject(cats.updated)) merged.categories.updated = clone(cats.updated);
    if (Array.isArray(cats.removed)) merged.categories.removed = [...new Set(cats.removed)];
    if (Array.isArray(cats.order)) merged.categories.order = [...cats.order];
  }
  if (isPlainObject(edits.items)) {
    const items = edits.items;
    if (Array.isArray(items.added)) merged.items.added = items.added.map(clone);
    if (isPlainObject(items.updated)) merged.items.updated = clone(items.updated);
    if (Array.isArray(items.removed)) merged.items.removed = [...new Set(items.removed)];
    if (isPlainObject(items.order)) {
      const order = {};
      for (const [cat, ids] of Object.entries(items.order)) {
        if (Array.isArray(ids)) order[cat] = [...ids];
      }
      merged.items.order = order;
    }
  }
  if (Array.isArray(edits.favorites)) merged.favorites = [...new Set(edits.favorites.filter((id) => typeof id === "string"))];
  if (isPlainObject(edits.metadata) && typeof edits.metadata.updatedAt === "string") {
    merged.metadata.updatedAt = edits.metadata.updatedAt;
  }
  merged.schema = USER_EDITS_SCHEMA;
  return merged;
}

export function loadUserEdits() {
  try {
    const raw = localStorage.getItem(USER_EDITS_KEY);
    if (!raw) return normaliseEdits();
    const parsed = JSON.parse(raw);
    return normaliseEdits(parsed);
  } catch (err) {
    console.warn("[phrases-store] loadUserEdits failed", err);
    return normaliseEdits();
  }
}

export function saveUserEdits(edits) {
  const payload = normaliseEdits(edits);
  payload.metadata.updatedAt = new Date().toISOString();
  try {
    localStorage.setItem(USER_EDITS_KEY, JSON.stringify(payload));
  } catch (err) {
    console.warn("[phrases-store] saveUserEdits failed", err);
    throw err;
  }
  return payload;
}

export function clearUserEdits() {
  localStorage.removeItem(USER_EDITS_KEY);
}

export function exportUserPack(edits) {
  const payload = normaliseEdits(edits);
  const body = JSON.stringify(payload, null, 2);
  const blob = new Blob([body], { type: "application/json" });
  const stamp = new Date().toISOString().replace(/[-:]/g, "").split(".")[0];
  const filename = `phrases-user-edits-${stamp}.json`;
  return { blob, filename, payload };
}

async function readSourceText(source) {
  if (typeof source === "string") return source;
  if (source instanceof Blob) return await source.text();
  throw new Error("importUserPack 僅支援字串或 Blob/File");
}

export async function importUserPack(source, { expectedBaseId } = {}) {
  const text = await readSourceText(source);
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error("JSON 解析失敗，請確認檔案格式");
  }
  const edits = normaliseEdits(parsed);
  if (typeof expectedBaseId === "string" && edits.baseId && edits.baseId !== expectedBaseId) {
    throw new Error(`基底句庫不符：預期 ${expectedBaseId}，收到 ${edits.baseId || "未知"}`);
  }
  return edits;
}

export function applyEdits(basePack, edits) {
  if (!edits || !edits.baseId || edits.baseId !== basePack.id) {
  return { pack: clone(basePack), favorites: [] };
  }
  const categories = new Map(basePack.categories.map((cat) => [cat.id, { ...cat }]));
  for (const id of edits.categories.removed) categories.delete(id);
  for (const [id, cat] of Object.entries(edits.categories.updated)) {
    if (categories.has(id)) categories.set(id, { ...categories.get(id), ...cat });
  }
  for (const cat of edits.categories.added) {
    if (cat && typeof cat.id === "string" && cat.id) {
      categories.set(cat.id, { ...cat });
    }
  }
  let categoryOrder = Array.isArray(edits.categories.order) ? edits.categories.order.filter((id) => categories.has(id)) : [];
  const missingCats = Array.from(categories.keys()).filter((id) => !categoryOrder.includes(id));
  categoryOrder = [...categoryOrder, ...missingCats];
  const mergedCategories = categoryOrder.map((id) => categories.get(id));

  const baseItems = new Map(basePack.items.map((item) => [item.id, { ...item }]));
  for (const id of edits.items.removed) baseItems.delete(id);
  for (const [id, itemPatch] of Object.entries(edits.items.updated)) {
    if (baseItems.has(id)) baseItems.set(id, { ...baseItems.get(id), ...itemPatch });
  }
  for (const item of edits.items.added) {
    if (item && typeof item.id === "string" && item.id) {
      baseItems.set(item.id, { ...item });
    }
  }
  const itemsByCat = new Map();
  for (const [id, item] of baseItems.entries()) {
    if (!categories.has(item.cat)) continue;
    if (!itemsByCat.has(item.cat)) itemsByCat.set(item.cat, []);
    itemsByCat.get(item.cat).push({ ...item, id });
  }
  const finalItems = [];
  for (const catId of categoryOrder) {
    const order = edits.items.order[catId] ?? [];
    const list = itemsByCat.get(catId) ?? [];
    const map = new Map(list.map((it) => [it.id, it]));
    const seq = [];
    for (const itemId of order) {
      if (map.has(itemId)) {
        seq.push(map.get(itemId));
        map.delete(itemId);
      }
    }
    for (const remaining of map.values()) seq.push(remaining);
    for (const item of seq) finalItems.push(item);
  }
  const favSet = new Set(edits.favorites);
  const favorites = Array.from(favSet).filter((id) => baseItems.has(id));
  const pack = {
    ...clone(basePack),
    categories: mergedCategories,
    items: finalItems,
  };
  return { pack, favorites };
}
