import localeStrings from "../js/phrases/i18n/zh-Hant.js";
import {
  loadPack,
  loadUserEdits,
  saveUserEdits,
  clearUserEdits,
  applyEdits,
  exportUserPack,
  importUserPack,
} from "../js/phrases/store.js";

const BASE_PACK_ID = "zh-Hant-core-36";

const dom = {};
const state = {
  basePack: null,
  appliedPack: null,
  edits: null,
  favorites: [],
  filters: {
    category: "all",
    difficulty: "all",
    favoritesOnly: false,
    search: "",
  },
  practice: {
    mode: "sequential",
    duration: 0,
    hideHints: false,
    active: false,
    paused: false,
    queue: [],
    pointer: 0,
    timerId: null,
    remainingMs: 0,
    startedAt: 0,
  },
};

function qs(id) {
  return document.getElementById(id);
}

function formatDate(iso) {
  if (!iso) return "";
  try {
    const date = new Date(iso);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
  } catch (err) {
    return iso;
  }
}

async function init() {
  bindDom();
  applyLocale();
  await bootstrapData();
  attachEvents();
  renderAll();
}

function bindDom() {
  dom.searchInput = qs("searchInput");
  dom.filterCategory = qs("filterCategory");
  dom.filterDifficulty = qs("filterDifficulty");
  dom.filterFavorites = qs("filterFavorites");
  dom.categoryList = qs("categoryList");
  dom.phraseList = qs("phraseList");
  dom.phraseCount = qs("phrase-count");
  dom.emptyState = qs("emptyState");
  dom.addCategoryBtn = qs("addCategoryBtn");
  dom.addPhraseBtn = qs("addPhraseBtn");
  dom.labTitle = qs("lab-title");
  dom.baseInfo = qs("lab-base-info");
  dom.importBtn = qs("importBtn");
  dom.exportBtn = qs("exportBtn");
  dom.resetBtn = qs("resetBtn");
  dom.importFile = qs("importFile");
  dom.categoryDialog = qs("categoryDialog");
  dom.categoryDialogTitle = qs("categoryDialogTitle");
  dom.categoryNameInput = qs("categoryNameInput");
  dom.categoryDialogSave = qs("categoryDialogSave");
  dom.phraseDialog = qs("phraseDialog");
  dom.phraseDialogTitle = qs("phraseDialogTitle");
  dom.phraseTextInput = qs("phraseTextInput");
  dom.phraseAltsInput = qs("phraseAltsInput");
  dom.phraseTagsInput = qs("phraseTagsInput");
  dom.phraseNotesInput = qs("phraseNotesInput");
  dom.phraseDifficultyInput = qs("phraseDifficultyInput");
  dom.phraseCategoryInput = qs("phraseCategoryInput");
  dom.phraseDialogSave = qs("phraseDialogSave");
  dom.practiceMode = qs("practiceMode");
  dom.practiceDuration = qs("practiceDuration");
  dom.practiceHideHints = qs("practiceHideHints");
  dom.practiceStart = qs("practiceStart");
  dom.practicePause = qs("practicePause");
  dom.practiceStop = qs("practiceStop");
  dom.practiceNext = qs("practiceNext");
  dom.practiceStatus = qs("practiceStatus");
  dom.practiceCard = qs("practiceCard");
  dom.practicePhrase = qs("practicePhrase");
  dom.practiceMeta = qs("practiceMeta");
  dom.practiceHints = qs("practiceHints");
  dom.categoryTemplate = document.getElementById("categoryItemTemplate");
  dom.phraseTemplate = document.getElementById("phraseCardTemplate");
}

function applyLocale() {
  document.title = `${localeStrings.pageTitle} | Voice Practice Assistant`;
  dom.labTitle.textContent = localeStrings.pageTitle;
  dom.searchInput.placeholder = localeStrings.searchPlaceholder;
  dom.filterFavorites.nextElementSibling.textContent = localeStrings.filterFavorites;
  qs("cat-title").textContent = localeStrings.filterCategory;
  qs("phrase-title").textContent = "句子";
  qs("cat-hint").textContent = localeStrings.reorderCategories;
  qs("phrase-hint").textContent = localeStrings.reorderPhrases;
  dom.addCategoryBtn.textContent = localeStrings.addCategory;
  dom.addPhraseBtn.textContent = localeStrings.addPhrase;
  dom.importBtn.textContent = localeStrings.import;
  dom.exportBtn.textContent = localeStrings.export;
  dom.resetBtn.textContent = localeStrings.reset;
  qs("practice-title").textContent = localeStrings.practicePanelTitle;
  dom.practiceMode.previousElementSibling.textContent = localeStrings.practiceMode;
  dom.practiceDuration.previousElementSibling.textContent = localeStrings.practiceTimer;
  dom.practiceHideHints.nextElementSibling.textContent = localeStrings.practiceHideHints;
  dom.practiceStart.textContent = localeStrings.practiceStart;
  dom.practicePause.textContent = localeStrings.practicePause;
  dom.practiceStop.textContent = localeStrings.practiceStop;
  dom.practiceNext.textContent = localeStrings.practiceNext;
  dom.emptyState.textContent = localeStrings.emptyState;
  dom.categoryDialogTitle.textContent = localeStrings.addCategory;
  dom.categoryDialogSave.textContent = localeStrings.save;
  dom.phraseDialogTitle.textContent = localeStrings.addPhrase;
  dom.phraseDialogSave.textContent = localeStrings.save;
  dom.categoryDialog.querySelector("label span").textContent = localeStrings.newCategoryPlaceholder;
  dom.categoryNameInput.placeholder = localeStrings.newCategoryPlaceholder;
  dom.phraseDialog.querySelectorAll("label span")[0].textContent = localeStrings.phraseText;
  dom.phraseDialog.querySelectorAll("label span")[1].textContent = localeStrings.phraseAlts;
  dom.phraseDialog.querySelectorAll("label span")[2].textContent = localeStrings.phraseTags;
  dom.phraseDialog.querySelectorAll("label span")[3].textContent = localeStrings.phraseNotes;
  dom.phraseDialog.querySelectorAll("label span")[4].textContent = localeStrings.phraseDifficulty;
  dom.phraseDialog.querySelectorAll("label span")[5].textContent = localeStrings.filterCategory;
  dom.categoryDialog.querySelector("button[value='cancel']").textContent = localeStrings.cancel;
  dom.phraseDialog.querySelector("button[value='cancel']").textContent = localeStrings.cancel;
  const sequentialOpt = dom.practiceMode.querySelector("option[value='sequential']");
  const randomOpt = dom.practiceMode.querySelector("option[value='random']");
  if (sequentialOpt) sequentialOpt.textContent = localeStrings.practiceModes.sequential;
  if (randomOpt) randomOpt.textContent = localeStrings.practiceModes.random;
  const durationOptions = dom.practiceDuration.querySelectorAll("option");
  for (const opt of durationOptions) {
    if (opt.value === "0") opt.textContent = localeStrings.practiceDurations.off;
    if (opt.value === "30") opt.textContent = localeStrings.practiceDurations[30];
    if (opt.value === "60") opt.textContent = localeStrings.practiceDurations[60];
    if (opt.value === "90") opt.textContent = localeStrings.practiceDurations[90];
  }
}

async function bootstrapData() {
  state.basePack = await loadPack(BASE_PACK_ID);
  state.edits = loadUserEdits();
  if (!state.edits.baseId) state.edits.baseId = state.basePack.id;
  if (!state.edits.baseVersion) state.edits.baseVersion = state.basePack.version;
  applyEditsAndPersist(false);
}

function attachEvents() {
  dom.searchInput.addEventListener("input", (e) => {
    state.filters.search = e.target.value;
    renderPhrases();
  });
  dom.filterCategory.addEventListener("change", (e) => {
    state.filters.category = e.target.value;
    renderPhrases();
  });
  dom.filterDifficulty.addEventListener("change", (e) => {
    state.filters.difficulty = e.target.value;
    renderPhrases();
  });
  dom.filterFavorites.addEventListener("change", (e) => {
    state.filters.favoritesOnly = e.target.checked;
    renderPhrases();
  });
  dom.addCategoryBtn.addEventListener("click", () => openCategoryDialog());
  dom.addPhraseBtn.addEventListener("click", () => openPhraseDialog());
  dom.importBtn.addEventListener("click", () => dom.importFile.click());
  dom.importFile.addEventListener("change", handleImportFile);
  dom.exportBtn.addEventListener("click", handleExport);
  dom.resetBtn.addEventListener("click", handleReset);
  dom.categoryDialog.addEventListener("close", handleCategoryDialogClose);
  dom.phraseDialog.addEventListener("close", handlePhraseDialogClose);
  dom.practiceStart.addEventListener("click", startPractice);
  dom.practicePause.addEventListener("click", togglePracticePause);
  dom.practiceStop.addEventListener("click", stopPractice);
  dom.practiceNext.addEventListener("click", nextPractice);
  dom.practiceMode.addEventListener("change", (e) => {
    state.practice.mode = e.target.value;
  });
  dom.practiceDuration.addEventListener("change", (e) => {
    state.practice.duration = Number(e.target.value);
  });
  dom.practiceHideHints.addEventListener("change", (e) => {
    state.practice.hideHints = e.target.checked;
    updatePracticeCard();
  });
  dom.categoryList.addEventListener("click", handleCategoryListClick);
  dom.phraseList.addEventListener("click", handlePhraseListClick);
  setupCategoryDrag();
  setupPhraseDrag();
}

function renderAll() {
  renderHeaderInfo();
  renderDifficultyOptions();
  renderCategoryOptions();
  renderCategories();
  renderPhrases();
  updatePracticeCard();
}

function renderDifficultyOptions() {
  const select = dom.filterDifficulty;
  const prev = select.value;
  select.innerHTML = "";
  const optAll = document.createElement("option");
  optAll.value = "all";
  optAll.textContent = localeStrings.filterDifficulty;
  select.appendChild(optAll);
  for (const key of ["E", "M", "H"]) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = localeStrings.difficultyLabels[key];
    select.appendChild(opt);
  }
  select.value = prev && [...select.options].some((opt) => opt.value === prev) ? prev : "all";
  state.filters.difficulty = select.value;
}

function renderHeaderInfo() {
  dom.baseInfo.textContent = `${localeStrings.basePackInfo}${state.basePack?.title || ""} · ${localeStrings.updatedAt} ${formatDate(state.edits?.metadata?.updatedAt)}`;
}

function renderCategoryOptions() {
  if (!state.appliedPack) return;
  const select = dom.filterCategory;
  const prev = select.value;
  select.innerHTML = "";
  const optionAll = document.createElement("option");
  optionAll.value = "all";
  optionAll.textContent = "全部";
  select.appendChild(optionAll);
  for (const cat of state.appliedPack.categories) {
    const opt = document.createElement("option");
    opt.value = cat.id;
    opt.textContent = cat.name;
    select.appendChild(opt);
  }
  select.value = prev && [...select.options].some((opt) => opt.value === prev) ? prev : "all";
  state.filters.category = select.value;
  const catSelect = dom.phraseCategoryInput;
  catSelect.innerHTML = "";
  for (const cat of state.appliedPack.categories) {
    const opt = document.createElement("option");
    opt.value = cat.id;
    opt.textContent = cat.name;
    catSelect.appendChild(opt);
  }
}

function renderCategories() {
  if (!state.appliedPack) return;
  const list = dom.categoryList;
  list.innerHTML = "";
  for (const cat of state.appliedPack.categories) {
    const item = dom.categoryTemplate.content.firstElementChild.cloneNode(true);
    item.dataset.id = cat.id;
    const label = item.querySelector(".category-item__label");
    label.textContent = cat.name;
    label.setAttribute("role", "option");
    label.setAttribute("aria-selected", state.filters.category === cat.id ? "true" : "false");
    if (state.filters.category === cat.id) item.classList.add("is-active");
    list.appendChild(item);
  }
}

function renderPhrases() {
  if (!state.appliedPack) return;
  const { items } = state.appliedPack;
  const filtered = getFilteredItems(items);
  dom.phraseCount.textContent = `${filtered.length} / ${items.length}`;
  dom.emptyState.hidden = filtered.length > 0;
  dom.phraseList.innerHTML = "";
  const catMap = new Map(state.appliedPack.categories.map((cat) => [cat.id, cat.name]));
  for (const item of filtered) {
    const card = dom.phraseTemplate.content.firstElementChild.cloneNode(true);
    card.dataset.id = item.id;
    card.dataset.cat = item.cat;
    card.querySelector(".phrase-text").textContent = item.text;
    const metaParts = [];
    metaParts.push(catMap.get(item.cat) || item.cat);
    if (item.difficulty && localeStrings.difficultyLabels[item.difficulty]) {
      metaParts.push(localeStrings.difficultyLabels[item.difficulty]);
    }
    if (item.tags?.length) metaParts.push(`#${item.tags.join(" · ")}`);
    card.querySelector(".phrase-meta").textContent = metaParts.join(" ｜ ");
    const altList = card.querySelector(".phrase-alts");
    altList.innerHTML = "";
    if (item.alts?.length) {
      for (const alt of item.alts) {
        const li = document.createElement("li");
        li.textContent = alt;
        altList.appendChild(li);
      }
    } else {
      altList.hidden = true;
    }
    const noteEl = card.querySelector(".phrase-notes");
    if (item.notes) {
      noteEl.textContent = item.notes;
    } else {
      noteEl.hidden = true;
    }
    const star = card.querySelector(".phrase-star");
    const isFav = state.favorites.includes(item.id);
    star.setAttribute("aria-pressed", isFav ? "true" : "false");
    star.textContent = isFav ? "★" : "☆";
    dom.phraseList.appendChild(card);
  }
}

function getFilteredItems(items) {
  const search = state.filters.search.trim().toLowerCase();
  const category = state.filters.category;
  const difficulty = state.filters.difficulty;
  const favoritesOnly = state.filters.favoritesOnly;
  return items.filter((item) => {
    if (category !== "all" && item.cat !== category) return false;
    if (difficulty !== "all" && item.difficulty !== difficulty) return false;
    if (favoritesOnly && !state.favorites.includes(item.id)) return false;
    if (!search) return true;
    const haystack = [item.text, ...(item.alts ?? []), ...(item.tags ?? []), item.notes ?? ""].join("\n").toLowerCase();
    return haystack.includes(search);
  });
}

function handleCategoryListClick(event) {
  const target = event.target;
  const item = target.closest(".category-item");
  if (!item) return;
  const id = item.dataset.id;
  if (target.classList.contains("rename")) {
    const cat = state.appliedPack.categories.find((c) => c.id === id);
    openCategoryDialog(cat);
  } else if (target.classList.contains("delete")) {
    if (confirm(localeStrings.confirmDeleteCategory)) {
      removeCategory(id);
    }
  } else if (target.classList.contains("category-item__label")) {
    state.filters.category = id;
    dom.filterCategory.value = id;
    renderCategories();
    renderPhrases();
  }
}

function handlePhraseListClick(event) {
  const target = event.target;
  const card = target.closest(".phrase-card");
  if (!card) return;
  const id = card.dataset.id;
  if (target.classList.contains("phrase-star")) {
    toggleFavorite(id);
  } else if (target.classList.contains("edit")) {
    const phrase = state.appliedPack.items.find((it) => it.id === id);
    openPhraseDialog(phrase);
  } else if (target.classList.contains("delete")) {
    if (confirm(localeStrings.confirmDeletePhrase)) {
      removePhrase(id);
    }
  }
}

function toggleFavorite(id) {
  const set = new Set(state.edits.favorites || []);
  if (set.has(id)) set.delete(id); else set.add(id);
  state.edits.favorites = Array.from(set);
  applyEditsAndPersist();
  renderPhrases();
}

function openCategoryDialog(category) {
  dom.categoryDialog.dataset.editing = category ? category.id : "";
  dom.categoryDialogTitle.textContent = category ? localeStrings.renameCategory : localeStrings.addCategory;
  dom.categoryNameInput.value = category ? category.name : "";
  dom.categoryDialog.showModal();
  dom.categoryNameInput.focus();
}

function handleCategoryDialogClose(event) {
  if (dom.categoryDialog.returnValue !== "default") return;
  const name = dom.categoryNameInput.value.trim();
  if (!name) return;
  const editingId = dom.categoryDialog.dataset.editing;
  if (editingId) {
    renameCategory(editingId, name);
  } else {
    addCategory(name);
  }
}

function openPhraseDialog(phrase) {
  dom.phraseDialog.dataset.editing = phrase ? phrase.id : "";
  dom.phraseDialogTitle.textContent = phrase ? localeStrings.editPhrase : localeStrings.addPhrase;
  dom.phraseTextInput.value = phrase ? phrase.text : "";
  dom.phraseAltsInput.value = phrase?.alts?.join("\n") ?? "";
  dom.phraseTagsInput.value = phrase?.tags?.join(",") ?? "";
  dom.phraseNotesInput.value = phrase?.notes ?? "";
  dom.phraseDifficultyInput.value = phrase?.difficulty ?? "";
  dom.phraseCategoryInput.value = phrase?.cat ?? state.filters.category ?? dom.phraseCategoryInput.value;
  dom.phraseDialog.showModal();
  dom.phraseTextInput.focus();
}

function handlePhraseDialogClose() {
  if (dom.phraseDialog.returnValue !== "default") return;
  const payload = {
    text: dom.phraseTextInput.value.trim(),
    alts: dom.phraseAltsInput.value
      .split(/\n+/)
      .map((s) => s.trim())
      .filter(Boolean),
    tags: dom.phraseTagsInput.value
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean),
    notes: dom.phraseNotesInput.value.trim(),
    difficulty: dom.phraseDifficultyInput.value || undefined,
    cat: dom.phraseCategoryInput.value,
  };
  if (!payload.text) return;
  const editingId = dom.phraseDialog.dataset.editing;
  if (editingId) {
    updatePhrase(editingId, payload);
  } else {
    createPhrase(payload);
  }
}

function addCategory(name) {
  const id = generateCategoryId(name);
  const existing = state.appliedPack.categories.find((c) => c.id === id);
  let finalId = id;
  let counter = 1;
  while (existing || state.edits.categories.added.some((c) => c.id === finalId)) {
    finalId = `${id}-${counter++}`;
  }
  state.edits.categories.added.push({ id: finalId, name });
  state.edits.categories.removed = state.edits.categories.removed.filter((cid) => cid !== finalId);
  state.edits.categories.order = state.edits.categories.order || [];
  state.edits.categories.order.push(finalId);
  applyEditsAndPersist();
  state.filters.category = finalId;
  dom.filterCategory.value = finalId;
  renderCategories();
  renderPhrases();
}

function renameCategory(id, name) {
  const added = state.edits.categories.added.find((cat) => cat.id === id);
  if (added) {
    added.name = name;
  } else {
    state.edits.categories.updated[id] = { ...(state.edits.categories.updated[id] || {}), name };
  }
  applyEditsAndPersist();
  renderCategories();
  renderPhrases();
}

function removeCategory(id) {
  const addedIndex = state.edits.categories.added.findIndex((cat) => cat.id === id);
  if (addedIndex >= 0) {
    state.edits.categories.added.splice(addedIndex, 1);
  } else {
    if (!state.edits.categories.removed.includes(id)) state.edits.categories.removed.push(id);
    delete state.edits.categories.updated[id];
  }
  // Remove phrases under category
  const items = state.appliedPack.items.filter((item) => item.cat === id);
  for (const item of items) {
    const addedIdx = state.edits.items.added.findIndex((p) => p.id === item.id);
    if (addedIdx >= 0) {
      state.edits.items.added.splice(addedIdx, 1);
    } else {
      if (!state.edits.items.removed.includes(item.id)) state.edits.items.removed.push(item.id);
      delete state.edits.items.updated[item.id];
    }
    state.edits.favorites = state.edits.favorites.filter((fav) => fav !== item.id);
  }
  delete state.edits.items.order[id];
  applyEditsAndPersist();
  state.filters.category = "all";
  dom.filterCategory.value = "all";
  renderCategories();
  renderPhrases();
}

function createPhrase(payload) {
  const id = generatePhraseId(payload.text);
  let finalId = id;
  let counter = 1;
  while (
    state.appliedPack.items.some((item) => item.id === finalId) ||
    state.edits.items.added.some((item) => item.id === finalId)
  ) {
    finalId = `${id}-${counter++}`;
  }
  state.edits.items.added.push({
    id: finalId,
    cat: payload.cat,
    text: payload.text,
    alts: payload.alts,
    tags: payload.tags,
    difficulty: payload.difficulty,
    notes: payload.notes,
  });
  state.edits.items.order[payload.cat] = state.edits.items.order[payload.cat] || [];
  state.edits.items.order[payload.cat].push(finalId);
  applyEditsAndPersist();
  renderPhrases();
}

function updatePhrase(id, payload) {
  const added = state.edits.items.added.find((item) => item.id === id);
  const targetCat = payload.cat;
  if (added) {
    Object.assign(added, payload);
  } else {
    state.edits.items.updated[id] = { ...(state.edits.items.updated[id] || {}), ...payload };
  }
  if (payload.cat) {
    ensureItemOrderMove(id, payload.cat);
  }
  applyEditsAndPersist();
  renderPhrases();
}

function ensureItemOrderMove(id, targetCat) {
  for (const cat of Object.keys(state.edits.items.order)) {
    state.edits.items.order[cat] = (state.edits.items.order[cat] || []).filter((pid) => pid !== id);
  }
  state.edits.items.order[targetCat] = state.edits.items.order[targetCat] || [];
  if (!state.edits.items.order[targetCat].includes(id)) {
    state.edits.items.order[targetCat].push(id);
  }
}

function removePhrase(id) {
  const addedIndex = state.edits.items.added.findIndex((item) => item.id === id);
  if (addedIndex >= 0) {
    state.edits.items.added.splice(addedIndex, 1);
  } else {
    if (!state.edits.items.removed.includes(id)) state.edits.items.removed.push(id);
    delete state.edits.items.updated[id];
  }
  for (const cat of Object.keys(state.edits.items.order)) {
    state.edits.items.order[cat] = (state.edits.items.order[cat] || []).filter((pid) => pid !== id);
  }
  state.edits.favorites = state.edits.favorites.filter((fav) => fav !== id);
  applyEditsAndPersist();
  renderPhrases();
}

function handleImportFile(event) {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  importUserPack(file, { expectedBaseId: BASE_PACK_ID })
    .then((edits) => {
      state.edits = { ...edits, baseId: BASE_PACK_ID, baseVersion: state.basePack.version };
      applyEditsAndPersist();
      alert(localeStrings.importSuccess);
    })
    .catch((err) => {
      alert(`${localeStrings.importError}${err.message}`);
    });
}

function handleExport() {
  const { blob, filename } = exportUserPack(state.edits);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 0);
}

function handleReset() {
  if (!confirm(localeStrings.confirmReset)) return;
  clearUserEdits();
  state.edits = loadUserEdits();
  state.edits.baseId = state.basePack.id;
  state.edits.baseVersion = state.basePack.version;
  applyEditsAndPersist(false);
  renderAll();
}

function applyEditsAndPersist(persist = true) {
  const { pack, favorites } = applyEdits(state.basePack, state.edits);
  state.appliedPack = pack;
  state.favorites = favorites;
  ensureOrderIntegrity();
  if (persist !== false) {
    saveUserEdits(state.edits);
  }
}

function ensureOrderIntegrity() {
  if (!state.edits.categories.order) state.edits.categories.order = [];
  state.edits.categories.order = state.appliedPack.categories.map((cat) => cat.id);
  const validCats = new Set(state.appliedPack.categories.map((cat) => cat.id));
  for (const cat of state.appliedPack.categories) {
    state.edits.items.order[cat.id] = state.appliedPack.items
      .filter((item) => item.cat === cat.id)
      .map((item) => item.id);
  }
  for (const key of Object.keys(state.edits.items.order)) {
    if (!validCats.has(key)) delete state.edits.items.order[key];
  }
}

function generateCategoryId(name) {
  return slugify(name || "cat");
}

function generatePhraseId(text) {
  return slugify(text || "phrase");
}

function slugify(input) {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-+/g, "-")
    .slice(0, 48) || "item";
}

function setupCategoryDrag() {
  let dragId = null;
  dom.categoryList.addEventListener("dragstart", (e) => {
    const li = e.target.closest(".category-item");
    if (!li) return;
    dragId = li.dataset.id;
    e.dataTransfer.setData("text/plain", dragId);
    e.dataTransfer.effectAllowed = "move";
  });
  dom.categoryList.addEventListener("dragover", (e) => {
    if (!dragId) return;
    e.preventDefault();
    const li = e.target.closest(".category-item");
    if (!li || li.dataset.id === dragId) return;
    li.classList.add("drag-over");
  });
  dom.categoryList.addEventListener("dragleave", (e) => {
    const li = e.target.closest(".category-item");
    if (li) li.classList.remove("drag-over");
  });
  dom.categoryList.addEventListener("drop", (e) => {
    e.preventDefault();
    const target = e.target.closest(".category-item");
    if (!target || !dragId) return;
    target.classList.remove("drag-over");
    reorderCategories(dragId, target.dataset.id);
  });
  dom.categoryList.addEventListener("dragend", () => {
    dragId = null;
    for (const li of dom.categoryList.querySelectorAll(".drag-over")) li.classList.remove("drag-over");
  });
}

function reorderCategories(sourceId, targetId) {
  const order = state.edits.categories.order.slice();
  const filtered = order.filter((id) => id !== sourceId);
  const idx = filtered.indexOf(targetId);
  if (idx === -1) filtered.push(sourceId);
  else filtered.splice(idx, 0, sourceId);
  state.edits.categories.order = filtered;
  applyEditsAndPersist();
  renderCategories();
  renderPhrases();
}

function setupPhraseDrag() {
  let dragData = null;
  dom.phraseList.addEventListener("dragstart", (e) => {
    const card = e.target.closest(".phrase-card");
    if (!card) return;
    dragData = {
      id: card.dataset.id,
      cat: card.dataset.cat,
    };
    e.dataTransfer.setData("text/plain", dragData.id);
    e.dataTransfer.effectAllowed = "move";
  });
  dom.phraseList.addEventListener("dragover", (e) => {
    if (!dragData) return;
    e.preventDefault();
    const card = e.target.closest(".phrase-card");
    if (card) card.classList.add("drag-over");
  });
  dom.phraseList.addEventListener("dragleave", (e) => {
    const card = e.target.closest(".phrase-card");
    if (card) card.classList.remove("drag-over");
  });
  dom.phraseList.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!dragData) return;
    const card = e.target.closest(".phrase-card");
    if (card) {
      card.classList.remove("drag-over");
      reorderPhrase(dragData.id, card.dataset.id, card.dataset.cat);
    } else {
      const targetCat = state.filters.category !== "all" ? state.filters.category : dragData.cat;
      movePhraseToCategory(dragData.id, targetCat);
      const order = state.edits.items.order[targetCat] || [];
      if (!order.includes(dragData.id)) order.push(dragData.id);
      state.edits.items.order[targetCat] = order;
      applyEditsAndPersist();
      renderPhrases();
    }
  });
  dom.phraseList.addEventListener("dragend", () => {
    dragData = null;
    for (const card of dom.phraseList.querySelectorAll(".drag-over")) card.classList.remove("drag-over");
  });
}

function reorderPhrase(sourceId, targetId, targetCat) {
  const sourceItem = state.appliedPack.items.find((item) => item.id === sourceId);
  if (!sourceItem) return;
  const sourceCat = sourceItem.cat;
  if (sourceCat !== targetCat) {
    movePhraseToCategory(sourceId, targetCat);
  }
  const order = state.edits.items.order[targetCat] || [];
  const without = order.filter((id) => id !== sourceId);
  const idx = without.indexOf(targetId);
  if (idx === -1) without.push(sourceId);
  else without.splice(idx, 0, sourceId);
  state.edits.items.order[targetCat] = without;
  applyEditsAndPersist();
  renderPhrases();
}

function movePhraseToCategory(id, targetCat) {
  const added = state.edits.items.added.find((item) => item.id === id);
  if (added) {
    added.cat = targetCat;
  } else {
    state.edits.items.updated[id] = { ...(state.edits.items.updated[id] || {}), cat: targetCat };
  }
  ensureItemOrderMove(id, targetCat);
}

function startPractice() {
  const pool = getFilteredItems(state.appliedPack.items);
  if (!pool.length) {
    dom.practiceStatus.textContent = localeStrings.searchNoResult;
    return;
  }
  state.practice.queue = state.practice.mode === "random" ? shuffleArray(pool) : pool.slice();
  state.practice.pointer = 0;
  state.practice.active = true;
  state.practice.paused = false;
  state.practice.hideHints = dom.practiceHideHints.checked;
  const duration = Number(dom.practiceDuration.value);
  state.practice.duration = duration;
  state.practice.remainingMs = duration ? duration * 1000 : 0;
  state.practice.startedAt = Date.now();
  startPracticeTimer();
  updatePracticeControls();
  updatePracticeCard();
}

function togglePracticePause() {
  if (!state.practice.active) return;
  state.practice.paused = !state.practice.paused;
  if (state.practice.paused) {
    stopPracticeTimer();
  } else {
    state.practice.startedAt = Date.now();
    startPracticeTimer();
  }
  updatePracticeControls();
}

function nextPractice() {
  if (!state.practice.active) return;
  state.practice.pointer = (state.practice.pointer + 1) % state.practice.queue.length;
  updatePracticeCard();
}

function stopPractice() {
  stopPracticeTimer();
  state.practice.active = false;
  state.practice.paused = false;
  state.practice.queue = [];
  state.practice.pointer = 0;
  dom.practiceStatus.textContent = "";
  updatePracticeControls();
  updatePracticeCard();
}

function startPracticeTimer() {
  stopPracticeTimer();
  if (!state.practice.duration) {
    dom.practiceStatus.textContent = `${localeStrings.practiceProgress}${state.practice.pointer + 1}/${state.practice.queue.length}`;
    return;
  }
  const tick = () => {
    if (!state.practice.active || state.practice.paused) return;
    const elapsed = Date.now() - state.practice.startedAt;
    state.practice.remainingMs = Math.max(0, state.practice.remainingMs - elapsed);
    state.practice.startedAt = Date.now();
    if (state.practice.remainingMs <= 0) {
      nextPractice();
      state.practice.remainingMs = state.practice.duration * 1000;
    }
    updatePracticeStatus();
    state.practice.timerId = setTimeout(tick, 500);
  };
  state.practice.startedAt = Date.now();
  state.practice.timerId = setTimeout(tick, 500);
  updatePracticeStatus();
}

function stopPracticeTimer() {
  if (state.practice.timerId) {
    clearTimeout(state.practice.timerId);
    state.practice.timerId = null;
  }
}

function updatePracticeStatus() {
  const remainingSec = state.practice.duration ? Math.ceil(state.practice.remainingMs / 1000) : null;
  const base = `${localeStrings.practiceProgress}${state.practice.pointer + 1}/${state.practice.queue.length}`;
  dom.practiceStatus.textContent = state.practice.duration ? `${base} ｜ ${remainingSec}s` : base;
}

function updatePracticeControls() {
  const active = state.practice.active;
  dom.practiceStart.disabled = active;
  dom.practiceStop.disabled = !active;
  dom.practiceNext.disabled = !active;
  dom.practicePause.disabled = !active;
  dom.practicePause.textContent = state.practice.paused ? localeStrings.practiceResume : localeStrings.practicePause;
}

function updatePracticeCard() {
  if (!state.practice.active || !state.practice.queue.length) {
    dom.practiceCard.hidden = true;
    return;
  }
  dom.practiceCard.hidden = false;
  const item = state.practice.queue[state.practice.pointer];
  dom.practicePhrase.textContent = item.text;
  const catName = state.appliedPack.categories.find((cat) => cat.id === item.cat)?.name || item.cat;
  const metaParts = [catName];
  if (item.difficulty && localeStrings.difficultyLabels[item.difficulty]) metaParts.push(localeStrings.difficultyLabels[item.difficulty]);
  if (item.tags?.length) metaParts.push(`#${item.tags.join(" · ")}`);
  dom.practiceMeta.textContent = metaParts.join(" ｜ ");
  dom.practiceHints.innerHTML = "";
  if (!state.practice.hideHints) {
    for (const alt of item.alts ?? []) {
      const li = document.createElement("li");
      li.textContent = alt;
      dom.practiceHints.appendChild(li);
    }
    if (item.notes) {
      const li = document.createElement("li");
      li.textContent = item.notes;
      dom.practiceHints.appendChild(li);
    }
  }
  updatePracticeStatus();
}

function shuffleArray(arr) {
  const list = arr.slice();
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}

init().catch((err) => {
  console.error("phrases lab init failed", err);
});
