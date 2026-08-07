export function createAdvancedSectionController(deps) {
  const {
    detailsKeyPrefix,
    modeKey,
    onLocaleChange,
    t,
  } = deps;

  function getAdvancedMode() {
    try { return localStorage.getItem(modeKey) || "beginner"; } catch { return "beginner"; }
  }

  function setAdvancedMode(mode) {
    try { localStorage.setItem(modeKey, mode); } catch { }
  }

  function getDetailsOpen(id, fallbackOpen) {
    try {
      const raw = localStorage.getItem(detailsKeyPrefix + id);
      return raw == null ? fallbackOpen : raw === "1";
    } catch { return fallbackOpen; }
  }

  function setDetailsOpen(id, open) {
    try { localStorage.setItem(detailsKeyPrefix + id, open ? "1" : "0"); } catch { }
  }

  function setupAdvancedSection(root) {
    if (!root) return;

    const toggleBtn = root.querySelector("[data-adv-toggle]");

    const labelFor = (mode) =>
      mode === "advanced"
        ? t("ui.advancedMode.beginner")
        : t("ui.advancedMode.advanced");

    // force: null=用記憶, "expand"=全部展開, "collapse"=全部收起
    // persist: 是否把這次結果寫回每塊的記憶
    function applyMode(force, persist) {
      const mode = getAdvancedMode(); // "beginner" | "advanced"
      root.setAttribute("data-mode", mode);
      if (toggleBtn) {
        toggleBtn.setAttribute("aria-pressed", mode === "advanced" ? "true" : "false");
        toggleBtn.textContent = labelFor(mode);
      }

      const blocks = root.querySelectorAll("details[data-adv], details.adv-details, details.adv");
      blocks.forEach(d => {
        const key = d.getAttribute("data-adv") || "";
        let open;
        if (force === "expand") open = true;
        else if (force === "collapse") open = false;
        else open = getDetailsOpen(key, mode === "advanced"); // 用記憶，沒有就依模式預設

        d.open = open;
        d.setAttribute("aria-expanded", open ? "true" : "false");
        if (persist === true) setDetailsOpen(key, open);
      });
    }

    // 初次套用：尊重既有記憶（不強制、不覆蓋）
    applyMode(null, false);

    // 一鍵切換：預設不覆蓋使用者記憶；按住 Shift/Alt 可「順便寫入記憶」
    if (toggleBtn) {
      toggleBtn.addEventListener("click", (ev) => {
        const next = getAdvancedMode() === "advanced" ? "beginner" : "advanced";
        setAdvancedMode(next);

        const force = next === "advanced" ? "expand" : "collapse";
        const persist = ev.shiftKey || ev.altKey; // Shift/Alt 點擊 = 覆蓋記憶
        applyMode(force, persist);
      });
    }

    // 使用者手動展開/收起某一塊時，更新該塊的記憶
    root.querySelectorAll("details[data-adv], details.adv-details, details.adv").forEach(d => {
      const key = d.getAttribute("data-adv") || "";
      d.addEventListener("toggle", () => setDetailsOpen(key, d.open));
    });

    // 語系切換時同步更新按鈕文案
    if (typeof onLocaleChange === "function") {
      onLocaleChange(() => {
        if (toggleBtn) toggleBtn.textContent = labelFor(getAdvancedMode());
      });
    }
  }

  return {
    getAdvancedMode,
    setAdvancedMode,
    getDetailsOpen,
    setDetailsOpen,
    setupAdvancedSection,
  };
}
