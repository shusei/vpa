export const ADVANCED_MODE_KEY = "ui:advancedMode";           // "beginner" | "advanced"
export const ADV_DETAILS_KEY_PREFIX = "ui:advOpen:";          // per-section open memory
export const WARMUP_CARD_OPEN_KEY = "vpa::warmup.open";

export function getAdvancedMode() {
    try { return localStorage.getItem(ADVANCED_MODE_KEY) || "beginner"; } catch { return "beginner"; }
}

export function setAdvancedMode(mode) {
    try { localStorage.setItem(ADVANCED_MODE_KEY, mode); } catch { }
}

export function getDetailsOpen(id, fallbackOpen) {
    try {
        const raw = localStorage.getItem(ADV_DETAILS_KEY_PREFIX + id);
        return raw == null ? fallbackOpen : raw === "1";
    } catch { return fallbackOpen; }
}

export function setDetailsOpen(id, open) {
    try { localStorage.setItem(ADV_DETAILS_KEY_PREFIX + id, open ? "1" : "0"); } catch { }
}
