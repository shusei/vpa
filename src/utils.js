export function fmt0(n) { return Number.isFinite(n) ? n.toFixed(0) : "—"; }
export function fmt1(n) { return Number.isFinite(n) ? n.toFixed(1) : "—"; }

export function fmtSec(sec) {
    if (!Number.isFinite(sec)) return "--:--";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}

export function clamp01(v) {
    return Math.max(0, Math.min(1, v));
}

export function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

export function escapeAttr(value) {
    if (value == null) return "";
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

export function percentile(arr, p) {
    const a = arr.slice().sort((x, y) => x - y);
    const idx = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
    return a[idx];
}

export function smoothMask(mask, k = 3) {
    // Fill short 0 gaps
    let count = 0;
    for (let i = 0; i <= mask.length; i++) {
        if (i < mask.length && !mask[i]) count++;
        else { if (count > 0 && count < k) { for (let j = i - count; j < i; j++) mask[j] = true; } count = 0; }
    }
    // Remove short 1 islands
    count = 0;
    for (let i = 0; i <= mask.length; i++) {
        if (i < mask.length && mask[i]) count++;
        else { if (count > 0 && count < k) { for (let j = i - count; j < i; j++) mask[j] = false; } count = 0; }
    }
}

export function toMap(arr) {
    const m = { female: 0, male: 0 };
    if (Array.isArray(arr)) {
        for (const r of arr) {
            if (r && typeof r.label === "string") m[r.label] = (typeof r.score === "number" ? r.score : 0);
        }
    }
    return m;
}

export function microYield() {
    return new Promise(r => setTimeout(r, 0));
}

export function resizeIntonationCanvas(canvas) {
    const pxRatio = Math.max(1, window.devicePixelRatio || 1);
    const box = canvas.parentElement || canvas;
    const cssWidth = Math.max(320, Math.floor(box.clientWidth || 600));
    const cssHeight = 160;
    canvas.style.width = cssWidth + "px";
    canvas.style.height = cssHeight + "px";
    canvas.width = Math.floor(cssWidth * pxRatio);
    canvas.height = Math.floor(cssHeight * pxRatio);
}

let heartbeatTimer = null;
export function startHeartbeat(fn) {
    stopHeartbeat();
    heartbeatTimer = setInterval(() => { try { fn(); } catch { } }, 1000);
}

export function stopHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

export function isOOMError(err) {
    const msg = (err?.message || "").toLowerCase();
    return msg.includes("memory") || msg.includes("allocation") || msg.includes("oom");
}

export function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}
