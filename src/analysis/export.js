import { t } from "../i18n.js";
import { setStatus } from "../ui.js";

// We need a way to get the latest analysis data.
// It seems `service.js` has `offlineFeatureStore`.
// We should probably import it or have a getter in service.js.
// Since `offlineFeatureStore` is exported from `service.js`, we can import it.
import { offlineFeatureStore } from "../service.js";

export function initExportUI() {
    const exportBtn = document.getElementById("exportBtn");
    if (exportBtn) {
        // Remove old listeners (by cloning? or just add new one if we removed the old one in main.js?)
        // In main.js we added a listener that alerts "not implemented".
        // We should probably replace the element or just add this listener and let main.js listener be removed/ignored?
        // main.js code: exportBtn.addEventListener("click", () => alert(...));
        // To cleanly override, we can use `onclick` or clone.
        // Let's clone node to strip existing listeners.
        const newBtn = exportBtn.cloneNode(true);
        exportBtn.parentNode.replaceChild(newBtn, exportBtn);

        newBtn.addEventListener("click", () => {
            exportAnalysis();
        });
    }
}

export function exportAnalysis() {
    try {
        // Check if we have data
        if (!offlineFeatureStore || (!offlineFeatureStore.pitchRaw.length && !offlineFeatureStore.pitchProcessed.length)) {
            setStatus(t("status.exportNoData") || "No analysis data to export");
            return;
        }

        const payload = {
            meta: {
                timestamp: new Date().toISOString(),
                version: "vpa-refactor", // or get from package?
                // userAgent?
            },
            features: {
                ...offlineFeatureStore
            }
        };

        const json = JSON.stringify(payload, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        link.href = url;
        link.download = `vpa-analysis-${timestamp}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        setStatus(t("status.exportReady") || "Export ready");
    } catch (err) {
        console.error("[export]", err);
        setStatus(t("status.errorPrefix", { message: err?.message || "export failed" }));
    }
}
