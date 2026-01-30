import { initI18n, t } from "./i18n.js";
import { setCallbacks, handleFileOrBlob, startPitchStream, stopPitchStream, isRecording, busy, hasLastRecording, getLastRecordingUrl } from "./service.js";
import { setStatus, render, updateRealtimeMonitor, startDrawLoop, finishStreamStats, initUI } from "./ui.js";
import { preventDefaults } from "./utils.js";
import { initManualUI } from "./manual-ui.js";
import { setupPracticeUI } from "./practice.js";
import { initSettingsUI } from "./ui/settings.js";
import { initExportUI } from "./analysis/export.js";
import "./theme.js";

// Initialize
document.addEventListener("DOMContentLoaded", async () => {
    await initI18n();

    // Inference Subscription
    const inferenceSubscribers = new Set();
    const subscribeInference = (cb) => {
        inferenceSubscribers.add(cb);
        return () => inferenceSubscribers.delete(cb);
    };

    // Set up service callbacks
    setCallbacks({
        onStatus: (msg, showLoading) => setStatus(msg, showLoading),
        onRender: (pf, pm) => {
            render(pf, pm);
            inferenceSubscribers.forEach(cb => cb({ pf, pm }));
        },
        onUpdateRealtimeMonitor: (data) => updateRealtimeMonitor(data),
        onStartDrawLoop: () => startDrawLoop(),
        onFinishStreamStats: () => {
            // Delegate to ui.js module
            finishStreamStats();
        }
    });

    // Recorder Adapter for Practice UI
    const recorder = {
        start: async () => {
            if (busy) return;
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                await startPitchStream(stream);
            } catch (err) {
                console.error("Mic error", err);
                setStatus(t("status.micError"));
                throw err;
            }
        },
        stop: async () => {
            await stopPitchStream();
        },
        get isRecording() { return isRecording; },
        get busy() { return busy; },
        get hasLastRecording() { return hasLastRecording(); },
        getLastRecordingUrl: () => getLastRecordingUrl()
    };

    // Initialize Sub-modules
    initManualUI();
    initUI();
    initSettingsUI();
    initExportUI();
    setupPracticeUI({ subscribeInference, recorder });

    // UI Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const recordBtn = document.getElementById("recordBtn");

    // Drag & Drop
    if (dropZone) {
        ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        dropZone.addEventListener("dragenter", () => dropZone.classList.add("highlight"), false);
        dropZone.addEventListener("dragover", () => dropZone.classList.add("highlight"), false);
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("highlight"), false);
        dropZone.addEventListener("drop", (e) => {
            dropZone.classList.remove("highlight");
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) handleFile(files[0]);
        }, false);


    }

    // File Input
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        });
    }

    // Upload FAB
    const uploadFab = document.getElementById("uploadFab");
    if (uploadFab) {
        uploadFab.addEventListener("click", (e) => {
            e.stopPropagation(); // Prevent dropZone click
            fileInput && fileInput.click();
        });
    }

    // Recording
    if (recordBtn) {
        recordBtn.addEventListener("click", async () => {
            if (busy) return;
            if (isRecording) {
                await recorder.stop();
                recordBtn.innerText = t("record.idle");
                recordBtn.classList.remove("recording");
            } else {
                try {
                    await recorder.start();
                    recordBtn.innerText = t("record.active");
                    recordBtn.classList.add("recording");
                } catch (err) {
                    // Error handled in recorder.start
                }
            }
        });
    }

    // Export Button
    // Logic is now handled by initExportUI


    // Helper to handle file
    function handleFile(file) {
        if (busy) return;
        handleFileOrBlob(file, "upload");
    }

    console.log("Application initialized");
});
