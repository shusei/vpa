export function bindMainUIEvents(deps) {
  const {
    recordBtn,
    dropZone,
    fileInput,
    uploadFab,
    isBusy,
    isRecording,
    getMediaRecorder,
    resetMeter,
    startRecording,
    stopRecording,
    setStatus,
    t,
    dismissOnboardTip,
    stopPlayback,
    handleFileOrBlob,
  } = deps;

  recordBtn?.addEventListener("click", async () => {
    if (isBusy() && !isRecording()) return;
    try {
      const mediaRecorder = getMediaRecorder();
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        resetMeter();
        await startRecording();
      } else {
        await stopRecording();
      }
    } catch (err) {
      console.error("[recordBtn]", err);
      setStatus(t("status.recordFailed"));
    }
  });

  fileInput?.addEventListener("change", async (e) => {
    if (isRecording()) {
      setStatus(t("status.uploadWhileRecording"));
      if (e.target) e.target.value = "";
      return;
    }
    try {
      const f = e.target.files?.[0];
      if (!f) return;
      dismissOnboardTip(true);
      resetMeter();
      stopPlayback();
      await handleFileOrBlob(f, "upload");
      e.target.value = "";
    } catch (err) {
      console.error("[fileInput]", err);
      setStatus(t("status.uploadFailed"));
    }
  });

  uploadFab?.addEventListener("click", () => {
    if (isRecording()) return;
    if (typeof window !== "undefined" && typeof window.scrollTo === "function") {
      try {
        window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
      } catch {
        window.scrollTo(0, 0);
      }
    }
    stopPlayback();
    fileInput?.click();
  });

  const dropZoneActiveClass = "dropzone-active";

  if (dropZone) {
    let dropZoneDragDepth = 0;

    const hasFilePayload = (event) => {
      if (!event?.dataTransfer) return false;
      const types = event.dataTransfer.types;
      if (!types) return false;
      return Array.from(types).includes("Files");
    };

    const clearDropZoneHighlight = () => {
      dropZoneDragDepth = 0;
      dropZone.classList.remove(dropZoneActiveClass);
    };

    dropZone.addEventListener("dragenter", (event) => {
      if (!hasFilePayload(event)) return;
      event.preventDefault();
      dropZoneDragDepth += 1;
      dropZone.classList.add(dropZoneActiveClass);
    });

    dropZone.addEventListener("dragover", (event) => {
      if (!hasFilePayload(event)) return;
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = isRecording() ? "none" : "copy";
      }
      dropZone.classList.add(dropZoneActiveClass);
    });

    dropZone.addEventListener("dragleave", (event) => {
      if (!hasFilePayload(event)) return;
      event.preventDefault();
      dropZoneDragDepth = Math.max(0, dropZoneDragDepth - 1);
      if (dropZoneDragDepth === 0) {
        dropZone.classList.remove(dropZoneActiveClass);
      }
    });

    dropZone.addEventListener("drop", async (event) => {
      if (!hasFilePayload(event)) return;
      event.preventDefault();
      clearDropZoneHighlight();
      const file = event.dataTransfer?.files?.[0];
      if (!file) return;
      if (isRecording()) {
        setStatus(t("status.uploadWhileRecording"));
        return;
      }
      try {
        dismissOnboardTip(true);
        resetMeter();
        stopPlayback();
        await handleFileOrBlob(file, "upload");
      } catch (err) {
        console.error("[dropZone]", err);
        setStatus(t("status.uploadFailed"));
      }
    });

    document.addEventListener("dragend", clearDropZoneHighlight);
    document.addEventListener("drop", clearDropZoneHighlight);
  }
}
