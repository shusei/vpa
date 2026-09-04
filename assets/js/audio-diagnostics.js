const AUDIO_DEBUG_QUERY_KEY = "vpaAudioDebug";
const MAX_AUDIO_DEBUG_EVENTS = 600;

function serializeError(error) {
  if (!error) return null;
  return {
    message: String(error.message || error),
    name: String(error.name || "Error"),
  };
}

function finiteNumber(value) {
  return Number.isFinite(value) ? value : null;
}

function safeUserActivation() {
  try {
    return {
      hasBeenActive: Boolean(navigator.userActivation?.hasBeenActive),
      isActive: Boolean(navigator.userActivation?.isActive),
    };
  } catch {
    return { hasBeenActive: false, isActive: false };
  }
}

function safeCanvasChecksum(canvas) {
  try {
    if (!canvas || canvas.width <= 0 || canvas.height <= 0) return 0;
    const context = canvas.getContext("2d");
    if (!context) return 0;
    const columns = Math.min(16, canvas.width);
    const rows = Math.min(16, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    for (let row = 0; row < rows; row += 1) {
      const y = Math.min(canvas.height - 1, Math.floor((row * canvas.height) / rows));
      for (let column = 0; column < columns; column += 1) {
        const x = Math.min(canvas.width - 1, Math.floor((column * canvas.width) / columns));
        const offset = ((y * canvas.width) + x) * 4;
        for (let channel = 0; channel < 4; channel += 1) {
          hash = Math.imul(hash ^ pixels[offset + channel], 16777619);
        }
      }
    }
    return hash >>> 0;
  } catch {
    return null;
  }
}

function safePanelSnapshot(dom) {
  const { pitchCanvas, pitchNowEl, pitchWrap } = dom || {};
  try {
    const wrapStyle = pitchWrap ? getComputedStyle(pitchWrap) : null;
    const wrapRect = pitchWrap?.getBoundingClientRect?.();
    const canvasRect = pitchCanvas?.getBoundingClientRect?.();
    return {
      canvas: {
        checksum: safeCanvasChecksum(pitchCanvas),
        clientHeight: finiteNumber(canvasRect?.height),
        clientWidth: finiteNumber(canvasRect?.width),
        height: finiteNumber(pitchCanvas?.height),
        width: finiteNumber(pitchCanvas?.width),
      },
      pitchNow: String(pitchNowEl?.textContent || ""),
      wrap: {
        display: String(wrapStyle?.display || ""),
        height: finiteNumber(wrapRect?.height),
        hidden: Boolean(pitchWrap?.hasAttribute?.("hidden")),
        visibility: String(wrapStyle?.visibility || ""),
        width: finiteNumber(wrapRect?.width),
      },
    };
  } catch (error) {
    return { error: serializeError(error) };
  }
}

function safeTrackSnapshot(stream) {
  try {
    return stream?.getAudioTracks?.().map((track) => {
      const settings = track.getSettings?.() || {};
      return {
        enabled: Boolean(track.enabled),
        kind: String(track.kind || ""),
        muted: Boolean(track.muted),
        readyState: String(track.readyState || ""),
        settings: {
          autoGainControl: settings.autoGainControl ?? null,
          channelCount: finiteNumber(settings.channelCount),
          echoCancellation: settings.echoCancellation ?? null,
          noiseSuppression: settings.noiseSuppression ?? null,
          sampleRate: finiteNumber(settings.sampleRate),
        },
      };
    }) || [];
  } catch (error) {
    return [{ error: serializeError(error) }];
  }
}

function audioDebugEnabled(locationObject) {
  try {
    return new URL(locationObject.href).searchParams.get(AUDIO_DEBUG_QUERY_KEY) === "1";
  } catch {
    return false;
  }
}

function defaultExperience() {
  try {
    return document.documentElement.dataset.experience || "unknown";
  } catch {
    return "unknown";
  }
}

export function createAudioDiagnostics({
  buildVersion = "development",
  dom = {},
  downloadButton = null,
  getExperience = defaultExperience,
  locationObject = window.location,
} = {}) {
  const enabled = audioDebugEnabled(locationObject);
  const events = [];
  const startedAt = Date.now();
  let sequence = 0;

  function record(type, detail = {}) {
    if (!enabled) return;
    events.push({
      at: new Date().toISOString(),
      detail,
      elapsedMs: Date.now() - startedAt,
      experience: getExperience(),
      sequence: ++sequence,
      type,
      userActivation: safeUserActivation(),
    });
    if (events.length > MAX_AUDIO_DEBUG_EVENTS) {
      events.splice(0, events.length - MAX_AUDIO_DEBUG_EVENTS);
    }
  }

  function recordError(type, error, detail = {}) {
    record(type, { ...detail, error: serializeError(error) });
  }

  function recordPanel(type, detail = {}) {
    record(type, { ...detail, panel: safePanelSnapshot(dom) });
  }

  function recordStream(type, stream, detail = {}) {
    record(type, { ...detail, tracks: safeTrackSnapshot(stream) });
  }

  function getReport() {
    return {
      buildVersion,
      collectedAt: new Date().toISOString(),
      currentExperience: getExperience(),
      events: events.map((event) => ({ ...event })),
      location: `${locationObject.origin}${locationObject.pathname}`,
      panel: safePanelSnapshot(dom),
      schemaVersion: 1,
      userAgent: navigator.userAgent,
    };
  }

  function download() {
    if (!enabled) return false;
    const blob = new Blob([JSON.stringify(getReport(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `vpa-audio-debug-${Date.now()}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    return true;
  }

  if (enabled) {
    downloadButton?.removeAttribute("hidden");
    downloadButton?.addEventListener("click", download);
    recordPanel("diagnostics.enabled", { buildVersion });
  }

  return {
    download,
    enabled,
    getReport,
    record,
    recordError,
    recordPanel,
    recordStream,
  };
}

export const audioDiagnosticsInternals = {
  AUDIO_DEBUG_QUERY_KEY,
  MAX_AUDIO_DEBUG_EVENTS,
  audioDebugEnabled,
  safeCanvasChecksum,
};
