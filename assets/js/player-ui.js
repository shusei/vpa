export function ensurePlayerUI(state, deps) {
  const container = document.querySelector("main.container");
  if (!container) return;
  if (document.getElementById("playBtn")) return;

  const wrap = document.createElement("div");
  wrap.className = "player";
  wrap.hidden = true;

  const btn = document.createElement("button");
  btn.id = "playBtn";
  btn.type = "button";
  btn.disabled = true;

  const hint = document.createElement("div");
  hint.className = "hint";

  const hintPrefix = document.createElement("span");
  hintPrefix.className = "hint-prefix";
  const hintSpacer = document.createTextNode("");
  const replayButton = document.createElement("button");
  replayButton.type = "button";
  replayButton.className = "replay-button";
  const hintSuffix = document.createElement("span");
  hintSuffix.className = "hint-suffix";

  replayButton.addEventListener("click", (e) => {
    e.preventDefault();
    btn.click();
  });

  hint.appendChild(hintPrefix);
  hint.appendChild(hintSpacer);
  hint.appendChild(replayButton);
  hint.appendChild(hintSuffix);

  const audio = document.createElement("audio");
  audio.id = "playback";
  audio.preload = "metadata";
  audio.style.display = "none";

  wrap.appendChild(btn);
  wrap.appendChild(hint);
  wrap.appendChild(audio);

  // Playback and the result panels anchored to it must follow live feedback.
  // Otherwise a previous result pushes the next recording's chart off screen.
  const anchor = deps.realtimeAnchor || container.querySelector(".start-wrap");
  const tipEl = container.querySelector(".callout");
  if (anchor) anchor.insertAdjacentElement("afterend", wrap);
  else if (tipEl) container.insertBefore(wrap, tipEl);
  else container.appendChild(wrap);

  state.playBtn = btn;
  state.audioEl = audio;
  state.playerHintEl = hint;
  state.replayBtn = replayButton;
  state.replayHintPrefixEl = hintPrefix;
  state.replayHintSuffixEl = hintSuffix;
  state.replayHintSpacerNode = hintSpacer;
  deps.updatePlayerCopy(false);
  deps.updatePlaybackAvailability();

  state.playBtn.onclick = async () => {
    if (!state.audioEl.src) return;
    if (state.audioEl.paused) {
      await playLastRecording(state, deps);
    } else {
      pausePlayback(state, deps);
    }
  };
  state.audioEl.onended = () => { deps.updatePlayerCopy(false); };
  state.audioEl.onpause = () => { deps.updatePlayerCopy(false); };
  state.audioEl.onplay = () => { deps.updatePlayerCopy(!state.audioEl.paused); };

  if (!document.getElementById("streamStats")) {
    const stats = document.createElement("div");
    stats.id = "streamStats";
    stats.className = "insight";
    stats.innerHTML = "";
    wrap.insertAdjacentElement("afterend", stats);
  }
}

function invalidatePlayback(state) {
  const nextGeneration = (Number.isSafeInteger(state.playbackGeneration) ? state.playbackGeneration : 0) + 1;
  state.playbackGeneration = nextGeneration;
  return nextGeneration;
}

export function setupExportButton({ exportBtn, getLatestAnalysisExport, setStatus, t }) {
  if (!exportBtn) return;
  exportBtn.addEventListener("click", () => {
    try {
      const menu = document.getElementById("themeMenu");
      const gear = document.getElementById("settingsBtn");
      if (menu && !menu.hasAttribute("hidden")) {
        menu.setAttribute("hidden", "");
        gear?.setAttribute("aria-expanded", "false");
      }
      exportBtn.blur?.();
      const payload = getLatestAnalysisExport();
      if (!payload) {
        setStatus(t("status.exportUnavailable"));
        return;
      }
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
      setStatus(t("status.exportReady"));
    } catch (err) {
      console.error("[export]", err);
      setStatus(t("status.errorPrefix", { message: err?.message || "export failed" }));
    }
  });
}

export function stopPlayback(state, deps) {
  invalidatePlayback(state);
  try {
    if (state.audioEl) {
      try { state.audioEl.pause(); } catch { }
      state.audioEl.currentTime = 0;
    }
  } catch (e) {
    console.error("[stopPlayback]", e);
  }
  deps.updatePlayerCopy(false);
}

export function pausePlayback(state, deps) {
  invalidatePlayback(state);
  try {
    if (state.audioEl && !state.audioEl.paused) {
      state.audioEl.pause();
    }
  } catch (e) {
    console.error("[pausePlayback]", e);
  }
  deps.updatePlayerCopy(false);
}

export function isPlaying(state) {
  return !!(state.audioEl && !state.audioEl.paused);
}


export async function playLastRecording(state, deps) {
  if (!state.audioEl || !state.audioEl.src) return false;
  const playGeneration = invalidatePlayback(state);
  try {
    if (state.audioEl.ended || (Number.isFinite(state.audioEl.duration) && state.audioEl.currentTime >= state.audioEl.duration)) {
      state.audioEl.currentTime = 0;
    }
    const playPromise = state.audioEl.play();
    if (playPromise && typeof playPromise.then === "function") {
      await playPromise;
    }
    if (playGeneration !== state.playbackGeneration || state.audioEl.paused) {
      return false;
    }
    deps.updatePlayerCopy(true);
    return true;
  } catch (err) {
    if (playGeneration !== state.playbackGeneration) {
      return false;
    }
    console.error("[playLastRecording]", err);
    return false;
  }
}

export function setPlaybackSource(state, blob, deps) {
  try {
    if (!state.audioEl || !state.playBtn) return;
    invalidatePlayback(state);
    const oldUrl = state.lastAudioUrl;
    state.lastAudioUrl = null;
    try { state.audioEl.pause(); } catch { }
    state.audioEl.removeAttribute("src");
    state.audioEl.load();
    if (oldUrl) {
      try { URL.revokeObjectURL(oldUrl); } catch { }
    }
    state.lastAudioUrl = URL.createObjectURL(blob);
    state.audioEl.src = state.lastAudioUrl;
    state.audioEl.load();
    deps.updatePlaybackAvailability();
    deps.updatePlayerCopy(false);
  } catch (e) {
    console.error("[setPlaybackSource]", e);
  }
}
