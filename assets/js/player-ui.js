export function ensurePlayerUI(state, deps) {
  const container = document.querySelector("main.container");
  if (!container) return;
  if (document.getElementById("playBtn")) return;

  const wrap = document.createElement("div");
  wrap.className = "player";

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

  const tipEl = container.querySelector(".callout");
  if (tipEl) container.insertBefore(wrap, tipEl);
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
    try {
      if (state.audioEl.paused) {
        await state.audioEl.play();
        deps.updatePlayerCopy(true);
      } else {
        state.audioEl.pause();
        deps.updatePlayerCopy(false);
      }
    } catch (e) {
      console.error("[audio play]", e);
    }
  };
  state.audioEl.onended = () => { deps.updatePlayerCopy(false); };
  state.audioEl.onpause = () => { deps.updatePlayerCopy(false); };
  state.audioEl.onplay = () => { deps.updatePlayerCopy(true); };

  if (!document.getElementById("streamStats")) {
    const stats = document.createElement("div");
    stats.id = "streamStats";
    stats.className = "insight";
    stats.innerHTML = "";
    wrap.insertAdjacentElement("afterend", stats);
  }
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
  try {
    if (state.audioEl && !state.audioEl.paused) {
      state.audioEl.pause();
      state.audioEl.currentTime = 0;
    }
  } catch (e) {
    console.error("[stopPlayback]", e);
  }
  deps.updatePlayerCopy(false);
}

export async function playLastRecording(state, deps) {
  if (!state.audioEl || !state.audioEl.src) return false;
  try {
    const playPromise = state.audioEl.play();
    if (playPromise && typeof playPromise.then === "function") {
      await playPromise;
    }
    deps.updatePlayerCopy(true);
    return true;
  } catch (err) {
    console.error("[playLastRecording]", err);
    return false;
  }
}

export function setPlaybackSource(state, blob, deps) {
  try {
    if (!state.audioEl || !state.playBtn) return;
    if (state.lastAudioUrl) {
      try {
        URL.revokeObjectURL(state.lastAudioUrl);
      } catch { }
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
