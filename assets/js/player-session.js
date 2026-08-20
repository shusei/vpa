export function createPlayerSessionController(deps) {
  const {
    sharedEnsurePlayerUI,
    sharedIsPlaying,
    sharedPausePlayback,
    sharedPlayLastRecording,
    sharedSetPlaybackSource,
    sharedSetupExportButton,
    sharedStopPlayback,
    t,
  } = deps;

  const state = {
    playBtn: null,
    audioEl: null,
    lastAudioUrl: null,
    playbackGeneration: 0,
    playerHintEl: null,
    replayBtn: null,
    replayHintPrefixEl: null,
    replayHintSuffixEl: null,
    replayHintSpacerNode: null,
  };

  function updatePlayerCopy(forcePlaying) {
    const isPlaying = forcePlaying ?? (state.audioEl ? !state.audioEl.paused : false);
    if (state.playBtn) {
      state.playBtn.textContent = t(isPlaying ? "player.pause" : "player.play");
      state.playBtn.setAttribute(
        "aria-label",
        t(isPlaying ? "player.ariaPause" : "player.ariaPlay"),
      );
    }
    if (state.playerHintEl) {
      if (state.replayHintPrefixEl) {
        state.replayHintPrefixEl.textContent = t("player.replayHintPrefix");
      }
      if (state.replayHintSpacerNode) {
        state.replayHintSpacerNode.textContent = t("player.replayHintSpacer");
      }
      if (state.replayBtn) {
        state.replayBtn.textContent = t("player.replayHintAction");
        state.replayBtn.setAttribute("aria-label", t("player.replayHintAria"));
      }
      if (state.replayHintSuffixEl) {
        state.replayHintSuffixEl.textContent = t("player.replayHintSuffix");
      }
    }
  }

  function ensurePlayerUI(updatePlaybackAvailability) {
    sharedEnsurePlayerUI(state, {
      updatePlayerCopy,
      updatePlaybackAvailability,
    });
  }

  function setupExportButton(config) {
    sharedSetupExportButton(config);
  }

  function stopPlayback() {
    sharedStopPlayback(state, { updatePlayerCopy });
  }

  function pausePlayback() {
    if (sharedPausePlayback) {
      sharedPausePlayback(state, { updatePlayerCopy });
    } else if (state.audioEl && !state.audioEl.paused) {
      state.audioEl.pause();
      updatePlayerCopy(false);
    }
  }

  function isPlaying() {
    if (sharedIsPlaying) return sharedIsPlaying(state);
    return !!(state.audioEl && !state.audioEl.paused);
  }

  async function playLastRecording() {
    return sharedPlayLastRecording(state, { updatePlayerCopy });
  }

  function setPlaybackSource(blob, updatePlaybackAvailability) {
    sharedSetPlaybackSource(state, blob, {
      updatePlaybackAvailability,
      updatePlayerCopy,
    });
  }

  function getPlayBtn() {
    return state.playBtn;
  }

  function getAudioEl() {
    return state.audioEl;
  }

  function hasPlaybackSource() {
    return !!(state.audioEl && state.audioEl.src);
  }

  function hasLastRecording() {
    return !!state.lastAudioUrl;
  }

  function getLastRecordingUrl() {
    return state.lastAudioUrl;
  }

  return {
    ensurePlayerUI,
    getAudioEl,
    getLastRecordingUrl,
    getPlayBtn,
    hasLastRecording,
    hasPlaybackSource,
    isPlaying,
    pausePlayback,
    playLastRecording,
    setPlaybackSource,
    setupExportButton,
    stopPlayback,
    updatePlayerCopy,
  };
}

