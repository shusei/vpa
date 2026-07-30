export function createPitchProfileController(deps) {
  const {
    clampPitchRange,
    CONFIDENCE_INCLUDE_THRESHOLD,
    DEFAULT_PITCH_RANGE,
    getOctaveCorrectedCount,
    PITCH_PROFILE_DEFAULT,
    PITCH_RANGE_HARD,
    PS_INTERVAL_MS,
    VOICE_PRESETS,
  } = deps;

  const PITCH_RANGE_KEY = "vpa:pitchRangeHz";
  const PITCH_PROFILE_KEY = "vpa:pitchProfile";
  const AUTO_WIDE_RANGE = { min: 70, max: 560 };
  const AUTO_SAMPLE_MS = 800;
  const AUTO_MIN_VALID_FRAMES = 12;
  const AUTO_REEVAL_WINDOW_MS = 2000;
  const AUTO_INVALID_RATIO_LIMIT = 0.4;
  const AUTO_OCTAVE_SPIKE_LIMIT = 3;
  const AUTO_HYSTERESIS_UP = 1.2;
  const AUTO_HYSTERESIS_DOWN = 0.85;
  const AUTO_NEUTRAL_MEDIAN = (VOICE_PRESETS.neutral.min + VOICE_PRESETS.neutral.max) / 2;

  let pitchProfileSetting = loadPitchProfileSetting();
  let pitchRangeSetting = loadPitchRangeSetting();

  const autoRangeState = {
    stage: "idle",
    currentRange: clampPitchRange(VOICE_PRESETS.neutral),
    sampleValues: [],
    sampleDurationMs: 0,
    windowFrames: [],
    windowMs: 0,
    windowInvalidMs: 0,
    octaveEvents: [],
    prevOctaveCount: 0,
    timelineMs: 0,
    lastMedian: null,
    lastUpdateMs: 0,
  };

  let pitchMinInput = null;
  let pitchMaxInput = null;
  let pitchRangeResetBtn = null;
  let voiceProfileButtons = [];
  let voiceSettingsContainer = null;

  function loadPitchProfileSetting() {
    try {
      const raw = localStorage.getItem(PITCH_PROFILE_KEY);
      if (!raw) return PITCH_PROFILE_DEFAULT;
      if (raw === "custom") return "custom";
      if (raw in VOICE_PRESETS) return raw;
      return PITCH_PROFILE_DEFAULT;
    } catch {
      return PITCH_PROFILE_DEFAULT;
    }
  }

  function savePitchProfileSetting(profile) {
    pitchProfileSetting = profile;
    if (typeof window === "undefined" || !window.localStorage) return;
    try { window.localStorage.setItem(PITCH_PROFILE_KEY, profile); } catch { }
  }

  function loadPitchRangeSetting() {
    try {
      const raw = localStorage.getItem(PITCH_RANGE_KEY);
      if (!raw) return { ...DEFAULT_PITCH_RANGE };
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return { ...DEFAULT_PITCH_RANGE };
      return clampPitchRange(parsed);
    } catch {
      return { ...DEFAULT_PITCH_RANGE };
    }
  }

  function savePitchRangeSetting(range) {
    pitchRangeSetting = clampPitchRange(range || DEFAULT_PITCH_RANGE);
    if (pitchProfileSetting !== "auto") {
      try { localStorage.setItem(PITCH_RANGE_KEY, JSON.stringify(pitchRangeSetting)); } catch { }
    }
    updatePitchRangeInputs();
    return pitchRangeSetting;
  }

  function getPitchProfileDisplayRange() {
    if (pitchProfileSetting === "auto") {
      return clampPitchRange(autoRangeState.currentRange || VOICE_PRESETS.neutral);
    }
    if (pitchProfileSetting && pitchProfileSetting in VOICE_PRESETS) {
      const preset = VOICE_PRESETS[pitchProfileSetting];
      if (preset) return clampPitchRange(preset);
    }
    return clampPitchRange(pitchRangeSetting || DEFAULT_PITCH_RANGE);
  }

  function getPitchDetectorRange() {
    if (pitchProfileSetting === "auto") {
      if (autoRangeState.stage === "bootstrap") {
        return clampPitchRange(AUTO_WIDE_RANGE);
      }
      return clampPitchRange(autoRangeState.currentRange || VOICE_PRESETS.neutral);
    }
    if (pitchProfileSetting && pitchProfileSetting in VOICE_PRESETS) {
      const preset = VOICE_PRESETS[pitchProfileSetting];
      if (preset) return clampPitchRange(preset);
    }
    return clampPitchRange(pitchRangeSetting || DEFAULT_PITCH_RANGE);
  }

  function updatePitchProfileControls() {
    const isAuto = pitchProfileSetting === "auto";
    const activePreset = !isAuto && pitchProfileSetting !== "custom" ? pitchProfileSetting : null;
    voiceProfileButtons.forEach((btn) => {
      if (!btn) return;
      const profile = btn.dataset?.profile;
      const pressed = profile === "auto" ? isAuto : (profile === activePreset);
      btn.setAttribute("aria-pressed", pressed ? "true" : "false");
    });
    if (voiceSettingsContainer) {
      voiceSettingsContainer.classList.toggle("is-auto", isAuto);
    }
    if (pitchMinInput) {
      pitchMinInput.disabled = isAuto;
      if (isAuto) pitchMinInput.setAttribute("aria-disabled", "true");
      else pitchMinInput.removeAttribute("aria-disabled");
    }
    if (pitchMaxInput) {
      pitchMaxInput.disabled = isAuto;
      if (isAuto) pitchMaxInput.setAttribute("aria-disabled", "true");
      else pitchMaxInput.removeAttribute("aria-disabled");
    }
    if (pitchRangeResetBtn) {
      if (isAuto) {
        pitchRangeResetBtn.setAttribute("disabled", "true");
        pitchRangeResetBtn.setAttribute("aria-disabled", "true");
      } else {
        pitchRangeResetBtn.removeAttribute("disabled");
        pitchRangeResetBtn.removeAttribute("aria-disabled");
      }
    }
  }

  function updatePitchRangeInputs() {
    const range = getPitchProfileDisplayRange();
    if (pitchMinInput) { pitchMinInput.value = Math.round(range.min); }
    if (pitchMaxInput) { pitchMaxInput.value = Math.round(range.max); }
  }

  function refreshVoiceProfileUI() {
    updatePitchProfileControls();
    updatePitchRangeInputs();
  }

  function applyVoiceProfile(profile) {
    if (!profile) return;
    if (profile === "auto") {
      savePitchProfileSetting("auto");
      startAutoRangeSession({ preserveRange: false });
      refreshVoiceProfileUI();
      return;
    }
    if (profile in VOICE_PRESETS && profile !== "auto") {
      const preset = VOICE_PRESETS[profile];
      savePitchProfileSetting(profile);
      autoRangeState.stage = "idle";
      const range = preset ? clampPitchRange(preset) : clampPitchRange(DEFAULT_PITCH_RANGE);
      savePitchRangeSetting(range);
      refreshVoiceProfileUI();
      return;
    }
    if (profile === "custom") {
      savePitchProfileSetting("custom");
      autoRangeState.stage = "idle";
      refreshVoiceProfileUI();
    }
  }

  function initPitchRangeControls() {
    pitchMinInput = document.getElementById("pitchMinInput");
    pitchMaxInput = document.getElementById("pitchMaxInput");
    pitchRangeResetBtn = document.getElementById("pitchRangeReset");
    voiceSettingsContainer = document.querySelector(".voice-settings");
    voiceProfileButtons = Array.from(document.querySelectorAll("[data-profile]"));

    voiceProfileButtons.forEach((btn) => {
      btn?.addEventListener("click", () => {
        const profile = btn.dataset?.profile;
        if (!profile) return;
        if (profile === pitchProfileSetting && profile !== "auto") return;
        applyVoiceProfile(profile);
      });
    });

    const applyChange = () => {
      const minVal = Number(pitchMinInput?.value);
      const maxVal = Number(pitchMaxInput?.value);
      const next = clampPitchRange({
        min: Number.isFinite(minVal) ? minVal : DEFAULT_PITCH_RANGE.min,
        max: Number.isFinite(maxVal) ? maxVal : DEFAULT_PITCH_RANGE.max,
      });
      savePitchProfileSetting("custom");
      autoRangeState.stage = "idle";
      savePitchRangeSetting(next);
      refreshVoiceProfileUI();
    };

    pitchMinInput?.addEventListener("change", applyChange);
    pitchMaxInput?.addEventListener("change", applyChange);
    pitchRangeResetBtn?.addEventListener("click", () => {
      applyVoiceProfile("neutral");
    });

    if (pitchProfileSetting === "auto") {
      startAutoRangeSession({ preserveRange: true });
    }
    refreshVoiceProfileUI();
  }

  function resetAutoWatchdogs() {
    autoRangeState.windowFrames.length = 0;
    autoRangeState.windowMs = 0;
    autoRangeState.windowInvalidMs = 0;
    autoRangeState.octaveEvents.length = 0;
    autoRangeState.prevOctaveCount = Number(getOctaveCorrectedCount?.() || 0);
  }

  function startAutoRangeSession({ preserveRange = false } = {}) {
    if (pitchProfileSetting !== "auto") {
      autoRangeState.stage = "idle";
      autoRangeState.sampleValues.length = 0;
      autoRangeState.sampleDurationMs = 0;
      autoRangeState.timelineMs = 0;
      resetAutoWatchdogs();
      return;
    }
    autoRangeState.stage = "bootstrap";
    autoRangeState.sampleValues.length = 0;
    autoRangeState.sampleDurationMs = 0;
    autoRangeState.timelineMs = 0;
    autoRangeState.lastUpdateMs = 0;
    if (!preserveRange) {
      autoRangeState.currentRange = clampPitchRange(VOICE_PRESETS.neutral);
      autoRangeState.lastMedian = null;
    }
    resetAutoWatchdogs();
    updatePitchRangeInputs();
  }

  function deriveAutoRange(median) {
    const rawMin = Math.round(median * 0.6);
    const rawMax = Math.round(median * 1.6);
    const min = Math.max(PITCH_RANGE_HARD.min, Math.min(PITCH_RANGE_HARD.max - 20, rawMin));
    const max = Math.max(min + 20, Math.min(PITCH_RANGE_HARD.max, rawMax));
    return { min, max };
  }

  function finalizeAutoBootstrap() {
    const values = autoRangeState.sampleValues.slice();
    autoRangeState.sampleValues.length = 0;
    autoRangeState.sampleDurationMs = 0;

    let median = NaN;
    if (values.length >= AUTO_MIN_VALID_FRAMES) {
      values.sort((a, b) => a - b);
      median = values[Math.floor(values.length / 2)];
    }

    let nextMedian = Number.isFinite(median) ? median : AUTO_NEUTRAL_MEDIAN;
    let nextRange = Number.isFinite(median)
      ? deriveAutoRange(nextMedian)
      : clampPitchRange(VOICE_PRESETS.neutral);

    const hadMedian = Number.isFinite(autoRangeState.lastMedian);
    let shouldApply = !hadMedian || !Number.isFinite(median);
    if (!shouldApply && hadMedian) {
      const lower = autoRangeState.lastMedian * AUTO_HYSTERESIS_DOWN;
      const upper = autoRangeState.lastMedian * AUTO_HYSTERESIS_UP;
      if (nextMedian < lower || nextMedian > upper) {
        shouldApply = true;
      }
    }

    if (shouldApply) {
      autoRangeState.currentRange = clampPitchRange(nextRange);
      autoRangeState.lastMedian = nextMedian;
      autoRangeState.lastUpdateMs = autoRangeState.timelineMs;
    }

    autoRangeState.stage = "ready";
    resetAutoWatchdogs();
    updatePitchRangeInputs();
  }

  function triggerAutoRangeRefresh() {
    if (pitchProfileSetting !== "auto") return;
    if (autoRangeState.stage === "bootstrap") return;
    autoRangeState.stage = "bootstrap";
    autoRangeState.sampleValues.length = 0;
    autoRangeState.sampleDurationMs = 0;
    resetAutoWatchdogs();
  }

  function handleAutoRangeFrame(result, { dtMs = PS_INTERVAL_MS } = {}) {
    if (pitchProfileSetting !== "auto") return;
    if (autoRangeState.stage === "idle") return;

    const dt = Number.isFinite(dtMs) && dtMs > 0 ? dtMs : PS_INTERVAL_MS;
    autoRangeState.timelineMs += dt;

    const processed = Number.isFinite(result?.processed) ? result.processed : NaN;
    const confidence = Number.isFinite(result?.confidence) ? result.confidence : 0;
    const isValid = Number.isFinite(processed) && confidence >= CONFIDENCE_INCLUDE_THRESHOLD;

    if (autoRangeState.stage === "bootstrap") {
      autoRangeState.sampleDurationMs += dt;
      if (isValid) {
        autoRangeState.sampleValues.push(processed);
      }
      if (autoRangeState.sampleDurationMs >= AUTO_SAMPLE_MS) {
        finalizeAutoBootstrap();
      }
      return;
    }

    autoRangeState.windowFrames.push({ dt, invalid: !isValid });
    autoRangeState.windowMs += dt;
    if (!isValid) autoRangeState.windowInvalidMs += dt;

    while (autoRangeState.windowFrames.length && autoRangeState.windowMs > AUTO_REEVAL_WINDOW_MS) {
      const head = autoRangeState.windowFrames.shift();
      autoRangeState.windowMs -= head.dt;
      if (head.invalid) autoRangeState.windowInvalidMs -= head.dt;
      autoRangeState.windowMs = Math.max(0, autoRangeState.windowMs);
      autoRangeState.windowInvalidMs = Math.max(0, autoRangeState.windowInvalidMs);
    }

    if (autoRangeState.windowMs >= AUTO_REEVAL_WINDOW_MS * 0.9) {
      const ratio = autoRangeState.windowInvalidMs / Math.max(autoRangeState.windowMs, 1);
      if (ratio > AUTO_INVALID_RATIO_LIMIT) {
        triggerAutoRangeRefresh();
        return;
      }
    }

    const currentOctave = Number(getOctaveCorrectedCount?.() || 0);
    if (currentOctave < autoRangeState.prevOctaveCount) {
      autoRangeState.prevOctaveCount = currentOctave;
      autoRangeState.octaveEvents.length = 0;
    } else if (currentOctave > autoRangeState.prevOctaveCount) {
      const diff = currentOctave - autoRangeState.prevOctaveCount;
      autoRangeState.octaveEvents.push({ time: autoRangeState.timelineMs, count: diff });
      autoRangeState.prevOctaveCount = currentOctave;
    }

    while (autoRangeState.octaveEvents.length && (autoRangeState.timelineMs - autoRangeState.octaveEvents[0].time) > AUTO_REEVAL_WINDOW_MS) {
      autoRangeState.octaveEvents.shift();
    }

    const octaveSum = autoRangeState.octaveEvents.reduce((acc, evt) => acc + (evt?.count || 0), 0);
    if (octaveSum > AUTO_OCTAVE_SPIKE_LIMIT) {
      triggerAutoRangeRefresh();
      return;
    }
  }

  return {
    getPitchDetectorRange,
    handleAutoRangeFrame,
    initPitchRangeControls,
    startAutoRangeSession,
  };
}
