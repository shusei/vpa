import { t } from "../i18n.js";
import {
    loadSettings,
    savePitchProfile,
    savePitchRange,
    getPitchProfile,
    getPitchRange,
    onSettingsChange,
    getPitchProfileDisplayRange
} from "../state/pitch-settings.js";
import { autoRangeState } from "../service.js"; // Import autoRangeState from service to use in display logic

let voiceProfileButtons = [];
let voiceSettingsContainer = null;
let pitchMinInput = null;
let pitchMaxInput = null;
let pitchRangeResetBtn = null;

export function initSettingsUI() {
    voiceProfileButtons = document.querySelectorAll("[data-profile]");
    voiceSettingsContainer = document.querySelector(".voice-settings");
    pitchMinInput = document.getElementById("pitchMinInput");
    pitchMaxInput = document.getElementById("pitchMaxInput");
    pitchRangeResetBtn = document.getElementById("pitchRangeReset");

    // Load initial state
    loadSettings();

    // Bind Events
    if (voiceProfileButtons.length) {
        voiceProfileButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const profile = btn.getAttribute("data-profile");
                if (profile) savePitchProfile(profile);
            });
        });
    }

    if (pitchMinInput) {
        pitchMinInput.addEventListener("change", () => {
            const current = getPitchRange();
            savePitchRange({ min: Number(pitchMinInput.value), max: current.max });
        });
    }

    if (pitchMaxInput) {
        pitchMaxInput.addEventListener("change", () => {
            const current = getPitchRange();
            savePitchRange({ min: current.min, max: Number(pitchMaxInput.value) });
        });
    }

    if (pitchRangeResetBtn) {
        pitchRangeResetBtn.addEventListener("click", () => {
            savePitchProfile("neutral"); // Reset to default or allow specific reset logic?
            // Original app reset calls savePitchRange(DEFAULT) if profile is custom, or just resets range?
            // Here we can just reset range.
            savePitchRange(null); // Force default
        });
    }

    // Initial UI Update
    updateUI();

    // Listen for changes
    onSettingsChange(updateUI);
}

function updateUI() {
    const profile = getPitchProfile();
    const isAuto = profile === "auto";
    const activePreset = !isAuto && profile !== "custom" ? profile : null;

    voiceProfileButtons.forEach((btn) => {
        const p = btn.getAttribute("data-profile");
        const pressed = p === "auto" ? isAuto : (p === activePreset);
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

    updateInputs();
}

function updateInputs() {
    // We need autoRangeState to show dynamic range if in auto mode?
    // Original app uses getPitchProfileDisplayRange() which uses autoRangeState.currentRange
    // We imported autoRangeState.
    const range = getPitchProfileDisplayRange(autoRangeState);
    if (pitchMinInput && document.activeElement !== pitchMinInput) {
        pitchMinInput.value = Math.round(range.min);
    }
    if (pitchMaxInput && document.activeElement !== pitchMaxInput) {
        pitchMaxInput.value = Math.round(range.max);
    }
}
