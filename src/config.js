export const RAW_MODEL_ID = (typeof window !== "undefined" && window.ONNX_MODEL_ID) || "prithivMLmods/Common-Voice-Gender-Detection-ONNX";
export const MODEL_ID = String(RAW_MODEL_ID).trim().replace(/^\{+|\}+$/g, "");

export const TARGET_SR = 16000;
export const MAX_WHOLE_SEC = 150;
export const WARN_LONG_SEC = 180;
export const STREAM_WIN_CAND = [12, 8, 6, 4];
export const STREAM_HOP_S = 3;
export const EPS = 1e-9;

export const VAD_MIN_APPLY_SEC = 20;
export const VAD_FRAME_MS = 30;
export const VAD_HOP_MS = 10;
export const VAD_PAD_MS = 60;
export const VAD_MIN_SEG_MS = 200;
export const VAD_MIN_VOICED_SEC = 2;
export const VAD_SILENCE_RATIO_TO_APPLY = 0.15;

export const IS_SAFARI = (typeof navigator !== "undefined") && /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// App Constants
export const THREAD_STORAGE_KEY = "vpa::onnxThreads";
export const VOLUME_DISPLAY_MODE = "relative";
export const MEDIA_RECORDER_DATA_TIMEOUT_MS = 5000;

// Pitch & Storage Keys
export const PITCH_RANGE_KEY = "vpa:pitchRangeHz";
export const PITCH_PROFILE_KEY = "vpa:pitchProfile";
export const INTONATION_RAW_KEY = "vpa:intonationShowRaw";

// Auto Range Constants
export const AUTO_WIDE_RANGE = { min: 70, max: 560 };
export const AUTO_SAMPLE_MS = 800;
export const AUTO_MIN_VALID_FRAMES = 12;
export const AUTO_REEVAL_WINDOW_MS = 2000;
export const AUTO_INVALID_RATIO_LIMIT = 0.4;
export const AUTO_OCTAVE_SPIKE_LIMIT = 3;
export const AUTO_HYSTERESIS_UP = 1.2;
export const AUTO_HYSTERESIS_DOWN = 0.85;

// Pitch Runtime Constants
export const PITCH_RUNTIME_BASE_BUDGET_MS = 26;
export const PITCH_RUNTIME_OVER_BUDGET_LIMIT = 3;
export const PITCH_RUNTIME_RECOVERY_MS = 1500;
export const PITCH_RUNTIME_OFFLINE_MULTIPLIER = 1.65;
export const PITCH_RUNTIME_MIN_BUDGET_MS = 18;
export const PITCH_RUNTIME_MAX_BUDGET_MS = 60;
export const PITCH_RETRY_MIN_INTERVAL_MS = 3000;
export const PITCH_RETRY_COOLDOWN_MS = 20000;
export const PITCH_RETRY_ERROR_COOLDOWN_MS = 45000;
export const PITCH_RETRY_ERROR_GUARD_MS = 6500;
export const PITCH_RETRY_ERROR_ATTEMPT_SPACING_MS = 8000;

// Resonance Constants
export const RESONANCE_PRIOR_WEIGHT = 0.02;
export const RESONANCE_PRIOR_MIN = 0.6;
export const RESONANCE_DOMINANCE_DELTA = 0.12;
export const RESONANCE_HEAD_MIN = 0.50;
export const RESONANCE_MASK_MIN = 0.46;
export const RESONANCE_CHEST_MIN = 0.54;
export const RESONANCE_MIN_SAMPLES = 18;
export const RESONANCE_MIN_COVERAGE = 0.2;
export const RESONANCE_COVERAGE_GOOD = 0.35;

// Formant Constants
export const FORMANT_MIN_SAMPLES = 24;
export const FORMANT_MIN_COVERAGE = 0.18;
export const FORMANT_GOOD_COVERAGE = 0.32;
export const FORMANT_MIN_SPREAD = 70;
export const FORMANT_CONFIDENCE_THRESHOLD = 0.7; // Assuming CONFIDENCE_INCLUDE_THRESHOLD is 0.7
export const FORMANT_MAX_GAP_FRAMES = 8;

// Breathiness & Brightness Constants
export const BREATHINESS_EMA_TAU_SEC = 0.2;
export const BRIGHTNESS_F3_LOW = 2400;
export const BRIGHTNESS_F3_HIGH = 3400;
export const BRIGHTNESS_WARM_Z = -0.7;
export const BRIGHTNESS_SPARKLE_Z = 0.4;
export const BRIGHTNESS_SWEET_Z = 1.0;
export const BRIGHTNESS_TILT_SHARP = -1.5;
export const BRIGHTNESS_BREATH_THRESHOLD = 0.45;
