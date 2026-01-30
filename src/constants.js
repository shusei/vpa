export const RAW_MODEL_ID = (window.ONNX_MODEL_ID || "prithivMLmods/Common-Voice-Gender-Detection-ONNX");
export const MODEL_ID = String(RAW_MODEL_ID).trim().replace(/^\{+|\}+$/g, "");

export const TARGET_SR       = 16000;
export const MAX_WHOLE_SEC   = 150;
export const WARN_LONG_SEC   = 180;
export const STREAM_WIN_CAND = [12,8,6,4];
export const STREAM_HOP_S    = 3;
export const EPS             = 1e-9;

export const VAD_MIN_APPLY_SEC   = 20;
export const VAD_FRAME_MS        = 30;
export const VAD_HOP_MS          = 10;
export const VAD_PAD_MS          = 60;
export const VAD_MIN_SEG_MS      = 200;
export const VAD_MIN_VOICED_SEC  = 2;
export const VAD_SILENCE_RATIO_TO_APPLY = 0.15;

export const IS_SAFARI = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
