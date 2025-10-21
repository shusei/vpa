// 更穩定的瀏覽器端版本：
// 1) 改用 @xenova/transformers 穩定版 CDN
// 2) 自動偵測 MediaRecorder 可用的 mimeType
// 3) 失敗時自動回退到 server 模式（/api/classify）
// 4) 若 ONNX 模型不存在或載入失敗，會切回 server 模式


import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0/dist/transformers.min.js";
// ONNXRuntime Web 的 WASM 靜態資源位置
env.backends.onnx.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/";


const recordBtn = document.getElementById('recordBtn');
const fileInput = document.getElementById('fileInput');
const statusEl = document.getElementById('status');
const meter = document.getElementById('meter');
const femaleVal = document.getElementById('femaleVal');
const maleVal = document.getElementById('maleVal');


let mediaRecorder; let chunks = [];
let pipe; // transformers.js pipeline (lazy)
const device = (navigator.gpu ? 'webgpu' : 'wasm');


function setStatus(text, spin=false){
statusEl.innerHTML = spin ? `<span class="spinner"></span> ${text}` : text;
}


recordBtn.addEventListener('click', async () => {
if (!mediaRecorder || mediaRecorder.state === 'inactive') {
await startRecording();
} else {
await stopRecording();
}
});


fileInput.addEventListener('change', async (e) => {
if (!e.target.files?.length) return;
const file = e.target.files[0];
await handleBlob(file);
e.target.value = '';
});


function pickSupportedMime(){
const cands = [
'audio/webm;codecs=opus',
'audio/webm',
'audio/mp4',
'audio/ogg'
];
for (const t of cands){ if (MediaRecorder.isTypeSupported?.(t)) return t; }
return '';
}


async function startRecording(){
try {
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
chunks = [];
const mimeType = pickSupportedMime();
}
