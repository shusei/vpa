// SAFE v3 — syntax‑clean, with strict try/catch blocks and early console banner.
}
}


// Decode the audio, resample to target SR, return Float32Array
async function decodeAndResample(blob, targetSR=16000){
const arrayBuf = await blob.arrayBuffer();
const ctx = new (window.AudioContext || window.webkitAudioContext)();
const audioBuf = await ctx.decodeAudioData(arrayBuf);
const offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
const src = offline.createBufferSource();
// Convert to mono buffer at original SR
const mono = new AudioBuffer({ length: audioBuf.length, numberOfChannels: 1, sampleRate: audioBuf.sampleRate });
const ch0 = audioBuf.getChannelData(0);
if (audioBuf.numberOfChannels > 1) {
const ch1 = audioBuf.getChannelData(1);
const out = mono.getChannelData(0);
for (let i=0;i<ch0.length;i++){ out[i] = (ch0[i] + ch1[i]) / 2; }
} else {
mono.copyToChannel(ch0, 0);
}
src.buffer = mono; src.connect(offline.destination); src.start(0);
const rendered = await offline.startRendering();
const out = rendered.getChannelData(0);
return { data: new Float32Array(out), sr: targetSR };
}


// In-browser classification using Transformers.js + ONNX
async function runInBrowser(float32PCM, samplingRate){
try {
setStatus('載入模型（首次需較久）…', true);
meter.classList.remove('hidden');
if (!pipe) {
pipe = await pipeline('audio-classification', window.ONNX_MODEL_ID, { device });
}
const results = await pipe(float32PCM, { sampling_rate: samplingRate, topk: 2 });
renderResults(results);
setStatus('完成');
return true;
} catch (e) {
console.error('[browser inference error]', e);
setStatus('瀏覽器端推論失敗，嘗試伺服器分析…', true);
return false;
}
}
