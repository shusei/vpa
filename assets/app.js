// ➊ Imports for client-side inference
try {
setStatus('載入模型（首次需較久）…', true);
meter.classList.remove('hidden');
if (!pipe) {
pipe = await pipeline('audio-classification', window.ONNX_MODEL_ID, { dtype: 'q8', device });
}
const results = await pipe(float32PCM, { sampling_rate: samplingRate, topk: 2 });
renderResults(results);
setStatus('完成');
} catch (e) {
console.error(e); setStatus('模型載入或推論失敗');
}
}


// Server route – send WAV to your Pages Function / API
async function runViaServer(wavBlob){
try {
setStatus('上傳到伺服器分析…', true);
meter.classList.remove('hidden');
const res = await fetch(window.API_BASE_URL, { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: await wavBlob.arrayBuffer() });
if (!res.ok) throw new Error(`API ${res.status}`);
const json = await res.json();
renderResults(json.results || json);
setStatus('完成');
} catch (e) {
console.error(e); setStatus('伺服器分析失敗');
}
}


function renderResults(arr){
// Expect: [{label:"female", score:0.98}, {label:"male", score:0.02}] order arbitrary
const map = { female: 0, male: 0 };
for (const r of arr){ map[r.label] = r.score; }
const f = (map.female ?? 0), m = (map.male ?? 0);
document.querySelector('.bar.female').style.setProperty('--p', f);
document.querySelector('.bar.male').style.setProperty('--p', m);
femaleVal.textContent = `${(f*100).toFixed(1)}%`;
maleVal.textContent = `${(m*100).toFixed(1)}%`;
}


// Helpers: make 16-bit PCM WAV blob
function floatToWavBlob(float32, sampleRate){
const buffer = new ArrayBuffer(44 + float32.length * 2);
const view = new DataView(buffer);
const writeStr = (off, str) => { for (let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); };
// RIFF header
writeStr(0, 'RIFF'); view.setUint32(4, 36 + float32.length * 2, true); writeStr(8, 'WAVE');
// fmt chunk
writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); // byte rate
view.setUint16(32, 2, true); view.setUint16(34, 16, true);
// data chunk
writeStr(36, 'data'); view.setUint32(40, float32.length * 2, true);
let offset = 44;
for (let i=0; i<float32.length; i++) {
const s = Math.max(-1, Math.min(1, float32[i]));
view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true); offset += 2;
}
return new Blob([view], { type: 'audio/wav' });
}
