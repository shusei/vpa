import { t } from "../i18n.js";
import { setStatus } from "../ui.js";
import { transcodeToFloat32 } from "../ffmpeg-transcode.js";
import { mixChannelDataToMono } from "../audio-utils.js";

// --- Audio Decoding ---
export async function decodeSmartToFloat32(blobOrFile, targetSR) {
    setStatus(t("status.webaudioDecode"), true);
    try {
        return await decodeViaWebAudio(blobOrFile, targetSR);
    } catch (err) {
        console.warn("[decode] WebAudio failed, trying FFmpeg...", err);
        try {
            setStatus(t("status.ffmpegDecode"), true);
            return await transcodeToFloat32(blobOrFile, targetSR, (ev) => {
                if (ev.type === "transcode-progress") {
                    setStatus(t("status.ffmpegProgress", { progress: Math.round(ev.progress * 100) }), true);
                }
            });
        } catch (ffmpegErr) {
            console.error("[decode] FFmpeg failed", ffmpegErr);
            throw err; // Throw original WebAudio error or FFmpeg error? Usually better to throw the last one or a combined one.
        }
    }
}

async function decodeViaWebAudio(blobOrFile, targetSR = 16000) {
    const arrayBuf = await blobOrFile.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    let offline = null;
    try {
        let audioBuf;
        try {
            audioBuf = await ctx.decodeAudioData(arrayBuf);
        } catch (err) {
            audioBuf = await new Promise((resolve, reject) => {
                try { ctx.decodeAudioData(arrayBuf.slice(0), resolve, reject); } catch (e) { reject(e); }
            });
        }
        const mono = ctx.createBuffer(1, audioBuf.length, audioBuf.sampleRate);
        const outCh = mono.getChannelData(0);
        const channels = [];
        for (let i = 0; i < audioBuf.numberOfChannels; i++) {
            const chData = audioBuf.getChannelData(i);
            if (chData) channels.push(chData);
        }
        mixChannelDataToMono(channels, outCh);

        let out;
        if (audioBuf.sampleRate === targetSR) {
            out = outCh.slice(0);
        } else {
            offline = new OfflineAudioContext(1, Math.ceil(audioBuf.duration * targetSR), targetSR);
            const src = offline.createBufferSource();
            src.buffer = mono; src.connect(offline.destination); src.start(0);
            const rendered = await offline.startRendering();
            out = rendered.getChannelData(0).slice(0);
        }
        return { float32: out, sr: targetSR, durationSec: out.length / targetSR };
    } finally {
        try { await ctx.close(); } catch { }
    }
}
