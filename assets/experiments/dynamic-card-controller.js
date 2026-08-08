import { t } from "../js/i18n.js?v=1.4.13";
import { buildShareText, shareResultFiles } from "./audio-share.js?v=1.4.13";
import {
  createSelectedAudioFile,
  defaultClipRange,
  generateDynamicVoiceCard,
  readAudioDuration,
} from "./dynamic-voice-card.js?v=1.4.13";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percentageBand(score) {
  return Math.floor(Math.max(0, Math.min(100, Number(score) || 0)) / 10) * 10;
}

export function createDynamicCardController({
  createResultCard,
  downloadBlob,
  ensureChallenge,
  formatResult,
  getAudioUrl,
  getShareUrl,
  render,
  track,
}) {
  let generationId = 0;
  let activeRoot = null;
  let state = initialState();

  function initialState() {
    return {
      audioDuration: 0,
      clip: null,
      errorKey: "",
      open: false,
      output: null,
      phase: "idle",
      progress: 0,
      shareUrlPromise: null,
      statusKey: "",
    };
  }

  function revokeOutput(output = state.output) {
    [output?.previewUrl, output?.audioPreviewUrl]
      .filter(Boolean)
      .forEach((url) => {
        try {
          URL.revokeObjectURL(url);
        } catch {
          // Ignore URL clean-up failures.
        }
      });
  }

  function reset() {
    generationId += 1;
    revokeOutput();
    state = initialState();
  }

  function progressMarkup() {
    const percent = Math.max(0, Math.min(100, Math.round(state.progress * 100)));
    const phaseKey = state.phase === "decoding"
      ? "experiment.quick.dynamic.progress.decoding"
      : "experiment.quick.dynamic.progress.encoding";
    return `
      <div class="quick-dynamic-progress" role="status" aria-live="polite">
        <span>${escapeHtml(t(phaseKey, { progress: percent }))}</span>
        <div aria-hidden="true"><i style="width:${percent}%"></i></div>
        <small>${escapeHtml(t("experiment.quick.dynamic.progress.keepOpen"))}</small>
      </div>
    `;
  }

  function makePreviewAudible(video) {
    if (!video) return;
    video.defaultMuted = false;
    video.muted = false;
    if (!(video.volume > 0)) video.volume = 1;
  }

  function preparePreview(root) {
    const video = root.querySelector("[data-dynamic-preview]");
    if (!video) return;
    makePreviewAudible(video);
    const enableAudio = () => makePreviewAudible(video);
    video.addEventListener("pointerdown", enableAudio, { once: true });
    video.addEventListener("touchstart", enableAudio, { once: true, passive: true });
    video.load();
  }

  function outputMarkup() {
    const output = state.output;
    if (!output) return "";
    const isVideo = output.kind === "video";
    return `
      <div class="quick-dynamic-output">
        <span class="quick-dynamic-output__badge">
          ${escapeHtml(t(isVideo
            ? "experiment.quick.dynamic.readyVideo"
            : "experiment.quick.dynamic.readyFallback", {
            format: isVideo ? output.extension.toUpperCase() : "",
          }))}
        </span>
        ${isVideo ? `
          <video controls playsinline preload="auto" data-dynamic-preview
            src="${escapeHtml(output.previewUrl)}"></video>
        ` : `
          <img src="${escapeHtml(output.previewUrl)}" alt="${escapeHtml(t("experiment.quick.dynamic.fallbackAlt"))}" />
          <audio controls preload="metadata" src="${escapeHtml(output.audioPreviewUrl)}"></audio>
          <p>${escapeHtml(t("experiment.quick.dynamic.fallbackHint"))}</p>
        `}
        <div class="quick-dynamic-actions">
          <button type="button" class="quick-primary" data-dynamic-share>
            ${escapeHtml(t("experiment.quick.dynamic.share"))}
          </button>
          <button type="button" class="quick-secondary" data-dynamic-download>
            ${escapeHtml(t("experiment.quick.dynamic.download"))}
          </button>
        </div>
        <p class="quick-share-status" role="status">
          ${state.statusKey ? escapeHtml(t(state.statusKey)) : ""}
        </p>
      </div>
    `;
  }

  function markup(result) {
    if (!result?.ready) return "";
    const audioUrl = getAudioUrl();
    if (!audioUrl) {
      return `
        <section class="quick-dynamic-card">
          <div>
            <h3>${escapeHtml(t("experiment.quick.dynamic.title"))}</h3>
            <p>${escapeHtml(t("experiment.quick.dynamic.noAudio"))}</p>
          </div>
        </section>
      `;
    }
    return `
      <section class="quick-dynamic-card">
        <div class="quick-dynamic-card__head">
          <div>
            <span>${escapeHtml(t("experiment.quick.dynamic.eyebrow"))}</span>
            <h3>${escapeHtml(t("experiment.quick.dynamic.title"))}</h3>
            <p>${escapeHtml(t("experiment.quick.dynamic.subtitle"))}</p>
          </div>
          ${state.open ? `
            <button type="button" class="quick-dynamic-close" data-dynamic-close
              aria-label="${escapeHtml(t("experiment.quick.dynamic.close"))}">✕</button>
          ` : ""}
        </div>
        ${state.open ? `
          ${state.phase === "loading" ? `
            <p class="quick-dynamic-loading" role="status">${escapeHtml(t("experiment.quick.dynamic.loadingAudio"))}</p>
          ` : ""}
          ${state.phase === "decoding" || state.phase === "encoding" ? progressMarkup() : ""}
          ${state.phase === "complete" ? outputMarkup() : ""}
          ${state.phase === "error" ? `
            <p class="quick-error" role="alert">${escapeHtml(t(state.errorKey || "experiment.quick.dynamic.failed"))}</p>
            <button type="button" class="quick-secondary" data-dynamic-retry>
              ${escapeHtml(t("experiment.quick.dynamic.retry"))}
            </button>
          ` : ""}
        ` : `
          <button type="button" class="quick-secondary" data-dynamic-open>
            ${escapeHtml(t("experiment.quick.dynamic.open"))}
          </button>
        `}
      </section>
    `;
  }

  async function open(result) {
    const audioUrl = getAudioUrl();
    if (!audioUrl || !result?.ready) return;
    const requestId = ++generationId;
    state.open = true;
    state.phase = "loading";
    state.errorKey = "";
    render();
    try {
      const duration = await readAudioDuration(audioUrl);
      if (requestId !== generationId) return;
      state.audioDuration = duration;
      state.clip = defaultClipRange(duration);
      await generate(result);
      return;
    } catch (error) {
      console.error("[dynamic-card] audio metadata failed", error);
      if (requestId !== generationId) return;
      state.phase = "error";
      state.errorKey = "experiment.quick.dynamic.audioFailed";
    }
    render();
  }

  function close() {
    reset();
    render();
  }

  function labelsFor(result, formatted) {
    return {
      age: t("experiment.quick.reveal.age"),
      ageValue: formatted.age,
      archetype: t("experiment.quick.reveal.archetype"),
      archetypeValue: formatted.archetype,
      brand: t("experiment.quick.dynamic.brand"),
      challenge: t("experiment.quick.dynamic.challenge", {
        score: result.score,
      }),
      challengeLabel: t("experiment.quick.dynamic.challengeLabel"),
      disclaimer: t("experiment.advanced.disclaimer"),
      feminine: t("experiment.quick.reveal.feminine"),
      masculine: t("experiment.quick.reveal.masculine"),
      title: t("experiment.quick.reveal.tendency"),
      waveform: t("experiment.quick.dynamic.waveform"),
    };
  }

  function prepareShareUrl(result, challenge) {
    if (state.shareUrlPromise) return state.shareUrlPromise;
    state.shareUrlPromise = Promise.resolve()
      .then(() => getShareUrl?.(result, challenge))
      .then((url) => url || challenge.url)
      .catch((error) => {
        console.error("[dynamic-card] short URL failed", error);
        return challenge.url;
      });
    return state.shareUrlPromise;
  }

  async function createFallback(result, shareUrl) {
    const cardUrl = new URL(shareUrl);
    cardUrl.hash = "";
    const [cardBlob, audioFile] = await Promise.all([
      createResultCard(result, { shareUrl: cardUrl.toString() }),
      createSelectedAudioFile({
        audioUrl: getAudioUrl(),
      }),
    ]);
    return {
      audioFile,
      audioPreviewUrl: URL.createObjectURL(audioFile),
      cardBlob,
      kind: "fallback",
      previewUrl: URL.createObjectURL(cardBlob),
    };
  }

  async function generate(result) {
    if (!state.clip || state.phase === "decoding" || state.phase === "encoding") return;
    const requestId = ++generationId;
    revokeOutput();
    state.output = null;
    state.phase = "decoding";
    state.progress = 0;
    state.statusKey = "";
    render();
    const challenge = ensureChallenge();
    const formatted = formatResult(result);
    const shareUrlPromise = prepareShareUrl(result, challenge);
    try {
      const output = await generateDynamicVoiceCard({
        audioUrl: getAudioUrl(),
        labels: labelsFor(result, formatted),
        onProgress: (event) => {
          if (requestId !== generationId) return;
          state.phase = event.phase === "decoding" ? "decoding" : "encoding";
          state.progress = Number(event.progress) || 0;
          const progress = activeRoot?.querySelector(".quick-dynamic-progress i");
          const status = activeRoot?.querySelector(".quick-dynamic-progress > span");
          if (progress) progress.style.width = `${Math.round(state.progress * 100)}%`;
          if (status) {
            status.textContent = t("experiment.quick.dynamic.progress.encoding", {
              progress: Math.round(state.progress * 100),
            });
          }
        },
        result,
        theme: document.documentElement.getAttribute("data-faction") === "light"
          ? "light"
          : "dark",
      });
      if (requestId !== generationId) return;
      state.output = {
        ...output,
        kind: "video",
        previewUrl: URL.createObjectURL(output.blob),
      };
    } catch (videoError) {
      console.warn("[dynamic-card] video output failed, using fallback", videoError);
      try {
        state.output = await createFallback(result, await shareUrlPromise);
      } catch (fallbackError) {
        console.error("[dynamic-card] fallback failed", fallbackError);
        if (requestId !== generationId) return;
        state.phase = "error";
        state.errorKey = "experiment.quick.dynamic.failed";
        render();
        return;
      }
    }
    if (requestId !== generationId) {
      revokeOutput(state.output);
      return;
    }
    state.phase = "complete";
    state.progress = 1;
    render();
  }

  async function shareVideo(output, result, shareUrl) {
    const fileType = String(output.mimeType || `video/${output.extension}`)
      .split(";")[0]
      .trim();
    const file = new File(
      [output.blob],
      `vpa-voice-card.${output.extension}`,
      { type: fileType },
    );
    const payload = {
      files: [file],
      text: buildShareText(formatResult(result).caption, shareUrl),
      title: t("experiment.quick.dynamic.shareTitle"),
    };
    if (
      typeof navigator.share === "function"
      && typeof navigator.canShare === "function"
      && navigator.canShare({ files: payload.files })
    ) {
      await navigator.share(payload);
      return "files";
    }
    downloadBlob(output.blob, file.name);
    return "download";
  }

  async function share(result) {
    const output = state.output;
    if (!output) return;
    const challenge = ensureChallenge();
    state.statusKey = "";
    try {
      const shareUrl = await prepareShareUrl(result, challenge);
      let method;
      if (output.kind === "video") {
        method = await shareVideo(output, result, shareUrl);
      } else {
        const response = await shareResultFiles({
          audioFile: output.audioFile,
          cardBlob: output.cardBlob,
          caption: formatResult(result).caption,
          title: t("experiment.quick.dynamic.shareTitle"),
          url: shareUrl,
        });
        method = response.method;
        if (method === "unsupported" || method === "unsupported-files") {
          downloadBlob(output.cardBlob, "vpa-result.png");
          downloadBlob(output.audioFile, output.audioFile.name);
          method = "download";
        }
      }
      state.statusKey = method === "download"
        ? "experiment.quick.dynamic.downloaded"
        : "experiment.quick.dynamic.shared";
      track("share_success", {
        media: output.kind === "video" ? output.extension : "png_audio",
        method,
        score_band: percentageBand(result.score),
      });
    } catch (error) {
      if (error?.name === "AbortError") {
        state.statusKey = "experiment.quick.share.cancelled";
      } else {
        console.error("[dynamic-card] share failed", error);
        state.statusKey = "experiment.quick.dynamic.shareFailed";
      }
    }
    render();
  }

  function download() {
    const output = state.output;
    if (!output) return;
    if (output.kind === "video") {
      downloadBlob(output.blob, `vpa-voice-card.${output.extension}`);
    } else {
      downloadBlob(output.cardBlob, "vpa-result.png");
      downloadBlob(output.audioFile, output.audioFile.name);
    }
    state.statusKey = "experiment.quick.dynamic.downloaded";
    render();
  }

  function bind(root, result) {
    preparePreview(root);
    activeRoot = root;
    root.querySelector("[data-dynamic-open]")?.addEventListener("click", () => {
      open(result);
    });
    root.querySelector("[data-dynamic-close]")?.addEventListener("click", () => {
      close();
    });
    root.querySelector("[data-dynamic-retry]")?.addEventListener("click", () => {
      open(result);
    });
    root.querySelector("[data-dynamic-share]")?.addEventListener("click", () => {
      share(result);
    });
    root.querySelector("[data-dynamic-download]")?.addEventListener("click", () => {
      download();
    });
  }

  return {
    bind,
    markup,
    open,
    reset,
  };
}
