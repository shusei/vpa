export function wireAdvancedIntonation(advRoot, advSummary, deps = {}) {
  const {
    drawIntonationCurve,
    setupIntonationLegend,
  } = deps;

  if (!advRoot || !advSummary) return;
  const det = advRoot.querySelector('details[data-adv="intonation"]');
  if (!det) return;

  function resizeIntonationCanvas(canvas) {
    const pxRatio = Math.max(1, window.devicePixelRatio || 1);
    const box = canvas.parentElement || canvas;
    const cssWidth = Math.max(1, Math.floor(box.clientWidth || 600));
    const cssHeight = 160;
    canvas.style.width = cssWidth + "px";
    canvas.style.height = cssHeight + "px";
    canvas.width = Math.floor(cssWidth * pxRatio);
    canvas.height = Math.floor(cssHeight * pxRatio);
  }

  function drawNow() {
    const canvas = advRoot.querySelector("#intonationCanvas");
    if (!canvas) return;
    resizeIntonationCanvas(canvas);
    try {
      if (Array.isArray(advSummary.intonation?.points) && advSummary.intonation.points.length) {
        drawIntonationCurve(canvas, advSummary.intonation);
      } else {
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      setupIntonationLegend(advSummary.intonation);
    } catch { }
  }

  det.addEventListener("toggle", () => { if (det.open) drawNow(); });
  if (det.open) drawNow();

  if (typeof window.__advIntonationOnResize === "function") {
    window.removeEventListener("resize", window.__advIntonationOnResize);
  }
  window.__advIntonationOnResize = () => { if (det.open) drawNow(); };
  window.addEventListener("resize", window.__advIntonationOnResize, { passive: true });
}
