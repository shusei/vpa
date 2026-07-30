export function drawIntonationCurve(canvas, intonation, deps = {}) {
  const {
    EPS,
    PS_MIN_HZ,
    PS_MAX_HZ,
    showIntonationRawPoints = false,
  } = deps;

  try {
    const pts = intonation?.points || [];
    const rawPts = intonation?.rawPoints || [];
    const shaded = intonation?.shadedRanges || [];
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.clientWidth || canvas.offsetWidth || canvas.width || 520;
    const height = canvas.clientHeight || canvas.offsetHeight || canvas.height || 140;
    const DPR = Math.max(1, window.devicePixelRatio || 1);

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.max(1, Math.round(width * DPR));
    canvas.height = Math.max(1, Math.round(height * DPR));
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8f8f8";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(0,0,0,.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 18);
    ctx.lineTo(width, height - 18);
    ctx.stroke();
    if (!pts.length) return;
    const minT = pts[0].t;
    const maxT = pts[pts.length - 1].t;
    const tRange = Math.max(maxT - minT, EPS);
    const minHz = Number.isFinite(intonation.minHz) ? intonation.minHz : Math.min(...pts.map((p) => p.hz));
    const maxHz = Number.isFinite(intonation.maxHz) ? intonation.maxHz : Math.max(...pts.map((p) => p.hz));
    const hzRange = Math.max(maxHz - minHz, 1);
    const projectX = (t) => 10 + ((t - minT) / tRange) * (width - 20);
    const projectY = (hz) => height - 20 - ((hz - minHz) / hzRange) * (height - 40);

    shaded.forEach(({ type, start, end }) => {
      const x0 = projectX(Math.max(minT, start));
      const x1 = projectX(Math.min(maxT, end));
      if (x1 <= x0) return;
      ctx.fillStyle = type === "mute" ? "rgba(110,110,110,0.24)" : "rgba(110,110,110,0.12)";
      ctx.fillRect(x0, 10, x1 - x0, height - 30);
    });

    if (showIntonationRawPoints && rawPts.length) {
      ctx.fillStyle = "rgba(60,60,60,0.22)";
      rawPts.forEach(({ t, hz }) => {
        const x = projectX(t);
        const y = projectY(Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, hz)));
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    ctx.strokeStyle = "rgba(239,93,168,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let drawing = false;
    pts.forEach((p) => {
      if (!Number.isFinite(p.hz)) { drawing = false; return; }
      const x = projectX(p.t);
      const y = projectY(Math.max(PS_MIN_HZ, Math.min(PS_MAX_HZ, p.hz)));
      if (!drawing) { ctx.moveTo(x, y); drawing = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  } catch (e) { console.error("[drawIntonationCurve]", e); }
}

export function setupIntonationLegend(intonation, deps = {}) {
  const {
    getShowIntonationRawPoints,
    setShowIntonationRawPoints,
    saveIntonationRawPreference,
    drawIntonationCurve,
  } = deps;

  try {
    const legend = document.querySelector(".intonation-legend") || document.getElementById("intonationLegend");
    if (!legend) return;

    const hasRaw = Array.isArray(intonation?.rawPoints) && intonation.rawPoints.length > 0;
    legend.setAttribute("data-has-raw", hasRaw ? "true" : "false");

    const cb = document.getElementById("toggleRawDots");
    const btn = document.getElementById("intonationRawToggle");

    const canvas = document.getElementById("intonationCanvas");

    const syncLegend = () => {
      const showRaw = !!getShowIntonationRawPoints?.();
      legend.setAttribute("data-show-raw", showRaw ? "true" : "false");
    };

    if (cb) {
      cb.disabled = !hasRaw;
      cb.checked = !!getShowIntonationRawPoints?.() && hasRaw;
      syncLegend();

      cb.onchange = () => {
        const next = !!cb.checked;
        setShowIntonationRawPoints?.(next);
        saveIntonationRawPreference?.(next);
        syncLegend();
        if (canvas) drawIntonationCurve?.(canvas, intonation || {});
      };
      return;
    }

    if (btn) {
      const updateBtn = () => {
        const showRaw = !!getShowIntonationRawPoints?.();
        btn.setAttribute("aria-pressed", showRaw ? "true" : "false");
        btn.setAttribute("aria-disabled", hasRaw ? "false" : "true");
        btn.disabled = !hasRaw;
        const off = btn.querySelector(".state-off");
        const on = btn.querySelector(".state-on");
        if (off) off.hidden = !!showRaw;
        if (on) on.hidden = !showRaw;
        syncLegend();
      };

      updateBtn();
      btn.onclick = () => {
        if (!hasRaw) return;
        const next = !getShowIntonationRawPoints?.();
        setShowIntonationRawPoints?.(next);
        saveIntonationRawPreference?.(next);
        updateBtn();
        if (canvas) drawIntonationCurve?.(canvas, intonation || {});
      };
      return;
    }

    syncLegend();
  } catch (err) {
    console.error("[setupIntonationLegend]", err);
  }
}
