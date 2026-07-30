export async function fillBuildMeta(selfUrl = "assets/app.js") {
  try {
    const verEl = document.getElementById("ver");
    const updEl = document.getElementById("updatedAt");
    if (!verEl && !updEl) return;

    let buildVersion = null;
    try {
      const resolved = new URL(selfUrl, window.location.href);
      buildVersion = resolved.searchParams.get("v");
    } catch { }
    const hasMeaningfulVersion = typeof buildVersion === "string" && buildVersion.length > 0 && !/^__.*__$/.test(buildVersion);

    if (verEl && hasMeaningfulVersion) {
      verEl.textContent = buildVersion;
    }

    const res = await fetch(selfUrl, { method: "HEAD", cache: "no-store" });
    let d = null;
    if (res.ok) {
      const lm = res.headers.get("last-modified");
      if (lm) d = new Date(lm);
    }
    if (!d || isNaN(d.getTime())) d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");

    if (updEl) updEl.textContent = `${y}-${m}-${day}`;
    if (verEl && !hasMeaningfulVersion) verEl.textContent = `build-${y}${m}${day}-${hh}${mm}`;
  } catch { }
}
