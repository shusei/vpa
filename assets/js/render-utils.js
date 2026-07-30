export function toMap(arr) {
  const m = { female: 0, male: 0 };
  if (Array.isArray(arr)) {
    for (const r of arr) {
      if (r && typeof r.label === "string") m[r.label] = (typeof r.score === "number" ? r.score : 0);
    }
  }
  return m;
}

export function renderScores(pf, pm, { femaleVal, maleVal } = {}) {
  const barF = document.querySelector(".bar.female");
  const barM = document.querySelector(".bar.male");
  if (barF) {
    barF.style.setProperty("--p", pf ?? 0);
    barF.setAttribute("aria-valuenow", Math.round(((pf ?? 0) * 100)));
  }
  if (barM) {
    barM.style.setProperty("--p", pm ?? 0);
    barM.setAttribute("aria-valuenow", Math.round(((pm ?? 0) * 100)));
  }
  if (femaleVal) femaleVal.textContent = `${((pf ?? 0) * 100).toFixed(1)}%`;
  if (maleVal) maleVal.textContent = `${((pm ?? 0) * 100).toFixed(1)}%`;

  return {
    female: pf ?? 0,
    male: pm ?? 0,
  };
}

export function startHeartbeat(timer, fn) {
  if (timer) {
    clearInterval(timer);
  }
  return setInterval(() => {
    try {
      fn();
    } catch { }
  }, 1000);
}

export function stopHeartbeat(timer) {
  if (timer) {
    clearInterval(timer);
  }
  return null;
}

export function microYield() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
