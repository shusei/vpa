function splitGuidanceLine(line) {
  const fullWidth = line.indexOf("：");
  if (fullWidth > 0) return [line.slice(0, fullWidth), line.slice(fullWidth + 1)];
  const ascii = line.indexOf(": ");
  if (ascii > 0) return [line.slice(0, ascii), line.slice(ascii + 2)];
  return null;
}

export function renderGuidance(value, escapeHtml) {
  const source = String(value ?? "").trim();
  if (!source) return "&nbsp;";

  const lines = source.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const rows = lines.map(splitGuidanceLine);
  if (lines.length < 3 || rows.some((row) => !row)) return escapeHtml(source);

  return `<span class="guidance-list">${rows.map(([label, body]) => `
    <span class="guidance-row">
      <b>${escapeHtml(label)}</b>
      <span>${escapeHtml(body)}</span>
    </span>
  `).join("")}</span>`;
}
