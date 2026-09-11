/** Small DOM and formatting helpers shared by the front-end modules. */

export const $ = (sel, root = document) => root.querySelector(sel);

export const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

export const fmtInt = (n) => (n == null ? "—" : Number(n).toLocaleString());

export const fmtBytes = (b) => {
  if (b == null) return "—";
  const u = ["B", "KiB", "MiB", "GiB"];
  let i = 0, v = b;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
};

export const fmtNum = (v) => {
  if (v == null || !Number.isFinite(v)) return "—";
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  if (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(2);
  return v.toFixed(3).replace(/\.?0+$/, "");
};

export const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
export const rgbHex = (c) => "#" + c.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
export const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

/** Scene ids may contain folders; keep the slashes but encode each segment. */
export const encId = (id) => id.split("/").map(encodeURIComponent).join("/");

export async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `${res.status} ${res.statusText}`);
  return body;
}

let toastTimer = null;
export function toast(message, bad = false) {
  const t = $("#toast");
  t.textContent = message;
  t.classList.toggle("bad", bad);
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), bad ? 6000 : 3000);
}

/**
 * Renders a packed thumbnail (see src/instances.mjs) into a canvas.
 * Each hex nibble is one cell: 0 is empty, 1-15 shade near to far.
 */
export function drawSilhouette(canvas, hex, size, color) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!hex) return;

  const cell = w / size;
  for (let i = 0; i < hex.length && i < size * size; i++) {
    const shade = parseInt(hex[i], 16);
    if (!shade) continue;
    const x = i % size, y = Math.floor(i / size);
    // Far points sit darker, so the object reads as a solid shape.
    const k = 0.4 + 0.6 * (shade / 15);
    ctx.fillStyle = `rgb(${Math.round(color[0] * k)},${Math.round(color[1] * k)},${Math.round(color[2] * k)})`;
    ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
  }
}

/**
 * A modal confirmation. Resolves true if the user confirms, false otherwise.
 * `bodyNode` may be any element; it is shown between the title and the buttons.
 */
export function confirmDialog({ title, bodyNode, note = null, confirmText = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    const backdrop = el("div", "modal-backdrop");
    const modal = el("div", "modal");

    modal.appendChild(el("h3", null, title));
    if (bodyNode) modal.appendChild(bodyNode);
    if (note) modal.appendChild(el("div", "note", note));

    const row = el("div", "modal-buttons");
    const cancel = el("button", "btn", "Cancel");
    const ok = el("button", `btn ${danger ? "danger" : "primary"}`, confirmText);
    row.appendChild(cancel);
    row.appendChild(ok);
    modal.appendChild(row);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    const close = (result) => {
      document.removeEventListener("keydown", onKey, true);
      backdrop.remove();
      resolve(result);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); close(false); }
      if (e.key === "Enter") { e.stopPropagation(); close(true); }
    };

    cancel.addEventListener("click", () => close(false));
    ok.addEventListener("click", () => close(true));
    backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) close(false); });
    document.addEventListener("keydown", onKey, true);
    ok.focus();
  });
}
