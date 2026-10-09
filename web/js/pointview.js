/**
 * A small point cloud in a canvas, coloured by a signed scalar.
 *
 * Potree owns the viewport; this is for the thumbnails beside an experiment,
 * where three clouds have to be compared side by side and a second WebGL
 * context each would be absurd. Orthographic, painter's algorithm, a few
 * thousand points — plenty for "which end of the sofa did the model look at".
 */

/**
 * The diverging ramp: grey at nothing, blue for negative, red for positive.
 *
 * Two hues with a neutral midpoint, which is the only honest encoding for a
 * signed quantity — a one-hue ramp would make "argues against" and "argues for"
 * look like more and less of the same thing. The poles are the validated
 * diverging pair; the midpoint sits near the panel surface so a point that
 * means nothing recedes into it.
 */
const MID = [56, 56, 53];        // #383835
const NEG = [57, 135, 229];      // #3987e5  blue  -- argues against
const POS = [230, 103, 103];     // #e66767  red   -- argues for

/** t in [-1, 1]. The curve lifts mid magnitudes clear of the midpoint. */
export function divergingColor(t) {
  const k = Math.min(1, Math.abs(t)) ** 0.65;
  const pole = t < 0 ? NEG : POS;
  return [
    Math.round(MID[0] + (pole[0] - MID[0]) * k),
    Math.round(MID[1] + (pole[1] - MID[1]) * k),
    Math.round(MID[2] + (pole[2] - MID[2]) * k),
  ];
}

export const rampCss = (steps = 9) => {
  const stops = [];
  for (let i = 0; i < steps; i++) {
    const t = -1 + (2 * i) / (steps - 1);
    const [r, g, b] = divergingColor(t);
    stops.push(`rgb(${r},${g},${b}) ${Math.round((i / (steps - 1)) * 100)}%`);
  }
  return `linear-gradient(90deg, ${stops.join(", ")})`;
};

/**
 * Draws one frame.
 *
 * @param xyz     flat [x,y,z, …] in scene coordinates
 * @param values  one signed number per point
 * @param skip    optional 0/1 per point; 1 is left out entirely, which is how a
 *                removed point is shown — absence, not a colour, so it reads
 *                without relying on hue
 * @param scale   the magnitude that maps to a pole; shared across frames so the
 *                three are directly comparable
 */
export function drawPointView(canvas, { xyz, values, skip = null, yaw = 0.6, pitch = 0.35, scale = 1 }) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const n = values?.length ?? 0;
  if (!n) return;

  // Centre on the object, not the scene.
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += xyz[i * 3]; cy += xyz[i * 3 + 1]; cz += xyz[i * 3 + 2]; }
  cx /= n; cy /= n; cz /= n;

  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch), sinP = Math.sin(pitch);

  // Project every point once, then fit the result to the canvas.
  //
  // This is the same camera as the library thumbnails (src/scene/thumbnail.mjs),
  // deliberately: the two views sit a panel apart and must not disagree about
  // which way an object faces. The earlier version here used `x·cos − y·sin`
  // for the horizontal, which swaps the roles of +x and +y and renders the
  // object mirrored -- a right-handed world seen from above the horizon has to
  // put x→y counter-clockwise on screen, and that version put it clockwise.
  const u = new Float32Array(n), v = new Float32Array(n), d = new Float32Array(n);
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xyz[i * 3] - cx, y = xyz[i * 3 + 1] - cy, z = xyz[i * 3 + 2] - cz;
    // In-plane distance along the view azimuth; it carries the depth and the
    // foreshortening, and never the horizontal.
    const h = x * cosY + y * sinY;
    u[i] = -x * sinY + y * cosY;
    v[i] = z * cosP - h * sinP;           // z up
    d[i] = h * cosP + z * sinP;           // larger is farther
    if (u[i] < minU) minU = u[i]; if (u[i] > maxU) maxU = u[i];
    if (v[i] < minV) minV = v[i]; if (v[i] > maxV) maxV = v[i];
  }
  const pad = 6;
  const fit = Math.min((w - 2 * pad) / Math.max(maxU - minU, 1e-6),
                       (h - 2 * pad) / Math.max(maxV - minV, 1e-6));
  const offU = (w - (maxU - minU) * fit) / 2 - minU * fit;
  const offV = (h - (maxV - minV) * fit) / 2 + maxV * fit;

  // Far points first, so near ones land on top.
  const order = Array.from({ length: n }, (_, i) => i)
    .filter((i) => !skip || !skip[i])
    .sort((a, b) => d[b] - d[a]);

  const size = Math.max(1.5, Math.min(3, fit * 0.02));
  for (const i of order) {
    const [r, g, b] = divergingColor(values[i] / (scale || 1));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(offU + u[i] * fit - size / 2, offV - v[i] * fit - size / 2, size, size);
  }
}

/** Drag to orbit. Calls back with new angles; the caller redraws. */
export function orbitable(canvas, get, onChange) {
  let from = null;
  canvas.addEventListener("pointerdown", (e) => {
    from = { x: e.clientX, y: e.clientY, ...get() };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!from) return;
    onChange({
      yaw: from.yaw + (e.clientX - from.x) * 0.012,
      pitch: Math.max(-1.4, Math.min(1.4, from.pitch + (e.clientY - from.y) * 0.012)),
    });
  });
  const stop = () => { from = null; };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
}
