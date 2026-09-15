/**
 * Turning a point set into a small picture.
 *
 * Two of them, sharing one camera so a scene card and a library row are drawn
 * from the same angle:
 *
 *   silhouette()      one object, depth-shaded, tinted by its class colour
 *   sceneThumbnail()  a whole scene, in its own colours
 *
 * Both project the points the way a viewer would actually look at them. A
 * straight-on elevation stretched to fill a square is unreadable -- every object
 * becomes the same filled rectangle.
 */

/** Thumbnail resolution for the object library. */
export const SILHOUETTE = 32;

/** Resolution for a scene card. Bigger: a room has more to tell apart. */
export const SCENE_THUMB = 48;

/** Exposure boost for scene thumbnails; see the shading in sceneThumbnail(). */
const LIFT = 1.35;

/** Azimuth and elevation of the three-quarter view. */
const VIEW_AZIMUTH = (40 * Math.PI) / 180;
const VIEW_ELEVATION = (25 * Math.PI) / 180;

const ca = Math.cos(VIEW_AZIMUTH), sa = Math.sin(VIEW_AZIMUTH);
const ce = Math.cos(VIEW_ELEVATION), se = Math.sin(VIEW_ELEVATION);

/** Screen position of a point, for a camera orbiting at (azimuth, elevation), z up. */
export const projectView = (x, y, z) => [
  -x * sa + y * ca,
  -x * se * ca - y * se * sa + z * ce,
];

/** Distance along the view direction; smaller is nearer the camera. */
export const depthAlongView = (x, y, z) => x * ce * ca + y * ce * sa + z * se;

/**
 * Fits the projected points into a square, aspect ratio intact, one cell of
 * margin. Returns what the raster loop needs to place a point in a cell.
 */
function fitToGrid(extent, size) {
  const pad = 1;
  const usable = size - 2 * pad;
  const spanU = Math.max(extent.maxU - extent.minU, 1e-6);
  const spanV = Math.max(extent.maxV - extent.minV, 1e-6);
  const scale = usable / Math.max(spanU, spanV);
  return {
    scale,
    offU: pad + (usable - spanU * scale) / 2,
    offV: pad + (usable - spanV * scale) / 2,
    spanD: Math.max(extent.maxD - extent.minD, 1e-6),
  };
}

/** Projected bounds of a set of points, relative to an origin. */
function viewExtent(indices, px, py, pz, origin) {
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  let minD = Infinity, maxD = -Infinity;
  for (const i of indices) {
    const x = px[i] - origin[0], y = py[i] - origin[1], z = pz[i] - origin[2];
    const [u, v] = projectView(x, y, z);
    const d = depthAlongView(x, y, z);
    if (u < minU) minU = u; if (u > maxU) maxU = u;
    if (v < minV) minV = v; if (v > maxV) maxV = v;
    if (d < minD) minD = d; if (d > maxD) maxD = d;
  }
  return { minU, maxU, minV, maxV, minD, maxD };
}

/**
 * Renders a small shaded three-quarter view of one object.
 *
 * Encoded one hex nibble per cell: 0 is empty, 1-15 is near-to-far shading.
 */
export function silhouette(indices, px, py, pz, anchor, size = SILHOUETTE) {
  const extent = viewExtent(indices, px, py, pz, anchor);
  const { scale, offU, offV, spanD } = fitToGrid(extent, size);

  const cells = new Uint8Array(size * size);   // 0 = empty
  for (const i of indices) {
    const x = px[i] - anchor[0], y = py[i] - anchor[1], z = pz[i] - anchor[2];
    const [u, v] = projectView(x, y, z);
    let cu = Math.floor(offU + (u - extent.minU) * scale);
    let cv = Math.floor(offV + (extent.maxV - v) * scale);      // v up -> row 0 on top
    if (cu < 0) cu = 0; else if (cu >= size) cu = size - 1;
    if (cv < 0) cv = 0; else if (cv >= size) cv = size - 1;

    // Nearest point wins the cell, so the shading reads as a surface.
    const shade = 15 - Math.min(14, Math.floor(((depthAlongView(x, y, z) - extent.minD) / spanD) * 14));
    const idx = cv * size + cu;
    if (shade > cells[idx]) cells[idx] = shade;
  }

  let hex = "";
  for (let i = 0; i < cells.length; i++) hex += cells[i].toString(16);
  return hex;
}

/**
 * The z above which points are a ceiling rather than part of the scene.
 *
 * An indoor scan viewed from above the horizon is a picture of its ceiling: one
 * flat rectangle, the same for every room. So the top slab is dropped -- but
 * only when it really is a lid, judged the way the object library judges a
 * backdrop: does it cover most of the scene's footprint? The tops of furniture
 * and walls do not, and an outdoor scan has nothing up there at all, so both
 * keep every point.
 *
 * Returns Infinity when there is nothing to cull.
 */
function ceilingCut(indices, px, py, pz) {
  let zMin = Infinity, zMax = -Infinity;
  let sMinX = Infinity, sMaxX = -Infinity, sMinY = Infinity, sMaxY = -Infinity;
  for (const i of indices) {
    if (pz[i] < zMin) zMin = pz[i]; if (pz[i] > zMax) zMax = pz[i];
    if (px[i] < sMinX) sMinX = px[i]; if (px[i] > sMaxX) sMaxX = px[i];
    if (py[i] < sMinY) sMinY = py[i]; if (py[i] > sMaxY) sMaxY = py[i];
  }
  const height = zMax - zMin;
  if (!(height > 0)) return Infinity;

  const cut = zMax - 0.08 * height;
  let n = 0, mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
  for (const i of indices) {
    if (pz[i] < cut) continue;
    n++;
    if (px[i] < mnX) mnX = px[i]; if (px[i] > mxX) mxX = px[i];
    if (py[i] < mnY) mnY = py[i]; if (py[i] > mxY) mxY = py[i];
  }
  if (n < indices.length * 0.02) return Infinity;      // too thin to be a lid

  const coverage = ((mxX - mnX) / Math.max(sMaxX - sMinX, 1e-6))
                 * ((mxY - mnY) / Math.max(sMaxY - sMinY, 1e-6));
  return coverage > 0.6 ? cut : Infinity;
}

/**
 * Renders a whole scene in its own colours.
 *
 * Four characters per cell: "0" or "1" for empty/filled, then one hex digit each
 * of r, g, b. Nearest point wins the cell and the colour is dimmed with distance,
 * so the room reads as a surface rather than a flat wash.
 *
 * @param indices  the scene's valid point indices
 * @param rgb      [r, g, b] arrays in 0..255, or null for an untextured cloud
 */
export function sceneThumbnail(indices, px, py, pz, rgb = null, size = SCENE_THUMB) {
  if (indices.length === 0) return null;

  const cut = ceilingCut(indices, px, py, pz);
  const shown = cut === Infinity ? indices : indices.filter((i) => pz[i] < cut);
  const pts = shown.length ? shown : indices;

  const origin = [0, 0, 0];
  const extent = viewExtent(pts, px, py, pz, origin);
  const { scale, offU, offV, spanD } = fitToGrid(extent, size);

  const depth = new Float64Array(size * size).fill(Infinity);
  const filled = new Uint8Array(size * size);
  const col = new Uint8Array(size * size * 3);

  for (const i of pts) {
    const x = px[i], y = py[i], z = pz[i];
    const [u, v] = projectView(x, y, z);
    let cu = Math.floor(offU + (u - extent.minU) * scale);
    let cv = Math.floor(offV + (extent.maxV - v) * scale);
    if (cu < 0) cu = 0; else if (cu >= size) cu = size - 1;
    if (cv < 0) cv = 0; else if (cv >= size) cv = size - 1;

    const d = depthAlongView(x, y, z);
    const idx = cv * size + cu;
    if (d >= depth[idx]) continue;                     // a nearer point owns this cell
    depth[idx] = d;
    filled[idx] = 1;

    // Dim with distance so the room reads as a surface, then lift the result:
    // indoor scans are dark, and a 48-pixel card of them on a dark panel is
    // unreadable at true exposure. Without colour there is nothing but the
    // shading, so a plain grey carries it.
    const k = (1 - 0.3 * ((d - extent.minD) / spanD)) * LIFT;
    const r = rgb ? rgb[0][i] : 190, g = rgb ? rgb[1][i] : 190, b = rgb ? rgb[2][i] : 190;
    col[idx * 3] = Math.min(255, Math.round(r * k));
    col[idx * 3 + 1] = Math.min(255, Math.round(g * k));
    col[idx * 3 + 2] = Math.min(255, Math.round(b * k));
  }

  const nibble = (v) => (v >> 4).toString(16);
  let out = "";
  for (let i = 0; i < size * size; i++) {
    out += filled[i]
      ? `1${nibble(col[i * 3])}${nibble(col[i * 3 + 1])}${nibble(col[i * 3 + 2])}`
      : "0000";
  }
  return { size, cells: out };
}
