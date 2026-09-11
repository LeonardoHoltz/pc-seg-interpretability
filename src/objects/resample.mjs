/**
 * Rescales an extracted object and resamples it to a target point spacing.
 *
 * Scaling an object without resampling spreads its points apart, so an enlarged
 * object looks sparse; shrinking one packs them together into a dense clot.
 * Neither matches the scene it is being placed into. This rebuilds the object at
 * the requested size with roughly the scene's own point density: gaps are filled
 * by inserting midpoints between neighbours, and excess points are thinned on a
 * grid.
 *
 * New points inherit every attribute from the neighbour they were derived from
 * rather than averaging, so class ids and instance ids stay exact.
 */
import { existsSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { writeOctree } from "../octree/write.mjs";
import { readOctree } from "../octree/read.mjs";

/** Cap so an extreme scale cannot explode into an unusable cloud. */
const MAX_POINTS = 4_000_000;
const MAX_PASSES = 6;

const cellKey = (x, y, z, inv) =>
  `${Math.floor(x * inv)},${Math.floor(y * inv)},${Math.floor(z * inv)}`;

/** Keeps one point per cell of a grid with the given cell size. */
function thin(points, cell) {
  const inv = 1 / cell;
  const seen = new Set();
  const keep = [];
  for (let i = 0; i < points.n; i++) {
    const k = cellKey(points.x[i], points.y[i], points.z[i], inv);
    if (seen.has(k)) continue;
    seen.add(k);
    keep.push(i);
  }
  return keep;
}

/**
 * Inserts midpoints between neighbouring points until no gap larger than
 * `target` remains. A midpoint of two nearby surface points still lies close to
 * the surface, so this densifies without inventing structure.
 *
 * Works from coarse to fine: after scaling by s the gaps are s x too wide, so
 * the neighbour search has to start at that spacing and halve each pass. A
 * search radius fixed at the target would find no neighbours at all once the
 * object is more than about twice its original size.
 */
function densify(points, target, startSpacing) {
  // Global de-duplication grid: nothing is inserted where a point already sits.
  const inv = 1 / (target * 0.9);
  const occupied = new Set();
  for (let i = 0; i < points.n; i++) {
    occupied.add(cellKey(points.x[i], points.y[i], points.z[i], inv));
  }

  let current = Math.max(startSpacing, target);
  for (let pass = 0; pass < MAX_PASSES && current > target * 1.05; pass++) {
    const search = current * 1.45;
    const sInv = 1 / search;
    const buckets = new Map();
    for (let i = 0; i < points.n; i++) {
      const k = cellKey(points.x[i], points.y[i], points.z[i], sInv);
      let b = buckets.get(k);
      if (!b) { b = []; buckets.set(k, b); }
      b.push(i);
    }

    const added = [];
    const t2 = search * search;
    for (let i = 0; i < points.n && points.n + added.length < MAX_POINTS; i++) {
      const bx = Math.floor(points.x[i] * sInv);
      const by = Math.floor(points.y[i] * sInv);
      const bz = Math.floor(points.z[i] * sInv);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const b = buckets.get(`${bx + dx},${by + dy},${bz + dz}`);
            if (!b) continue;
            for (const j of b) {
              if (j <= i) continue;
              const ddx = points.x[j] - points.x[i];
              const ddy = points.y[j] - points.y[i];
              const ddz = points.z[j] - points.z[i];
              const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 > t2) continue;
              const mx = points.x[i] + ddx / 2;
              const my = points.y[i] + ddy / 2;
              const mz = points.z[i] + ddz / 2;
              const k = cellKey(mx, my, mz, inv);
              if (occupied.has(k)) continue;
              occupied.add(k);
              added.push({ x: mx, y: my, z: mz, from: i });
            }
          }
        }
      }
    }

    current /= 2;
    if (added.length === 0) continue;

    // Grow the arrays and copy attributes from the point each midpoint came from.
    const total = points.n + added.length;
    const grow = (src, elems) => {
      const out = new Float64Array(total * elems);
      out.set(src.subarray(0, points.n * elems));
      return out;
    };
    const nx = grow(points.x, 1), ny = grow(points.y, 1), nz = grow(points.z, 1);
    const cols = new Map();
    for (const [name, col] of points.columns) {
      cols.set(name, { ...col, data: grow(col.data, col.numElements) });
    }
    added.forEach((a, idx) => {
      const w = points.n + idx;
      nx[w] = a.x; ny[w] = a.y; nz[w] = a.z;
      for (const [, col] of cols) {
        const e = col.numElements;
        for (let k = 0; k < e; k++) col.data[w * e + k] = col.data[a.from * e + k];
      }
    });
    points = { n: total, x: nx, y: ny, z: nz, columns: cols };
    if (total >= MAX_POINTS) break;
  }
  return points;
}

/**
 * @param opts.dir           the instance octree to rescale
 * @param opts.outDir        where the rescaled octree is written
 * @param opts.scale         requested size multiplier
 * @param opts.targetSpacing the point spacing to aim for (the scene's)
 */
export function resampleInstance({ dir, outDir, scale, targetSpacing }) {
  const src = readOctree(dir);

  // The instance octree is already centred on its anchor, so scaling is a plain
  // multiply; the anchor stays at the origin.
  const n = src.count;
  let points = {
    n,
    x: new Float64Array(n), y: new Float64Array(n), z: new Float64Array(n),
    columns: new Map([...src.columns].map(([k, v]) => [k, { ...v }])),
  };
  for (let i = 0; i < n; i++) {
    points.x[i] = src.x[i] * scale;
    points.y[i] = src.y[i] * scale;
    points.z[i] = src.z[i] * scale;
  }

  const before = points.n;
  // The source octree was written at the scene's spacing, so after scaling the
  // gaps are exactly `scale` times too wide.
  const startSpacing = (src.metadata.spacing || targetSpacing) * scale;
  if (scale > 1.001) {
    points = densify(points, targetSpacing, startSpacing);
  } else if (scale < 0.999) {
    // Shrinking packs points together; thin them back to the target spacing.
    const keep = thin(points, targetSpacing);
    const cols = new Map();
    for (const [name, col] of points.columns) {
      const e = col.numElements;
      const out = new Float64Array(keep.length * e);
      keep.forEach((src_i, w) => {
        for (let k = 0; k < e; k++) out[w * e + k] = col.data[src_i * e + k];
      });
      cols.set(name, { ...col, data: out });
    }
    const nx = new Float64Array(keep.length), ny = new Float64Array(keep.length), nz = new Float64Array(keep.length);
    keep.forEach((src_i, w) => { nx[w] = points.x[src_i]; ny[w] = points.y[src_i]; nz[w] = points.z[src_i]; });
    points = { n: keep.length, x: nx, y: ny, z: nz, columns: cols };
  }

  // ---- write the rescaled octree, reusing the source's attribute layout ----
  const attributes = [];
  for (const att of src.metadata.attributes) {
    if (att.name === "position") continue;
    const col = points.columns.get(att.name);
    if (!col) continue;
    const e = att.numElements;
    const data = col.data;
    const write = {
      uint8: (v, o, i) => { for (let k = 0; k < e; k++) v.setUint8(o + k, data[i * e + k]); },
      uint16: (v, o, i) => { for (let k = 0; k < e; k++) v.setUint16(o + k * 2, data[i * e + k], true); },
      int32: (v, o, i) => { for (let k = 0; k < e; k++) v.setInt32(o + k * 4, data[i * e + k], true); },
      uint32: (v, o, i) => { for (let k = 0; k < e; k++) v.setUint32(o + k * 4, data[i * e + k], true); },
      float: (v, o, i) => { for (let k = 0; k < e; k++) v.setFloat32(o + k * 4, data[i * e + k], true); },
      double: (v, o, i) => { for (let k = 0; k < e; k++) v.setFloat64(o + k * 8, data[i * e + k], true); },
    }[att.type];
    if (!write) continue;

    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lo) lo = data[i];
      if (data[i] > hi) hi = data[i];
    }
    if (!(hi > lo)) hi = lo + 1;

    attributes.push({
      name: att.name, type: att.type, numElements: e, description: att.description ?? "",
      min: e === 1 ? [lo] : att.min, max: e === 1 ? [hi] : att.max,
      write,
    });
  }

  const indices = new Uint32Array(points.n);
  for (let i = 0; i < points.n; i++) indices[i] = i;

  mkdirSync(outDir, { recursive: true });
  const result = writeOctree({
    indices,
    getX: (i) => points.x[i], getY: (i) => points.y[i], getZ: (i) => points.z[i],
    attributes, outDir,
    name: `rescaled x${scale}`,
    maxPointsPerNode: 60000,
    spacing: targetSpacing,
  });

  return {
    scale,
    pointsBefore: before,
    pointsAfter: points.n,
    anchorLocal: [
      -result.metadata.boundingBox.min[0],
      -result.metadata.boundingBox.min[1],
      -result.metadata.boundingBox.min[2],
    ],
    size: [
      result.tightBoundingBox.max[0] - result.tightBoundingBox.min[0],
      result.tightBoundingBox.max[1] - result.tightBoundingBox.min[1],
      result.tightBoundingBox.max[2] - result.tightBoundingBox.min[2],
    ],
    capped: points.n >= MAX_POINTS,
  };
}
