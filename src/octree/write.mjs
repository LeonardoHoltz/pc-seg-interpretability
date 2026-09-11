/**
 * Writes a Potree 2.0 octree (metadata.json + hierarchy.bin + octree.bin).
 *
 * Format notes, taken from Potree's own loader (modules/loader/2.0):
 *
 *  hierarchy.bin  22-byte records: uint8 type, uint8 childMask, uint32 numPoints,
 *                 int64 byteOffset, int64 byteSize. Records appear in
 *                 breadth-first order with children in ascending child index --
 *                 exactly the order OctreeLoader.parseHierarchy walks them.
 *                 We emit a single chunk, so `hierarchy.firstChunkSize` is the
 *                 whole file and no proxy (type 2) nodes are needed.
 *
 *  octree.bin     per node, points interleaved in attribute order.
 *
 *  positions      int32 triples; world = value * scale + offset.
 *
 *  scalars        Potree's decoder normalises an attribute into 0..1 only when
 *                 its type is larger than 4 bytes; smaller types reach the
 *                 shader raw. The generic `getExtra()` colour path assumes a
 *                 normalised input, so every generic scalar is written as
 *                 `double`. `classification` is the exception: its shader path
 *                 indexes a 256-entry LUT with the raw value, so it stays uint8.
 */
import { openSync, writeSync, closeSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const TYPE_SIZES = {
  int8: 1, uint8: 1, int16: 2, uint16: 2,
  int32: 4, uint32: 4, float: 4, double: 8,
};

const NODE_TYPE_NORMAL = 0;
const NODE_TYPE_LEAF = 1;

/**
 * Estimates the real point spacing of a set of points.
 *
 * `spacing` drives adaptive point size, and the usual cubeSize/gridSize is only
 * right when the root was actually grid-subsampled. A node that kept every point
 * has whatever spacing the source data has -- and for an elongated object (a
 * wall spanning the scene, say) the bounding cube is huge while the points are
 * dense, so the cube-derived figure comes out several times too large and its
 * points render as oversized blobs.
 *
 * Finds the grid resolution at which most cells hold a single point; that cell
 * size is the natural spacing.
 */
function estimateSpacing(indices, getX, getY, getZ, cubeMin, cubeSize) {
  const n = indices.length;
  if (n < 8) return cubeSize / 128;

  // A sample is plenty: this is an average, not a measurement of every point.
  const step = Math.max(1, Math.floor(n / 30000));
  const sample = [];
  for (let k = 0; k < n; k += step) sample.push(indices[k]);
  const m = sample.length;

  // A finer sweep than powers of two: each step is ~1.25x, so the answer is
  // quantised by at most that much.
  const RESOLUTIONS = [4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100,
                       128, 160, 200, 256, 320, 400, 512, 640, 800, 1024];
  let spacing = cubeSize / 128;
  for (const g of RESOLUTIONS) {
    const cells = new Set();
    const inv = g / cubeSize;
    for (const i of sample) {
      let ix = Math.floor((getX(i) - cubeMin[0]) * inv);
      let iy = Math.floor((getY(i) - cubeMin[1]) * inv);
      let iz = Math.floor((getZ(i) - cubeMin[2]) * inv);
      if (ix < 0) ix = 0; else if (ix >= g) ix = g - 1;
      if (iy < 0) iy = 0; else if (iy >= g) iy = g - 1;
      if (iz < 0) iz = 0; else if (iz >= g) iz = g - 1;
      cells.add(ix + iy * g + iz * g * g);
    }
    spacing = cubeSize / g;
    // Once nearly every cell holds a single point the grid has reached the
    // data's own resolution; going finer just spreads the same points thinner.
    if (cells.size >= m * 0.9) break;
  }
  return spacing;
}

/**
 * @param opts.numPoints   total points to place
 * @param opts.getX/Y/Z    (i) => world coordinate
 * @param opts.indices     Uint32Array of point indices to include
 * @param opts.attributes  extra attributes, in the order they should be interleaved:
 *                     [{ name, type, numElements, min[], max[], description,
 *                        write(view, byteOffset, pointIndex) }].
 *                     The `position` attribute is prepended here, because it
 *                     needs the quantisation scale computed below.
 * @param opts.outDir      destination directory
 * @param opts.gridSize    subsampling grid per node (default 128)
 * @param opts.maxPointsPerNode  a node holding at most this many points becomes a leaf
 * @param opts.grid        reuses an existing quantisation grid
 *                     ({ cubeMin, cubeSize, scale }). Rebuilding a scene from
 *                     its own decoded points must keep the same grid, or every
 *                     coordinate is re-rounded against a slightly shifted one
 *                     and drifts by a quantisation step.
 * @param opts.spacing     overrides the root spacing. Used when cutting an object
 *                     out of a scene: what matters is not the object's own point
 *                     density but the size the scene draws those same points at,
 *                     so a placed copy matches the cloud it came from.
 */
export function writeOctree(opts) {
  const {
    indices, getX, getY, getZ, attributes, outDir,
    name = "cloud", description = "", projection = "",
    gridSize = 128, maxPointsPerNode = 40000, maxDepth = 14,
    spacing: spacingOverride = null,
    grid = null,
    onProgress = () => {},
  } = opts;

  const numPoints = indices.length;
  if (numPoints === 0) throw new Error("refusing to build an octree with no points");

  // ---- bounding cube ---------------------------------------------------
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let k = 0; k < numPoints; k++) {
    const i = indices[k];
    const x = getX(i), y = getY(i), z = getZ(i);
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  let cubeMin, cubeSize, scale;
  if (grid) {
    // Reusing the grid keeps re-encoding exactly idempotent.
    cubeMin = grid.cubeMin.slice();
    cubeSize = grid.cubeSize;
    scale = grid.scale;
  } else {
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
    // A cube keeps `spacing` halving cleanly with each octree level, which is
    // what Potree's point-size and LOD logic assumes.
    const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
    cubeSize = extent * 1.001;                   // nudge so no point sits on the far face
    const half = cubeSize / 2;
    cubeMin = [cx - half, cy - half, cz - half];
    // int32 quantisation step; ~1.5e9 steps across the cube is well inside int32.
    scale = Math.max(cubeSize / 1.5e9, 1e-9);
  }
  const cubeMax = [cubeMin[0] + cubeSize, cubeMin[1] + cubeSize, cubeMin[2] + cubeSize];

  // ---- build the tree --------------------------------------------------
  const stamp = new Int32Array(gridSize * gridSize * gridSize);
  let generation = 0;

  const root = {
    name: "r", level: 0, min: cubeMin, size: cubeSize,
    points: null, children: new Array(8).fill(null),
  };

  const ordered = [root];          // creation order == the order Potree reads records
  const queue = [{ node: root, pending: indices }];
  let maxLevel = 0;
  let processed = 0;

  while (queue.length > 0) {
    const { node, pending } = queue.shift();
    maxLevel = Math.max(maxLevel, node.level);

    if (pending.length <= maxPointsPerNode || node.level >= maxDepth) {
      node.points = pending;
      processed += pending.length;
      onProgress(processed, numPoints);
      continue;
    }

    // Keep one point per occupied cell of a gridSize^3 lattice over this node;
    // the rest are pushed down to the children. This is the classic MNO layout:
    // every level is a uniformly thinned view of the whole cloud.
    generation++;
    const cell = gridSize / node.size;
    const keep = [];
    const rest = [];
    for (let k = 0; k < pending.length; k++) {
      const i = pending[k];
      let ix = ((getX(i) - node.min[0]) * cell) | 0;
      let iy = ((getY(i) - node.min[1]) * cell) | 0;
      let iz = ((getZ(i) - node.min[2]) * cell) | 0;
      if (ix < 0) ix = 0; else if (ix >= gridSize) ix = gridSize - 1;
      if (iy < 0) iy = 0; else if (iy >= gridSize) iy = gridSize - 1;
      if (iz < 0) iz = 0; else if (iz >= gridSize) iz = gridSize - 1;

      const c = ix + iy * gridSize + iz * gridSize * gridSize;
      if (stamp[c] !== generation) {
        stamp[c] = generation;
        keep.push(i);
      } else {
        rest.push(i);
      }
    }

    node.points = Uint32Array.from(keep);
    processed += keep.length;
    onProgress(processed, numPoints);

    if (rest.length === 0) continue;

    // Split the remainder into octants. Potree's createChildAABB uses
    // bit 0 = upper half in z, bit 1 = y, bit 2 = x.
    const midX = node.min[0] + node.size / 2;
    const midY = node.min[1] + node.size / 2;
    const midZ = node.min[2] + node.size / 2;
    const buckets = [[], [], [], [], [], [], [], []];
    for (let k = 0; k < rest.length; k++) {
      const i = rest[k];
      const c = (getX(i) >= midX ? 4 : 0) | (getY(i) >= midY ? 2 : 0) | (getZ(i) >= midZ ? 1 : 0);
      buckets[c].push(i);
    }

    const childSize = node.size / 2;
    for (let c = 0; c < 8; c++) {
      if (buckets[c].length === 0) continue;
      const child = {
        name: node.name + c,
        level: node.level + 1,
        min: [
          node.min[0] + ((c & 4) ? childSize : 0),
          node.min[1] + ((c & 2) ? childSize : 0),
          node.min[2] + ((c & 1) ? childSize : 0),
        ],
        size: childSize,
        points: null,
        children: new Array(8).fill(null),
      };
      node.children[c] = child;
      ordered.push(child);
      queue.push({ node: child, pending: Uint32Array.from(buckets[c]) });
    }
  }

  // ---- attribute layout ------------------------------------------------
  mkdirSync(outDir, { recursive: true });

  // Positions are quantised against the cube origin: world = value * scale + offset.
  // Divide rather than multiply by a precomputed reciprocal: 1/scale is itself
  // rounded, which shifts the odd point across a .5 boundary and off by one step.
  const position = {
    name: "position",
    type: "int32",
    numElements: 3,
    description: "",
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    write: (view, o, i) => {
      view.setInt32(o + 0, Math.round((getX(i) - cubeMin[0]) / scale), true);
      view.setInt32(o + 4, Math.round((getY(i) - cubeMin[1]) / scale), true);
      view.setInt32(o + 8, Math.round((getZ(i) - cubeMin[2]) / scale), true);
    },
  };
  const allAttributes = [position, ...attributes];

  const bytesPerPoint = allAttributes.reduce((a, at) => a + TYPE_SIZES[at.type] * at.numElements, 0);
  const attrOffsets = [];
  {
    let acc = 0;
    for (const at of allAttributes) {
      attrOffsets.push(acc);
      acc += TYPE_SIZES[at.type] * at.numElements;
    }
  }

  const octreePath = join(outDir, "octree.bin");
  const fd = openSync(octreePath, "w");
  let byteCursor = 0;

  try {
    for (const node of ordered) {
      const pts = node.points;
      node.numPoints = pts.length;
      node.byteOffset = byteCursor;
      node.byteSize = pts.length * bytesPerPoint;

      if (pts.length === 0) continue;

      const buf = Buffer.allocUnsafe(node.byteSize);
      const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
      for (let k = 0; k < pts.length; k++) {
        const base = k * bytesPerPoint;
        const i = pts[k];
        for (let a = 0; a < allAttributes.length; a++) {
          allAttributes[a].write(view, base + attrOffsets[a], i);
        }
      }
      writeSync(fd, buf, 0, buf.length);
      byteCursor += buf.length;
    }
  } finally {
    closeSync(fd);
  }

  // A root with no children was never subsampled, so its spacing is the data's
  // own, not the grid's.
  const rootSubsampled = root.children.some((c) => c !== null);
  const spacing = spacingOverride ?? (rootSubsampled
    ? cubeSize / gridSize
    : estimateSpacing(indices, getX, getY, getZ, cubeMin, cubeSize));

  // ---- write hierarchy.bin --------------------------------------------
  const RECORD = 22;
  const hierarchy = Buffer.alloc(ordered.length * RECORD);
  {
    const view = new DataView(hierarchy.buffer, hierarchy.byteOffset, hierarchy.byteLength);
    ordered.forEach((node, idx) => {
      let childMask = 0;
      for (let c = 0; c < 8; c++) if (node.children[c]) childMask |= (1 << c);

      const o = idx * RECORD;
      view.setUint8(o + 0, childMask === 0 ? NODE_TYPE_LEAF : NODE_TYPE_NORMAL);
      view.setUint8(o + 1, childMask);
      view.setUint32(o + 2, node.numPoints, true);
      view.setBigInt64(o + 6, BigInt(node.byteOffset), true);
      view.setBigInt64(o + 14, BigInt(node.byteSize), true);
    });
  }
  writeFileSync(join(outDir, "hierarchy.bin"), hierarchy);

  // ---- write metadata.json --------------------------------------------
  const metadata = {
    version: "2.0",
    name,
    description,
    points: numPoints,
    projection,
    hierarchy: {
      firstChunkSize: hierarchy.length,
      stepSize: 100,           // unused: the whole hierarchy is one chunk
      depth: maxLevel,
    },
    offset: cubeMin,
    scale: [scale, scale, scale],
    spacing,
    boundingBox: { min: cubeMin, max: cubeMax },
    encoding: "DEFAULT",
    attributes: allAttributes.map((at) => ({
      name: at.name,
      description: at.description ?? "",
      size: TYPE_SIZES[at.type] * at.numElements,
      numElements: at.numElements,
      elementSize: TYPE_SIZES[at.type],
      type: at.type,
      min: at.min,
      max: at.max,
    })),
  };
  writeFileSync(join(outDir, "metadata.json"), JSON.stringify(metadata, null, 2));

  return {
    metadata,
    numNodes: ordered.length,
    depth: maxLevel,
    octreeBytes: byteCursor,
    hierarchyBytes: hierarchy.length,
    bytesPerPoint,
    tightBoundingBox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    scale,
    offset: cubeMin,
  };
}

export { TYPE_SIZES };
