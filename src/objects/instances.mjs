/**
 * Extracts individual objects from a segmented scene into a reusable library.
 *
 * Each instance is written as its own small Potree octree, recentred on an
 * anchor point (footprint centre at ground level). That means a spawned copy is
 * a real PointCloudOctree and renders through exactly the same pipeline as the
 * scene itself -- eye-dome lighting, adaptive point size, the shared point
 * budget -- rather than as a second-class overlay.
 */
import { join } from "node:path";
import { writeOctree } from "../octree/write.mjs";
import { silhouette, SILHOUETTE } from "../scene/thumbnail.mjs";

export { silhouette, SILHOUETTE };

/**
 * Convex hull of the footprint, relative to the anchor.
 *
 * Used to outline a highlighted object with a contour that follows its actual
 * shape instead of an axis-aligned box. Andrew's monotone chain.
 */
export function footprintHull(indices, px, py, anchor, maxPoints = 28) {
  const pts = [];
  // Thin dense objects first: the hull only needs the extremes.
  const step = Math.max(1, Math.floor(indices.length / 4000));
  for (let k = 0; k < indices.length; k += step) {
    const i = indices[k];
    pts.push([px[i] - anchor[0], py[i] - anchor[1]]);
  }
  if (pts.length < 3) return pts;

  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const build = (source) => {
    const out = [];
    for (const p of source) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  let hull = [...build(pts), ...build([...pts].reverse())];

  // Keep the outline light: drop the least significant vertices if needed.
  while (hull.length > maxPoints) {
    let worst = 1, worstArea = Infinity;
    for (let i = 1; i < hull.length - 1; i++) {
      const area = Math.abs(cross(hull[i - 1], hull[i], hull[i + 1]));
      if (area < worstArea) { worstArea = area; worst = i; }
    }
    hull.splice(worst, 1);
  }
  return hull.map(([x, y]) => [Number(x.toFixed(3)), Number(y.toFixed(3))]);
}

/**
 * @param opts.described    result of describeFields()
 * @param opts.indices      the valid point indices of the whole scene
 * @param opts.attributes   the same attribute writers used for the scene octree
 * @param opts.outDir       the scene's cache directory
 * @param opts.instanceField / opts.classField  entries from described.scalars
 * @param opts.classTable   [{ value, index, name, color }] for the class field
 * @param opts.spacing      the scene's finest-level spacing, so extracted objects
 *                      render at the same point size as the cloud they came from
 */
export function extractInstances(opts) {
  const {
    described, indices, attributes, outDir,
    instanceField, classField, classTable = [],
    minPoints = 24, maxInstances = 4000, gridSize = 128,
    spacing = null,
    onProgress = () => {},
  } = opts;

  if (!instanceField) return null;

  const [px, py, pz] = described.position.map((f) => f.data);
  const idData = instanceField.data;
  const classData = classField?.data ?? null;
  const classByValue = new Map(classTable.map((c) => [c.value, c]));

  // ---- group point indices by instance id ------------------------------
  const groups = new Map();
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    const id = idData[i];
    let g = groups.get(id);
    if (!g) { g = []; groups.set(id, g); }
    g.push(i);
  }

  const total = indices.length;
  const skipped = { tooSmall: 0, background: 0, overflow: 0 };

  // Extent of the whole cloud, to judge what counts as a backdrop.
  let sMinX = Infinity, sMinY = Infinity, sMaxX = -Infinity, sMaxY = -Infinity;
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    if (px[i] < sMinX) sMinX = px[i]; if (px[i] > sMaxX) sMaxX = px[i];
    if (py[i] < sMinY) sMinY = py[i]; if (py[i] > sMaxY) sMaxY = py[i];
  }
  const sceneW = Math.max(sMaxX - sMinX, 1e-6);
  const sceneD = Math.max(sMaxY - sMinY, 1e-6);

  /**
   * Is this group the ground / backdrop rather than an object?
   *
   * Judged from the data, never from the id: "instance 0" is a genuine object in
   * plenty of datasets and the unassigned bucket in plenty of others. A backdrop
   * is something that covers most of the scene's footprint, or most of its
   * points. A long thin wall spans one axis but almost none of the other, so it
   * survives this test.
   */
  const isBackground = (members) => {
    if (members.length > total * 0.5) return true;
    let mnX = Infinity, mnY = Infinity, mxX = -Infinity, mxY = -Infinity;
    for (const i of members) {
      if (px[i] < mnX) mnX = px[i]; if (px[i] > mxX) mxX = px[i];
      if (py[i] < mnY) mnY = py[i]; if (py[i] > mxY) mxY = py[i];
    }
    const coverage = ((mxX - mnX) / sceneW) * ((mxY - mnY) / sceneD);
    return coverage > 0.5;
  };

  const kept = [];
  const background = [];
  for (const [id, members] of groups) {
    if (members.length < minPoints) { skipped.tooSmall++; continue; }
    if (isBackground(members)) { background.push([id, members]); skipped.background++; continue; }
    kept.push([id, members]);
  }

  // Biggest objects first: those are the ones worth having in a library.
  kept.sort((a, b) => b[1].length - a[1].length);
  if (kept.length > maxInstances) {
    skipped.overflow = kept.length - maxInstances;
    kept.length = maxInstances;
  }

  // ---- write one octree per instance -----------------------------------
  const instances = [];
  let done = 0;

  for (const [id, members] of kept) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const i of members) {
      if (px[i] < minX) minX = px[i]; if (px[i] > maxX) maxX = px[i];
      if (py[i] < minY) minY = py[i]; if (py[i] > maxY) maxY = py[i];
      if (pz[i] < minZ) minZ = pz[i]; if (pz[i] > maxZ) maxZ = pz[i];
    }

    // Anchor: centre of the footprint, at the object's lowest point. Placing by
    // this anchor makes an object sit on whatever surface it is dropped onto.
    const anchor = [(minX + maxX) / 2, (minY + maxY) / 2, minZ];

    // majority class
    let klass = null;
    if (classData) {
      const tally = new Map();
      for (const i of members) tally.set(classData[i], (tally.get(classData[i]) ?? 0) + 1);
      let best = null, bestN = -1;
      for (const [v, n] of tally) if (n > bestN) { bestN = n; best = v; }
      klass = classByValue.get(best) ?? { value: best, index: 0, name: `class ${best}`, color: [150, 150, 150] };
    }

    const dir = `instances/${id}`;
    const memberIdx = Uint32Array.from(members);

    const result = writeOctree({
      indices: memberIdx,
      getX: (i) => px[i] - anchor[0],
      getY: (i) => py[i] - anchor[1],
      getZ: (i) => pz[i] - anchor[2],
      attributes,
      outDir: join(outDir, "instances", String(id)),
      name: `instance ${id}`,
      gridSize,
      maxPointsPerNode: 60000,
      spacing,
    });

    instances.push({
      id,
      dir,
      points: members.length,
      class: klass,
      size: [maxX - minX, maxY - minY, maxZ - minZ],
      anchor,
      // Anchor expressed in the instance octree's own local frame; the viewer
      // needs it to rotate a placed object about its centre rather than the
      // corner of its bounding cube.
      anchorLocal: [
        -result.metadata.boundingBox.min[0],
        -result.metadata.boundingBox.min[1],
        -result.metadata.boundingBox.min[2],
      ],
      bytes: result.octreeBytes,
      nodes: result.numNodes,
      silhouette: silhouette(members, px, py, pz, anchor),
      footprint: footprintHull(members, px, py, anchor),
    });

    done++;
    onProgress(done, kept.length);
  }

  // Backdrops get an entry too -- without an octree, so they can be clicked and
  // inspected in the scene even though there is nothing sensible to spawn.
  for (const [id, members] of background) {
    let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
    let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
    for (const i of members) {
      if (px[i] < mnX) mnX = px[i]; if (px[i] > mxX) mxX = px[i];
      if (py[i] < mnY) mnY = py[i]; if (py[i] > mxY) mxY = py[i];
      if (pz[i] < mnZ) mnZ = pz[i]; if (pz[i] > mxZ) mxZ = pz[i];
    }
    const anchor = [(mnX + mxX) / 2, (mnY + mxY) / 2, mnZ];

    let klass = null;
    if (classData) {
      const tally = new Map();
      for (const i of members) tally.set(classData[i], (tally.get(classData[i]) ?? 0) + 1);
      let best = null, bestN = -1;
      for (const [v, n] of tally) if (n > bestN) { bestN = n; best = v; }
      klass = classByValue.get(best) ?? { value: best, index: 0, name: `class ${best}`, color: [150, 150, 150] };
    }

    instances.push({
      id,
      dir: null,
      background: true,
      points: members.length,
      class: klass,
      size: [mxX - mnX, mxY - mnY, mxZ - mnZ],
      anchor,
      anchorLocal: null,
      bytes: 0,
      nodes: 0,
      silhouette: silhouette(members, px, py, pz, anchor),
      footprint: footprintHull(members, px, py, anchor),
    });
  }

  instances.sort((a, b) => (a.class?.name ?? "").localeCompare(b.class?.name ?? "") || b.points - a.points);

  return {
    field: instanceField.source,
    fieldName: instanceField.name,
    classField: classField?.source ?? null,
    silhouetteSize: SILHOUETTE,
    count: instances.length,
    skipped,
    instances,
  };
}
