/**
 * Adds or replaces per-point scalar attributes on a converted scene.
 *
 * Shared by everything that brings values back from an inference service --
 * predicted labels, confidence scores, saliency. The octree is rebuilt from its
 * own decoded points with the extra columns appended, reusing the existing
 * quantisation grid so coordinates come out bit-identical rather than drifting
 * by a rounding step on every rewrite.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { writeOctree } from "../octree/write.mjs";
import { readOctree } from "../octree/read.mjs";

const WRITERS = {
  uint8: (data, e) => (v, o, i) => { for (let k = 0; k < e; k++) v.setUint8(o + k, data[i * e + k]); },
  uint16: (data, e) => (v, o, i) => { for (let k = 0; k < e; k++) v.setUint16(o + k * 2, data[i * e + k], true); },
  int32: (data, e) => (v, o, i) => { for (let k = 0; k < e; k++) v.setInt32(o + k * 4, data[i * e + k], true); },
  uint32: (data, e) => (v, o, i) => { for (let k = 0; k < e; k++) v.setUint32(o + k * 4, data[i * e + k], true); },
  float: (data, e) => (v, o, i) => { for (let k = 0; k < e; k++) v.setFloat32(o + k * 4, data[i * e + k], true); },
  double: (data, e) => (v, o, i) => { for (let k = 0; k < e; k++) v.setFloat64(o + k * 8, data[i * e + k], true); },
};

export function minMax(arr) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo > hi) { lo = 0; hi = 0; }
  // Potree divides by (max - min) when normalising a wide attribute.
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

/**
 * @param add    [{ name, label, kind, values (per point), classes?, sceneExtra? }]
 * @param remove attribute names to drop before adding
 */
export function rewriteSceneWithAttributes(sceneId, { cacheDir, add = [], remove = [], onProgress = () => {} }) {
  const sceneJsonPath = join(cacheDir, "scene.json");
  if (!existsSync(sceneJsonPath)) throw new Error(`${sceneId} has not been converted yet`);
  const scene = JSON.parse(readFileSync(sceneJsonPath, "utf8"));
  const octree = readOctree(cacheDir);
  const n = octree.count;

  const dropped = new Set([...remove, ...add.map((a) => a.name)]);
  const attributes = [];

  for (const att of octree.metadata.attributes) {
    if (att.name === "position" || dropped.has(att.name)) continue;
    const col = octree.columns.get(att.name);
    const make = WRITERS[att.type];
    if (!col || !make) continue;
    attributes.push({
      name: att.name, type: att.type, numElements: att.numElements,
      description: att.description ?? "", min: att.min, max: att.max,
      write: make(col.data, att.numElements),
    });
  }

  for (const entry of add) {
    if (entry.values.length !== n) {
      throw new Error(`"${entry.name}" has ${entry.values.length} values for ${n} points`);
    }
    const [lo, hi] = minMax(entry.values);
    entry.range = [lo, hi];
    attributes.push({
      name: entry.name, type: "double", numElements: 1,
      description: entry.kind ?? "continuous", min: [lo], max: [hi],
      write: WRITERS.double(entry.values, 1),
    });
  }

  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  const bb = octree.metadata.boundingBox;
  let lastReport = 0;
  const result = writeOctree({
    indices,
    getX: (i) => octree.x[i], getY: (i) => octree.y[i], getZ: (i) => octree.z[i],
    attributes, outDir: cacheDir,
    name: sceneId,
    description: scene.description ?? "",
    // The points are unchanged; only columns are added.
    grid: { cubeMin: bb.min, cubeSize: bb.max[0] - bb.min[0], scale: octree.metadata.scale[0] },
    spacing: octree.metadata.spacing,
    onProgress: (done, total) => {
      const now = Date.now();
      if (now - lastReport < 200) return;
      lastReport = now;
      onProgress({ phase: "building", progress: 0.7 + 0.3 * (done / total), message: "Rebuilding the octree" });
    },
  });

  const updated = { ...scene };
  updated.numPoints = n;
  updated.boundingBox = result.metadata.boundingBox;
  updated.tightBoundingBox = result.tightBoundingBox;
  updated.spacing = result.metadata.spacing;
  updated.depth = result.depth;
  updated.numNodes = result.numNodes;
  updated.bytesPerPoint = result.bytesPerPoint;
  updated.octreeBytes = result.octreeBytes;

  updated.attributes = [
    ...(scene.attributes ?? []).filter((a) => !dropped.has(a.name)),
    ...add.map((entry) => ({
      name: entry.name,
      source: entry.name,
      label: entry.label ?? entry.name,
      kind: entry.kind ?? "continuous",
      potreeMode: "extra",
      min: entry.range[0],
      max: entry.range[1],
      ...(entry.classes ? { classes: entry.classes, numClasses: entry.classes.length } : {}),
    })),
  ];

  writeFileSync(sceneJsonPath, JSON.stringify(updated, null, 2));
  return { scene: updated, result, octree };
}
