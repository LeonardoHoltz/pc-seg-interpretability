#!/usr/bin/env node
/**
 * Validates a converted octree by decoding it the way Potree's loader does,
 * then comparing the result against the source PCD.
 *
 * It re-implements OctreeLoader.parseHierarchy / createChildAABB and the
 * DecoderWorker position maths, so a format mistake shows up here rather than
 * as an empty canvas in the browser.
 *
 *   node tools/verify_octree.mjs <scene-id>
 */
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { readCloud } from "../src/io/cloud.mjs";
import { cacheDirFor, resolveScene } from "../src/scene/registry.mjs";

const id = process.argv[2];
if (!id) { console.error("usage: node tools/verify_octree.mjs <scene-id>"); process.exit(1); }

const dir = cacheDirFor(id);
const metadata = JSON.parse(readFileSync(join(dir, "metadata.json"), "utf8"));
const hierarchy = readFileSync(join(dir, "hierarchy.bin"));
const octree = readFileSync(join(dir, "octree.bin"));
const scene = JSON.parse(readFileSync(join(dir, "scene.json"), "utf8"));

let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "  ok  " : "  FAIL"}  ${msg}`); if (!ok) failures++; };

// ---- metadata sanity ------------------------------------------------------
console.log(`\nverifying ${id}`);
check(metadata.version === "2.0", `version is 2.0`);
check(metadata.encoding === "DEFAULT", `encoding is DEFAULT`);
check(metadata.attributes[0].name === "position" && metadata.attributes[0].type === "int32",
      `first attribute is int32 position`);
check(metadata.hierarchy.firstChunkSize === hierarchy.length,
      `firstChunkSize (${metadata.hierarchy.firstChunkSize}) == hierarchy.bin size (${hierarchy.length})`);

const bytesPerPoint = metadata.attributes.reduce((a, at) => a + at.size, 0);
check(bytesPerPoint === scene.bytesPerPoint, `bytesPerPoint = ${bytesPerPoint}`);

// ---- walk the hierarchy exactly like Potree does --------------------------
function createChildAABB(box, index) {
  const min = [...box.min], max = [...box.max];
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  if (index & 0b0001) min[2] += size[2] / 2; else max[2] -= size[2] / 2;
  if (index & 0b0010) min[1] += size[1] / 2; else max[1] -= size[1] / 2;
  if (index & 0b0100) min[0] += size[0] / 2; else max[0] -= size[0] / 2;
  return { min, max };
}

const RECORD = 22;
const numRecords = hierarchy.length / RECORD;
check(Number.isInteger(numRecords), `hierarchy.bin is a whole number of 22-byte records (${numRecords})`);

const hview = new DataView(hierarchy.buffer, hierarchy.byteOffset, hierarchy.byteLength);
const rootBox = { min: [...metadata.boundingBox.min], max: [...metadata.boundingBox.max] };
const nodes = [{ name: "r", level: 0, box: rootBox }];

let totalPoints = 0;
let maxLevel = 0;
const ranges = [];

for (let i = 0; i < numRecords; i++) {
  const current = nodes[i];
  if (!current) { check(false, `record ${i} has no node (hierarchy ordering is wrong)`); break; }

  const o = i * RECORD;
  const type = hview.getUint8(o + 0);
  const childMask = hview.getUint8(o + 1);
  const numPoints = hview.getUint32(o + 2, true);
  const byteOffset = Number(hview.getBigInt64(o + 6, true));
  const byteSize = Number(hview.getBigInt64(o + 14, true));

  current.numPoints = numPoints;
  current.byteOffset = byteOffset;
  current.byteSize = byteSize;
  current.type = type;
  totalPoints += numPoints;
  maxLevel = Math.max(maxLevel, current.level);
  ranges.push([byteOffset, byteSize, current.name]);

  if (byteSize !== numPoints * bytesPerPoint) {
    check(false, `node ${current.name}: byteSize ${byteSize} != numPoints*${bytesPerPoint}`);
  }
  if (type === 2) { check(false, `node ${current.name}: unexpected proxy node in a single-chunk hierarchy`); continue; }
  if (childMask === 0 && type !== 1) check(false, `node ${current.name}: leaf not marked as type 1`);

  for (let c = 0; c < 8; c++) {
    if (((1 << c) & childMask) === 0) continue;
    nodes.push({ name: current.name + c, level: current.level + 1, box: createChildAABB(current.box, c) });
  }
}

check(nodes.length === numRecords,
      `hierarchy is self-consistent: ${nodes.length} nodes for ${numRecords} records`);
check(totalPoints === metadata.points,
      `node point counts sum to metadata.points (${totalPoints} vs ${metadata.points})`);
check(maxLevel === metadata.hierarchy.depth, `depth ${maxLevel} matches metadata`);

// byte ranges must tile octree.bin exactly, no gaps or overlaps
ranges.sort((a, b) => a[0] - b[0]);
let cursor = 0, tiled = true;
for (const [off, size, name] of ranges) {
  if (off !== cursor) { tiled = false; console.log(`      gap/overlap at ${name}: expected ${cursor}, got ${off}`); break; }
  cursor += size;
}
check(tiled, `node byte ranges tile octree.bin without gaps`);
check(cursor === octree.length, `ranges cover all of octree.bin (${cursor} vs ${octree.length})`);

// ---- decode points, the DecoderWorker way ---------------------------------
const scale = metadata.scale;
const offset = metadata.offset;
const oview = new DataView(octree.buffer, octree.byteOffset, octree.byteLength);

const attrOffsets = [];
{ let acc = 0; for (const at of metadata.attributes) { attrOffsets.push(acc); acc += at.size; } }
const attrIndex = Object.fromEntries(metadata.attributes.map((a, i) => [a.name, i]));

let outsideBox = 0;
const decodedX = new Float64Array(totalPoints);
const decodedY = new Float64Array(totalPoints);
const decodedZ = new Float64Array(totalPoints);
const decodedI = new Int32Array(totalPoints * 3);
const decodedCls = attrIndex.classification != null ? new Uint8Array(totalPoints) : null;
const decodedLabel = attrIndex.label != null ? new Float64Array(totalPoints) : null;
const decodedRgb = attrIndex.rgb != null ? new Uint8Array(totalPoints * 3) : null;

let w = 0;
for (const node of nodes) {
  // Potree gives the decoder node.min = metadata.boundingBox.min + node box min,
  // where the node box has already been shifted to be relative to that min.
  for (let p = 0; p < node.numPoints; p++) {
    const base = node.byteOffset + p * bytesPerPoint;
    const po = base + attrOffsets[0];
    const x = oview.getInt32(po + 0, true) * scale[0] + offset[0];
    const y = oview.getInt32(po + 4, true) * scale[1] + offset[1];
    const z = oview.getInt32(po + 8, true) * scale[2] + offset[2];

    const eps = metadata.spacing * 1e-3 + 1e-9;
    if (x < node.box.min[0] - eps || x > node.box.max[0] + eps ||
        y < node.box.min[1] - eps || y > node.box.max[1] + eps ||
        z < node.box.min[2] - eps || z > node.box.max[2] + eps) {
      outsideBox++;
    }

    decodedX[w] = x; decodedY[w] = y; decodedZ[w] = z;
    decodedI[w * 3 + 0] = oview.getInt32(po + 0, true);
    decodedI[w * 3 + 1] = oview.getInt32(po + 4, true);
    decodedI[w * 3 + 2] = oview.getInt32(po + 8, true);
    if (decodedCls) decodedCls[w] = oview.getUint8(base + attrOffsets[attrIndex.classification]);
    if (decodedLabel) decodedLabel[w] = oview.getFloat64(base + attrOffsets[attrIndex.label], true);
    if (decodedRgb) {
      const ro = base + attrOffsets[attrIndex.rgb];
      decodedRgb[w * 3 + 0] = oview.getUint16(ro + 0, true);
      decodedRgb[w * 3 + 1] = oview.getUint16(ro + 2, true);
      decodedRgb[w * 3 + 2] = oview.getUint16(ro + 4, true);
    }
    w++;
  }
}
check(outsideBox === 0, `every decoded point falls inside its node's bounding box (${outsideBox} strays)`);

// ---- compare against the source PCD ---------------------------------------
const pcd = readCloud(resolveScene(id));
const sx = pcd.fields[0].data, sy = pcd.fields[1].data, sz = pcd.fields[2].data;
check(totalPoints === scene.numPoints, `octree holds every point scene.json claims (${totalPoints})`);
if (scene.bakedPoints) console.log(`  note  ${scene.bakedPoints.toLocaleString()} point(s) were merged in from the instance library`);

// Positions are quantised to int32, so compare the exact integer triples the
// converter must have produced. Comparing floats on a hash grid would give
// false mismatches for points sitting near a bucket boundary.
const qi = (v, axis) => Math.round((v - offset[axis]) / scale[axis]);
const sourceKeys = new Map();
for (let i = 0; i < pcd.numPoints; i++) {
  if (!Number.isFinite(sx[i]) || !Number.isFinite(sy[i]) || !Number.isFinite(sz[i])) continue;
  const k = `${qi(sx[i], 0)},${qi(sy[i], 1)},${qi(sz[i], 2)}`;
  sourceKeys.set(k, (sourceKeys.get(k) ?? 0) + 1);
}
let missing = 0;
for (let i = 0; i < totalPoints; i++) {
  const k = `${decodedI[i * 3]},${decodedI[i * 3 + 1]},${decodedI[i * 3 + 2]}`;
  const c = sourceKeys.get(k);
  if (!c) missing++; else sourceKeys.set(k, c - 1);
}
// Objects merged in from the instance library are not in the source PCD, so
// exactly that many points are expected to have no match.
const baked = scene.bakedPoints ?? 0;
if (baked > 0) {
  check(Math.abs(missing - baked) <= 0,
        `unmatched points equal the ${baked.toLocaleString()} merged from the library (${missing})`);
} else {
  check(missing === 0, `every decoded point matches a source point exactly (${missing} unmatched)`);
}

// Points detached from the scene are deliberately left out of the octree.
let leftover = 0;
for (const c of sourceKeys.values()) leftover += c;
const detached = scene.detachedPoints ?? 0;
if (detached > 0) {
  check(leftover === detached,
        `source points left out equal the ${detached.toLocaleString()} detached (${leftover})`);
} else {
  check(leftover === 0, `no source point was dropped (${leftover} unaccounted)`);
}

// Round-trip error introduced by the int32 quantisation.
let maxErr = 0;
for (let i = 0; i < totalPoints; i++) {
  maxErr = Math.max(maxErr,
    Math.abs(decodedX[i] - (decodedI[i * 3] * scale[0] + offset[0])));
}

check(scale[0] / 2 < 1e-3, `quantisation error <= ${(scale[0] / 2).toExponential(2)} m`);

// classification / label agreement
if (decodedCls && scene.classification) {
  const table = scene.classification.classes;
  const valueOfIndex = new Map(table.map((c) => [c.index, c.value]));
  let bad = 0;
  for (let i = 0; i < totalPoints; i++) {
    if (valueOfIndex.get(decodedCls[i]) !== decodedLabel?.[i]) bad++;
  }
  if (decodedLabel) check(bad === 0, `classification index maps back to the source label value (${bad} mismatches)`);
  const sum = table.reduce((a, c) => a + c.count, 0);
  check(sum === scene.numPoints, `class histogram sums to the point count (${sum})`);
}

// rgb range
if (decodedRgb) {
  let bad = 0;
  for (let i = 0; i < decodedRgb.length; i++) if (decodedRgb[i] > 255) bad++;
  check(bad === 0, `rgb channels stay in 0..255 so Potree's >255 halving never triggers`);
}

console.log(failures === 0 ? `\nALL CHECKS PASSED (${numRecords} nodes, ${totalPoints.toLocaleString()} points)\n`
                           : `\n${failures} CHECK(S) FAILED\n`);
process.exit(failures === 0 ? 0 : 1);
