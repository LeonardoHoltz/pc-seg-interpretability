/**
 * Reads a Potree 2.0 octree back into flat arrays.
 *
 * The inverse of src/octree.mjs, used when merging placed instances into a
 * scene: an instance's points are recovered from its own octree rather than by
 * re-reading (and re-grouping) the PCD it originally came from, so merging works
 * even across scenes.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

const RECORD = 22;

const READERS = {
  int8: (v, o) => v.getInt8(o),
  uint8: (v, o) => v.getUint8(o),
  int16: (v, o) => v.getInt16(o, true),
  uint16: (v, o) => v.getUint16(o, true),
  int32: (v, o) => v.getInt32(o, true),
  uint32: (v, o) => v.getUint32(o, true),
  float: (v, o) => v.getFloat32(o, true),
  double: (v, o) => v.getFloat64(o, true),
};

/**
 * @returns {{
 *   metadata: object, count: number,
 *   x: Float64Array, y: Float64Array, z: Float64Array,
 *   columns: Map<string, {type: string, numElements: number, data: Float64Array}>
 * }}
 */
export function readOctree(dir) {
  const metadata = JSON.parse(readFileSync(join(dir, "metadata.json"), "utf8"));
  const hierarchy = readFileSync(join(dir, "hierarchy.bin"));
  const octree = readFileSync(join(dir, "octree.bin"));

  const bytesPerPoint = metadata.attributes.reduce((a, at) => a + at.size, 0);
  const offsets = [];
  { let acc = 0; for (const at of metadata.attributes) { offsets.push(acc); acc += at.size; } }

  // Walk the hierarchy the way Potree does, collecting each node's byte range.
  const hview = new DataView(hierarchy.buffer, hierarchy.byteOffset, hierarchy.byteLength);
  const numRecords = hierarchy.length / RECORD;
  const nodes = [{}];
  const ranges = [];
  for (let i = 0; i < numRecords; i++) {
    const o = i * RECORD;
    const childMask = hview.getUint8(o + 1);
    const numPoints = hview.getUint32(o + 2, true);
    const byteOffset = Number(hview.getBigInt64(o + 6, true));
    ranges.push({ byteOffset, numPoints });
    for (let c = 0; c < 8; c++) if ((1 << c) & childMask) nodes.push({});
  }

  const count = ranges.reduce((a, r) => a + r.numPoints, 0);
  const x = new Float64Array(count);
  const y = new Float64Array(count);
  const z = new Float64Array(count);

  const columns = new Map();
  for (const at of metadata.attributes) {
    if (at.name === "position") continue;
    columns.set(at.name, {
      type: at.type,
      numElements: at.numElements,
      data: new Float64Array(count * at.numElements),
    });
  }

  const view = new DataView(octree.buffer, octree.byteOffset, octree.byteLength);
  const [sx, sy, sz] = metadata.scale;
  const [ox, oy, oz] = metadata.offset;

  let w = 0;
  for (const range of ranges) {
    for (let p = 0; p < range.numPoints; p++) {
      const base = range.byteOffset + p * bytesPerPoint;
      for (let a = 0; a < metadata.attributes.length; a++) {
        const at = metadata.attributes[a];
        const at0 = base + offsets[a];
        if (at.name === "position") {
          x[w] = view.getInt32(at0 + 0, true) * sx + ox;
          y[w] = view.getInt32(at0 + 4, true) * sy + oy;
          z[w] = view.getInt32(at0 + 8, true) * sz + oz;
          continue;
        }
        const col = columns.get(at.name);
        const read = READERS[at.type];
        for (let e = 0; e < at.numElements; e++) {
          col.data[w * at.numElements + e] = read(view, at0 + e * at.elementSize);
        }
      }
      w++;
    }
  }

  return { metadata, count, x, y, z, columns };
}
