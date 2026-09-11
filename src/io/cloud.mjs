/**
 * Loads a scene from whichever on-disk form it takes.
 *
 * Two are supported and both come back in the same shape, so everything
 * downstream -- field classification, the octree writer, the library -- is
 * unaware of the difference:
 *
 *   a `.pcd` file          one file, fields described by its header
 *   a directory of `.npy`  one array per field, as Pointcept and similar
 *                          pipelines export ScanNet and friends
 *
 * A directory scene takes whatever arrays it happens to contain: ScanNet's test
 * split ships coord/color/normal with no labels, and that loads fine with the
 * label-derived features simply absent.
 */
import { existsSync, readdirSync, statSync } from "node:fs";
import { join, basename, extname } from "node:path";
import { readPcd, readPcdHeader } from "./pcd.mjs";
import { readNpy, readNpyHeader } from "./npy.mjs";

/** The array that must be present for a directory to count as a cloud. */
export const NPY_REQUIRED = "coord.npy";

/** Names to split a multi-column array into, per column count. */
const COLUMN_SUFFIXES = { 3: ["x", "y", "z"], 2: ["u", "v"], 4: ["x", "y", "z", "w"] };

/**
 * How known arrays map onto field names. Anything not listed keeps its own
 * name, so a dataset with extra arrays still loads.
 */
const KNOWN = {
  coord: ["x", "y", "z"],
  color: ["r", "g", "b"],
  normal: ["normal_x", "normal_y", "normal_z"],
};

export const isNpyScene = (path) =>
  existsSync(path) && statSync(path).isDirectory() && existsSync(join(path, NPY_REQUIRED));

/** PCD type letter for a numpy dtype. */
function pcdTypeOf(dtype) {
  if (dtype.includes("f")) return { type: "F", size: dtype.endsWith("8") ? 8 : 4 };
  if (dtype.includes("u") || dtype.includes("b")) {
    return { type: "U", size: Math.min(4, Number(dtype.slice(-1)) || 1) };
  }
  return { type: "I", size: 4 };
}

function npyFieldNames(stem, columns) {
  if (KNOWN[stem]) return KNOWN[stem].slice(0, columns);
  if (columns === 1) return [stem];
  const suffix = COLUMN_SUFFIXES[columns] ?? Array.from({ length: columns }, (_, i) => String(i));
  return suffix.map((s) => `${stem}_${s}`);
}

function listNpy(dir) {
  return readdirSync(dir)
    .filter((f) => extname(f).toLowerCase() === ".npy")
    .sort((a, b) => {
      // coord first, then colour, then the rest alphabetically.
      const rank = (n) => (n === "coord.npy" ? 0 : n === "color.npy" ? 1 : 2);
      return rank(a) - rank(b) || a.localeCompare(b);
    });
}

/** Reads a directory of .npy arrays into the same shape readPcd() returns. */
export function readNpyScene(dir) {
  const files = listNpy(dir);
  if (!files.includes(NPY_REQUIRED)) throw new Error(`${dir} has no ${NPY_REQUIRED}`);

  const coord = readNpy(join(dir, NPY_REQUIRED));
  const numPoints = coord.shape[0];
  const fields = [];
  const arrays = [];

  for (const file of files) {
    const stem = basename(file, ".npy");
    const array = stem === "coord" ? coord : readNpy(join(dir, file));
    const rows = array.shape[0];
    if (rows !== numPoints) {
      throw new Error(
        `${file} has ${rows} rows but ${NPY_REQUIRED} has ${numPoints}; they must describe the same points`);
    }

    const columns = array.shape.length > 1 ? array.shape[1] : 1;
    const names = npyFieldNames(stem, columns);
    const { type, size } = pcdTypeOf(array.dtype);

    for (let c = 0; c < columns; c++) {
      // De-interleave: the pipeline wants one contiguous array per field.
      const out = columns === 1
        ? array.data
        : (() => {
            const dst = new array.data.constructor(numPoints);
            for (let i = 0; i < numPoints; i++) dst[i] = array.data[i * columns + c];
            return dst;
          })();
      fields.push({ name: names[c], type, size, count: 1, data: out, source: file });
      arrays.push(file);
    }
  }

  return {
    header: {
      version: "npy",
      fields: fields.map((f) => f.name),
      size: fields.map((f) => f.size),
      type: fields.map((f) => f.type),
      count: fields.map(() => 1),
      width: numPoints, height: 1, points: numPoints,
      viewpoint: [0, 0, 0, 1, 0, 0, 0],
      data: "npy",
      arrays: files,
    },
    numPoints,
    fields,
    path: dir,
  };
}

/** Header-only read, for listing a folder without touching the point data. */
export function readNpySceneHeader(dir) {
  const files = listNpy(dir);
  const coord = readNpyHeader(join(dir, NPY_REQUIRED));
  const names = [];
  const size = [];
  const type = [];

  for (const file of files) {
    const stem = basename(file, ".npy");
    const h = readNpyHeader(join(dir, file));
    const columns = h.shape.length > 1 ? h.shape[1] : 1;
    const { type: t, size: s } = pcdTypeOf(h.dtype);
    for (const n of npyFieldNames(stem, columns)) { names.push(n); size.push(s); type.push(t); }
  }

  return {
    version: "npy", fields: names, size, type, count: names.map(() => 1),
    width: coord.shape[0], height: 1, points: coord.shape[0],
    viewpoint: [0, 0, 0, 1, 0, 0, 0], data: "npy", arrays: files,
  };
}

export const readCloud = (path) => (isNpyScene(path) ? readNpyScene(path) : readPcd(path));
export const readCloudHeader = (path) => (isNpyScene(path) ? readNpySceneHeader(path) : readPcdHeader(path));
