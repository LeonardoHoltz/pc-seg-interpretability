/**
 * Reader for NumPy `.npy` arrays.
 *
 * Enough of the format to load preprocessed point cloud datasets that ship one
 * array per field (Pointcept's ScanNet export, for instance): version 1/2/3
 * headers, little-endian numeric dtypes, C or Fortran order. Pickled object
 * arrays are refused rather than silently mis-read.
 */
import { readFileSync, openSync, readSync, closeSync } from "node:fs";

const MAGIC = "\x93NUMPY";

const DTYPES = {
  "|i1": { ctor: Int8Array, bytes: 1 }, "|u1": { ctor: Uint8Array, bytes: 1 },
  "|b1": { ctor: Uint8Array, bytes: 1 },
  "<i2": { ctor: Int16Array, bytes: 2 }, "<u2": { ctor: Uint16Array, bytes: 2 },
  "<i4": { ctor: Int32Array, bytes: 4 }, "<u4": { ctor: Uint32Array, bytes: 4 },
  "<i8": { ctor: BigInt64Array, bytes: 8 }, "<u8": { ctor: BigUint64Array, bytes: 8 },
  "<f4": { ctor: Float32Array, bytes: 4 }, "<f8": { ctor: Float64Array, bytes: 8 },
};
// numpy writes '=' or no prefix for single-byte and native-endian types.
const ALIASES = { i1: "|i1", u1: "|u1", b1: "|b1", "=i4": "<i4", "=i8": "<i8", "=f4": "<f4", "=f8": "<f8" };

const normalise = (descr) => ALIASES[descr] ?? descr;

/** Parses the header out of the first bytes of a .npy file. */
export function parseNpyHeader(buf) {
  if (buf.length < 10 || buf.toString("latin1", 0, 6) !== MAGIC) {
    throw new Error("not a .npy file (bad magic)");
  }
  const major = buf[6];
  const headerLength = major === 1 ? buf.readUInt16LE(8) : buf.readUInt32LE(8);
  const start = major === 1 ? 10 : 12;
  const text = buf.toString("latin1", start, start + headerLength);

  const descr = /'descr'\s*:\s*'([^']+)'/.exec(text)?.[1];
  if (!descr) throw new Error("`.npy` header has no dtype");
  const fortranOrder = /'fortran_order'\s*:\s*True/.test(text);
  const shapeText = /'shape'\s*:\s*\(([^)]*)\)/.exec(text)?.[1] ?? "";
  const shape = shapeText.split(",").map((s) => s.trim()).filter(Boolean).map(Number);

  return { dtype: normalise(descr), shape, fortranOrder, dataOffset: start + headerLength };
}

/** Reads only the header, so a directory of large arrays can be listed cheaply. */
export function readNpyHeader(path) {
  const fd = openSync(path, "r");
  try {
    const buf = Buffer.alloc(1024);
    const bytes = readSync(fd, buf, 0, buf.length, 0);
    return parseNpyHeader(buf.subarray(0, bytes));
  } finally {
    closeSync(fd);
  }
}

/**
 * @returns { dtype, shape, data } where `data` is a flat typed array in C order.
 *          int64/uint64 come back as Float64Array -- point cloud ids and labels
 *          are far inside the range doubles represent exactly, and the rest of
 *          the pipeline has no use for BigInt.
 */
export function readNpy(path) {
  const buf = readFileSync(path);
  const { dtype, shape, fortranOrder, dataOffset } = parseNpyHeader(buf);

  const spec = DTYPES[dtype];
  if (!spec) throw new Error(`unsupported .npy dtype "${dtype}" in ${path}`);

  const count = shape.reduce((a, b) => a * b, 1);
  const need = dataOffset + count * spec.bytes;
  if (buf.length < need) {
    throw new Error(`${path} is truncated: need ${need} bytes, file has ${buf.length}`);
  }

  // The data offset is padded to 64 bytes, so a view is normally aligned; copy
  // when it is not rather than throwing.
  const absolute = buf.byteOffset + dataOffset;
  let view;
  if (absolute % spec.bytes === 0) {
    view = new spec.ctor(buf.buffer, absolute, count);
  } else {
    view = new spec.ctor(buf.buffer.slice(absolute, absolute + count * spec.bytes));
  }

  let data;
  if (spec.ctor === BigInt64Array || spec.ctor === BigUint64Array) {
    data = new Float64Array(count);
    for (let i = 0; i < count; i++) data[i] = Number(view[i]);
  } else {
    data = view;
  }

  if (fortranOrder && shape.length === 2) {
    // Transpose into C order so every caller can assume row-major.
    const [rows, cols] = shape;
    const out = new data.constructor(count);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) out[r * cols + c] = data[c * rows + r];
    }
    data = out;
  }

  return { dtype, shape, data };
}
