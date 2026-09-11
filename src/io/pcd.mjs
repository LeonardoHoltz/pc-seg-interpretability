/**
 * PCD (Point Cloud Data) reader.
 *
 * Supports the three PCL encodings: `ascii`, `binary` and `binary_compressed`.
 * Only the header is ASCII; everything after the DATA line depends on the
 * encoding. Note that `binary_compressed` stores the data column-major
 * (all x, then all y, ...) while the other two are row-major.
 */
import { readFileSync, openSync, readSync, closeSync } from "node:fs";
import { lzfDecompress } from "./lzf.mjs";

const HEADER_KEYS = new Set([
  "VERSION", "FIELDS", "SIZE", "TYPE", "COUNT",
  "WIDTH", "HEIGHT", "VIEWPOINT", "POINTS", "DATA",
]);

/** TypedArray constructor for a PCD (type, size) pair. */
function typedArrayFor(type, size) {
  const key = `${type}${size}`;
  const map = {
    I1: Int8Array, I2: Int16Array, I4: Int32Array, I8: BigInt64Array,
    U1: Uint8Array, U2: Uint16Array, U4: Uint32Array, U8: BigUint64Array,
    F4: Float32Array, F8: Float64Array,
  };
  const ctor = map[key];
  if (!ctor) throw new Error(`Unsupported PCD field type ${type}${size}`);
  return ctor;
}

/** Reads a DataView value for a (type, size) pair, little-endian. */
function readerFor(type, size) {
  const key = `${type}${size}`;
  switch (key) {
    case "I1": return (v, o) => v.getInt8(o);
    case "I2": return (v, o) => v.getInt16(o, true);
    case "I4": return (v, o) => v.getInt32(o, true);
    case "I8": return (v, o) => Number(v.getBigInt64(o, true));
    case "U1": return (v, o) => v.getUint8(o);
    case "U2": return (v, o) => v.getUint16(o, true);
    case "U4": return (v, o) => v.getUint32(o, true);
    case "U8": return (v, o) => Number(v.getBigUint64(o, true));
    case "F4": return (v, o) => v.getFloat32(o, true);
    case "F8": return (v, o) => v.getFloat64(o, true);
    default: throw new Error(`Unsupported PCD field type ${key}`);
  }
}

/**
 * Parses the ASCII header out of a buffer.
 * Returns the parsed header plus the byte offset where the data section starts.
 */
export function parseHeader(buf) {
  const header = {
    version: "0.7", fields: [], size: [], type: [], count: [],
    width: 0, height: 1, viewpoint: [0, 0, 0, 1, 0, 0, 0], points: null, data: "ascii",
  };

  let offset = 0;
  let sawData = false;

  while (offset < buf.length && !sawData) {
    let eol = buf.indexOf(0x0a, offset); // '\n'
    if (eol === -1) throw new Error("PCD header is missing a DATA line");
    const line = buf.toString("ascii", offset, eol).replace(/\r$/, "").trim();
    offset = eol + 1;

    if (line === "" || line.startsWith("#")) continue;

    const sp = line.indexOf(" ");
    const key = (sp === -1 ? line : line.slice(0, sp)).toUpperCase();
    const value = sp === -1 ? "" : line.slice(sp + 1).trim();
    if (!HEADER_KEYS.has(key)) continue;

    switch (key) {
      case "VERSION": header.version = value; break;
      case "FIELDS": header.fields = value.split(/\s+/); break;
      case "SIZE": header.size = value.split(/\s+/).map(Number); break;
      case "TYPE": header.type = value.split(/\s+/).map((t) => t.toUpperCase()); break;
      case "COUNT": header.count = value.split(/\s+/).map(Number); break;
      case "WIDTH": header.width = Number(value); break;
      case "HEIGHT": header.height = Number(value); break;
      case "VIEWPOINT": header.viewpoint = value.split(/\s+/).map(Number); break;
      case "POINTS": header.points = Number(value); break;
      case "DATA": header.data = value.toLowerCase(); sawData = true; break;
    }
  }

  if (header.fields.length === 0) throw new Error("PCD header has no FIELDS");
  if (header.count.length === 0) header.count = header.fields.map(() => 1);
  if (header.points == null) header.points = header.width * header.height;

  const n = header.fields.length;
  if (header.size.length !== n || header.type.length !== n || header.count.length !== n) {
    throw new Error("PCD header FIELDS/SIZE/TYPE/COUNT lengths disagree");
  }

  return { header, dataOffset: offset };
}

/**
 * Reads only the header of a PCD file, without touching the point data.
 * Used by the scene registry so that browsing a folder stays cheap.
 */
export function readPcdHeader(path) {
  const fd = openSync(path, "r");
  try {
    // Headers are small; 64 KiB is far more than any real PCD needs.
    const buf = Buffer.alloc(65536);
    const bytes = readSync(fd, buf, 0, buf.length, 0);
    return parseHeader(buf.subarray(0, bytes)).header;
  } finally {
    closeSync(fd);
  }
}

/**
 * Reads a full PCD file.
 *
 * Returns `{ header, numPoints, fields }` where each field carries its own
 * TypedArray of `numPoints * count` values, in file order.
 */
export function readPcd(path) {
  const buf = readFileSync(path);
  const { header, dataOffset } = parseHeader(buf);
  const numPoints = header.points;
  const nf = header.fields.length;

  const arrays = header.fields.map((_, i) =>
    new (typedArrayFor(header.type[i], header.size[i]))(numPoints * header.count[i]));

  if (header.data === "ascii") {
    readAscii(buf, dataOffset, header, arrays, numPoints);
  } else if (header.data === "binary") {
    readBinary(buf, dataOffset, header, arrays, numPoints);
  } else if (header.data === "binary_compressed") {
    readBinaryCompressed(buf, dataOffset, header, arrays, numPoints);
  } else {
    throw new Error(`Unsupported PCD DATA encoding: ${header.data}`);
  }

  const fields = header.fields.map((name, i) => ({
    name,
    type: header.type[i],
    size: header.size[i],
    count: header.count[i],
    data: arrays[i],
  }));

  return { header, numPoints, fields, path };
}

function readAscii(buf, offset, header, arrays, numPoints) {
  const text = buf.toString("utf8", offset);
  const nf = header.fields.length;
  const stride = header.count.reduce((a, b) => a + b, 0);
  // Column index -> [fieldIndex, elementIndex]
  const slot = [];
  for (let f = 0; f < nf; f++) {
    for (let c = 0; c < header.count[f]; c++) slot.push([f, c]);
  }

  let row = 0;
  for (const rawLine of text.split("\n")) {
    if (row >= numPoints) break;
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    if (parts.length < stride) continue;
    for (let k = 0; k < stride; k++) {
      const [f, c] = slot[k];
      const v = Number(parts[k]);
      arrays[f][row * header.count[f] + c] = Number.isNaN(v) ? NaN : v;
    }
    row++;
  }
  if (row < numPoints) {
    throw new Error(`PCD claims ${numPoints} points but only ${row} ASCII rows were parsed`);
  }
}

function readBinary(buf, offset, header, arrays, numPoints) {
  const nf = header.fields.length;
  const pointStride = header.size.reduce((a, s, i) => a + s * header.count[i], 0);
  const need = offset + pointStride * numPoints;
  if (buf.length < need) {
    throw new Error(`PCD binary payload truncated: need ${need} bytes, file has ${buf.length}`);
  }

  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const readers = header.fields.map((_, i) => readerFor(header.type[i], header.size[i]));
  const fieldOffsets = [];
  let acc = 0;
  for (let f = 0; f < nf; f++) {
    fieldOffsets.push(acc);
    acc += header.size[f] * header.count[f];
  }

  for (let p = 0; p < numPoints; p++) {
    const base = offset + p * pointStride;
    for (let f = 0; f < nf; f++) {
      const cnt = header.count[f];
      const sz = header.size[f];
      const read = readers[f];
      for (let c = 0; c < cnt; c++) {
        arrays[f][p * cnt + c] = read(view, base + fieldOffsets[f] + c * sz);
      }
    }
  }
}

function readBinaryCompressed(buf, offset, header, arrays, numPoints) {
  if (buf.length < offset + 8) throw new Error("PCD binary_compressed payload truncated");
  const compressedSize = buf.readUInt32LE(offset);
  const uncompressedSize = buf.readUInt32LE(offset + 4);
  const body = buf.subarray(offset + 8, offset + 8 + compressedSize);
  if (body.length < compressedSize) {
    throw new Error("PCD binary_compressed payload shorter than its declared size");
  }

  const raw = lzfDecompress(body, uncompressedSize);

  // binary_compressed is stored column-major: every field's values are contiguous.
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const nf = header.fields.length;
  let cursor = 0;
  for (let f = 0; f < nf; f++) {
    const cnt = header.count[f];
    const sz = header.size[f];
    const read = readerFor(header.type[f], sz);
    for (let c = 0; c < cnt; c++) {
      for (let p = 0; p < numPoints; p++) {
        arrays[f][p * cnt + c] = read(view, cursor + p * sz);
      }
      cursor += sz * numPoints;
    }
  }
}
