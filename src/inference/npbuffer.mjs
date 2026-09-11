/**
 * Binary framing for shipping point cloud arrays to and from an inference
 * service.
 *
 * Layout:
 *   [8]  uint64 LE  length of the JSON header
 *   [H]  UTF-8 JSON header
 *   [..] the arrays' raw bytes, back to back, 8-byte aligned
 *
 * The header names each array with a numpy dtype string and a shape, so the
 * receiving end can wrap the payload with `np.frombuffer(...).reshape(...)` and
 * get real numpy arrays without copying or parsing anything per point.
 *
 * The layout the caller asked for is also the fastest one: three contiguous
 * per-axis arrays laid end to end *are* a C-contiguous [3, N] array, so xyz and
 * rgb need no interleaving or transposing on either side. [N, 3] would have been
 * the slower choice.
 */

const DTYPES = {
  int8: "|i1", uint8: "|u1",
  int16: "<i2", uint16: "<u2",
  int32: "<i4", uint32: "<u4",
  float32: "<f4", float64: "<f8",
};

const CTORS = {
  "|i1": Int8Array, "|u1": Uint8Array,
  "<i2": Int16Array, "<u2": Uint16Array,
  "<i4": Int32Array, "<u4": Uint32Array,
  "<f4": Float32Array, "<f8": Float64Array,
  // Tolerate the receiving end using native-endian or 64-bit integer labels.
  "=i4": Int32Array, "=f4": Float32Array, "=f8": Float64Array,
  "<i8": BigInt64Array, "<u8": BigUint64Array,
};

export const dtypeOf = (name) => DTYPES[name] ?? name;

const align8 = (n) => (n + 7) & ~7;

/**
 * @param arrays [{ name, dtype, shape, data: TypedArray }]
 * @param meta   extra fields merged into the header
 */
export function encodeArrays(arrays, meta = {}) {
  const entries = [];
  let offset = 0;
  for (const a of arrays) {
    const bytes = a.data.byteLength;
    entries.push({
      name: a.name,
      dtype: dtypeOf(a.dtype),
      shape: a.shape,
      offset,
      nbytes: bytes,
    });
    offset = align8(offset + bytes);
  }

  const header = Buffer.from(JSON.stringify({
    format: "pcit-arrays/1",
    ...meta,
    arrays: entries,
  }), "utf8");

  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length));

  const body = Buffer.alloc(offset);
  arrays.forEach((a, i) => {
    const src = Buffer.from(a.data.buffer, a.data.byteOffset, a.data.byteLength);
    src.copy(body, entries[i].offset);
  });

  return Buffer.concat([prefix, header, body]);
}

/** Inverse of encodeArrays. Views alias the input buffer; nothing is copied. */
export function decodeArrays(buf) {
  if (buf.length < 8) throw new Error("payload too short to hold a header length");
  const headerLength = Number(buf.readBigUInt64LE(0));
  if (headerLength <= 0 || 8 + headerLength > buf.length) {
    throw new Error("payload header length is out of range");
  }

  let meta;
  try {
    meta = JSON.parse(buf.toString("utf8", 8, 8 + headerLength));
  } catch (err) {
    throw new Error(`payload header is not valid JSON: ${err.message}`);
  }
  if (!Array.isArray(meta.arrays)) throw new Error("payload header has no arrays");

  const base = 8 + headerLength;
  const arrays = new Map();
  for (const a of meta.arrays) {
    const Ctor = CTORS[a.dtype];
    if (!Ctor) throw new Error(`unsupported dtype "${a.dtype}" for array "${a.name}"`);
    const count = (a.shape ?? []).reduce((x, y) => x * y, 1);
    const start = base + a.offset;
    if (start + count * Ctor.BYTES_PER_ELEMENT > buf.length) {
      throw new Error(`array "${a.name}" runs past the end of the payload`);
    }
    // Copy only when the slice is not aligned for the element type.
    const aligned = (buf.byteOffset + start) % Ctor.BYTES_PER_ELEMENT === 0;
    const view = aligned
      ? new Ctor(buf.buffer, buf.byteOffset + start, count)
      : new Ctor(buf.buffer.slice(buf.byteOffset + start,
                                 buf.byteOffset + start + count * Ctor.BYTES_PER_ELEMENT));
    arrays.set(a.name, { dtype: a.dtype, shape: a.shape ?? [count], data: view });
  }
  return { meta, arrays };
}
