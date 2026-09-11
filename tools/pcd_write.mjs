/** Minimal PCD writer used by the demo-scene generator. */
import { writeFileSync } from "node:fs";

const SIZES = { I1: 1, I2: 2, I4: 4, U1: 1, U2: 2, U4: 4, F4: 4, F8: 8 };

/**
 * @param path    destination file
 * @param columns [{ name, type: 'I'|'U'|'F', size, values: number[]|TypedArray }]
 * @param encoding 'ascii' | 'binary'
 */
export function writePcd(path, columns, encoding = "binary") {
  const n = columns[0].values.length;
  const header =
    `# .PCD v0.7 - Point Cloud Data file format\n` +
    `VERSION 0.7\n` +
    `FIELDS ${columns.map((c) => c.name).join(" ")}\n` +
    `SIZE ${columns.map((c) => c.size).join(" ")}\n` +
    `TYPE ${columns.map((c) => c.type).join(" ")}\n` +
    `COUNT ${columns.map(() => 1).join(" ")}\n` +
    `WIDTH ${n}\n` +
    `HEIGHT 1\n` +
    `VIEWPOINT 0 0 0 1 0 0 0\n` +
    `POINTS ${n}\n` +
    `DATA ${encoding}\n`;

  if (encoding === "ascii") {
    const lines = new Array(n);
    for (let i = 0; i < n; i++) {
      lines[i] = columns
        .map((c) => (c.type === "F" ? Number(c.values[i]).toPrecision(8) : String(c.values[i])))
        .join(" ");
    }
    writeFileSync(path, header + lines.join("\n") + "\n");
    return;
  }

  const stride = columns.reduce((a, c) => a + SIZES[c.type + c.size], 0);
  const body = Buffer.alloc(stride * n);
  const view = new DataView(body.buffer);
  let fieldOffset = 0;
  for (const c of columns) {
    const key = c.type + c.size;
    for (let i = 0; i < n; i++) {
      const o = i * stride + fieldOffset;
      const v = c.values[i];
      switch (key) {
        case "F4": view.setFloat32(o, v, true); break;
        case "F8": view.setFloat64(o, v, true); break;
        case "U1": view.setUint8(o, v); break;
        case "U2": view.setUint16(o, v, true); break;
        case "U4": view.setUint32(o, v, true); break;
        case "I1": view.setInt8(o, v); break;
        case "I2": view.setInt16(o, v, true); break;
        case "I4": view.setInt32(o, v, true); break;
        default: throw new Error(`unsupported ${key}`);
      }
    }
    fieldOffset += SIZES[key];
  }
  writeFileSync(path, Buffer.concat([Buffer.from(header, "ascii"), body]));
}
