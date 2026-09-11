/**
 * Turns the raw fields of a PCD file into the attribute set Potree renders.
 *
 * Three things matter here:
 *  - finding x/y/z and colour,
 *  - telling apart continuous scalars (intensity, confidence, curvature, ...)
 *    from categorical ones (semantic labels, instance ids), because they want
 *    completely different colour treatment,
 *  - keeping attribute names clear of the ones Potree reserves for its own
 *    shader paths.
 */

/**
 * Names Potree either binds to a dedicated vertex-attribute slot or turns into
 * a `color_type_*` shader define. A generic attribute must not collide with
 * these, or it silently renders through the wrong code path.
 */
const RESERVED = new Set([
  "position", "rgba", "rgb", "color", "indices", "normal", "spacing",
  "intensity", "intensity_gradient", "classification", "elevation", "height",
  "rgb_height", "depth", "gps_time", "gpstime", "level_of_detail",
  "return_number", "returnnumber", "returns", "number_of_returns",
  "numberofreturns", "source_id", "point_source_id", "pointsourceid",
  "phong", "composite", "matcap", "aextra",
  "normalx", "normaly", "normalz",
]);

/** Potree derives a shader define from the attribute name this way. */
const sanitize = (name) => name.replace(/[^a-zA-Z0-9]/g, "_");

const POSITION_NAMES = { x: 0, y: 1, z: 2 };

/** Field names that are almost certainly a segmentation / class id. */
const SEGMENTATION_PATTERN =
  /^(label|labels|classification|class|classes|semantic\d*|semantics|sem_seg|seg|segment\d*|segmentation|category|categories|panoptic|instance\d*|instances|object|object_id|obj_id|cluster|cluster_id|part|part_id)$/i;

/** Preference order when picking the one field that drives Potree's class LUT. */
const PRIMARY_PREFERENCE = [
  "classification", "label", "labels", "semantic", "semantics",
  "segmentation", "seg", "class", "category", "panoptic",
  // Datasets that ship several label sets: prefer the coarse one, which is the
  // more useful default to open on.
  "segment20", "segment200",
];

const isIntegerType = (f) => f.type === "I" || f.type === "U";

/** How many distinct values an *unnamed* integer field may have and still count
 *  as categorical. Fields named like labels bypass this entirely. */
const UNNAMED_CATEGORICAL_LIMIT = 32;

/**
 * True when r/g/b hold the same values as the packed colour field, i.e. they are
 * the unpacked duplicate that some exporters write alongside `rgb`. Checked on a
 * sample rather than assumed, so genuinely different channels are kept.
 */
function duplicatesPackedColor(color, r, g, b, numPoints) {
  const step = Math.max(1, Math.floor(numPoints / 512));
  let checked = 0;
  for (let i = 0; i < numPoints; i += step) {
    const scale = r.type === "F" ? 255 : 1;
    if (Math.abs(r.data[i] * scale - color.r[i]) > 1 ||
        Math.abs(g.data[i] * scale - color.g[i]) > 1 ||
        Math.abs(b.data[i] * scale - color.b[i]) > 1) return false;
    checked++;
  }
  return checked > 0;
}

/** Collects distinct values, bailing out once a field is clearly continuous. */
function distinctValues(data, limit = 4096) {
  const seen = new Set();
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    seen.add(v);
    if (seen.size > limit) return null;
  }
  return [...seen].sort((a, b) => a - b);
}

function minMax(data, valid) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (valid && !valid[i]) continue;
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo > hi) { lo = 0; hi = 0; }
  return [lo, hi];
}

/** Unpacks PCL's float32/uint32 packed RGB into three 0-255 channels. */
function unpackRgb(data, numPoints) {
  const r = new Uint8Array(numPoints);
  const g = new Uint8Array(numPoints);
  const b = new Uint8Array(numPoints);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  const packedIsFloat = data instanceof Float32Array;

  for (let i = 0; i < numPoints; i++) {
    let v;
    if (packedIsFloat) { f32[0] = data[i]; v = u32[0]; }
    else v = data[i] >>> 0;
    r[i] = (v >> 16) & 255;
    g[i] = (v >> 8) & 255;
    b[i] = v & 255;
  }
  return { r, g, b };
}

/**
 * @param pcd     result of readPcd()
 * @param options { classesSidecar, primaryField }
 */
export function describeFields(pcd, options = {}) {
  const { numPoints, fields } = pcd;
  const byName = new Map(fields.map((f) => [f.name.toLowerCase(), f]));

  // ---- positions -------------------------------------------------------
  const pos = ["x", "y", "z"].map((n) => byName.get(n));
  if (pos.some((f) => !f)) {
    throw new Error(`PCD is missing x/y/z fields (has: ${fields.map((f) => f.name).join(", ")})`);
  }

  // Points with a non-finite coordinate are placeholders in organised clouds.
  const valid = new Uint8Array(numPoints);
  let numValid = 0;
  for (let i = 0; i < numPoints; i++) {
    if (Number.isFinite(pos[0].data[i]) && Number.isFinite(pos[1].data[i]) && Number.isFinite(pos[2].data[i])) {
      valid[i] = 1;
      numValid++;
    }
  }

  // ---- colour ----------------------------------------------------------
  let color = null;
  const packed = byName.get("rgb") || byName.get("rgba");
  if (packed) {
    color = { ...unpackRgb(packed.data, numPoints), source: packed.name };
  } else if (byName.get("r") && byName.get("g") && byName.get("b")) {
    const to8 = (f) => {
      const out = new Uint8Array(numPoints);
      // Float colour channels are conventionally 0..1, integers already 0..255.
      const scale = f.type === "F" ? 255 : 1;
      for (let i = 0; i < numPoints; i++) out[i] = Math.max(0, Math.min(255, Math.round(f.data[i] * scale)));
      return out;
    };
    color = {
      r: to8(byName.get("r")), g: to8(byName.get("g")), b: to8(byName.get("b")),
      source: "r/g/b",
    };
  }

  // Fields already consumed by position or colour.
  const consumed = new Set(["x", "y", "z"]);
  if (packed) {
    consumed.add(packed.name.toLowerCase());
    // Some exporters write both `rgb` and its unpacked r/g/b. Keeping the
    // duplicates would add three attributes and ~24 bytes per point for nothing.
    const [r, g, b] = ["r", "g", "b"].map((n) => byName.get(n));
    if (r && g && b && duplicatesPackedColor(color, r, g, b, numPoints)) {
      ["r", "g", "b"].forEach((n) => consumed.add(n));
    }
  } else if (color) {
    ["r", "g", "b"].forEach((n) => consumed.add(n));
  }

  // ---- scalars ---------------------------------------------------------
  const sidecar = options.classesSidecar?.fields ?? {};
  const scalars = [];

  for (const f of fields) {
    const lower = f.name.toLowerCase();
    if (consumed.has(lower)) continue;
    if (f.count !== 1) continue;   // multi-element scalars have no sensible colour mapping

    const [lo, hi] = minMax(f.data, valid);
    const nameMatches = SEGMENTATION_PATTERN.test(f.name);
    let classes = null;

    if (isIntegerType(f)) {
      const distinct = distinctValues(f.data);
      // A named label field is categorical whatever its cardinality. An unnamed
      // integer field needs to be genuinely low-cardinality: things like voxel
      // indices have hundreds of values and are really continuous, and a
      // 200-entry legend is worse than a gradient.
      if (distinct && (nameMatches || distinct.length <= UNNAMED_CATEGORICAL_LIMIT)) classes = distinct;
    }

    const sanitized = sanitize(f.name);
    const attributeName = RESERVED.has(sanitized.toLowerCase()) && lower !== "intensity"
      ? `${sanitized}_field`
      : sanitized;

    scalars.push({
      name: attributeName,
      source: f.name,
      kind: classes ? "categorical" : "continuous",
      pcdType: `${f.type}${f.size}`,
      min: lo,
      max: hi,
      classes,
      numClasses: classes ? classes.length : null,
      labels: sidecar[f.name]?.classes ?? sidecar[attributeName]?.classes ?? null,
      displayName: sidecar[f.name]?.name ?? null,
      data: f.data,
    });
  }

  // ---- which categorical field drives Potree's classification LUT ------
  const categorical = scalars.filter((s) => s.kind === "categorical");
  let primary = null;
  if (options.primaryField) {
    primary = categorical.find((s) => s.source === options.primaryField || s.name === options.primaryField) ?? null;
  }
  if (!primary) {
    for (const want of PRIMARY_PREFERENCE) {
      primary = categorical.find((s) => s.source.toLowerCase() === want);
      if (primary) break;
    }
  }
  if (!primary) primary = categorical[0] ?? null;

  return { numPoints, numValid, valid, position: pos, color, scalars, categorical, primary };
}

export { sanitize, RESERVED };
