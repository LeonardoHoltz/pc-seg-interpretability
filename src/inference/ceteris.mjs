/**
 * Ceteris paribus analysis for a segmentation model.
 *
 * Holds a scene fixed, moves one object along a direction, and asks the model how
 * its prediction for that object changes. Plotting the object's mean probability
 * for a chosen class against X, Y or Z shows how much the prediction depends on
 * where the object sits rather than on what it looks like.
 *
 * The whole sweep is **one request**. The scene goes out once, together with a
 * uint8 mask marking the object's points, a direction and the offsets to try;
 * the service moves the masked points itself and replies with the object's
 * logits at every position. Sending the scene once instead of once per step cuts
 * the traffic by the number of steps, and the reply covers only the object
 * rather than the whole cloud.
 *
 * Nothing is written to the scene: this is analysis, so the octree is read once
 * and never rebuilt.
 */
import { encodeArrays, decodeArrays } from "./npbuffer.mjs";
import { readSceneForInference, sceneArrays } from "./payload.mjs";
import { config } from "../config.mjs";

/** Reads the scene once and lays out the arrays the service expects. */
function prepare(sceneId, fields) {
  const { scene, octree, n } = readSceneForInference(sceneId);
  const { arrays } = sceneArrays(scene, octree, { fields });
  return { scene, octree, n, arrays };
}

/** Shared by the HTTP route and direct callers; validate before allocating a range. */
export function parseCeterisSweep(opts = {}) {
  const axis = opts.axis ?? "z";
  if (!["x", "y", "z"].includes(axis)) throw new Error("axis must be x, y or z");
  if (opts.heights !== undefined && (axis !== "z" || opts.positions !== undefined)) {
    throw new Error("heights is only supported for Z; use positions for an axis sweep");
  }
  let positions = opts.positions ?? opts.heights;
  if (positions === undefined) {
    const { min, max, steps } = opts.range ?? {};
    if (![min, max].every(Number.isFinite) || !Number.isInteger(steps) || steps < 2 || steps > 256) {
      throw new Error("give positions[] or range {min, max, steps: integer from 2 to 256}");
    }
    positions = Array.from({ length: steps }, (_, i) => {
      const t = i / (steps - 1);
      return min * (1 - t) + max * t;
    });
  }
  if (!Array.isArray(positions) || positions.length < 1 || positions.length > 256
      || !positions.every(Number.isFinite)) {
    throw new Error("positions must contain 1 to 256 finite numbers");
  }
  return { axis, positions };
}

/** Point indices belonging to one object, via the scene's instance field. */
function objectIndices(scene, octree, instanceId) {
  // The scene says which field enumerates objects; see scene/roles.mjs.
  const field = scene.roles?.instance?.source ?? scene.library?.field
    ?? (scene.attributes ?? []).find((a) => a.kind === "categorical" && /instance|object/i.test(a.name))?.name;
  const col = octree.columns.get(field) ?? octree.columns.get("instance");
  if (!col) throw new Error("this scene has no instance field, so objects cannot be isolated");

  const idx = [];
  for (let i = 0; i < octree.count; i++) if (col.data[i] === instanceId) idx.push(i);
  if (idx.length === 0) throw new Error(`no points carry instance id ${instanceId}`);
  return Uint32Array.from(idx);
}

/** softmax over one column of logits, returning the probability of one row. */
function softmaxAt(logits, base, stride, numClasses, row) {
  let max = -Infinity;
  for (let c = 0; c < numClasses; c++) {
    const v = logits[base + c * stride];
    if (v > max) max = v;
  }
  let sum = 0, wanted = 0;
  for (let c = 0; c < numClasses; c++) {
    const e = Math.exp(logits[base + c * stride] - max);
    sum += e;
    if (c === row) wanted = e;
  }
  return sum > 0 ? wanted / sum : 0;
}

/**
 * Turns the service's reply into one mean probability per sweep position.
 *
 * Expects `logits` (or already-normalised `probs`) shaped [S, C, M]: sweep
 * position, class, then the masked points in ascending index order -- the order
 * numpy produces from `xyz[:, mask.astype(bool)]`.
 */
function readSweep(decoded, { numSteps, numObjectPoints, classValue }) {
  const { meta, arrays } = decoded;
  const entry = arrays.get("logits") ?? arrays.get("probs");
  if (!entry) throw new Error("the response contained neither `logits` nor `probs`");
  const alreadyProbabilities = !arrays.get("logits");

  const shape = entry.shape ?? [];
  if (shape.length !== 3) {
    throw new Error(`expected ${alreadyProbabilities ? "probs" : "logits"} shaped [S, C, M], got [${shape}]`);
  }
  const [S, C, M] = shape;
  if (S !== numSteps) throw new Error(`the reply covers ${S} positions, ${numSteps} were requested`);
  if (M !== numObjectPoints) {
    throw new Error(`the reply covers ${M} object points, the mask marked ${numObjectPoints}`);
  }

  const classes = meta.classes ?? null;
  const row = classes ? classes.indexOf(classValue) : classValue;
  if (row < 0 || row >= C) {
    throw new Error(`class ${classValue} is not among the ${C} classes returned`);
  }

  const data = entry.data;
  const values = [];
  for (let s = 0; s < S; s++) {
    const stepBase = s * C * M;
    let sum = 0;
    for (let m = 0; m < M; m++) {
      sum += alreadyProbabilities
        ? data[stepBase + row * M + m]
        : softmaxAt(data, stepBase + m, M, C, row);
    }
    values.push(sum / M);
  }
  return { measure: alreadyProbabilities ? "probability" : "probability", values };
}

/**
 * @param opts.instanceId  which object to move
 * @param opts.classValue  the class whose probability is plotted
 * @param opts.axis        x, y or z (default z); only this coordinate changes
 * @param opts.positions   absolute anchor coordinates in metres along that axis
 * @param opts.heights     legacy alias for positions on Z
 * @param opts.range       alternative to positions: {min, max, steps}
 */
export async function runCeterisParibus(sceneId, opts = {}) {
  const {
    endpoint, instanceId, classValue,
    fields = null, timeoutMs = config.inference.timeouts.ceteris, onProgress = () => {},
  } = opts;

  if (!endpoint) throw new Error("no inference endpoint configured");
  if (!Number.isFinite(instanceId)) throw new Error("instanceId is required");
  if (!Number.isFinite(classValue)) throw new Error("classValue is required");
  const { axis, positions } = parseCeterisSweep(opts);
  const axisIndex = ["x", "y", "z"].indexOf(axis);
  // Retain the direct caller's direction hook; the UI uses the three unit axes.
  const direction = opts.direction ?? [0, 1, 2].map((a) => Number(a === axisIndex));
  if (!Array.isArray(direction) || direction.length !== 3
      || !direction.every(Number.isFinite) || Math.hypot(...direction) === 0) {
    throw new Error("direction must be a nonzero finite XYZ vector");
  }

  onProgress({ phase: "reading", progress: 0.05, message: "Reading the scene" });
  const { scene, octree, n, arrays } = prepare(sceneId, fields);
  const indices = objectIndices(scene, octree, instanceId);

  // Same anchor as the object library: footprint centre X/Y, lowest point Z.
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const coords = [octree.x, octree.y, octree.z];
  for (const i of indices) for (let a = 0; a < 3; a++) {
    min[a] = Math.min(min[a], coords[a][i]);
    max[a] = Math.max(max[a], coords[a][i]);
  }
  const objectAnchor = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, min[2]];
  const objectBase = objectAnchor[2];
  const offsets = positions.map((p) => p - objectAnchor[axisIndex]);
  const translations = offsets.map((offset) => direction.map((d) => d * offset));
  const anchors = translations.map((delta) => objectAnchor.map((p, a) => p + delta[a]));
  if (anchors.some((p) => !p.every(Number.isFinite))) throw new Error("sweep coordinates overflow");
  const heights = anchors.map((p) => p[2]);

  // A uint8 mask is the shape numpy wants: `sel = mask.astype(bool)` then
  // `xyz[:, sel]`. It also fixes the order the reply's object columns follow.
  const mask = new Uint8Array(n);
  for (const i of indices) mask[i] = 1;

  const body = encodeArrays([...arrays, { name: "mask", dtype: "uint8", shape: [n], data: mask }], {
    request: "ceteris_paribus",
    scene: sceneId,
    num_points: n,
    ceteris_paribus: {
      instance: instanceId,
      object_points: indices.length,
      direction,
      offsets,
      axis,
      positions,
      anchor: objectAnchor,
      heights,
      base: objectBase,
      // What the reply should contain, so the service need not guess.
      expects: { logits: [positions.length, "C", indices.length] },
    },
  });

  onProgress({
    phase: "sending", progress: 0.2,
    message: `Sending the scene once with ${positions.length} ${axis.toUpperCase()} positions ` +
             `(${(body.length / 1048576).toFixed(1)} MiB)`,
  });

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/octet-stream",
        "x-pcit-scene": sceneId,
        "x-pcit-points": String(n),
        "x-pcit-request": "ceteris-paribus",
      },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    throw new Error(`could not reach the inference service: ${err.message}`);
  }

  const raw = Buffer.from(await response.arrayBuffer());
  if (!response.ok) {
    let detail = raw.toString("utf8", 0, Math.min(raw.length, 300));
    try { detail = JSON.parse(detail).error ?? detail; } catch { /* keep raw */ }
    throw new Error(`inference service returned ${response.status}: ${detail}`);
  }

  onProgress({ phase: "decoding", progress: 0.85, message: "Reading the logits" });
  const decoded = decodeArrays(raw);
  const { measure, values } = readSweep(decoded, {
    numSteps: positions.length,
    numObjectPoints: indices.length,
    classValue,
  });

  onProgress({ phase: "done", progress: 1, message: "Done" });
  return {
    scene: sceneId,
    instanceId,
    classValue,
    classNames: decoded.meta.class_names ?? null,
    classes: decoded.meta.classes ?? null,
    objectPoints: indices.length,
    objectBase,
    objectAnchor,
    axis,
    direction,
    measure,
    requestBytes: body.length,
    responseBytes: raw.length,
    steps: positions.map((coordinate, i) => ({
      coordinate, position: anchors[i], translation: translations[i],
      offset: offsets[i], height: heights[i], value: values[i],
    })),
  };
}
