/**
 * Ceteris paribus analysis for a segmentation model.
 *
 * Holds a scene fixed, moves one object along a direction, and asks the model how
 * its prediction for that object changes. Plotting the object's mean probability
 * for a chosen class against height shows how much the prediction depends on
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
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readOctree } from "../octree/read.mjs";
import { encodeArrays, decodeArrays } from "./npbuffer.mjs";
import { cacheDirFor } from "../scene/registry.mjs";
import { config } from "../config.mjs";

/** Reads the scene once and lays out the arrays the service expects. */
function prepare(sceneId, fields) {
  const dir = cacheDirFor(sceneId);
  const sceneJson = join(dir, "scene.json");
  if (!existsSync(sceneJson)) throw new Error(`${sceneId} has not been converted yet`);
  const scene = JSON.parse(readFileSync(sceneJson, "utf8"));
  const octree = readOctree(dir);
  const n = octree.count;

  const kindOf = new Map((scene.attributes ?? []).map((a) => [a.name, a.kind]));
  const arrays = [];

  // xyz as [3, N]; z occupies the last third and is the only part that varies.
  const xyz = new Float32Array(3 * n);
  xyz.set(octree.x, 0);
  xyz.set(octree.y, n);
  xyz.set(octree.z, 2 * n);
  arrays.push({ name: "xyz", dtype: "float32", shape: [3, n], data: xyz });

  const rgbCol = octree.columns.get("rgb") ?? octree.columns.get("rgba");
  if (rgbCol && rgbCol.numElements === 3) {
    const rgb = new Uint8Array(3 * n);
    for (let i = 0; i < n; i++) {
      rgb[i] = rgbCol.data[i * 3];
      rgb[n + i] = rgbCol.data[i * 3 + 1];
      rgb[2 * n + i] = rgbCol.data[i * 3 + 2];
    }
    arrays.push({ name: "rgb", dtype: "uint8", shape: [3, n], data: rgb });
  }

  for (const [name, col] of octree.columns) {
    if (name === "rgb" || name === "rgba" || name === "position") continue;
    if (col.numElements !== 1) continue;
    if (fields && !fields.includes(name)) continue;
    const kind = kindOf.get(name);
    const categorical = kind === "categorical" || kind === "classification" || name === "classification";
    const data = categorical ? new Int32Array(n) : new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = col.data[i];
    arrays.push({ name, dtype: categorical ? "int32" : "float32", shape: [n], data });
  }

  return { scene, octree, n, arrays, xyz, baseZ: Float32Array.from(octree.z) };
}

/** Point indices belonging to one object, via the scene's instance field. */
function objectIndices(scene, octree, instanceId) {
  const field = scene.library?.field
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
 * @param opts.heights     absolute heights (metres) for the object's base
 */
export async function runCeterisParibus(sceneId, opts = {}) {
  const {
    endpoint, instanceId, classValue, heights,
    direction = [0, 0, 1],
    fields = null, timeoutMs = config.inference.timeouts.ceteris, onProgress = () => {},
  } = opts;

  if (!endpoint) throw new Error("no inference endpoint configured");
  if (!Array.isArray(heights) || heights.length === 0) throw new Error("no heights to sweep");
  if (heights.length > 256) throw new Error("keep the sweep under 256 positions");

  onProgress({ phase: "reading", progress: 0.05, message: "Reading the scene" });
  const { scene, octree, n, arrays, baseZ } = prepare(sceneId, fields);
  const indices = objectIndices(scene, octree, instanceId);

  // The object's base is its lowest point; the sweep moves that to each height.
  let objectBase = Infinity;
  for (const i of indices) if (baseZ[i] < objectBase) objectBase = baseZ[i];
  const offsets = heights.map((h) => h - objectBase);

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
      heights,
      base: objectBase,
      // What the reply should contain, so the service need not guess.
      expects: { logits: [heights.length, "C", indices.length] },
    },
  });

  onProgress({
    phase: "sending", progress: 0.2,
    message: `Sending the scene once with ${heights.length} positions ` +
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
    numSteps: heights.length,
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
    direction,
    measure,
    requestBytes: body.length,
    responseBytes: raw.length,
    steps: heights.map((height, i) => ({ height, value: values[i] })),
  };
}
