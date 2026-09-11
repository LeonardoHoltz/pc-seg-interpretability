/**
 * Sends a scene's points to a segmentation service and folds the returned
 * labels back into the scene as a new attribute.
 *
 * Throughput notes: the points never touch the browser and are never turned into
 * JSON. They go straight from the cached octree into one contiguous binary body
 * (see npbuffer.mjs) that the receiving end can wrap in numpy views without
 * copying. The layout asked for -- xyz and rgb as [3, N], every scalar field as
 * [N] -- is also the fastest one available, because per-axis arrays laid end to
 * end already are a C-contiguous [3, N] array. No transposing happens anywhere.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readOctree } from "../octree/read.mjs";
import { rewriteSceneWithAttributes } from "../scene/attributes.mjs";
import { encodeArrays, decodeArrays } from "./npbuffer.mjs";
import { classColor, UNLABELED_COLOR, UNLABELED_NAMES } from "../scene/palette.mjs";
import { cacheDirFor } from "../scene/registry.mjs";

/** Name of the attribute the returned labels are written into. */
export const PREDICTION_ATTRIBUTE = "prediction";

/** Builds the request body from a scene's octree. */
export function buildPayload(sceneId, { fields = null } = {}) {
  const dir = cacheDirFor(sceneId);
  const sceneJson = join(dir, "scene.json");
  if (!existsSync(sceneJson)) throw new Error(`${sceneId} has not been converted yet`);
  const scene = JSON.parse(readFileSync(sceneJson, "utf8"));
  const octree = readOctree(dir);
  const n = octree.count;

  const kindOf = new Map((scene.attributes ?? []).map((a) => [a.name, a.kind]));
  const arrays = [];
  const described = [];

  // xyz as [3, N]: x, then y, then z, each contiguous.
  const xyz = new Float32Array(3 * n);
  xyz.set(octree.x, 0);
  xyz.set(octree.y, n);
  xyz.set(octree.z, 2 * n);
  arrays.push({ name: "xyz", dtype: "float32", shape: [3, n], data: xyz });
  described.push({ name: "xyz", dtype: "<f4", shape: [3, n], role: "coordinates" });

  // rgb as [3, N]: the octree stores it interleaved, so split it out per channel.
  const rgbCol = octree.columns.get("rgb") ?? octree.columns.get("rgba");
  if (rgbCol && rgbCol.numElements === 3) {
    const rgb = new Uint8Array(3 * n);
    for (let i = 0; i < n; i++) {
      rgb[i] = rgbCol.data[i * 3];
      rgb[n + i] = rgbCol.data[i * 3 + 1];
      rgb[2 * n + i] = rgbCol.data[i * 3 + 2];
    }
    arrays.push({ name: "rgb", dtype: "uint8", shape: [3, n], data: rgb });
    described.push({ name: "rgb", dtype: "|u1", shape: [3, n], role: "colour" });
  }

  // One [N] array per scalar field.
  for (const [name, col] of octree.columns) {
    if (name === "rgb" || name === "rgba" || name === "position") continue;
    if (col.numElements !== 1) continue;
    if (fields && !fields.includes(name)) continue;

    const kind = kindOf.get(name);
    const categorical = kind === "categorical" || kind === "classification" || name === "classification";
    const data = categorical ? new Int32Array(n) : new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = col.data[i];

    arrays.push({ name, dtype: categorical ? "int32" : "float32", shape: [n], data });
    described.push({
      name, dtype: categorical ? "<i4" : "<f4", shape: [n],
      role: categorical ? "categorical" : "continuous",
    });
  }

  const body = encodeArrays(arrays, {
    scene: sceneId,
    num_points: n,
    bounding_box: scene.tightBoundingBox,
  });

  return { body, numPoints: n, arrays: described, scene, octree };
}

/** What would be sent, without sending it. */
export function previewPayload(sceneId, options = {}) {
  const { body, numPoints, arrays } = buildPayload(sceneId, options);
  return {
    numPoints,
    bytes: body.length,
    arrays: arrays.map((a) => ({
      ...a,
      nbytes: a.shape.reduce((x, y) => x * y, 1) *
        ({ "<f4": 4, "|u1": 1, "<i4": 4, "<f8": 8 }[a.dtype] ?? 4),
    })),
  };
}

function buildClassTable(values, names) {
  const counts = new Map();
  for (let i = 0; i < values.length; i++) {
    const v = Number(values[i]);
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  return [...counts.keys()].sort((a, b) => a - b).map((value, index) => {
    const name = names?.[value] ?? names?.[String(value)] ?? `class ${value}`;
    const unlabeled = UNLABELED_NAMES.has(String(name).toLowerCase());
    return {
      value, index, name,
      color: unlabeled ? UNLABELED_COLOR : classColor(index),
      count: counts.get(value),
    };
  });
}

/**
 * @param sceneId   the scene to segment
 * @param endpoint  the service's URL
 */
export async function runSegmentation(sceneId, endpoint, options = {}) {
  const { fields = null, timeoutMs = 15 * 60 * 1000, onProgress = () => {} } = options;
  if (!endpoint) throw new Error("no inference endpoint configured");

  onProgress({ phase: "reading", progress: 0.05, message: "Reading the scene" });
  const { body, numPoints, scene, octree } = buildPayload(sceneId, { fields });

  onProgress({
    phase: "sending", progress: 0.25,
    message: `Sending ${numPoints.toLocaleString()} points (${(body.length / 1048576).toFixed(1)} MiB)`,
  });

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/octet-stream",
        "x-pcit-scene": sceneId,
        "x-pcit-points": String(numPoints),
      },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    throw new Error(`could not reach the inference service at ${endpoint}: ${err.message}`);
  }

  const raw = Buffer.from(await response.arrayBuffer());
  if (!response.ok) {
    let detail = raw.toString("utf8", 0, Math.min(raw.length, 400));
    try { detail = JSON.parse(detail).error ?? detail; } catch { /* keep raw */ }
    throw new Error(`inference service returned ${response.status}: ${detail}`);
  }

  onProgress({ phase: "decoding", progress: 0.6, message: "Reading the labels" });

  let labels = null, scores = null, classNames = null;
  const type = response.headers.get("content-type") ?? "";
  if (type.includes("json")) {
    const parsed = JSON.parse(raw.toString("utf8"));
    labels = parsed.labels;
    scores = parsed.scores ?? null;
    classNames = parsed.class_names ?? null;
  } else {
    const { meta, arrays } = decodeArrays(raw);
    labels = arrays.get("labels")?.data ?? null;
    scores = arrays.get("scores")?.data ?? null;
    classNames = meta.class_names ?? null;
  }

  if (!labels) throw new Error("the response contained no `labels` array");
  if (labels.length !== numPoints) {
    throw new Error(`the service returned ${labels.length} labels for ${numPoints} points`);
  }

  // ---- fold the labels back in as new attributes ------------------------
  onProgress({ phase: "building", progress: 0.7, message: "Rebuilding the octree" });

  const classes = buildClassTable(labels, classNames);
  const predicted = new Float64Array(numPoints);
  for (let i = 0; i < numPoints; i++) predicted[i] = Number(labels[i]);

  const scoreCol = scores && scores.length === numPoints
    ? Float64Array.from(scores, Number)
    : null;

  const add = [{
    name: PREDICTION_ATTRIBUTE,
    label: "Predicted class",
    kind: "categorical",
    values: predicted,
    classes,
  }];
  if (scoreCol) {
    add.push({
      name: `${PREDICTION_ATTRIBUTE}_score`,
      label: "Prediction score",
      kind: "continuous",
      values: scoreCol,
    });
  }

  const { scene: updated } = rewriteSceneWithAttributes(sceneId, {
    cacheDir: cacheDirFor(sceneId),
    add,
    onProgress,
  });

  updated.prediction = {
    endpoint, at: Date.now(),
    numPoints, numClasses: classes.length,
    hasScores: Boolean(scoreCol),
    classes: classes.map((c) => ({ value: c.value, name: c.name, count: c.count })),
  };

  writeFileSync(join(cacheDirFor(sceneId), "scene.json"), JSON.stringify(updated, null, 2));
  onProgress({ phase: "done", progress: 1, message: "Done" });
  return updated;
}
