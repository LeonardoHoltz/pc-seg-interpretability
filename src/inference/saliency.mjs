/**
 * Saliency for one object.
 *
 * Sends the scene with a mask marking the object of interest and asks the
 * service for one scalar per point. What that scalar means, and how it is
 * aggregated across classes or layers, is entirely the service's business --
 * this only carries the numbers back and makes them a scene attribute, so they
 * can be coloured, ramped and range-filtered like any other scalar field.
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readOctree } from "../octree/read.mjs";
import { encodeArrays, decodeArrays } from "./npbuffer.mjs";
import { cacheDirFor } from "../scene/registry.mjs";
import { rewriteSceneWithAttributes } from "../scene/attributes.mjs";
import { config } from "../config.mjs";

export const SALIENCY_ATTRIBUTE = "saliency";

/** Lays out the scene arrays plus a mask for the object under study. */
function prepare(sceneId, instanceId, fields) {
  const dir = cacheDirFor(sceneId);
  const sceneJson = join(dir, "scene.json");
  if (!existsSync(sceneJson)) throw new Error(`${sceneId} has not been converted yet`);
  const scene = JSON.parse(readFileSync(sceneJson, "utf8"));
  const octree = readOctree(dir);
  const n = octree.count;

  const kindOf = new Map((scene.attributes ?? []).map((a) => [a.name, a.kind]));
  const arrays = [];

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
    if (name === SALIENCY_ATTRIBUTE) continue;         // never feed a result back in
    if (fields && !fields.includes(name)) continue;
    const kind = kindOf.get(name);
    const categorical = kind === "categorical" || kind === "classification" || name === "classification";
    const data = categorical ? new Int32Array(n) : new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = col.data[i];
    arrays.push({ name, dtype: categorical ? "int32" : "float32", shape: [n], data });
  }

  // Which points make up the object, as the mask numpy wants.
  const instanceField = scene.library?.field
    ?? (scene.attributes ?? []).find((a) => a.kind === "categorical" && /instance|object/i.test(a.name))?.name;
  const idCol = octree.columns.get(instanceField) ?? octree.columns.get("instance");
  if (!idCol) throw new Error("this scene has no instance field, so objects cannot be isolated");

  const mask = new Uint8Array(n);
  let objectPoints = 0;
  for (let i = 0; i < n; i++) {
    if (idCol.data[i] === instanceId) { mask[i] = 1; objectPoints++; }
  }
  if (objectPoints === 0) throw new Error(`no points carry instance id ${instanceId}`);
  arrays.push({ name: "mask", dtype: "uint8", shape: [n], data: mask });

  return { scene, octree, n, arrays, mask, objectPoints };
}

/**
 * @param opts.instanceId  the object the saliency is about
 * @param opts.classValue  optional target class, passed through for the service
 */
export async function runSaliency(sceneId, opts = {}) {
  const {
    endpoint, instanceId, classValue = null,
    fields = null, timeoutMs = config.inference.timeouts.saliency, onProgress = () => {},
  } = opts;

  if (!endpoint) throw new Error("no inference endpoint configured");
  if (!Number.isFinite(instanceId)) throw new Error("an object must be chosen");

  onProgress({ phase: "reading", progress: 0.05, message: "Reading the scene" });
  const { n, arrays, mask, objectPoints } = prepare(sceneId, instanceId, fields);

  const body = encodeArrays(arrays, {
    request: "saliency",
    scene: sceneId,
    num_points: n,
    saliency: {
      instance: instanceId,
      object_points: objectPoints,
      target_class: classValue,
      // Either shape is accepted; the service picks whichever suits it.
      expects: { saliency: [`${n} (whole scene) or ${objectPoints} (masked points)`] },
    },
  });

  onProgress({
    phase: "sending", progress: 0.25,
    message: `Sending ${n.toLocaleString()} points (${(body.length / 1048576).toFixed(1)} MiB)`,
  });

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/octet-stream",
        "x-pcit-scene": sceneId,
        "x-pcit-points": String(n),
        "x-pcit-request": "saliency",
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

  onProgress({ phase: "decoding", progress: 0.6, message: "Reading the saliency" });
  const decoded = decodeArrays(raw);
  const entry = decoded.arrays.get("saliency") ?? decoded.arrays.get("scores");
  if (!entry) throw new Error("the response contained no `saliency` array");

  const incoming = entry.data;
  const values = new Float64Array(n);
  if (incoming.length === n) {
    for (let i = 0; i < n; i++) values[i] = Number(incoming[i]);
  } else if (incoming.length === objectPoints) {
    // Scoped to the object: scatter it back, leaving the rest of the scene at 0.
    let w = 0;
    for (let i = 0; i < n; i++) if (mask[i]) values[i] = Number(incoming[w++]);
  } else {
    throw new Error(
      `saliency has ${incoming.length} values; expected ${n} (whole scene) or ${objectPoints} (the object)`);
  }

  onProgress({ phase: "building", progress: 0.7, message: "Rebuilding the octree" });
  const scoped = incoming.length === objectPoints;
  const { scene: updated } = rewriteSceneWithAttributes(sceneId, {
    cacheDir: cacheDirFor(sceneId),
    add: [{
      name: SALIENCY_ATTRIBUTE,
      label: `Saliency (object #${instanceId})`,
      kind: "continuous",
      values,
    }],
    onProgress,
  });

  updated.saliency = {
    endpoint, at: Date.now(),
    instanceId, classValue,
    objectPoints,
    scope: scoped ? "object" : "scene",
    range: [Math.min(...values.slice(0, 1)), 0],
  };
  // Record the real range rather than a placeholder.
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { if (values[i] < lo) lo = values[i]; if (values[i] > hi) hi = values[i]; }
  updated.saliency.range = [lo, hi];

  const { writeFileSync } = await import("node:fs");
  writeFileSync(join(cacheDirFor(sceneId), "scene.json"), JSON.stringify(updated, null, 2));

  onProgress({ phase: "done", progress: 1, message: "Done" });
  return updated;
}
