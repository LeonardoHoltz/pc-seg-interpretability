/**
 * What gets sent to a segmentation service, and what does not.
 *
 * All three requests -- a prediction, a saliency map, a ceteris paribus sweep --
 * send the same scene arrays and differ only in what they add (a mask) and what
 * they vary. That layout lives here once.
 *
 * **Only the observations go out: coordinates, colour and normals.** Labels
 * stay behind. A field like `segment`, `instance`, `classification` or a
 * previous `prediction` is an answer, not an input: a service that can see the
 * ground truth can score perfectly without looking at the geometry, and an
 * interpretability result measured that way says nothing about the model. The
 * shape of a payload therefore does not depend on how well annotated a scene
 * happens to be, which also makes two scenes comparable.
 *
 * Throughput: xyz and rgb go as [3, N] because three contiguous per-axis arrays
 * laid end to end already *are* a C-contiguous [3, N], so nothing is transposed
 * or interleaved anywhere. Normals go as one [N] array per axis, which is how
 * the viewer holds every scalar field.
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readOctree } from "../octree/read.mjs";
import { cacheDirFor } from "../scene/registry.mjs";

/**
 * The scalar fields a service may receive. Normals only -- and spelled every way
 * the converter can produce, including the `_field` suffix it appends when a
 * name collides with one of Potree's reserved attribute slots.
 */
export const SENDABLE_SCALAR = /^normal_?[xyz](_field)?$/i;

/** Columns that are never a scalar field in their own right. */
const NOT_A_FIELD = new Set(["rgb", "rgba", "position"]);

/** Reads a converted scene: its description and its points. */
export function readSceneForInference(sceneId) {
  const dir = cacheDirFor(sceneId);
  const sceneJson = join(dir, "scene.json");
  if (!existsSync(sceneJson)) throw new Error(`${sceneId} has not been converted yet`);
  const scene = JSON.parse(readFileSync(sceneJson, "utf8"));
  const octree = readOctree(dir);
  return { dir, scene, octree, n: octree.count };
}

/**
 * The arrays for one scene, in wire order: xyz, rgb, then one per sendable
 * scalar field.
 *
 * @param fields  optional subset of the sendable fields, as the viewer's payload
 *                panel offers -- it can narrow what goes out, never widen it.
 * @returns {{ arrays, described, xyz }}  `arrays` for encodeArrays(), `described`
 *          for the payload preview, `xyz` so a caller can vary it in place.
 */
export function sceneArrays(scene, octree, { fields = null } = {}) {
  const n = octree.count;
  const kindOf = new Map((scene.attributes ?? []).map((a) => [a.name, a.kind]));
  const arrays = [];
  const described = [];

  const xyz = new Float32Array(3 * n);
  xyz.set(octree.x, 0);
  xyz.set(octree.y, n);
  xyz.set(octree.z, 2 * n);
  arrays.push({ name: "xyz", dtype: "float32", shape: [3, n], data: xyz });
  described.push({ name: "xyz", dtype: "<f4", shape: [3, n], role: "coordinates" });

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

  for (const [name, col] of octree.columns) {
    if (NOT_A_FIELD.has(name)) continue;
    if (col.numElements !== 1) continue;
    if (!SENDABLE_SCALAR.test(name)) continue;
    if (fields && !fields.includes(name)) continue;

    // Normals are continuous, so this is float32 in practice; the categorical
    // branch is what keeps the rule and the encoding independent of each other.
    const kind = kindOf.get(name);
    const categorical = kind === "categorical" || kind === "classification" || name === "classification";
    const data = categorical ? new Int32Array(n) : new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = col.data[i];

    arrays.push({ name, dtype: categorical ? "int32" : "float32", shape: [n], data });
    described.push({
      name, dtype: categorical ? "<i4" : "<f4", shape: [n],
      role: SENDABLE_SCALAR.test(name) ? "normal" : "continuous",
    });
  }

  return { arrays, described, xyz };
}

/**
 * Fields a scene holds that are deliberately *not* sent, so the viewer can say
 * so rather than leaving a gap where a field used to be listed.
 */
export function withheldFields(scene, octree) {
  const out = [];
  for (const [name, col] of octree.columns) {
    if (NOT_A_FIELD.has(name) || col.numElements !== 1) continue;
    if (SENDABLE_SCALAR.test(name)) continue;
    const kind = (scene.attributes ?? []).find((a) => a.name === name)?.kind ?? null;
    out.push({ name, kind });
  }
  return out;
}
