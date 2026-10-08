/**
 * Does an object stop being a sofa once you take away the points that argue it
 * is one?
 *
 * Two attribution maps for one object -- one per class -- their difference, and
 * the object re-segmented without the points at the top of that difference.
 * The MNIST version of this erases pixels from an 8 until the network reads a
 * 3; here the points genuinely leave the cloud, which is the same operation for
 * a sparse representation.
 *
 * Nothing is written to the scene. This is analysis: the octree is read, the
 * service answers, and the result lives in the panel until the next run.
 */
import { encodeArrays, decodeArrays } from "./npbuffer.mjs";
import { readSceneForInference, sceneArrays } from "./payload.mjs";
import { config } from "../config.mjs";

/** How many object points the viewer is sent for its three small previews. */
const PREVIEW_LIMIT = 12000;

/** The field that enumerates objects, as the scene recorded it. */
function instanceColumn(scene, octree) {
  const field = scene.roles?.instance?.source ?? scene.library?.field
    ?? (scene.attributes ?? []).find((a) => a.kind === "categorical" && /instance|object/i.test(a.name))?.name;
  const col = octree.columns.get(field) ?? octree.columns.get("instance");
  if (!col) throw new Error("this scene has no instance field, so objects cannot be isolated");
  return col;
}

export async function runAblation(sceneId, opts = {}) {
  const {
    endpoint, instanceId, classA, classB, remove = 0,
    method = "deeplift", baseline = "noise", steps = 16, input: attributeOver = "both",
    fields = null, timeoutMs = config.inference.timeouts.saliency, onProgress = () => {},
  } = opts;

  if (!endpoint) throw new Error("no inference endpoint configured");
  if (!Number.isFinite(instanceId)) throw new Error("an object must be chosen");
  if (!Number.isFinite(classA) || !Number.isFinite(classB)) {
    throw new Error("two classes must be chosen");
  }
  if (classA === classB) throw new Error("the two classes must differ");

  onProgress({ phase: "reading", progress: 0.05, message: "Reading the scene" });
  const { scene, octree, n } = readSceneForInference(sceneId);
  const { arrays } = sceneArrays(scene, octree, { fields });

  const idCol = instanceColumn(scene, octree);
  const mask = new Uint8Array(n);
  let objectPoints = 0;
  for (let i = 0; i < n; i++) if (idCol.data[i] === instanceId) { mask[i] = 1; objectPoints++; }
  if (objectPoints === 0) throw new Error(`no points carry instance id ${instanceId}`);

  const body = encodeArrays([...arrays, { name: "mask", dtype: "uint8", shape: [n], data: mask }], {
    request: "ablation",
    scene: sceneId,
    num_points: n,
    ablation: {
      instance: instanceId,
      object_points: objectPoints,
      class_a: classA,
      class_b: classB,
      remove,
      method, baseline, steps,
      // Removing a point takes away its position as well as its colour, so the
      // ranking has to see both.
      input: attributeOver,
    },
  });

  onProgress({
    phase: "sending", progress: 0.2,
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
        "x-pcit-request": "ablation",
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

  onProgress({ phase: "decoding", progress: 0.75, message: "Reading the attributions" });
  const decoded = decodeArrays(raw);
  const meta = decoded.meta?.ablation ?? {};
  const pick = (name) => {
    const entry = decoded.arrays.get(name);
    if (!entry) throw new Error(`the response contained no \`${name}\` array`);
    return entry.data;
  };
  const a = pick("attribution_a");
  const b = pick("attribution_b");
  const diff = pick("attribution_diff");
  const removedRaw = decoded.arrays.get("removed")?.data ?? [];

  // Attributions may come back scene-length or object-length; normalise to the
  // object's own points, which is all the previews draw.
  const scoped = a.length === objectPoints;
  if (!scoped && a.length !== n) {
    throw new Error(`attribution has ${a.length} values; expected ${n} or ${objectPoints}`);
  }

  const removed = new Set();
  for (const v of removedRaw) removed.add(Number(v));

  // ---- what the viewer draws -------------------------------------------
  // One entry per object point: position, the two maps, their difference, and
  // whether it was taken away. Subsampled on a regular stride, so a big object
  // still arrives as a page of JSON rather than a download.
  const indices = [];
  for (let i = 0; i < n; i++) if (mask[i]) indices.push(i);
  const stride = Math.max(1, Math.ceil(indices.length / PREVIEW_LIMIT));

  const pts = [];
  const va = [], vb = [], vd = [], gone = [];
  for (let k = 0; k < indices.length; k += stride) {
    const i = indices[k];
    const at = scoped ? k : i;
    pts.push(
      Math.round(octree.x[i] * 1000) / 1000,
      Math.round(octree.y[i] * 1000) / 1000,
      Math.round(octree.z[i] * 1000) / 1000,
    );
    va.push(Number(a[at]));
    vb.push(Number(b[at]));
    vd.push(Number(diff[at]));
    gone.push(removed.has(i) ? 1 : 0);
  }

  onProgress({ phase: "done", progress: 1, message: "Done" });
  return {
    sceneId,
    at: Date.now(),
    instanceId,
    classA: meta.class_a ?? classA,
    classB: meta.class_b ?? classB,
    method: meta.method ?? method,
    baseline, steps,
    objectPoints,
    removedCount: meta.removed ?? removed.size,
    requestBytes: body.length,
    responseBytes: raw.length,
    before: meta.before ?? null,
    after: meta.after ?? null,
    preview: { stride, count: gone.length, xyz: pts, a: va, b: vb, diff: vd, removed: gone },
  };
}
