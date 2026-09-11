#!/usr/bin/env node
/**
 * Merges placed instances into a scene's octree.
 *
 * The scene is rebuilt from its original PCD plus the points of every placed
 * object, transformed into world space. Instance points are recovered from the
 * instance octrees (see octree_read.mjs), so objects borrowed from another scene
 * merge just as well as ones from this scene.
 *
 * The same routine also *detaches* objects: `exclude` lists instance ids whose
 * points are dropped from the scene, which is how an object is lifted out of a
 * scene so it can be moved somewhere else.
 *
 * Note this rewrites the *cached octree*, not the source PCD, so re-converting
 * the scene afterwards discards the changes. The UI says so before it asks for
 * confirmation.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readCloud } from "../io/cloud.mjs";
import { describeFields } from "../scene/fields.mjs";
import { writeOctree } from "../octree/write.mjs";
import { readOctree } from "../octree/read.mjs";
import { silhouette, footprintHull, SILHOUETTE } from "./instances.mjs";
import { classColor, UNLABELED_COLOR, UNLABELED_NAMES } from "../scene/palette.mjs";
import { cacheDirFor, resolveScene, readSidecar } from "../scene/registry.mjs";

/** Rotation about z, then uniform scale, then translation. */
function makeTransform({ pos, yaw, scale }) {
  const c = Math.cos(yaw ?? 0), s = Math.sin(yaw ?? 0);
  const k = scale ?? 1;
  return (x, y, z) => [
    (x * c - y * s) * k + pos[0],
    (x * s + y * c) * k + pos[1],
    z * k + pos[2],
  ];
}

function minMaxOf(data, stride = 1, element = 0) {
  let lo = Infinity, hi = -Infinity;
  for (let i = element; i < data.length; i += stride) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo > hi) { lo = 0; hi = 0; }
  return [lo, hi];
}

/** Refreshes class counts, adding entries for values that appeared after a merge. */
function recountClasses(existing, values) {
  const counts = new Map();
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const byValue = new Map((existing ?? []).map((c) => [c.value, { ...c, count: 0 }]));
  let nextIndex = Math.max(-1, ...[...byValue.values()].map((c) => c.index)) + 1;

  for (const [value, count] of counts) {
    if (byValue.has(value)) {
      byValue.get(value).count = count;
    } else {
      const name = `class ${value}`;
      byValue.set(value, {
        value, index: nextIndex,
        name,
        color: UNLABELED_NAMES.has(name) ? UNLABELED_COLOR : classColor(nextIndex),
        count,
      });
      nextIndex++;
    }
  }
  return [...byValue.values()].sort((a, b) => a.value - b.value);
}

/**
 * @param sceneId     the scene whose octree is rebuilt
 * @param placements  [{ sourceSceneId, instanceId, pos:[x,y,z], yaw, scale }]
 */
export function bakeInstances(sceneId, placements, options = {}) {
  const { onProgress = () => {}, gridSize = 128, exclude = [] } = options;
  placements = placements ?? [];
  if (placements.length === 0 && exclude.length === 0) {
    throw new Error("nothing to add or remove");
  }

  const outDir = cacheDirFor(sceneId);
  const sceneJsonPath = join(outDir, "scene.json");
  if (!existsSync(sceneJsonPath)) throw new Error(`${sceneId} has not been converted yet`);
  const sceneInfo = JSON.parse(readFileSync(sceneJsonPath, "utf8"));

  // ---- the scene as it stands ------------------------------------------
  onProgress({ phase: "reading", progress: 0, message: "Reading the scene" });
  const pcdPath = resolveScene(sceneId);
  const pcd = readCloud(pcdPath);
  const described = describeFields(pcd, {
    classesSidecar: readSidecar(pcdPath),
    primaryField: sceneInfo.classification?.source ?? null,
  });

  const { valid, numValid, position, color, scalars, primary } = described;
  const [bx, by, bz] = position.map((f) => f.data);

  const libraryPath0 = join(outDir, "instances.json");
  const instanceSource = existsSync(libraryPath0)
    ? JSON.parse(readFileSync(libraryPath0, "utf8")).field
    : null;

  // Detaching an object means rebuilding the scene without its points. Ids
  // already removed by an earlier detach are remembered so they stay gone.
  const previouslyRemoved = sceneInfo.detached?.map((d) => d.instanceId) ?? [];
  const removeIds = new Set([...previouslyRemoved, ...exclude].map(Number));
  const instanceData = instanceSource
    ? (scalars.find((s) => s.source === instanceSource)?.data ?? null)
    : null;

  let kept = 0;
  const baseIndices = new Uint32Array(numValid);
  for (let i = 0; i < pcd.numPoints; i++) {
    if (!valid[i]) continue;
    if (instanceData && removeIds.has(instanceData[i])) continue;
    baseIndices[kept++] = i;
  }
  const numBase = kept;
  const removedPoints = numValid - numBase;

  // ---- the objects to merge in ------------------------------------------
  onProgress({ phase: "reading", progress: 0.2, message: "Reading placed objects" });
  const targetClasses = sceneInfo.classification?.classes ?? [];
  const targetIndexByValue = new Map(targetClasses.map((c) => [c.value, c.index]));
  const targetIndexByName = new Map(targetClasses.map((c) => [c.name, c.index]));

  const instanceFieldSource = instanceSource;

  // Fresh ids so every merged object stays a distinct instance. Ids reclaimed by
  // restored objects are reserved up front, or a new object could be handed an
  // id a carried-over one is already using.
  let nextInstanceId = 1;
  if (instanceFieldSource) {
    const field = scalars.find((s) => s.source === instanceFieldSource);
    if (field?.classes?.length) nextInstanceId = Math.max(...field.classes) + 1;
  }
  const usedIds = new Set();
  const allocateId = () => {
    while (usedIds.has(nextInstanceId)) nextInstanceId++;
    return nextInstanceId++;
  };

  // Each bake rebuilds the whole octree from the PCD, so objects merged by an
  // earlier bake are not in that source and would silently vanish. They are
  // carried over here: a merged object was written back out as its own instance
  // octree in its final pose, so re-placing it at its anchor reproduces it.
  const excludeNow = new Set(exclude.map(Number));
  const restoringNow = new Set(
    placements.map((p) => p.restoreInstanceId).filter((v) => v != null).map(Number));

  const libraryEntries = existsSync(libraryPath0)
    ? JSON.parse(readFileSync(libraryPath0, "utf8")).instances
    : [];
  const carryOver = [];
  const seenCarry = new Set();
  for (const b of sceneInfo.baked ?? []) {
    const id = Number(b.newInstanceId);
    if (excludeNow.has(id) || restoringNow.has(id) || seenCarry.has(id)) continue;
    const entry = libraryEntries.find((e) => e.id === id);
    if (!entry) continue;
    seenCarry.add(id);
    carryOver.push({
      sourceSceneId: sceneId,
      instanceId: id,
      restoreInstanceId: id,
      className: entry.class?.name ?? null,
      pos: entry.anchor,
      yaw: 0,
      scale: 1,
      carriedOver: true,
    });
  }
  placements = [...carryOver, ...placements];

  for (const p of placements) {
    if (p.restoreInstanceId != null) usedIds.add(Number(p.restoreInstanceId));
  }

  const parts = [];
  for (const placement of placements) {
    const srcScene = placement.sourceSceneId ?? sceneId;
    // A resampled object has its own octree; fall back to the original.
    const dir = placement.octreeDir
      ? join(cacheDirFor(srcScene), placement.octreeDir)
      : join(cacheDirFor(srcScene), "instances", String(placement.instanceId));
    if (!existsSync(dir)) throw new Error(`missing instance octree: ${srcScene}#${placement.instanceId}`);

    const data = readOctree(dir);

    // The source scene's class indices mean nothing here; map them by value.
    let indexMap = null;
    const srcJson = join(cacheDirFor(srcScene), "scene.json");
    if (srcScene !== sceneId && existsSync(srcJson)) {
      const src = JSON.parse(readFileSync(srcJson, "utf8"));
      indexMap = new Map();
      for (const c of src.classification?.classes ?? []) {
        indexMap.set(c.index, targetIndexByValue.get(c.value) ?? targetIndexByName.get(c.name) ?? 0);
      }
    }

    // An object that was detached from this scene keeps its original id when it
    // comes back, so its library entry stays valid instead of being orphaned.
    const restore = placement.restoreInstanceId != null ? Number(placement.restoreInstanceId) : null;
    const canRestore = restore != null && !parts.some((p) => p.newInstanceId === restore);

    parts.push({
      placement, data, indexMap,
      sourceSceneId: srcScene,
      transform: makeTransform(placement),
      newInstanceId: canRestore ? restore : allocateId(),
      restored: canRestore,
    });
    usedIds.add(parts[parts.length - 1].newInstanceId);
  }

  const extra = parts.reduce((a, p) => a + p.data.count, 0);
  const total = numBase + extra;

  // ---- merged columns, one per attribute the scene already has ----------
  onProgress({
    phase: "merging", progress: 0.35,
    message: removedPoints > 0 && extra === 0
      ? `Detaching ${removedPoints.toLocaleString()} points`
      : `Merging ${extra.toLocaleString()} points`,
  });

  const X = new Float64Array(total);
  const Y = new Float64Array(total);
  const Z = new Float64Array(total);
  for (let k = 0; k < numBase; k++) {
    const i = baseIndices[k];
    X[k] = bx[i]; Y[k] = by[i]; Z[k] = bz[i];
  }

  const specs = [];
  for (const att of sceneInfo.attributes) {
    if (att.kind === "color") {
      specs.push({ att, kind: "color", octreeName: "rgb",
        r: new Uint16Array(total), g: new Uint16Array(total), b: new Uint16Array(total) });
    } else if (att.kind === "classification") {
      specs.push({ att, kind: "classification", octreeName: "classification", data: new Uint8Array(total) });
    } else if (att.name === "intensity") {
      specs.push({ att, kind: "intensity", octreeName: "intensity", data: new Float32Array(total) });
    } else {
      specs.push({ att, kind: "scalar", octreeName: att.name, data: new Float64Array(total) });
    }
  }

  // base points
  for (const spec of specs) {
    const { att } = spec;
    if (spec.kind === "color") {
      for (let k = 0; k < numBase; k++) {
        const i = baseIndices[k];
        spec.r[k] = color.r[i]; spec.g[k] = color.g[i]; spec.b[k] = color.b[i];
      }
    } else if (spec.kind === "classification") {
      const src = primary?.data;
      for (let k = 0; k < numBase; k++) {
        spec.data[k] = src ? (targetIndexByValue.get(src[baseIndices[k]]) ?? 0) : 0;
      }
    } else {
      const field = scalars.find((s) => s.source === att.source || s.name === att.name);
      if (field) for (let k = 0; k < numBase; k++) spec.data[k] = field.data[baseIndices[k]];
    }
  }

  // merged objects
  let cursor = numBase;
  for (const part of parts) {
    const { data, transform, indexMap, newInstanceId } = part;
    part.start = cursor;
    for (let p = 0; p < data.count; p++) {
      const [wx, wy, wz] = transform(data.x[p], data.y[p], data.z[p]);
      X[cursor + p] = wx; Y[cursor + p] = wy; Z[cursor + p] = wz;
    }

    for (const spec of specs) {
      const col = data.columns.get(spec.octreeName);
      const isInstanceField = instanceFieldSource &&
        (spec.att.source === instanceFieldSource || spec.att.name === instanceFieldSource);

      if (spec.kind === "color") {
        if (col) {
          for (let p = 0; p < data.count; p++) {
            spec.r[cursor + p] = col.data[p * 3 + 0];
            spec.g[cursor + p] = col.data[p * 3 + 1];
            spec.b[cursor + p] = col.data[p * 3 + 2];
          }
        }
      } else if (spec.kind === "classification") {
        if (col) {
          for (let p = 0; p < data.count; p++) {
            const srcIdx = col.data[p];
            spec.data[cursor + p] = indexMap ? (indexMap.get(srcIdx) ?? 0) : srcIdx;
          }
        }
      } else if (isInstanceField) {
        // Each merged object becomes its own instance in the scene.
        for (let p = 0; p < data.count; p++) spec.data[cursor + p] = newInstanceId;
      } else if (col) {
        for (let p = 0; p < data.count; p++) spec.data[cursor + p] = col.data[p];
      }
    }
    cursor += data.count;
  }

  // ---- rebuild the octree ----------------------------------------------
  onProgress({ phase: "building", progress: 0.5, message: "Rebuilding the octree" });

  const safeRange = (lo, hi) => (hi > lo ? [lo, hi] : [lo, lo + 1]);
  const attributes = [];
  for (const spec of specs) {
    if (spec.kind === "color") {
      attributes.push({
        name: "rgb", type: "uint16", numElements: 3, description: spec.att.source ?? "",
        min: [0, 0, 0], max: [255, 255, 255],
        write: (view, o, i) => {
          view.setUint16(o + 0, spec.r[i], true);
          view.setUint16(o + 2, spec.g[i], true);
          view.setUint16(o + 4, spec.b[i], true);
        },
      });
    } else if (spec.kind === "classification") {
      const [, hi] = minMaxOf(spec.data);
      attributes.push({
        name: "classification", type: "uint8", numElements: 1, description: "",
        min: [0], max: [Math.max(0, hi)],
        write: (view, o, i) => view.setUint8(o, spec.data[i]),
      });
    } else if (spec.kind === "intensity") {
      const [lo, hi] = safeRange(...minMaxOf(spec.data));
      spec.range = [lo, hi];
      attributes.push({
        name: "intensity", type: "float", numElements: 1, description: "",
        min: [lo], max: [hi],
        write: (view, o, i) => view.setFloat32(o, spec.data[i], true),
      });
    } else {
      const [lo, hi] = safeRange(...minMaxOf(spec.data));
      spec.range = [lo, hi];
      attributes.push({
        name: spec.att.name, type: "double", numElements: 1, description: spec.att.kind,
        min: [lo], max: [hi],
        write: (view, o, i) => view.setFloat64(o, spec.data[i], true),
      });
    }
  }

  const allIndices = new Uint32Array(total);
  for (let i = 0; i < total; i++) allIndices[i] = i;

  let lastReport = 0;
  const result = writeOctree({
    indices: allIndices,
    getX: (i) => X[i], getY: (i) => Y[i], getZ: (i) => Z[i],
    attributes, outDir, gridSize,
    name: sceneId,
    description: sceneInfo.description ?? "",
    onProgress: (done, n) => {
      const now = Date.now();
      if (now - lastReport > 120) {
        lastReport = now;
        onProgress({ phase: "building", progress: 0.5 + 0.5 * (done / n), message: "Rebuilding the octree" });
      }
    },
  });

  // ---- refresh the object library --------------------------------------
  // Every merged object is re-cut as its own instance octree, from the points as
  // they now sit. Without this an integrated object has no library entry, so
  // clicking it in the viewport resolves to nothing and it can never be
  // inspected or detached again.
  const updated0Classes = sceneInfo.classification?.classes ?? [];
  if (existsSync(libraryPath0)) {
    const library = JSON.parse(readFileSync(libraryPath0, "utf8"));
    const byId = new Map(library.instances.map((e) => [e.id, e]));

    for (const part of parts) {
      const id = part.newInstanceId;
      const n = part.data.count;
      const idx = new Uint32Array(n);
      for (let i = 0; i < n; i++) idx[i] = part.start + i;

      let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
      let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
      for (let i = 0; i < n; i++) {
        const k = part.start + i;
        if (X[k] < mnX) mnX = X[k]; if (X[k] > mxX) mxX = X[k];
        if (Y[k] < mnY) mnY = Y[k]; if (Y[k] > mxY) mxY = Y[k];
        if (Z[k] < mnZ) mnZ = Z[k]; if (Z[k] > mxZ) mxZ = Z[k];
      }
      const anchor = [(mnX + mxX) / 2, (mnY + mxY) / 2, mnZ];

      const written = writeOctree({
        indices: idx,
        getX: (i) => X[i] - anchor[0],
        getY: (i) => Y[i] - anchor[1],
        getZ: (i) => Z[i] - anchor[2],
        attributes,
        outDir: join(outDir, "instances", String(id)),
        name: `instance ${id}`,
        maxPointsPerNode: 60000,
        spacing: result.metadata.spacing / Math.pow(2, result.depth),
      });

      // Read the class back out of the merged data: looking it up by the source
      // scene's instance id would hit an unrelated object in this scene's library.
      let klass = byId.get(id)?.class ?? null;
      const clsSpec = specs.find((sp) => sp.kind === "classification");
      if (clsSpec) {
        const tally = new Map();
        for (let i = 0; i < n; i++) {
          const v = clsSpec.data[part.start + i];
          tally.set(v, (tally.get(v) ?? 0) + 1);
        }
        let bestIdx = null, bestN = -1;
        for (const [v, c] of tally) if (c > bestN) { bestN = c; bestIdx = v; }
        const match = (updated0Classes ?? []).find((c) => c.index === bestIdx);
        if (match) klass = { value: match.value, index: match.index, name: match.name, color: match.color };
      }

      byId.set(id, {
        id,
        dir: `instances/${id}`,
        points: n,
        class: klass,
        size: [mxX - mnX, mxY - mnY, mxZ - mnZ],
        anchor,
        anchorLocal: [
          -written.metadata.boundingBox.min[0],
          -written.metadata.boundingBox.min[1],
          -written.metadata.boundingBox.min[2],
        ],
        bytes: written.octreeBytes,
        nodes: written.numNodes,
        silhouette: silhouette(idx, X, Y, Z, anchor),
        footprint: footprintHull(idx, X, Y, anchor),
      });
    }

    library.instances = [...byId.values()]
      .sort((a, b) => (a.class?.name ?? "").localeCompare(b.class?.name ?? "") || b.points - a.points);
    library.count = library.instances.length;
    library.silhouetteSize = SILHOUETTE;
    writeFileSync(libraryPath0, JSON.stringify(library, null, 2));
  }

  // ---- refresh scene.json ----------------------------------------------
  const updated = { ...sceneInfo };
  updated.numPoints = total;
  updated.boundingBox = result.metadata.boundingBox;
  updated.tightBoundingBox = result.tightBoundingBox;
  updated.spacing = result.metadata.spacing;
  updated.depth = result.depth;
  updated.numNodes = result.numNodes;
  updated.bytesPerPoint = result.bytesPerPoint;
  updated.octreeBytes = result.octreeBytes;

  updated.attributes = sceneInfo.attributes.map((att) => {
    const spec = specs.find((s) => s.att.name === att.name);
    if (!spec) return att;
    const next = { ...att };
    if (spec.range) { next.min = spec.range[0]; next.max = spec.range[1]; }
    if (att.classes) next.classes = recountClasses(att.classes, spec.data);
    return next;
  });

  if (updated.classification) {
    const spec = specs.find((s) => s.kind === "classification");
    const counts = new Map();
    for (let i = 0; i < spec.data.length; i++) counts.set(spec.data[i], (counts.get(spec.data[i]) ?? 0) + 1);
    updated.classification = {
      ...updated.classification,
      classes: updated.classification.classes.map((c) => ({ ...c, count: counts.get(c.index) ?? 0 })),
    };
  }

  // `baked` and `detached` describe the octree as it now stands, not a history:
  // every bake rebuilds from scratch, so accumulating entries would double-count.
  updated.baked = parts.map((p) => ({
    sourceSceneId: p.sourceSceneId,
    instanceId: p.placement.instanceId,
    newInstanceId: p.newInstanceId,
    restored: p.restored,
    className: p.placement.className ?? null,
    points: p.data.count,
    pos: p.placement.pos,
    yaw: p.placement.yaw ?? 0,
    scale: p.placement.scale ?? 1,
    at: Date.now(),
  }));
  updated.bakedPoints = extra;

  const detachedIds = new Set([...(sceneInfo.detached ?? []).map((d) => Number(d.instanceId)), ...excludeNow]);
  updated.detached = [...detachedIds].map((instanceId) => ({ instanceId }));
  updated.detachedPoints = numValid - numBase;

  writeFileSync(sceneJsonPath, JSON.stringify(updated, null, 2));
  onProgress({ phase: "done", progress: 1, message: "Done" });
  return updated;
}
