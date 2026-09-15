#!/usr/bin/env node
/**
 * Converts a mapped PCD scene into a Potree 2.0 octree plus a scene.json that
 * tells the front-end how to colour it.
 *
 *   npm run convert -- <scene-id> [--force] [--grid 128] [--primary label]
 *   npm run convert -- --all
 */
import { existsSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { readCloud } from "../io/cloud.mjs";
import { describeFields } from "./fields.mjs";
import { writeOctree } from "../octree/write.mjs";
import { classColor, UNLABELED_COLOR, UNLABELED_NAMES } from "./palette.mjs";
import { extractInstances } from "../objects/instances.mjs";
import { roleMapFrom, mergeRoleMaps, describeRoles } from "./roles.mjs";
import { sceneThumbnail } from "./thumbnail.mjs";
import { cacheDirFor, resolveScene, readSidecar, listScenes, describeScene, datasetForId } from "./registry.mjs";
import { statSync } from "node:fs";

/** Potree's classification LUT has 256 entries. */
const MAX_LUT_CLASSES = 256;

/**
 * Potree normalises a wide attribute by (value - min) / (max - min), so a
 * constant field would divide by zero. Widen the stored range in that case;
 * the true min/max still goes to the UI.
 */
const safeRange = (lo, hi) => (hi > lo ? [lo, hi] : [lo, lo + 1]);

function buildClassTable(scalar, counts) {
  const provided = scalar.labels ?? null;
  return scalar.classes.map((value, index) => {
    const meta = provided ? (provided[value] ?? provided[String(value)]) : null;
    const name = meta?.name ?? `class ${value}`;
    const isUnlabeled = UNLABELED_NAMES.has(String(name).toLowerCase());
    const color = meta?.color ?? (isUnlabeled ? UNLABELED_COLOR : classColor(index));
    return { value, index, name, color, count: counts.get(value) ?? 0 };
  });
}

function histogram(data, valid) {
  const counts = new Map();
  for (let i = 0; i < data.length; i++) {
    if (valid && !valid[i]) continue;
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  return counts;
}

export function convertScene(id, options = {}) {
  const {
    force = false, gridSize = 128, primaryField = null,
    instances: extractLibrary = true, instanceField: instanceFieldOverride = null,
    onProgress = () => {},
  } = options;

  const pcdPath = resolveScene(id);
  const outDir = cacheDirFor(id);
  const sceneJson = join(outDir, "scene.json");
  const sourceModified = statSync(pcdPath).mtimeMs;

  if (!force && existsSync(sceneJson)) {
    const cached = JSON.parse(readFileSync(sceneJson, "utf8"));
    if (cached.sourceModified >= sourceModified) {
      onProgress({ phase: "cached", progress: 1 });
      return cached;
    }
  }

  onProgress({ phase: "reading", progress: 0, message: "Reading the cloud" });
  const pcd = readCloud(pcdPath);

  onProgress({ phase: "analysing", progress: 0, message: "Inspecting fields" });
  const sidecar = readSidecar(pcdPath);
  // Which field is the semantic one and which enumerates objects is declared by
  // the dataset, not guessed at here: the dataset's entry in app.json, then any
  // classes.json from the dataset root down to this scene, then whatever this
  // particular request asked for. See scene/roles.mjs.
  const roles = mergeRoleMaps(
    roleMapFrom(datasetForId(id)),
    roleMapFrom(sidecar),
    { semantic: primaryField ?? undefined, instance: instanceFieldOverride ?? undefined },
  );
  const described = describeFields(pcd, { classesSidecar: sidecar, roles });

  const { valid, numValid, position, color, scalars, primary } = described;
  if (numValid === 0) throw new Error("every point in this scene has a non-finite coordinate");

  // Indices of the points that actually make it into the octree.
  const indices = new Uint32Array(numValid);
  {
    let w = 0;
    for (let i = 0; i < pcd.numPoints; i++) if (valid[i]) indices[w++] = i;
  }

  const [px, py, pz] = position.map((f) => f.data);
  const getX = (i) => px[i];
  const getY = (i) => py[i];
  const getZ = (i) => pz[i];

  // ---- assemble the Potree attribute list ------------------------------
  const attributes = [];
  const attributeInfo = [];

  if (color) {
    const { r, g, b } = color;
    attributes.push({
      name: "rgb",                     // Potree's loader renames this to "rgba"
      type: "uint16", numElements: 3,
      description: `unpacked from ${color.source}`,
      min: [0, 0, 0], max: [255, 255, 255],
      write: (view, o, i) => {
        view.setUint16(o + 0, r[i], true);
        view.setUint16(o + 2, g[i], true);
        view.setUint16(o + 4, b[i], true);
      },
    });
    attributeInfo.push({
      name: "rgba", source: color.source, label: "RGB",
      kind: "color", potreeMode: "rgba",
    });
  }

  // The primary segmentation field is duplicated into Potree's `classification`
  // slot, which unlocks its per-class colour LUT and visibility toggles. Class
  // values are remapped to a dense 0..N-1 index so ids like 10000 still work.
  let classification = null;
  let classTable = [];
  if (primary && primary.numClasses <= MAX_LUT_CLASSES) {
    const counts = histogram(primary.data, valid);
    const table = buildClassTable(primary, counts);
    const indexOf = new Map(table.map((c) => [c.value, c.index]));
    const data = primary.data;

    attributes.push({
      name: "classification",
      type: "uint8", numElements: 1,
      description: `dense index into the classes of "${primary.source}"`,
      min: [0], max: [Math.max(0, table.length - 1)],
      // uint8 stays raw through Potree's decoder, which is what the LUT lookup needs.
      write: (view, o, i) => view.setUint8(o, indexOf.get(data[i]) ?? 0),
    });

    classification = { field: primary.name, source: primary.source, classes: table };
    classTable = table;
    attributeInfo.push({
      name: "classification", source: primary.source,
      label: primary.displayName ?? `${primary.source} (classes)`,
      kind: "classification", potreeMode: "classification",
      numClasses: table.length,
    });
  }

  for (const s of scalars) {
    const isIntensity = s.source.toLowerCase() === "intensity";
    const counts = s.kind === "categorical" ? histogram(s.data, valid) : null;
    const data = s.data;

    if (isIntensity) {
      // Potree has a dedicated intensity path (greyscale + gradient) that reads
      // the raw value, so a 4-byte type is what we want here.
      const [ilo, ihi] = safeRange(s.min, s.max);
      attributes.push({
        name: "intensity", type: "float", numElements: 1,
        description: "", min: [ilo], max: [ihi],
        write: (view, o, i) => view.setFloat32(o, data[i], true),
      });
      attributeInfo.push({
        name: "intensity", source: s.source, label: "Intensity",
        kind: "continuous", potreeMode: "intensity", min: s.min, max: s.max,
      });
      continue;
    }

    // Everything else is a generic attribute. Potree's decoder only normalises
    // attributes wider than 4 bytes, and its getExtra() colour path expects a
    // normalised value -- so these are written as double.
    const [slo, shi] = safeRange(s.min, s.max);
    attributes.push({
      name: s.name, type: "double", numElements: 1,
      description: s.kind, min: [slo], max: [shi],
      write: (view, o, i) => view.setFloat64(o, data[i], true),
    });

    attributeInfo.push({
      name: s.name, source: s.source,
      label: s.displayName ?? s.source,
      kind: s.kind, potreeMode: "extra",
      min: s.min, max: s.max,
      isPrimary: primary ? s.name === primary.name : false,
      classes: counts ? buildClassTable(s, counts) : null,
      numClasses: s.numClasses,
    });
  }

  // ---- build ------------------------------------------------------------
  onProgress({ phase: "building", progress: 0, message: "Building octree" });
  if (force && existsSync(outDir)) rmSync(outDir, { recursive: true, force: true });

  let lastReport = 0;
  const result = writeOctree({
    indices, getX, getY, getZ, attributes, outDir,
    name: id, gridSize,
    description: sidecar?.description ?? "",
    onProgress: (done, total) => {
      const now = Date.now();
      if (now - lastReport > 120) {
        lastReport = now;
        onProgress({ phase: "building", progress: done / total, message: "Building octree" });
      }
    },
  });

  // ---- instance library -------------------------------------------------
  // Objects are pulled out of the same in-memory data and reuse the very same
  // attribute writers, so a library entry colours identically to the scene.
  let library = null;
  if (extractLibrary) {
    const instanceField = described.roles.instance;
    if (instanceField) {
      onProgress({ phase: "instances", progress: 0, message: "Extracting instances" });
      library = extractInstances({
        described, indices, attributes, outDir,
        instanceField, classField: primary, classTable,
        // The scene draws its finest detail at this spacing; extracted objects
        // must match it or a placed copy renders with different point sizes.
        spacing: result.metadata.spacing / Math.pow(2, result.depth),
        onProgress: (done, totalInstances) => {
          const now = Date.now();
          if (now - lastReport > 120) {
            lastReport = now;
            onProgress({
              phase: "instances", progress: done / totalInstances,
              message: `Extracting instances (${done}/${totalInstances})`,
            });
          }
        },
      });
      writeFileSync(join(outDir, "instances.json"), JSON.stringify(library, null, 2));
    }
  }

  // ---- the card picture --------------------------------------------------
  // Drawn from the points that are already in memory, so the scene browser can
  // show what a scene looks like without loading it.
  onProgress({ phase: "thumbnail", progress: 0, message: "Drawing the thumbnail" });
  const [tx, ty, tz] = position.map((f) => f.data);
  const thumbnail = sceneThumbnail(indices, tx, ty, tz, color ? [color.r, color.g, color.b] : null);

  const scene = {
    id,
    name: id.split("/").pop(),
    source: pcd.path,
    sourceModified,
    description: sidecar?.description ?? "",
    encoding: pcd.header.data,
    numPoints: numValid,
    numPointsInFile: pcd.numPoints,
    numSkipped: pcd.numPoints - numValid,
    boundingBox: result.metadata.boundingBox,
    tightBoundingBox: result.tightBoundingBox,
    spacing: result.metadata.spacing,
    depth: result.depth,
    numNodes: result.numNodes,
    bytesPerPoint: result.bytesPerPoint,
    octreeBytes: result.octreeBytes,
    hasColor: Boolean(color),
    roles: describeRoles(described.roles),
    thumbnail,
    attributes: attributeInfo,
    classification,
    library: library
      ? { field: library.field, count: library.count, skipped: library.skipped }
      : null,
    convertedAt: Date.now(),
  };

  writeFileSync(sceneJson, JSON.stringify(scene, null, 2));
  onProgress({ phase: "done", progress: 1, message: "Done" });
  return scene;
}

// ---- CLI ----------------------------------------------------------------
if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const flag = (name, fallback = null) => {
    const i = argv.indexOf(`--${name}`);
    return i === -1 ? fallback : argv[i + 1];
  };
  const has = (name) => argv.includes(`--${name}`);

  const targets = has("all")
    ? listScenes().map((s) => s.id)
    : argv.filter((a) => !a.startsWith("--") && argv[argv.indexOf(a) - 1] !== "--grid"
        && argv[argv.indexOf(a) - 1] !== "--primary");

  if (targets.length === 0) {
    console.error("usage: npm run convert -- <scene-id> [--force] [--grid 128] [--primary <field>]");
    console.error("                            [--instance-field <field>] [--no-instances]");
    console.error("       npm run convert -- --all");
    console.error("\navailable scenes:");
    for (const s of listScenes()) console.error(`  ${s.id}  [${s.status}]`);
    process.exit(1);
  }

  for (const id of targets) {
    const t0 = Date.now();
    process.stdout.write(`converting ${id} ... `);
    try {
      const scene = convertScene(id, {
        force: has("force"),
        gridSize: Number(flag("grid", 128)),
        primaryField: flag("primary"),
        instances: !has("no-instances"),
        instanceField: flag("instance-field"),
        onProgress: ({ phase, progress }) => {
          if (phase === "building") {
            process.stdout.write(`\rconverting ${id} ... ${(progress * 100).toFixed(0)}%   `);
          }
        },
      });
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`\rconverting ${id} ... done in ${secs}s`);
      console.log(`   ${scene.numPoints.toLocaleString()} points, ${scene.numNodes} nodes, ` +
                  `depth ${scene.depth}, ${(scene.octreeBytes / 1048576).toFixed(1)} MiB`);
      console.log(`   colour by: ${scene.attributes.map((a) => a.name).join(", ")}`);
      if (scene.library) {
        console.log(`   library:   ${scene.library.count} instances from "${scene.library.field}"`);
      }
    } catch (err) {
      console.log(`\rconverting ${id} ... FAILED`);
      console.error(`   ${err.message}`);
      process.exitCode = 1;
    }
  }
}
