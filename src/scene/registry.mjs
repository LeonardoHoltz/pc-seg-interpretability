/**
 * Discovers the scenes mapped under scenes/.
 *
 * A scene is either a `.pcd` file or a directory of `.npy` arrays (one per
 * field). Only headers are read, never the point data, so browsing a folder of
 * multi-gigabyte scans costs nothing. A scene is "ready" once a converted
 * octree exists in the cache that is newer than the source.
 */
import { readdirSync, statSync, existsSync, readFileSync } from "node:fs";
import { join, relative, sep, extname, basename } from "node:path";
import { CACHE_DIR, CONFIG_DIR, DATASETS } from "../paths.mjs";
import { readCloudHeader, isNpyScene, NPY_REQUIRED } from "../io/cloud.mjs";

/** Scene ids are POSIX-style relative paths without the .pcd extension. */
export const idToPosix = (id) => id.split("/").join(sep);

function walk(dir, out = [], stopAt = new Set()) {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  // A directory holding the required array is itself a scene, not a folder of
  // scenes, so stop descending there.
  if (existsSync(join(dir, NPY_REQUIRED))) {
    out.push(dir);
    return out;
  }
  for (const e of entries) {
    const full = join(dir, e.name);
    // A nested dataset declares its own root; it belongs to that dataset, not
    // to whichever one happens to sit above it on disk.
    if (stopAt.has(full)) continue;
    if (e.isDirectory()) walk(full, out, stopAt);
    else if (e.isFile() && extname(e.name).toLowerCase() === ".pcd") out.push(full);
  }
  return out;
}

/** Joins an id prefix and a path within a dataset into a scene id. */
const joinId = (prefix, rel) => (prefix ? `${prefix}/${rel}` : rel);

/** The dataset a scene id belongs to: the one with the longest matching prefix. */
export function datasetForId(id) {
  let best = null;
  for (const d of DATASETS) {
    if (d.prefix && id !== d.prefix && !id.startsWith(`${d.prefix}/`)) continue;
    if (!best || d.prefix.length > best.prefix.length) best = d;
  }
  return best;
}

export function cacheDirFor(id) {
  return join(CACHE_DIR, idToPosix(id));
}

function readJsonIfPresent(path) {
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    console.warn(`[registry] ignoring malformed ${basename(path)}: ${err.message}`);
    return null;
  }
}

/**
 * The `classes.json` files that may describe a scene, under one root, in
 * increasing priority. `root` is either config/ or scenes/ -- the two are
 * searched with identical rules, config/ being a shadow tree of scenes/.
 */
function sidecarChain(root, rel, isDir) {
  const dir = join(root, isDir ? rel : join(rel, ".."));
  const chain = [];
  for (let d = dir; d.startsWith(root); d = join(d, "..")) {
    chain.unshift(join(d, "classes.json"));
    if (d === root) break;
  }
  if (!isDir) chain.push(join(root, `${rel}.classes.json`));
  return chain;
}

/**
 * Class names and colours for a scene.
 *
 * A `classes.json` in any folder between the root and the scene applies to
 * everything beneath it, so a whole dataset is described once. Closer files win,
 * and a per-scene `<scene>.classes.json` (or `classes.json` inside a directory
 * scene) wins over all of them. Per-field entries merge, so a dataset config can
 * name the classes while one scene overrides a single field.
 *
 * Both config/ and scenes/ are searched, in that order, with the same rules.
 * Checked-in dataset descriptions therefore live in config/ while scenes/ stays
 * pure bulk data -- and a file dropped next to the points still wins, which
 * keeps a one-off scene easy to annotate without touching the repo's config.
 */
export function readSidecar(scenePath, id = idForPath(scenePath)) {
  const isDir = isNpyScene(scenePath);
  const dataset = datasetForId(id) ?? DATASETS[0];
  // Within config/ a scene is addressed by its id, so a dataset keeps its
  // descriptions when its points move to another disk. Within the dataset's own
  // root it is addressed by the path, so a classes.json can sit by the points.
  const withinDataset = relative(dataset.path, isDir ? scenePath : scenePath.replace(/\.pcd$/i, ""));
  const chain = [
    ...sidecarChain(CONFIG_DIR, idToPosix(id), isDir),
    ...sidecarChain(dataset.path, withinDataset, isDir),
  ];

  let merged = null;
  for (const path of chain) {
    const cfg = readJsonIfPresent(path);
    if (!cfg) continue;
    merged = {
      ...(merged ?? {}), ...cfg,
      fields: { ...(merged?.fields ?? {}), ...(cfg.fields ?? {}) },
    };
  }
  return merged;
}

/** Total bytes and newest mtime of a directory scene's arrays. */
function directorySize(dir) {
  let bytes = 0, modified = 0;
  for (const name of readdirSync(dir)) {
    if (extname(name).toLowerCase() !== ".npy") continue;
    const st = statSync(join(dir, name));
    bytes += st.size;
    modified = Math.max(modified, st.mtimeMs);
  }
  return { bytes, modified };
}

/** The dataset whose root contains this path: the deepest one that does. */
function datasetForPath(scenePath) {
  let best = null;
  for (const d of DATASETS) {
    if (scenePath !== d.path && !scenePath.startsWith(d.path + sep)) continue;
    if (!best || d.path.length > best.path.length) best = d;
  }
  return best;
}

/** Scene id for a path on disk: the dataset's prefix plus the path inside it. */
export function idForPath(scenePath, dataset = datasetForPath(scenePath)) {
  const ds = dataset ?? DATASETS[0];
  const rel = relative(ds.path, scenePath).split(sep).join("/").replace(/\.pcd$/i, "");
  return joinId(ds.prefix, rel);
}

export function describeScene(scenePath, dataset = datasetForPath(scenePath)) {
  const isDir = isNpyScene(scenePath);
  const id = idForPath(scenePath, dataset);
  const rel = relative((dataset ?? DATASETS[0]).path, scenePath);
  const st = isDir ? directorySize(scenePath) : statSync(scenePath);

  const scene = {
    id,
    dataset: (dataset ?? DATASETS[0]).name,
    name: isDir ? basename(scenePath) : basename(scenePath, extname(scenePath)),
    folder: id.split("/").slice(0, -1).join("/"),
    file: rel.split(sep).join("/"),
    bytes: st.bytes ?? st.size,
    modified: st.modified ?? st.mtimeMs,
    status: "unconverted",
  };

  try {
    const header = readCloudHeader(scenePath);
    scene.points = header.points;
    scene.encoding = header.data;
    scene.fields = header.fields;
    scene.fieldTypes = header.fields.map((f, i) => `${header.type[i]}${header.size[i]}`);
  } catch (err) {
    scene.status = "error";
    scene.error = err.message;
    return scene;
  }

  // A cached octree counts only if it is newer than the source.
  const cached = join(cacheDirFor(id), "scene.json");
  if (existsSync(cached)) {
    try {
      const info = JSON.parse(readFileSync(cached, "utf8"));
      if (info.sourceModified >= (st.modified ?? st.mtimeMs)) {
        scene.status = "ready";
        scene.converted = info;
      } else {
        scene.status = "stale";
      }
    } catch {
      scene.status = "unconverted";
    }
  }

  return scene;
}

export function listScenes() {
  // Roots of other datasets, so a dataset nested inside another is walked once,
  // by the dataset that declares it.
  const roots = new Set(DATASETS.map((d) => d.path));
  const out = [];
  for (const dataset of DATASETS) {
    const nested = new Set([...roots].filter((r) => r !== dataset.path));
    for (const p of walk(dataset.path, [], nested)) out.push(describeScene(p, dataset));
  }
  return out.sort((a, b) => a.id.localeCompare(b.id));
}

export function resolveScene(id) {
  const dataset = datasetForId(id);
  if (!dataset) throw new Error(`no dataset owns scene id: ${id}`);

  const within = dataset.prefix ? id.slice(dataset.prefix.length + 1) : id;
  const base = join(dataset.path, idToPosix(within));
  // Keep the id from escaping its dataset root.
  if (base !== dataset.path && !base.startsWith(dataset.path + sep)) {
    throw new Error(`invalid scene id: ${id}`);
  }

  const asPcd = `${base}.pcd`;
  if (existsSync(asPcd)) return asPcd;
  if (isNpyScene(base)) return base;
  throw new Error(`no such scene: ${id}`);
}

// CLI: `npm run scan`
if (import.meta.url === `file://${process.argv[1]}`) {
  const scenes = listScenes();
  const roots = DATASETS.map((d) => `${d.name} -> ${d.path}`).join("\n    ");
  if (scenes.length === 0) {
    console.log(`No scenes found. Dataset roots:\n    ${roots}`);
    console.log("Drop some in, or run `npm run demo` to generate a sample scene.");
  } else {
    console.log(`${scenes.length} scene(s) mapped:\n    ${roots}\n`);
    for (const s of scenes) {
      const mb = (s.bytes / 1048576).toFixed(1);
      const pts = s.points ? s.points.toLocaleString() : "?";
      console.log(`  [${s.status.padEnd(11)}] ${s.id}`);
      console.log(`      ${pts} points, ${mb} MiB, ${s.encoding ?? "?"}`);
      if (s.fields) console.log(`      fields: ${s.fields.join(", ")}`);
      if (s.error) console.log(`      error: ${s.error}`);
    }
  }
}
