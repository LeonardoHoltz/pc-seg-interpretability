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
import { SCENES_DIR, CACHE_DIR } from "../paths.mjs";
import { readCloudHeader, isNpyScene, NPY_REQUIRED } from "../io/cloud.mjs";

/** Scene ids are POSIX-style relative paths without the .pcd extension. */
export const idToPosix = (id) => id.split("/").join(sep);

function walk(dir, out = []) {
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
    if (e.isDirectory()) walk(full, out);
    else if (e.isFile() && extname(e.name).toLowerCase() === ".pcd") out.push(full);
  }
  return out;
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
 * Class names and colours for a scene.
 *
 * A `classes.json` in any folder between scenes/ and the scene applies to
 * everything beneath it, so a whole dataset is described once. Closer files win,
 * and a per-scene `<scene>.classes.json` (or `classes.json` inside a directory
 * scene) wins over all of them. Per-field entries merge, so a dataset config can
 * name the classes while one scene overrides a single field.
 */
export function readSidecar(scenePath) {
  const dir = isNpyScene(scenePath) ? scenePath : join(scenePath, "..");
  const chain = [];
  for (let d = dir; d.startsWith(SCENES_DIR); d = join(d, "..")) {
    chain.unshift(join(d, "classes.json"));
    if (d === SCENES_DIR) break;
  }
  if (!isNpyScene(scenePath)) chain.push(scenePath.replace(/\.pcd$/i, ".classes.json"));

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

export function describeScene(scenePath) {
  const isDir = isNpyScene(scenePath);
  const rel = relative(SCENES_DIR, scenePath);
  const id = rel.split(sep).join("/").replace(/\.pcd$/i, "");
  const st = isDir ? directorySize(scenePath) : statSync(scenePath);

  const scene = {
    id,
    name: isDir ? basename(scenePath) : basename(scenePath, extname(scenePath)),
    folder: rel.split(sep).slice(0, isDir ? -1 : -1).join("/"),
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
  return walk(SCENES_DIR)
    .map(describeScene)
    .sort((a, b) => a.id.localeCompare(b.id));
}

export function resolveScene(id) {
  const base = join(SCENES_DIR, idToPosix(id));
  // Keep the id from escaping the scenes directory.
  if (!base.startsWith(SCENES_DIR)) throw new Error(`invalid scene id: ${id}`);

  const asPcd = `${base}.pcd`;
  if (existsSync(asPcd)) return asPcd;
  if (isNpyScene(base)) return base;
  throw new Error(`no such scene: ${id}`);
}

// CLI: `npm run scan`
if (import.meta.url === `file://${process.argv[1]}`) {
  const scenes = listScenes();
  if (scenes.length === 0) {
    console.log(`No .pcd files found under ${SCENES_DIR}`);
    console.log("Drop some in, or run `npm run demo` to generate a sample scene.");
  } else {
    console.log(`${scenes.length} scene(s) mapped under ${SCENES_DIR}:\n`);
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
