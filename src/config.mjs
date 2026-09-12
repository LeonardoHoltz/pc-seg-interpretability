/**
 * Application configuration.
 *
 * One file, `config/app.json`, holds the things that were previously scattered
 * as literals: where the server listens, where datasets live, the default
 * inference endpoint, conversion defaults and request timeouts.
 *
 * Precedence, highest first:
 *
 *   1. CLI flag        --port 9000
 *   2. environment     PCIT_PORT=9000
 *   3. config/app.json
 *   4. the defaults below
 *
 * so an operator can override one value without editing the file, and the file
 * itself never has to list everything -- anything omitted falls back to DEFAULTS.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const MINUTE = 60 * 1000;

/** Every setting the app understands, with the value it had when hardcoded. */
const DEFAULTS = {
  server: {
    host: "127.0.0.1",
    port: 8080,
  },
  paths: {
    scenes: "scenes",
    cache: "cache",
    config: "config",
  },
  datasets: {},
  inference: {
    // Shown in the Segmentation tab when the browser has no saved endpoint.
    endpoint: "http://127.0.0.1:8500/",
    timeouts: {
      predict: 15 * MINUTE,
      saliency: 30 * MINUTE,
      ceteris: 30 * MINUTE,
    },
  },
  conversion: {
    gridSize: 128,
    // Cut every object out into its own small octree while converting.
    instances: true,
  },
};

/** Deep-merges plain objects; arrays and scalars replace wholesale. */
function merge(base, override) {
  if (!isPlainObject(base) || !isPlainObject(override)) return override ?? base;
  const out = { ...base };
  for (const [k, v] of Object.entries(override)) out[k] = merge(base[k], v);
  return out;
}

const isPlainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

function readConfigFile(path) {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    throw new Error(`${path} is not valid JSON: ${err.message}`);
  }
}

const num = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};

/** Environment overrides, as a sparse tree merged over the file. */
function fromEnv(env) {
  const out = {};
  const set = (path, value) => {
    if (value === undefined) return;
    let node = out;
    const keys = path.split(".");
    for (const k of keys.slice(0, -1)) node = node[k] ??= {};
    node[keys.at(-1)] = value;
  };
  set("server.host", env.PCIT_HOST ?? env.HOST);
  set("server.port", num(env.PCIT_PORT ?? env.PORT));
  set("paths.scenes", env.PCIT_SCENES_DIR);
  set("paths.cache", env.PCIT_CACHE_DIR);
  set("paths.config", env.PCIT_CONFIG_DIR);
  set("inference.endpoint", env.PCIT_INFERENCE_ENDPOINT);
  set("conversion.gridSize", num(env.PCIT_GRID_SIZE));
  return out;
}

/** Resolves a configured path against the repo root unless already absolute. */
const resolvePath = (p) => (isAbsolute(p) ? p : resolve(ROOT, p));

/**
 * Datasets are declared roots. Each becomes an id namespace: a scene's id is
 * `<prefix>/<path within the root>`, so a dataset can be moved to another disk
 * without any id -- or any cache entry -- changing.
 *
 * With no `datasets` block the whole scenes directory is one unnamed dataset,
 * which is exactly the old behaviour.
 */
function normalizeDatasets(cfg) {
  const scenesDir = resolvePath(cfg.paths.scenes);
  const entries = Object.entries(cfg.datasets ?? {});
  if (entries.length === 0) {
    return [{ name: "scenes", path: scenesDir, prefix: "", primaryField: null, description: null }];
  }
  return entries.map(([name, raw]) => {
    const spec = typeof raw === "string" ? { path: raw } : (raw ?? {});
    if (!spec.path) throw new Error(`dataset "${name}" has no path`);
    return {
      name,
      path: resolvePath(spec.path),
      // The id namespace. Defaults to the dataset name; "" puts the dataset's
      // scenes at the top level, which is how the bundled demo scenes keep the
      // bare ids they have always had.
      prefix: (spec.prefix ?? name).replace(/^\/+|\/+$/g, ""),
      primaryField: spec.primaryField ?? null,
      description: spec.description ?? null,
    };
  });
}

export function loadConfig({ env = process.env, file } = {}) {
  const configDir = resolvePath(env.PCIT_CONFIG_DIR ?? DEFAULTS.paths.config);
  const path = file ?? join(configDir, "app.json");

  let cfg = merge(DEFAULTS, readConfigFile(path));
  cfg = merge(cfg, fromEnv(env));

  cfg.paths = {
    scenes: resolvePath(cfg.paths.scenes),
    cache: resolvePath(cfg.paths.cache),
    config: resolvePath(cfg.paths.config),
  };
  cfg.datasets = normalizeDatasets(cfg);
  cfg.sourceFile = existsSync(path) ? path : null;

  const dupes = cfg.datasets.map((d) => d.prefix).filter((p, i, a) => a.indexOf(p) !== i);
  if (dupes.length) throw new Error(`datasets share an id prefix: ${[...new Set(dupes)].join(", ")}`);

  return cfg;
}

/** The loaded configuration. One read per process. */
export const config = loadConfig();

/** The subset the browser needs; served by /api/config. */
export const publicConfig = () => ({
  inference: {
    endpoint: config.inference.endpoint,
  },
  datasets: config.datasets.map((d) => ({
    name: d.name, prefix: d.prefix, description: d.description,
  })),
});
