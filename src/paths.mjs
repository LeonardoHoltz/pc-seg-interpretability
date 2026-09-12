/**
 * Directories the application works with.
 *
 * These are now views onto `config/app.json` (see config.mjs for the file
 * format and the flag/env/file precedence). They stay exported as constants
 * because most of the code only ever needs the resolved directory.
 */
import { join } from "node:path";
import { config, ROOT } from "./config.mjs";

export { ROOT };

/**
 * Where scenes are *mapped* from. The registry only reads PCD headers here, so
 * listing a folder of large scans stays instant -- nothing is loaded until a
 * scene is actually opened. Datasets may declare roots of their own outside
 * this one; see config.datasets.
 */
export const SCENES_DIR = config.paths.scenes;

/** Converted Potree octrees, one subdirectory per scene. */
export const CACHE_DIR = config.paths.cache;

/**
 * Checked-in dataset descriptions: class names and colours, keyed by scene id,
 * plus app.json itself. Scenes are bulk data and stay untracked; what a label
 * *means* is part of the repo, so it lives here instead of next to the points.
 */
export const CONFIG_DIR = config.paths.config;

/** Declared dataset roots, each an id namespace. */
export const DATASETS = config.datasets;

/** Static front-end assets. */
export const WEB_DIR = join(ROOT, "web");
