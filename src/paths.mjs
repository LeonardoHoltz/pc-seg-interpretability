/** Directories the application works with. */
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const envDir = (name, fallback) =>
  process.env[name] ? resolve(process.env[name]) : fallback;

/**
 * Where scenes are *mapped* from. The registry only reads PCD headers here, so
 * listing a folder of large scans stays instant -- nothing is loaded until a
 * scene is actually opened.
 */
export const SCENES_DIR = envDir("PCIT_SCENES_DIR", join(ROOT, "scenes"));

/** Converted Potree octrees, one subdirectory per scene. */
export const CACHE_DIR = envDir("PCIT_CACHE_DIR", join(ROOT, "cache"));

/**
 * Checked-in dataset descriptions: class names and colours, as a shadow tree of
 * SCENES_DIR. Scenes are bulk data and stay untracked; what a label *means* is
 * part of the repo, so it lives here instead of next to the points.
 */
export const CONFIG_DIR = envDir("PCIT_CONFIG_DIR", join(ROOT, "config"));

/** Static front-end assets. */
export const WEB_DIR = join(ROOT, "web");
