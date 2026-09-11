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

/** Static front-end assets. */
export const WEB_DIR = join(ROOT, "web");
