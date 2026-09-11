#!/usr/bin/env node
/**
 * Builds the vendored Potree bundle used by the front-end.
 *
 * Potree is not published as a usable npm package, so we clone the upstream
 * repository, run its own gulp build, and copy only the runtime pieces we need
 * into web/vendor/. Everything the browser loads is served from there, so the
 * app has no CDN dependency at runtime.
 *
 *   pixi run build-viewer
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const WORK = join(ROOT, ".potree-build");
const SRC = join(WORK, "potree");
const OUT = join(ROOT, "web", "vendor", "potree");

const POTREE_REPO = "https://github.com/potree/potree.git";

const run = (cmd, args, cwd) =>
  execFileSync(cmd, args, { cwd, stdio: "inherit", env: process.env });

function fetchSource() {
  if (existsSync(join(SRC, "package.json"))) {
    console.log(`[build-viewer] reusing existing checkout at ${SRC}`);
    return;
  }
  mkdirSync(WORK, { recursive: true });
  console.log("[build-viewer] cloning Potree ...");
  run("git", ["clone", "--depth", "1", POTREE_REPO, SRC], WORK);
}

function buildSource() {
  if (existsSync(join(SRC, "build", "potree", "potree.js"))) {
    console.log("[build-viewer] reusing existing Potree build");
    return;
  }
  console.log("[build-viewer] npm install + gulp build (this takes a minute) ...");
  run("npm", ["install", "--no-audit", "--no-fund"], SRC);
}

// [source relative to the Potree checkout, destination relative to web/vendor/potree]
const ASSETS = [
  ["build/potree/potree.js", "potree.js"],
  ["build/potree/potree.css", "potree.css"],
  ["build/potree/LICENSE", "LICENSE"],
  ["build/potree/workers", "workers"],
  ["build/potree/resources", "resources"],
  ["build/potree/lazylibs", "lazylibs"],
  ["libs/jquery/jquery-3.1.1.min.js", "libs/jquery.js"],
  ["libs/jquery-ui/jquery-ui.min.js", "libs/jquery-ui.js"],
  ["libs/jquery-ui/jquery-ui.min.css", "libs/jquery-ui.css"],
  ["libs/spectrum/spectrum.js", "libs/spectrum.js"],
  ["libs/spectrum/spectrum.css", "libs/spectrum.css"],
  ["libs/other/BinaryHeap.js", "libs/BinaryHeap.js"],
  ["libs/tween/tween.min.js", "libs/tween.js"],
  ["libs/d3/d3.js", "libs/d3.js"],
  ["libs/proj4/proj4.js", "libs/proj4.js"],
  ["libs/i18next/i18next.js", "libs/i18next.js"],
];

function vendor() {
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(join(OUT, "libs"), { recursive: true });
  for (const [from, to] of ASSETS) {
    const src = join(SRC, from);
    if (!existsSync(src)) {
      console.warn(`[build-viewer] skipping missing ${from}`);
      continue;
    }
    const dst = join(OUT, to);
    mkdirSync(dirname(dst), { recursive: true });
    cpSync(src, dst, { recursive: true });
  }
  console.log(`[build-viewer] vendored Potree into ${OUT}`);
}

fetchSource();
buildSource();
vendor();
