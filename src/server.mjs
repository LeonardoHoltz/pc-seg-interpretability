#!/usr/bin/env node
/**
 * HTTP server for the point cloud interpretability tool.
 *
 *   /                       front-end
 *   /api/scenes             mapped scenes (headers only, nothing loaded)
 *   /api/scenes/:id         one scene, including its converted description
 *   /api/scenes/:id/convert start a conversion (returns a job id)
 *   /api/scenes/:id/instances  the instance library extracted from one scene
 *   /api/scenes/:id/bake    merge placed objects into the scene's octree
 *   /api/scenes/:id/instances/:n/resample  rescale one object, resampling it
 *                           to the scene's point density
 *   /api/scenes/:id/segment/preview  what a segmentation request would send
 *   /api/scenes/:id/segment  send the points to an inference service and fold
 *                           the returned labels back in as an attribute
 *   /api/scenes/:id/ceteris  sweep one object through a range of heights and
 *                           report its mean class probability at each
 *   /api/scenes/:id/saliency  ask the service for a per-point scalar about one
 *                           object and add it to the scene as an attribute
 *   /api/library            every extracted instance, across all scenes
 *   /api/jobs/:id/events    server-sent conversion progress
 *   /octree/:id/...         the converted octree, served with Range support
 *
 * Range support is not optional: Potree fetches slices of octree.bin and
 * hierarchy.bin with `Range: bytes=a-b` and cannot load a cloud without it.
 */
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve, extname, normalize } from "node:path";
import { Worker } from "node:worker_threads";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { WEB_DIR, CACHE_DIR, SCENES_DIR, ROOT } from "./paths.mjs";
import { config, publicConfig } from "./config.mjs";
import { listScenes, describeScene, resolveScene, cacheDirFor } from "./scene/registry.mjs";
import { previewPayload } from "./inference/predict.mjs";
import { sceneThumbnail } from "./scene/thumbnail.mjs";
import { readOctree } from "./octree/read.mjs";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".wasm": "application/wasm",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".ttf": "font/ttf",
};

const sendJson = (res, status, body) => {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
    "cache-control": "no-store",
  });
  res.end(payload);
};

const sendError = (res, status, message) => sendJson(res, status, { error: message });

/**
 * Serves a file, honouring a single `Range: bytes=a-b` header.
 * Potree always requests one contiguous range, so multipart ranges are not
 * implemented -- an unsatisfiable or multi-range request falls back to 200.
 */
function sendFile(req, res, path, { cache = "no-cache" } = {}) {
  let st;
  try {
    st = statSync(path);
    if (!st.isFile()) return sendError(res, 404, "not found");
  } catch {
    return sendError(res, 404, "not found");
  }

  const type = MIME[extname(path).toLowerCase()] ?? "application/octet-stream";
  const range = req.headers.range;

  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (m) {
      let start = m[1] === "" ? null : Number(m[1]);
      let end = m[2] === "" ? null : Number(m[2]);

      if (start === null && end !== null) {
        // suffix range: last `end` bytes
        start = Math.max(0, st.size - end);
        end = st.size - 1;
      } else {
        if (start === null) start = 0;
        if (end === null || end >= st.size) end = st.size - 1;
      }

      if (start > end || start >= st.size) {
        res.writeHead(416, { "content-range": `bytes */${st.size}` });
        return res.end();
      }

      res.writeHead(206, {
        "content-type": type,
        "content-length": end - start + 1,
        "content-range": `bytes ${start}-${end}/${st.size}`,
        "accept-ranges": "bytes",
        "cache-control": cache,
      });
      if (req.method === "HEAD") return res.end();
      return createReadStream(path, { start, end }).pipe(res);
    }
  }

  res.writeHead(200, {
    "content-type": type,
    "content-length": st.size,
    "accept-ranges": "bytes",
    "cache-control": cache,
  });
  if (req.method === "HEAD") return res.end();
  createReadStream(path).pipe(res);
}

/** Resolves a URL path inside a root directory, refusing to escape it. */
function safeJoin(root, urlPath) {
  const decoded = decodeURIComponent(urlPath).replace(/^\/+/, "");
  const full = resolve(join(root, normalize(decoded)));
  if (full !== root && !full.startsWith(root + "/")) return null;
  return full;
}

// ---- conversion jobs ------------------------------------------------------
const jobs = new Map();   // jobId -> { id, sceneId, state, progress, message, scene, error, listeners }

/** Reads a JSON request body, with a size cap. */
function readJsonBody(req, limit = 4 << 20) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) { reject(new Error("request body too large")); req.destroy(); return; }
      chunks.push(c);
    });
    req.on("end", () => {
      if (chunks.length === 0) return resolve({});
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
      catch (err) { reject(new Error(`malformed JSON body: ${err.message}`)); }
    });
    req.on("error", reject);
  });
}

/** Spawns a worker and streams its progress to any listening EventSource. */
function startJob(sceneId, workerUrl, workerData) {
  const jobId = randomUUID();
  const job = {
    id: jobId, sceneId, state: "running", progress: 0,
    message: "Starting", scene: null, error: null, listeners: new Set(),
  };
  jobs.set(jobId, job);

  const emit = () => {
    const payload = `data: ${JSON.stringify({
      state: job.state, progress: job.progress, message: job.message,
      scene: job.scene, error: job.error,
    })}\n\n`;
    for (const res of job.listeners) res.write(payload);
    if (job.state !== "running") {
      for (const res of job.listeners) res.end();
      job.listeners.clear();
    }
  };
  job.emit = emit;

  const worker = new Worker(new URL(workerUrl, import.meta.url), { workerData });

  worker.on("message", (msg) => {
    if (msg.type === "progress") {
      job.progress = msg.progress ?? job.progress;
      job.message = msg.message ?? job.phase ?? job.message;
      emit();
    } else if (msg.type === "done") {
      job.state = "done"; job.progress = 1; job.message = "Done"; job.scene = msg.scene;
      emit();
    } else if (msg.type === "error") {
      job.state = "error"; job.error = msg.message;
      emit();
    }
  });
  worker.on("error", (err) => { job.state = "error"; job.error = err.message; emit(); });
  worker.on("exit", (code) => {
    if (job.state === "running") {
      job.state = "error";
      job.error = `worker exited with code ${code}`;
      emit();
    }
    // Keep finished jobs around briefly so a late EventSource still sees the result.
    setTimeout(() => jobs.delete(jobId), 60_000).unref();
  });

  return job;
}

const startConversion = (sceneId, opts) =>
  startJob(sceneId, "./workers/convert.mjs", { id: sceneId, ...opts });

const startBake = (sceneId, placements, exclude = []) =>
  startJob(sceneId, "./workers/bake.mjs", { id: sceneId, placements, exclude });

const startResample = (sceneId, workerData) =>
  startJob(sceneId, "./workers/resample.mjs", workerData);

const startSegmentation = (sceneId, endpoint, fields) =>
  startJob(sceneId, "./workers/predict.mjs", { id: sceneId, endpoint, fields });

const startCeteris = (sceneId, workerData) =>
  startJob(sceneId, "./workers/ceteris.mjs", { id: sceneId, ...workerData });

const startSaliency = (sceneId, workerData) =>
  startJob(sceneId, "./workers/saliency.mjs", { id: sceneId, ...workerData });

// ---- routing --------------------------------------------------------------
function handle(req, res) {
  const url = new URL(req.url, `http://${req.headers.host ?? "localhost"}`);
  const path = url.pathname;

  // --- API ---
  // Settings the front end needs, so nothing is duplicated as a literal in JS.
  if (path === "/api/config" && req.method === "GET") {
    return sendJson(res, 200, publicConfig());
  }

  if (path === "/api/scenes" && req.method === "GET") {
    return sendJson(res, 200, {
      scenesDir: SCENES_DIR,
      datasets: config.datasets.map((d) => ({ name: d.name, prefix: d.prefix, description: d.description })),
      scenes: listScenes(),
    });
  }

  let m = /^\/api\/scenes\/(.+)\/convert$/.exec(path);
  if (m && req.method === "POST") {
    const id = decodeURIComponent(m[1]);
    try {
      resolveScene(id);
    } catch (err) {
      return sendError(res, 404, err.message);
    }
    const job = startConversion(id, {
      force: url.searchParams.get("force") === "1",
      gridSize: Number(url.searchParams.get("grid") ?? config.conversion.gridSize),
      primaryField: url.searchParams.get("primary") ?? null,
      instances: url.searchParams.get("instances") !== "0" && config.conversion.instances,
      instanceField: url.searchParams.get("instanceField") ?? null,
    });
    return sendJson(res, 202, { jobId: job.id });
  }

  // The scene browser's card picture. Written during conversion; a cache built
  // before thumbnails existed gets one computed from its octree, once.
  m = /^\/api\/scenes\/(.+)\/thumbnail$/.exec(path);
  if (m && req.method === "GET") {
    const id = decodeURIComponent(m[1]);
    const file = join(cacheDirFor(id), "scene.json");
    if (!existsSync(file)) return sendError(res, 404, "not converted");
    let scene;
    try { scene = JSON.parse(readFileSync(file, "utf8")); } catch { return sendError(res, 404, "unreadable"); }

    if (!scene.thumbnail) {
      try {
        const cloud = readOctree(cacheDirFor(id));
        const indices = new Uint32Array(cloud.count);
        for (let i = 0; i < indices.length; i++) indices[i] = i;

        // The octree stores colour interleaved; the renderer wants a plane each.
        const packed = cloud.columns.get("rgb");
        let rgb = null;
        if (packed?.numElements === 3) {
          rgb = [0, 1, 2].map(() => new Uint8Array(cloud.count));
          for (let i = 0; i < cloud.count; i++) {
            for (let c = 0; c < 3; c++) rgb[c][i] = packed.data[i * 3 + c];
          }
        }

        scene.thumbnail = sceneThumbnail(indices, cloud.x, cloud.y, cloud.z, rgb);
        if (scene.thumbnail) writeFileSync(file, JSON.stringify(scene, null, 2));
      } catch (err) {
        return sendError(res, 404, `no thumbnail: ${err.message}`);
      }
    }
    if (!scene.thumbnail) {
      return sendError(res, 404, "this scene has no points to draw");
    }
    // Immutable for the life of this conversion: the cache directory is rewritten
    // wholesale when a scene is reconverted.
    res.setHeader("cache-control", "private, max-age=86400");
    return sendJson(res, 200, scene.thumbnail ?? null);
  }

  m = /^\/api\/jobs\/([^/]+)\/events$/.exec(path);
  if (m && req.method === "GET") {
    const job = jobs.get(m[1]);
    if (!job) return sendError(res, 404, "no such job");

    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    res.write(`data: ${JSON.stringify({
      state: job.state, progress: job.progress, message: job.message,
      scene: job.scene, error: job.error,
    })}\n\n`);

    if (job.state === "running") {
      job.listeners.add(res);
      req.on("close", () => job.listeners.delete(res));
    } else {
      res.end();
    }
    return;
  }

  // Every instance the cache holds, so the library can be grouped by class
  // across scenes rather than only within one.
  if (path === "/api/library" && req.method === "GET") {
    const out = [];
    const classes = new Map();

    for (const scene of listScenes()) {
      if (scene.status !== "ready") continue;
      const file = join(cacheDirFor(scene.id), "instances.json");
      if (!existsSync(file)) continue;
      let lib;
      try { lib = JSON.parse(readFileSync(file, "utf8")); } catch { continue; }

      // Instances lifted out of the scene are still placeable, but they are no
      // longer sitting in it, so the viewer must not offer to inspect them.
      // An id that was detached and later merged back in is present again.
      const detached = new Set((scene.converted?.detached ?? []).map((d) => d.instanceId));
      for (const b of scene.converted?.baked ?? []) detached.delete(b.newInstanceId);

      for (const inst of lib.instances) {
        const className = inst.class?.name ?? "unclassified";
        const entry = {
          key: `${scene.id}#${inst.id}`,
          sceneId: scene.id,
          sceneName: scene.name,
          id: inst.id,
          // Backdrops have no instance octree, so there is nothing to spawn.
          url: inst.dir
            ? `/octree/${scene.id.split("/").map(encodeURIComponent).join("/")}/${inst.dir}/metadata.json`
            : null,
          background: Boolean(inst.background),
          points: inst.points,
          size: inst.size,
          anchor: inst.anchor,
          anchorLocal: inst.anchorLocal,
          footprint: inst.footprint ?? null,
          silhouette: inst.silhouette,
          className,
          classColor: inst.class?.color ?? [150, 150, 150],
          classValue: inst.class?.value ?? null,
          detached: detached.has(inst.id),
        };
        out.push(entry);

        if (!classes.has(className)) {
          classes.set(className, { name: className, color: entry.classColor, count: 0 });
        }
        classes.get(className).count++;
      }
    }

    return sendJson(res, 200, {
      count: out.length,
      silhouetteSize: 32,
      classes: [...classes.values()].sort((a, b) => b.count - a.count),
      instances: out,
    });
  }

  // What a segmentation request would send, without sending it.
  m = /^\/api\/scenes\/(.+)\/segment\/preview$/.exec(path);
  if (m && req.method === "GET") {
    const id = decodeURIComponent(m[1]);
    try {
      return sendJson(res, 200, previewPayload(id));
    } catch (err) {
      return sendError(res, 400, err.message);
    }
  }

  // Send the scene's points to an inference service and fold the labels back in.
  m = /^\/api\/scenes\/(.+)\/segment$/.exec(path);
  if (m && req.method === "POST") {
    const id = decodeURIComponent(m[1]);
    if (!existsSync(join(cacheDirFor(id), "scene.json"))) {
      return sendError(res, 404, "convert this scene before segmenting it");
    }
    return readJsonBody(req).then((body) => {
      const endpoint = String(body.endpoint ?? "").trim();
      if (!/^https?:\/\//i.test(endpoint)) {
        return sendError(res, 400, "endpoint must be an http(s) URL");
      }
      const job = startSegmentation(id, endpoint, Array.isArray(body.fields) ? body.fields : null);
      sendJson(res, 202, { jobId: job.id });
    }).catch((err) => sendError(res, 400, err.message));
  }

  // Per-point saliency for one object, added to the scene as a scalar field.
  m = /^\/api\/scenes\/(.+)\/saliency$/.exec(path);
  if (m && req.method === "POST") {
    const id = decodeURIComponent(m[1]);
    if (!existsSync(join(cacheDirFor(id), "scene.json"))) {
      return sendError(res, 404, "convert this scene before analysing it");
    }
    return readJsonBody(req).then((body) => {
      const endpoint = String(body.endpoint ?? "").trim();
      if (!/^https?:\/\//i.test(endpoint)) {
        return sendError(res, 400, "endpoint must be an http(s) URL");
      }
      const instanceId = Number(body.instanceId);
      if (!Number.isFinite(instanceId)) return sendError(res, 400, "instanceId is required");
      const classValue = Number.isFinite(Number(body.classValue)) ? Number(body.classValue) : null;

      const job = startSaliency(id, {
        endpoint, instanceId, classValue,
        fields: Array.isArray(body.fields) ? body.fields : null,
      });
      sendJson(res, 202, { jobId: job.id });
    }).catch((err) => sendError(res, 400, err.message));
  }

  // Move one object through a range of heights, segmenting the scene at each.
  m = /^\/api\/scenes\/(.+)\/ceteris$/.exec(path);
  if (m && req.method === "POST") {
    const id = decodeURIComponent(m[1]);
    if (!existsSync(join(cacheDirFor(id), "scene.json"))) {
      return sendError(res, 404, "convert this scene before analysing it");
    }
    return readJsonBody(req).then((body) => {
      const endpoint = String(body.endpoint ?? "").trim();
      if (!/^https?:\/\//i.test(endpoint)) {
        return sendError(res, 400, "endpoint must be an http(s) URL");
      }
      const instanceId = Number(body.instanceId);
      const classValue = Number(body.classValue);
      if (!Number.isFinite(instanceId)) return sendError(res, 400, "instanceId is required");
      if (!Number.isFinite(classValue)) return sendError(res, 400, "classValue is required");

      let heights = body.heights;
      if (!Array.isArray(heights)) {
        const { min, max, steps } = body.range ?? {};
        if (![min, max, steps].every(Number.isFinite) || steps < 2) {
          return sendError(res, 400, "give either heights[] or range {min, max, steps>=2}");
        }
        heights = Array.from({ length: steps },
          (_, i) => min + ((max - min) * i) / (steps - 1));
      }

      const job = startCeteris(id, {
        endpoint, instanceId, classValue, heights,
        fields: Array.isArray(body.fields) ? body.fields : null,
      });
      sendJson(res, 202, { jobId: job.id, heights });
    }).catch((err) => sendError(res, 400, err.message));
  }

  // Rescale one library object, resampling it so its point density still
  // matches the scene it is being placed into.
  m = /^\/api\/scenes\/(.+)\/instances\/(\d+)\/resample$/.exec(path);
  if (m && req.method === "POST") {
    const sceneId = decodeURIComponent(m[1]);
    const instanceId = m[2];
    const scale = Number(url.searchParams.get("scale") ?? 1);
    if (!Number.isFinite(scale) || scale <= 0 || scale > 20) {
      return sendError(res, 400, "scale must be between 0 and 20");
    }

    const sceneDir = cacheDirFor(sceneId);
    const srcDir = join(sceneDir, "instances", instanceId);
    if (!existsSync(srcDir)) return sendError(res, 404, "no such object");

    // The density to aim for belongs to the scene being *viewed*, which the
    // client knows and passes in; fall back to the object's own scene.
    let targetSpacing = Number(url.searchParams.get("spacing"));
    if (!Number.isFinite(targetSpacing) || targetSpacing <= 0) {
      const meta = JSON.parse(readFileSync(join(sceneDir, "metadata.json"), "utf8"));
      targetSpacing = meta.spacing / Math.pow(2, meta.hierarchy.depth);
    }

    const key = `${scale.toFixed(2)}_${targetSpacing.toExponential(3)}`.replace(/[.+]/g, "_");
    const relDir = `instances/${instanceId}/scaled/${key}`;
    const outDir = join(sceneDir, relDir);
    const urlPath = `/octree/${sceneId.split("/").map(encodeURIComponent).join("/")}/${relDir}/metadata.json`;

    if (existsSync(join(outDir, "metadata.json"))) {
      const meta = JSON.parse(readFileSync(join(outDir, "metadata.json"), "utf8"));
      const bb = meta.boundingBox;
      return sendJson(res, 200, {
        cached: true, url: urlPath, dir: relDir, scale,
        points: meta.points,
        anchorLocal: [-bb.min[0], -bb.min[1], -bb.min[2]],
      });
    }

    const job = startResample(sceneId, { dir: srcDir, outDir, scale, targetSpacing });
    job.resultUrl = urlPath;
    job.resultDir = relDir;
    return sendJson(res, 202, { jobId: job.id, url: urlPath, dir: relDir, scale });
  }

  m = /^\/api\/scenes\/(.+)\/bake$/.exec(path);
  if (m && req.method === "POST") {
    const id = decodeURIComponent(m[1]);
    try {
      resolveScene(id);
    } catch (err) {
      return sendError(res, 404, err.message);
    }
    return readJsonBody(req).then((body) => {
      const placements = Array.isArray(body.placements) ? body.placements : [];
      const exclude = Array.isArray(body.exclude) ? body.exclude : [];
      if (placements.length === 0 && exclude.length === 0) {
        return sendError(res, 400, "nothing to add or remove");
      }
      const job = startBake(id, placements, exclude);
      sendJson(res, 202, { jobId: job.id });
    }).catch((err) => sendError(res, 400, err.message));
  }

  m = /^\/api\/scenes\/(.+)\/instances$/.exec(path);
  if (m && req.method === "GET") {
    const id = decodeURIComponent(m[1]);
    const file = join(cacheDirFor(id), "instances.json");
    if (!existsSync(file)) return sendError(res, 404, "this scene has no instance library yet");
    return sendFile(req, res, file);
  }

  m = /^\/api\/scenes\/(.+)$/.exec(path);
  if (m && req.method === "GET") {
    const id = decodeURIComponent(m[1]);
    let pcdPath;
    try {
      pcdPath = resolveScene(id);
    } catch (err) {
      return sendError(res, 404, err.message);
    }
    const scene = describeScene(pcdPath);
    const sceneJson = join(cacheDirFor(id), "scene.json");
    if (existsSync(sceneJson)) {
      try { scene.converted = JSON.parse(readFileSync(sceneJson, "utf8")); } catch { /* ignore */ }
    }
    return sendJson(res, 200, scene);
  }

  // --- converted octrees ---
  if (path.startsWith("/octree/")) {
    const target = safeJoin(CACHE_DIR, path.slice("/octree".length));
    if (!target) return sendError(res, 403, "forbidden");
    // Octree payloads are immutable for a given conversion; a short cache is safe.
    return sendFile(req, res, target, { cache: "no-cache" });
  }

  // --- static front-end ---
  if (req.method === "GET" || req.method === "HEAD") {
    const rel = path === "/" ? "/index.html" : path;
    const target = safeJoin(WEB_DIR, rel);
    if (!target) return sendError(res, 403, "forbidden");
    if (existsSync(target) && statSync(target).isDirectory()) {
      return sendFile(req, res, join(target, "index.html"));
    }
    return sendFile(req, res, target);
  }

  sendError(res, 404, "not found");
}

const server = createServer((req, res) => {
  try {
    handle(req, res);
  } catch (err) {
    console.error(err);
    if (!res.headersSent) sendError(res, 500, err.message);
    else res.end();
  }
});

if (import.meta.url === `file://${process.argv[1]}`) {
  const argv = process.argv.slice(2);
  const flag = (n, d) => { const i = argv.indexOf(`--${n}`); return i === -1 ? d : argv[i + 1]; };
  // Flags beat the environment, which beats config/app.json (see config.mjs).
  const port = Number(flag("port", config.server.port));
  const host = flag("host", config.server.host);

  if (!existsSync(join(WEB_DIR, "vendor", "potree", "potree.js"))) {
    console.warn("!! Potree is not vendored yet -- run `npm run build-viewer` first.\n");
  }

  server.listen(port, host, () => {
    console.log(`point cloud interpretability tool`);
    console.log(`  serving   http://${host}:${port}`);
    console.log(`  config    ${config.sourceFile ?? "(defaults -- no config/app.json)"}`);
    console.log(`  cache     ${CACHE_DIR}`);
    for (const d of config.datasets) console.log(`  dataset   ${d.name.padEnd(16)} ${d.path}`);
    const n = listScenes().length;
    console.log(`  ${n} scene(s) mapped` + (n === 0 ? " -- run `npm run demo` for a sample" : ""));
  });
}

export { server, handle };
