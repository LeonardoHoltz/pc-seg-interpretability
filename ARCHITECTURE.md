# Architecture

How this software is put together, and why. For what it *does*, see the
[README](README.md).

---

## Layout

```
src/
  server.mjs      the API, SSE progress, Range-capable static files
  paths.mjs       where scenes and the cache live

  io/             getting points off disk
    pcd.mjs         PCD reader (ascii / binary / binary_compressed)
    lzf.mjs         LZF decompressor, for binary_compressed
    npy.mjs         NumPy .npy reader
    cloud.mjs       picks the reader; one shape out for either format

  octree/         the Potree 2.0 format itself
    write.mjs       builds and writes an octree
    read.mjs        reads one back into flat arrays

  scene/          a scene as the app understands it
    registry.mjs    discovery, scene ids, class configs
    fields.mjs      sorts raw fields into position / colour / scalar / categorical
    palette.mjs     colours for classes
    convert.mjs     source cloud -> octree + scene.json  (also the CLI)
    attributes.mjs  adds scalar fields to an already-converted scene

  objects/        the object library
    instances.mjs   cuts objects out of a segmented scene
    resample.mjs    rescales one object to a target point density
    bake.mjs        merges placed objects back in, and detaches them

  inference/      talking to a segmentation service
    npbuffer.mjs    the binary wire format
    predict.mjs     segment the scene, fold labels back in
    saliency.mjs    per-point saliency for one object
    ceteris.mjs     sweep one object through heights, in one request

  workers/        thin worker_thread entry points, one per long job
    convert.mjs  bake.mjs  resample.mjs
    predict.mjs  saliency.mjs  ceteris.mjs

web/              front-end (vendor/potree is generated, not committed)
  index.html      the shell: top tab bar, three panels
  css/app.css     all styling
  js/app.js       viewer, scene browser, colouring, rendering controls
  js/instances.js object library, placement, transforms, inspect and detach
  js/segmentation.js  the segmentation and interpretability tab
  js/util.js      DOM helpers, thumbnails, modal dialogs

scenes/           your scenes: .pcd files, or folders of .npy arrays
cache/            generated octrees, one folder per scene
tools/            viewer build, demo scenes, octree verifier
examples/         reference segmentation service (documents the wire format)
```

Folders group modules by **what they are responsible for**, not by kind. Three
files were renamed when the folder made their old prefix redundant:
`octree.mjs` → `octree/write.mjs`, `octree_read.mjs` → `octree/read.mjs`,
`scene_attribute.mjs` → `scene/attributes.mjs`.

---

## Entry points

There are only two things meant to be run directly:

| | |
| --- | --- |
| `src/server.mjs` | serves the front-end and the API (`npm run serve`) |
| `src/scene/convert.mjs` | doubles as the conversion CLI (`npm run convert`) |

`src/scene/registry.mjs` also runs standalone to list scenes (`npm run scan`),
and the files in `src/workers/` are spawned by Node, never invoked by hand.

---

## The conversion pipeline

Everything starts here. A scene is converted once, and the result is what the
rest of the software works with.

```
io/                 scene/fields         octree/write        cache/<scene>/
read a cloud   ->   sort the fields  ->  write the      ->   metadata.json
(.pcd file or       into meaning         octree              hierarchy.bin
 .npy folder)                                                octree.bin
                                                             scene.json
                                                             instances/<id>/…
```

**`io/` hides the file format.** `cloud.mjs` chooses a reader and returns one
shape, so nothing downstream knows or cares whether the points came from a PCD
or a directory of `.npy` arrays. Adding a third format means adding a reader
here and nothing else.

**`scene/fields.mjs` is where raw columns become meaning.** It decides which
fields are position, which is colour, which are continuous scalars, which are
categorical labels, and which single categorical field becomes *the* class
field. Nearly every affordance in the UI — the legend, per-class visibility, the
object library, the interpretability tools — follows from that one
classification.

**`octree/` owns the Potree format and nothing else.** `write.mjs` builds one;
`read.mjs` reads it back. That second half is what makes everything after
conversion possible.

**`scene/convert.mjs` orchestrates**: read, classify, write the octree, cut out
the objects, and write `scene.json` — the manifest the front-end reads to know
what a scene contains.

---

## Everything after conversion reads the octree back

This is the load-bearing idea. Once a scene is converted, the source file is
barely touched again; the octree is both the render format *and* the working
store.

- **`objects/`** — `instances.mjs` cuts each object into its own small octree at
  conversion time. `resample.mjs` rebuilds one at a new size with the scene's
  point density. `bake.mjs` merges placed objects into the scene and detaches
  them again.
- **`inference/`** — `npbuffer.mjs` defines the binary wire format;
  `predict.mjs`, `saliency.mjs` and `ceteris.mjs` each read the octree, ship the
  points to a service, and fold the results back in.
- **`scene/attributes.mjs`** is the shared "add a scalar column to a converted
  scene" step used by both `predict` and `saliency`. It reuses the octree's
  existing quantisation grid, so coordinates come out bit-identical on every
  rewrite instead of drifting a rounding step each time.

---

## Long jobs run in workers

Building an octree is CPU-bound and would block the single-threaded server for
the whole job — and while it was blocked, no progress could reach the browser.

So every long job runs in a `worker_thread`. `server.mjs` has one generic
`startJob(sceneId, workerUrl, workerData)` that spawns a worker, tracks the job,
and streams its progress to any listening `EventSource`:

```
POST /api/scenes/:id/<job>   ->  { jobId }
GET  /api/jobs/:jobId/events ->  SSE: progress … progress … done | error
```

The six files in `src/workers/` are near-identical adapters: unpack
`workerData`, call the real function, post progress and the result. The logic
lives in `scene/`, `objects/` and `inference/`, where it can also be called
directly — which is how the CLI and the tests use it.

---

## The front-end

`web/js/app.js` owns the Potree viewer and `styleMaterial()`, the single
definition of how *any* point cloud is drawn — colour mode, ramp, value ranges,
point size, shape, opacity. It is applied to the scene and to every placed
object together, which is what keeps a spawned copy looking identical to the
scene it came from.

`instances.js` and `segmentation.js` are feature modules. They own their panels
and their state, and reach the viewer only through callbacks handed to them by
`app.js` — so neither one manipulates the viewer directly.

The server never renders anything. It converts, stores, proxies inference, and
serves bytes; all rendering is Potree in the browser.

---

## Two honest notes

**No import cycles** among the 26 modules. At *folder* level, though, `scene`
and `objects` do depend on each other: conversion produces the object library,
while `bake` needs scene-level helpers (`fields`, `registry`, `palette`). That
is a real mutual relationship rather than an accident of layering, and no
individual file takes part in a cycle.

**`paths.mjs` stayed at the root** instead of getting a `config/` folder of its
own. It is one small module that everything shares, and a folder holding a
single file is worse than no folder at all.
