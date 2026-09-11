# Point Cloud Interpretability Tool

A [Potree](https://github.com/potree/potree) viewer for point clouds, built for looking at
*scalar fields* — especially **segmentation** ones. Reads **`.pcd` files** and
**directories of `.npy` arrays** (the shape Pointcept exports ScanNet and friends in).

Scenes live in `scenes/`. They are **mapped, not loaded**: the browser only reads headers,
so a folder of multi-gigabyte scans lists instantly. Picking a scene converts it to a
Potree 2.0 octree once (cached in `cache/`) and streams it into the viewer.

Node-only. No Python, and no runtime npm dependencies — the server is plain `node:http`.

For how the code is organised and why, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick start

```bash
npm run build-viewer     # clone + build Potree, vendor it into web/vendor/  (once)
npm run demo             # generate a synthetic segmented street scene
npm run serve            # http://127.0.0.1:8080
```

`npm run setup` does the first two together.

The window is a full-width bar carrying the three tabs — **Scenes**, **Object library** and
**Segmentation** — with that tab's options in the left panel, the cloud in the middle, and
colouring and rendering on the right. Either side panel collapses (☰ and ⚙ in the
viewport) to give the cloud the whole width.

Then drop your own scenes into `scenes/` — `.pcd` files or folders of `.npy` arrays,
subfolders are fine — and hit ⟳ in the scene browser.

---

## Colouring

The converter inspects every field and sorts it into one of three buckets:

| Bucket | Detected from | How you colour by it |
| --- | --- | --- |
| **RGB** | `rgb` / `rgba` packed float, or `r`,`g`,`b` | true colour |
| **Continuous scalar** | any float field (`intensity`, `confidence`, `curvature`, …) | colour ramp + min/max range slider |
| **Categorical / segmentation** | integer fields named `label`, `classification`, `semantic`, `instance`, `seg`, … — or any integer field with few distinct values | flat colour bands + class legend |

### Segmentation fields

One categorical field becomes the scene's **class field** and is written into Potree's
`classification` slot. That unlocks the good stuff:

- a legend with class names, colours and point percentages,
- **per-class show/hide** and **“only”** (solo) buttons,
- **live recolouring** of any class with a colour picker.

Other categorical fields still get distinct flat colour bands and a legend, but not
per-class visibility — that is a limit of Potree's shader, which has exactly one class
LUT. The panel offers a **“Make *field* the class field”** button that reconverts the
scene so a different field takes that slot.

Class order, names and colours can be pinned with a config file. A `classes.json` in any
folder between `scenes/` and the scene applies to **everything beneath it**, so a whole
dataset is described once:

```
scenes/scannet_subset/classes.json      <- applies to all 12 scenes below
scenes/my_scan.classes.json             <- applies to that one scene
```

Closer files win over further ones and a per-scene sidecar wins over all of them, merging
field by field — so a dataset config can name the classes while one scene overrides a
single field. A config may also nominate the **class field** with `primaryField`, which
matters for a dataset shipping several label sets (ScanNet has both `segment20` and
`segment200`).

`scenes/scannet_subset/classes.json` is a worked example, generated from Pointcept's own
`scannet200_constants.py`, so the 20 and 200 class names and the official ScanNet colours
are the real ones rather than approximations. `segment20`/`segment200` store the *index*
into `VALID_CLASS_IDS_*` with `-1` for unlabelled, and the config maps that back to names.

The per-scene form sits next to the PCD:

```jsonc
// scenes/my_scan.classes.json
{
  "description": "Shown in the viewer header",
  "fields": {
    "label": {
      "name": "Semantic class",
      "classes": {
        "0": { "name": "unlabeled", "color": [90, 90, 95] },
        "1": { "name": "road",      "color": [128, 64, 128] }
      }
    }
  }
}
```

Anything not listed gets a generated colour from a well-separated palette.

---

## Object library

When a scene has both a class field and a per-object id field (`instance`,
`object_id`, `cluster`, …), the converter also cuts every object out into its own
small Potree octree under `cache/<scene>/instances/<id>/`.

The **Object library** tab lists them all, across every converted scene, grouped **by class**
or **by scene**, each with a shaded three-quarter thumbnail, point count and real-world size. The
thumbnails are rendered at conversion time from the object's own points, with the aspect
ratio preserved — a pole looks like a pole, a tree like a tree.

Click one to place it:

- a live preview follows the cursor and **sits on whatever surface is under it** — it
  uses Potree's own GPU picking, so objects land on the road, a rooftop, or another
  object, not on an imaginary plane;
- click to drop, **Shift-click** to keep placing more, **Esc** to cancel.

Once placed, objects can be moved around:

| Action | How |
| --- | --- |
| Move | drag it in the viewport (follows the surface) |
| Move vertically | Shift-drag |
| Rotate | `[` / `]`, or the rotation slider |
| Scale | the scale slider |
| Nudge | arrow keys (Shift for fine steps) |
| Duplicate | `Ctrl+D` |
| Remove | `Delete` |
| Lock / unlock | `L`, or the 🔓 button |
| Exact placement | type X/Y/Z in the transform panel |
| Rest on what's below | “Drop to surface” |

**Locking.** Once an object is where you want it, lock it. A locked object cannot be
dragged, nudged, rotated, scaled or removed, its transform controls grey out, and its
outline turns amber. Dragging it passes straight through to the camera, so you can orbit
around your composition without knocking anything out of place. It stays selectable, so
unlocking is one click. "Remove all" skips locked objects.

Dragging an object suppresses the camera orbit for that drag only; dragging empty space
still orbits as usual.

### Inspecting and detaching what is already in the scene

Objects belonging to the scene you have open are marked **in scene** in the library and
get two extra actions:

- **Inspect** — highlights the object where it sits and frames the camera on it. Clicking
  an object directly in the viewport does the same thing. Identification is exact: Potree's
  GPU pick returns the picked point's `instance` value, so the object under the cursor is
  read from the data rather than guessed from bounding boxes (a long thin wall has a box
  that swallows half the scene).
- **Detach** — lifts the object out of the scene. The scene's octree is rebuilt without its
  points and a movable copy appears in its place, already selected. Move it and use
  *Integrate into scene* to put it back, or leave it out entirely. Detached objects are
  greyed out in the library, and re-converting the scene from its PCD restores them.

### Reloading a scene

Reloading the scene you already have open — which happens after every detach and every
integrate — is treated as a refresh, not a fresh start. Placed objects stay where they
are, the camera does not move, and the colour mode is kept. Switching to a *different*
scene clears placed objects and frames the new cloud as usual.

### Highlight style

Selection and inspection outlines come in two styles, switchable in **Placed objects**:

- **Contour** (default) — follows the convex hull of the object's footprint, drawn at the
  base and at full height. Much tighter than a box for anything round or off-axis.
- **Bounding box** — the plain axis-aligned box.

Outlines are drawn in a scene of their own, rendered on the `render.pass.end` event.
Potree draws `viewer.scene.scene` *before* the eye-dome-lighting pass, and that pass
composites the point cloud over the whole frame — so an overlay added there is painted
over by the points regardless of `depthTest`. WebGL also ignores line width, so the
outline colours (white for selected, amber for locked, cyan for inspected) deliberately
avoid the class palette; a blue outline is invisible against a blue car.

A spawned instance is a **real `PointCloudOctree`** added to `viewer.scene.pointclouds`,
not an overlay — so it renders through the same pipeline as the scene and keeps eye-dome
lighting, adaptive point size and the shared point budget.

**Everything on screen is drawn identically.** `styleMaterial()` in `web/js/app.js` is the
single definition of how a cloud looks — colour mode, ramp, value ranges, point size, size
mode, shape, opacity, elevation range, bounding box — and it is applied to the scene and
every placed object together. Changing anything in the Rendering panel or the Mapping
panel re-styles them all, and the settings survive a scene rebuild rather than snapping
back to defaults.

**Scale resizes the object, not its points.** Potree's `getPointSize()` multiplies the
projection factor by the model matrix's scale, so scaling a placed object up would fatten
its points along with it — at 4x they bloat until the whole object fuses into one blob.
That term is cancelled out, so scale moves the points apart and nothing else; point size
stays frozen to the scene's. (Fixed size mode is in screen pixels and is untouched by the
model matrix, so it needs no compensation and gets none.) The trade-off is that scaling an
object up spreads its points, so a heavily enlarged object will look sparser rather than
blurrier — which is the honest depiction of what its data actually is.

Point *size* is the other half of looking identical. In adaptive mode Potree sizes a point
from its cloud's declared `spacing`, and the usual `cubeSize / gridSize` is only correct
when the root was actually grid-subsampled. An instance keeps every one of its points in a
single node, so for an elongated object — a wall spanning the whole scene — the bounding
cube is huge while the points are dense, and the cube-derived figure came out 4x too large:
its points rendered as oversized blobs next to the scene. Extracted objects therefore
**inherit the scene's finest-level spacing** (`sceneSpacing / 2^depth`) rather than being
given one derived from their own geometry.

Inheriting is the right rule, not measuring. What matters is not how dense an object
actually is but the size the scene draws those same points at: in the scene an object's
points sit in leaf nodes and are drawn at the finest level's spacing, so a copy cut out
into its own single-node octree has to use that same figure. Measuring the object's own
density gets this wrong for anything unusually sparse — a large, thinly-sampled object in
one real scan came out 6.5x oversized that way. Every extracted object now matches its
scene's finest spacing exactly.

`estimateSpacing()` in `src/octree.mjs` remains for clouds converted without a parent to
inherit from — a small scene whose root was never subsampled — where it finds the grid
resolution at which nearly every cell holds a single point.

That covers an object placed back into the scene it came from. An object placed into a
*different* scene needs one more step: its baked-in spacing belongs to its source, so a
tree cut from an 80 m street and dropped into an 8 m indoor scan renders its points ~8x
too big. `styleMaterial()` therefore also overrides `pcoGeometry.spacing` on every placed
object to match the scene currently open. Potree re-reads that from the geometry each
frame, which is why the override lives there rather than on the material.

### Resampling when scaling

Freezing point size means scaling only moves points apart, so an enlarged object turns
sparse and a shrunken one clots together. **Resample when scaling** (on by default, in the
transform panel) fixes that properly: on releasing the scale slider the object is rebuilt
at its new size with the scene's point density, and the new octree is swapped in.

Gaps are filled by inserting midpoints between neighbours — a midpoint of two nearby
surface points still lies on the surface, so nothing is invented — working coarse to fine,
halving the search radius each pass until the target spacing is reached. New points inherit
every attribute from the neighbour they came from rather than averaging, so class and
instance ids stay exact. Shrinking thins on a grid instead.

Measured nearest-neighbour distance across 0.5x to 4x stays within 0.65-1.2x of the
scene's spacing, where stretching alone would degrade it linearly (4x the gaps at 4x the
size). Results are cached per (object, scale, spacing) under
`cache/<scene>/instances/<id>/scaled/`, and merging uses the resampled geometry.

Turn the toggle off to get plain stretching, where scale only multiplies coordinates and
the object keeps exactly the points it started with.

The value ranges are the subtle part. An instance octree carries its own per-object
min/max, so a spawned car left to itself would stretch the colour ramp across only its own
values and render the same number as a different colour than the scene does. Every cloud
is given the *scene's* range instead; the instance's own range cancels out of Potree's
normalisation, so identical values come out identically coloured. The categorical ramp is
built once and shared, so per-class colour edits reach every cloud at the same time.

### Integrating objects into the scene

The **Integrate into scene** button merges every placed object into the scene's own
octree, permanently. It asks first, showing how many objects and points are involved,
a breakdown by class, and what the operation actually does.

Each merged object keeps its class and is given a **new instance id**, so it stays a
distinct object in the segmentation rather than dissolving into the cloud. Class counts,
attribute ranges and the legend are all recomputed. Objects borrowed from another scene
are merged too — their class indices are remapped by value into the target scene's table.

Merged points are recovered from the instance octrees rather than by re-reading the
source PCD, so this works even when the object came from a different scene.

Every merged object is **re-cut as its own library entry** in its final pose, so it stays
a first-class object: click it, inspect it, detach it again. An object that was detached
from this scene reclaims its original instance id when it comes back, so its library
entry is refreshed rather than orphaned.

Because each merge rebuilds the octree from the PCD, objects merged by an *earlier* merge
are carried over automatically — integrating a second object does not discard the first.
`scene.json` records `baked` and `detached` as the octree's current state, not a history.

> **This rewrites the cached octree, not your PCD.** Re-converting the scene from its
> source file afterwards discards the merged objects. `scene.json` records what was
> merged under `baked`, and `tools/verify_octree.mjs` accounts for those points instead
> of reporting them as mismatches.

Objects rotate and scale about an **anchor** at the centre of their footprint, at their
lowest point, rather than about the corner of their bounding cube — that is what makes
dropping and spinning them feel right. (Potree puts an octree's origin at its bounding
cube's minimum corner, so `src/instances.mjs` records `anchorLocal` and the viewer
compensates for it on every transform.)

Instances with fewer than 24 points are skipped.

**Backdrops** — the ground, a floor, a wall spanning the whole scan — are detected from
the data, never from the id: a group counts as a backdrop when its footprint covers more
than half the scene's, or it holds more than half the points. Backdrops still get a
library entry so they can be clicked and inspected, but they carry no instance octree, so
they cannot be placed or detached.

Judging by id would be wrong. `instance 0` is the unassigned bucket in some datasets and
a perfectly ordinary object in others — in one real scan here it is a 0.9 × 1.1 m object
of 3,367 points, while in the synthetic demo it is the entire road surface. The extent
test gets both right, and a long thin wall (spanning one axis but almost none of the
other) is correctly kept as an object.

Use `--no-instances` to skip extraction, or `--instance-field <field>` to pick the id
field yourself.

---

## Segmentation inference

The **Segmentation** tab sends the open scene's points to an external service and folds
the labels it returns back into the scene as a `prediction` attribute — which then behaves
like any other categorical field: colour bands, a legend, per-class counts, and a
`prediction_score` scalar if the service supplies one.

The points never pass through the browser. The viewer asks the server to run the request,
so the payload goes straight from the cached octree to the service as one binary body.

### Wire format

Both directions use the same framing:

```
[8]  uint64 little-endian   length of the JSON header
[H]  UTF-8 JSON header
[..] raw array bytes, back to back, each on an 8-byte boundary
```

The header names each array with a numpy dtype and a shape, so the receiving end wraps the
body in `np.frombuffer(...).reshape(...)` and gets real arrays with **no copying and no
per-point parsing**:

| array | dtype | shape |
| --- | --- | --- |
| `xyz` | float32 | `(3, N)` |
| `rgb` | uint8 | `(3, N)` — only if the scene has colour |
| each continuous field | float32 | `(N,)` |
| each categorical field | int32 | `(N,)` |

Reply with at least `labels` int32 `(N,)`, in the order the points were sent. `scores`
float32 `(N,)` and a `class_names` map in the header are used if present. A JSON reply is
accepted too, for convenience while developing.

**The requested shapes are also the fastest ones.** Three contiguous per-axis arrays laid
end to end already *are* a C-contiguous `[3, N]` array, so nothing is transposed or
interleaved on either side — `[N, 3]` would have been the slower choice. Coordinates go as
float32 rather than float64, halving that array for sub-micron precision on a scene of any
realistic size. A 422k-point scan with nine fields is a single ~19 MiB POST.

`examples/segmentation_service.py` is a runnable reference receiver with the decode/encode
helpers; point the tab at it to try the whole path:

```bash
python examples/segmentation_service.py --port 8500
```

The endpoint is typed in the tab and remembered per browser. The tab shows exactly what
would be sent — every array with its dtype, shape and size — and individual scalar fields
can be excluded before sending. Confirming rewrites the scene's cached octree to add the
prediction; running again replaces it, and re-converting from the PCD discards it.

Point positions are untouched: the rebuild reuses the octree's existing quantisation grid,
so coordinates come out bit-identical.

### Interpretability: ceteris paribus

The tab's **Interpretability** section holds the scene fixed, moves one object through a
range of heights, and plots the object's **mean probability for a class you choose**
against the height it was placed at. If the curve is flat the model is judging the object
on its own shape; if it slopes, the model is leaning on where the object sits.

The focus object can be chosen from the dropdown or by **clicking it in the viewport** —
press **◎ Pick**, then click the object; Esc cancels. Picking is armed explicitly rather
than always-on, so exploring the scene by clicking objects (which inspects them) never
quietly retargets an analysis you have already set up.

Pick the object, the class and a height sweep (from / to / steps), and the result appears
as a line chart with a table view beside it. Nothing is written to the scene — this is
analysis, so the octree is read but never rebuilt.

**Hovering a point on the curve draws the object where it actually was** when the model
scored it: a translucent copy at that height, outlined so it can be found inside a dense
cloud. Hovering a table row does the same, so the feature is not mouse-only. The copy is
loaded once when the chart appears and only attached and detached as the pointer moves, so
tracing the curve reloads nothing. It is excluded from picking, so it can never be
mistaken for the scene or become a surface to drop objects onto.

**The whole sweep is one request.** The scene goes out once, together with

    mask         uint8    (N,)   1 for the points of the object being moved

and a `ceteris_paribus` block in the header carrying `direction` (a unit vector),
`offsets` along it and the absolute `heights` they correspond to. The service moves the
masked points itself and replies with

    logits       float32  (S, C, M)   sweep position, class, masked point

where M is `mask.sum()` and the columns follow the masked points in ascending index order
— exactly `xyz[:, mask.astype(bool)]`. Send `probs` with the same shape instead if your
model already normalises; the viewer applies the softmax otherwise.

Sending the scene once rather than once per position is what makes this affordable: a
nine-step sweep over a 422k-point scan is a single 23 MiB request with 11 MiB back, not
nine 23 MiB requests. The reply covers only the object rather than the whole cloud.

### Interpretability: saliency

**Compute saliency** asks the service for one scalar per point about the focused object.
It arrives as a scalar field named `saliency` and behaves like any other — pick it in
**Colour by**, get a ramp and a min/max range slider.

The scene goes out with the same `mask` marking the object, under
`request: "saliency"`, with the chosen class passed along as `target_class` in case the
method needs a target. Reply with

    saliency     float32  (N,)   one scalar per scene point
                 float32  (M,)   or just the masked points, in ascending index
                                 order; the rest of the scene reads as 0

**What the scalar measures and how it is aggregated is entirely the service's business.**
The viewer makes no assumptions: it carries the numbers back, records their range, and
hands them to the colour pipeline. Both reply shapes are accepted, so a method that only
scores the object costs no more than one that scores the whole cloud.

Each run replaces the previous `saliency` field. `prediction`, `prediction_score` and
`saliency` coexist happily — they are separate attributes and any of them can be the
active colour mode.

---

## Commands

| Command | What it does |
| --- | --- |
| `npm run serve` | start the viewer (`--port`, `--host`) |
| `npm run scan` | list mapped scenes and their conversion status |
| `npm run convert -- <scene-id>` | convert one scene (`--force`, `--grid N`, `--primary <field>`, `--instance-field <field>`, `--no-instances`) |
| `npm run convert -- --all` | convert everything |
| `npm run demo` | write synthetic demo scenes |
| `npm run build-viewer` | rebuild the vendored Potree bundle |
| `node tools/verify_octree.mjs <scene-id>` | decode a converted octree the way Potree does and check it against the source PCD |

Environment: `PCIT_SCENES_DIR`, `PCIT_CACHE_DIR`, `PORT`, `HOST`.

---

## Input formats

A scene is either a **`.pcd` file** or a **directory of `.npy` arrays**, one per field.
Both load through the same path, so everything downstream — field classification, the
octree, the object library, the interpretability tools — is unaware of the difference.

### PCD

All three PCL encodings are read natively in JS: `ascii`, `binary` and
`binary_compressed` (including the LZF decoder, cross-checked against reference liblzf).
Points with a non-finite coordinate — common in organised clouds — are dropped and
reported in the scene details panel.

### Directories of `.npy`

Any folder containing `coord.npy` is a scene. This is what Pointcept and similar
pipelines export for ScanNet, ScanNet200, S3DIS and friends:

```
scenes/scannet_subset/val/scene0011_00/
    coord.npy       float32 (N, 3)   ->  x, y, z
    color.npy       uint8   (N, 3)   ->  r, g, b        (becomes RGB colour)
    normal.npy      float32 (N, 3)   ->  normal_x/y/z   (three scalar fields)
    segment20.npy   int64   (N,)     ->  segment20      (categorical)
    segment200.npy  int64   (N,)     ->  segment200     (categorical)
    instance.npy    int64   (N,)     ->  instance       (drives the object library)
```

`coord`, `color` and `normal` get the names above; **any other array keeps its own
name**, with `_x/_y/_z` appended when it has several columns — so a dataset with extra
arrays loads without code changes. Multi-column arrays are de-interleaved into one
contiguous field each.

**Only the arrays that are there are used.** ScanNet's test split ships just
coord/color/normal with the labels withheld, and it loads fine: RGB and normals to colour
by, no class legend, no object library, and the interpretability panel says plainly that
the scene has no instance field.

The reader covers `.npy` v1/v2/v3 headers, little-endian numeric dtypes and both C and
Fortran order; `int64`/`uint64` come back as doubles, which hold point ids and labels
exactly. Every array read was checked element-for-element against numpy.

Files that carry both a packed `rgb` field *and* its unpacked `r`,`g`,`b` duplicates (some
exporters write both) get the duplicates dropped, after checking on a sample that they
really do match — that alone saved 24% of the octree size on a real 422k-point scan.
Integer fields that are not named like labels need 32 or fewer distinct values to be
treated as categorical, so things like voxel indices stay continuous instead of producing
a 200-entry legend.

---

## How the conversion works

`src/convert.mjs` reads the PCD, classifies its fields, and hands a set of attributes to
`src/octree.mjs`, which writes a Potree 2.0 octree: `metadata.json`, `hierarchy.bin`
(22-byte breadth-first node records) and `octree.bin` (interleaved attributes per node).
Each node holds a grid-subsampled view of its subtree, so the viewer streams coarse
detail first.

Two encoding details are load-bearing, both dictated by Potree's decoder:

- **Generic scalars are written as `double`.** Potree only normalises an attribute into
  0..1 when its type is wider than 4 bytes, and the generic colour path assumes a
  normalised input. A `uint16` label would reach the shader raw and clamp to solid colour.
- **`classification` stays `uint8`.** Its shader path indexes a 256-entry LUT with the raw
  value, so it must *not* be normalised. Class values are remapped to a dense `0..N-1`
  index, which is why label ids like `10000` still work.

`tools/verify_octree.mjs` re-implements Potree's `parseHierarchy`, `createChildAABB` and
the decoder's position maths, then checks the result against the source PCD point by
point. Run it after touching the converter.

---

## Layout

```
scenes/          your scenes: .pcd files, or folders of .npy arrays
                 (+ optional classes.json configs, per folder or per scene)
cache/           generated octrees, one folder per scene
examples/        reference segmentation service (documents the wire format)
tools/           viewer build, demo scenes, octree verifier

src/
  server.mjs     HTTP server: the API, SSE progress, Range-capable static files
  paths.mjs      where scenes and the cache live

  io/            getting points off disk
    pcd.mjs        PCD reader (ascii / binary / binary_compressed)
    lzf.mjs        LZF decompressor for binary_compressed
    npy.mjs        NumPy .npy reader
    cloud.mjs      picks the reader; one shape out for either format

  octree/        the Potree 2.0 format itself
    write.mjs      builds and writes an octree
    read.mjs       reads one back into flat arrays

  scene/         a scene as the app understands it
    registry.mjs   discovery, ids, class configs
    fields.mjs     sorts raw fields into position / colour / scalar / categorical
    palette.mjs    colours for classes
    convert.mjs    source cloud -> octree + scene.json  (also the CLI)
    attributes.mjs adds scalar fields to an already-converted scene

  objects/       the object library
    instances.mjs  cuts objects out of a segmented scene
    resample.mjs   rescales one object to a target point density
    bake.mjs       merges placed objects back in, and detaches them

  inference/     talking to a segmentation service
    npbuffer.mjs   the binary wire format
    predict.mjs    segment the scene, fold labels back in
    saliency.mjs   per-point saliency for one object
    ceteris.mjs    sweep one object through heights, one request

  workers/       thin worker_thread entry points, one per long job
    convert.mjs  bake.mjs  resample.mjs
    predict.mjs  saliency.mjs  ceteris.mjs

web/             front-end (vendor/potree is generated, not committed)
  js/app.js        scene browser, colouring, rendering controls
  js/instances.js  object library, placement, transforms, inspect and detach
  js/segmentation.js  the segmentation and interpretability tab
  js/util.js       DOM helpers, thumbnails, modal dialogs
```
