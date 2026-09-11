/**
 * Front-end for the point cloud interpretability tool.
 *
 * Drives Potree directly rather than loading its stock sidebar, so the
 * colouring controls can be built around what actually matters here:
 * segmentation fields and per-class inspection.
 */

import {
  $, el, fmtInt, fmtBytes, fmtNum, rgbCss, rgbHex, hexRgb, encId, fetchJson, toast,
} from "./util.js";
import { createInstanceManager } from "./instances.js";
import { createSegmentation } from "./segmentation.js";

// ------------------------------------------------------------------- state
const state = {
  scenes: [],
  filter: "",
  currentId: null,
  info: null,          // scene.json of the loaded scene
  pointcloud: null,
  attribute: null,     // active attribute descriptor
  attrState: new Map(),   // attribute name -> { gradient, range: [lo, hi] }
  // Rendering settings apply to every cloud on screen, the scene and any placed
  // object alike, so a spawned copy never looks different from the scene it
  // came from.
  render: {
    size: 1, sizeType: "ADAPTIVE", shape: "SQUARE", opacity: 1,
    budget: 2, background: "gradient", edl: true, bbox: false,
  },
  classState: new Map(),  // attribute name -> Map(classIndex -> { visible, color })
  converting: new Set(),
};

// ------------------------------------------------------------------ potree
if (typeof Potree === "undefined") {
  document.body.innerHTML =
    '<div style="padding:40px;font-family:system-ui;color:#e6ebf1;background:#0d1013;height:100vh">' +
    "<h2>Potree is not vendored yet</h2><p>Run <code>npm run build-viewer</code>, then reload.</p></div>";
  throw new Error("Potree missing");
}

const viewer = new Potree.Viewer($("#render-area"));
window.viewer = viewer;
viewer.setEDLEnabled(true);
viewer.setFOV(60);
viewer.setPointBudget(2_000_000);
viewer.setBackground("gradient");
viewer.setControls(viewer.orbitControls);
viewer.useHQ = false;

/** Potree bundles its own three.js and does not expose it, so borrow the Color type. */
const ThreeColor = Potree.Gradients.SPECTRAL[0][1].constructor;

const GRADIENTS = ["SPECTRAL", "VIRIDIS", "PLASMA", "INFERNO", "TURBO", "RAINBOW", "YELLOW_GREEN", "GRAYSCALE"];

/** Instance library: browse extracted objects, spawn them, move them around. */
const instances = createInstanceManager({
  viewer,
  getSceneInfo: () => state.info,
  getActiveAttributeName: () => state.attribute?.potreeName ?? "rgba",
  styleMaterial: (pointcloud) => styleMaterial(pointcloud),
  // After a merge the scene's octree has changed on disk; reload it.
  reloadScene: async (scene) => { await refreshScenes(); await loadOctree(scene); },
});
window.instances = instances;

/** Segmentation inference: send the scene out, fold the labels back in. */
const segmentation = createSegmentation({
  getSceneInfo: () => state.info,
  reloadScene: async (scene) => { await refreshScenes(); await loadOctree(scene); },
  onPrediction: () => {
    const opt = attributeOptions().find((a) => a.name === "prediction");
    if (opt) selectAttribute(opt);
  },
  // Hovering the ceteris paribus curve puts a translucent copy of the object
  // where it sat when the model scored it at that height.
  onHoverStep: ({ instanceId, height }) => {
    const entry = instances.sceneInstances().find((e) => e.id === instanceId);
    if (!entry) return;
    instances.showGhost(entry, [entry.anchor[0], entry.anchor[1], height]);
  },
  onHoverEnd: () => instances.hideGhost(),
  onPreload: ({ instanceId }) => {
    const entry = instances.sceneInstances().find((e) => e.id === instanceId);
    if (entry) instances.preloadGhost(entry);
  },
  onSelectAttribute: (name) => {
    const opt = attributeOptions().find((a) => a.name === name);
    if (opt) selectAttribute(opt);
  },
  onPickObject: (done) => instances.armInstancePick(done),
  onCancelPick: () => instances.disarmInstancePick(),
});
window.segmentation = segmentation;

/**
 * Builds a stepped gradient so a categorical field reads as flat colour bands
 * instead of a continuous ramp. Each class owns the interval halfway to its
 * neighbours, and the two stops at each boundary share a position to keep the
 * edge as hard as a 64px gradient texture allows.
 */
function categoricalGradient(classes, min, max) {
  const span = (max - min) || 1;
  const t = classes.map((c) => Math.min(1, Math.max(0, (c.value - min) / span)));
  const stops = [];
  for (let i = 0; i < classes.length; i++) {
    const color = new ThreeColor(
      classes[i].color[0] / 255, classes[i].color[1] / 255, classes[i].color[2] / 255);
    const lo = i === 0 ? 0 : (t[i - 1] + t[i]) / 2;
    const hi = i === classes.length - 1 ? 1 : (t[i] + t[i + 1]) / 2;
    stops.push([lo, color], [hi, color]);
  }
  return stops;
}

function gradientCss(gradient) {
  const stops = gradient.map(([t, c]) => `#${c.getHexString()} ${(t * 100).toFixed(2)}%`);
  return `linear-gradient(to right, ${stops.join(",")})`;
}

// ------------------------------------------------------------ scene browser
async function refreshScenes() {
  try {
    const data = await fetchJson("/api/scenes");
    state.scenes = data.scenes;
    $("#scenes-dir").textContent = data.scenesDir;
    $("#scenes-dir").title = data.scenesDir;
    renderSceneList();
    instances.loadLibrary();
  } catch (err) {
    toast(`Could not list scenes: ${err.message}`, true);
  }
}

function sceneMatches(scene, needle) {
  if (!needle) return true;
  const hay = [scene.id, scene.folder, scene.encoding, ...(scene.fields ?? [])].join(" ").toLowerCase();
  return hay.includes(needle);
}

const SEG_HINT = /^(label|labels|classification|class|semantic|seg|segmentation|instance|category|panoptic|object)/i;

function renderSceneList() {
  const list = $("#scene-list");
  list.innerHTML = "";

  const needle = state.filter.trim().toLowerCase();
  const visible = state.scenes.filter((s) => sceneMatches(s, needle));

  if (visible.length === 0) {
    const note = el("div", "empty-note");
    note.innerHTML = state.scenes.length === 0
      ? `No <code>.pcd</code> files under the scenes folder.<br><br>Drop some in and hit ⟳, or run <code>npm run demo</code>.`
      : "No scene matches that filter.";
    list.appendChild(note);
    return;
  }

  // Group by folder so a mapped directory tree stays readable.
  const groups = new Map();
  for (const s of visible) {
    const key = s.folder || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  }

  for (const [folder, scenes] of [...groups].sort((a, b) => a[0].localeCompare(b[0]))) {
    list.appendChild(el("div", "group-label", folder || "scenes"));
    for (const scene of scenes) list.appendChild(renderSceneCard(scene));
  }
}

function renderSceneCard(scene) {
  const card = el("div", "scene");
  card.dataset.id = scene.id;
  if (scene.id === state.currentId) card.classList.add("selected");

  const row = el("div", "row1");
  row.appendChild(el("div", "nm", scene.name));
  row.appendChild(el("span", `pill ${scene.status}`, scene.status));
  card.appendChild(row);

  card.appendChild(el("div", "meta",
    `${fmtInt(scene.points)} pts · ${fmtBytes(scene.bytes)} · ${scene.encoding ?? "?"}`));

  if (scene.fields) {
    const chips = el("div", "fields");
    for (const f of scene.fields) {
      const chip = el("span", SEG_HINT.test(f) ? "chip seg" : "chip", f);
      chips.appendChild(chip);
    }
    card.appendChild(chips);
  }
  if (scene.error) card.appendChild(el("div", "err", scene.error));

  card.addEventListener("click", () => openScene(scene));
  return card;
}

// -------------------------------------------------------------- conversion
/** Converts a scene if needed, streaming progress into its card. */
function convertScene(scene, { force = false } = {}) {
  return new Promise((resolve, reject) => {
    const card = $(`.scene[data-id="${CSS.escape(scene.id)}"]`);
    let bar = null, statusLine = null;
    if (card) {
      statusLine = el("div", "status-line", "Queued…");
      const wrap = el("div", "scene-progress");
      bar = el("div");
      wrap.appendChild(bar);
      card.appendChild(statusLine);
      card.appendChild(wrap);
    }
    const cleanup = () => {
      state.converting.delete(scene.id);
      if (card) {
        card.querySelector(".status-line")?.remove();
        card.querySelector(".scene-progress")?.remove();
      }
    };

    state.converting.add(scene.id);
    fetchJson(`/api/scenes/${encId(scene.id)}/convert?force=${force ? 1 : 0}`, { method: "POST" })
      .then(({ jobId }) => {
        const es = new EventSource(`/api/jobs/${jobId}/events`);
        es.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          if (statusLine) statusLine.textContent = msg.message ?? "Working…";
          if (bar) bar.style.width = `${Math.round((msg.progress ?? 0) * 100)}%`;
          setStage(`Converting ${scene.name}`, msg.message ?? "", null);

          if (msg.state === "done") { es.close(); cleanup(); resolve(msg.scene); }
          else if (msg.state === "error") { es.close(); cleanup(); reject(new Error(msg.error)); }
        };
        es.onerror = () => { es.close(); cleanup(); reject(new Error("lost contact with the conversion job")); };
      })
      .catch((err) => { cleanup(); reject(err); });
  });
}

// ------------------------------------------------------------ scene loading
function setStage(title, text, hint) {
  const stage = $("#stage-message");
  stage.classList.remove("hidden");
  $("#stage-title").textContent = title;
  $("#stage-text").textContent = text;
  const h = $("#stage-hint");
  if (hint) { h.style.display = ""; h.textContent = hint; } else { h.style.display = "none"; }
}
const hideStage = () => $("#stage-message").classList.add("hidden");

function unloadCurrent() {
  if (!state.pointcloud) return;
  viewer.scene.scenePointCloud.remove(state.pointcloud);
  const i = viewer.scene.pointclouds.indexOf(state.pointcloud);
  if (i !== -1) viewer.scene.pointclouds.splice(i, 1);
  state.pointcloud = null;
}

async function openScene(scene) {
  if (state.converting.has(scene.id)) return;

  state.currentId = scene.id;
  renderSceneList();
  setStage(`Loading ${scene.name}`, "Preparing the octree…", null);

  try {
    if (scene.status !== "ready") {
      await convertScene(scene, { force: scene.status === "stale" });
      await refreshScenes();
    }
    const detail = await fetchJson(`/api/scenes/${encId(scene.id)}`);
    if (!detail.converted) throw new Error("conversion produced no scene description");
    await loadOctree(detail.converted);
  } catch (err) {
    console.error(err);
    setStage("Could not load this scene", err.message, null);
    toast(err.message, true);
  }
}

function loadOctree(info) {
  return new Promise((resolve, reject) => {
    const url = `/octree/${encId(info.id)}/metadata.json`;
    let settled = false;

    // Reloading the scene you are already in -- after a detach or an integrate --
    // should feel like a refresh, not like starting over: keep the objects you
    // have placed and stay exactly where the camera was.
    const sameScene = state.info?.id === info.id;
    const view = viewer.scene.view;
    const camera = sameScene
      ? { position: view.position.clone(), yaw: view.yaw, pitch: view.pitch, radius: view.radius }
      : null;

    Potree.loadPointCloud(url, info.name, (e) => {
      if (settled) return;      // Potree's loader can fire more than once
      settled = true;

      if (!sameScene) instances.resetPlaced();
      unloadCurrent();
      const pointcloud = e.pointcloud;
      state.pointcloud = pointcloud;
      const previousAttribute = sameScene ? state.attribute?.potreeName : null;
      state.info = info;
      if (!sameScene) {
        state.attrState = new Map();
        state.classState = new Map();
      }

      viewer.scene.addPointCloud(pointcloud);

      if (camera) {
        viewer.scene.view.position.copy(camera.position);
        viewer.scene.view.yaw = camera.yaw;
        viewer.scene.view.pitch = camera.pitch;
        viewer.scene.view.radius = camera.radius;
      } else {
        viewer.fitToScreen(1.0);
      }
      hideStage();

      $("#scene-title").textContent = info.name;
      $("#scene-sub").textContent =
        `${fmtInt(info.numPoints)} pts · ${info.numNodes} nodes · depth ${info.depth}`;
      $("#control-sub").textContent = info.description || "Colouring and rendering";

      // Re-apply the viewer-wide settings the user already chose, rather than
      // snapping back to defaults every time the scene is rebuilt.
      viewer.setPointBudget(state.render.budget * 1e6);
      viewer.setBackground(state.render.background);
      viewer.setEDLEnabled(state.render.edl);

      buildAttributePanel();
      buildRenderPanel();
      buildInfoPanel();
      // The library marks which objects belong to the scene that is open.
      instances.refreshView();
      segmentation.loadPreview(true);

      // Open on the segmentation field when the scene has one -- that is the
      // reason this tool exists.
      const options = attributeOptions();
      const preferred = (previousAttribute && options.find((a) => a.potreeName === previousAttribute))
        ?? options.find((a) => a.kind === "classification")
        ?? options.find((a) => a.kind === "categorical")
        ?? options.find((a) => a.kind === "color")
        ?? options[0];
      if (preferred) selectAttribute(preferred);

      resolve(pointcloud);
    });

    setTimeout(() => {
      if (!settled) { settled = true; reject(new Error("timed out loading the octree")); }
    }, 60_000);
  });
}

// ------------------------------------------------------- attribute picking
/** The colour modes offered for the loaded scene, in display order. */
function attributeOptions() {
  const info = state.info;
  if (!info) return [];
  const out = [];

  for (const a of info.attributes) {
    if (a.kind === "color") {
      out.push({ ...a, potreeName: "rgba", label: "RGB colour", kind: "color" });
    } else if (a.kind === "classification") {
      out.push({ ...a, potreeName: "classification", kind: "classification" });
    } else if (a.name === "intensity") {
      out.push({ ...a, potreeName: "intensity", kind: "intensity" });
      out.push({ ...a, potreeName: "intensity gradient", label: "Intensity (gradient)", kind: "intensity" });
    } else {
      out.push({ ...a, potreeName: a.name });
    }
  }

  out.push({ name: "__elevation", potreeName: "elevation", label: "Elevation", kind: "builtin" });
  out.push({ name: "__lod", potreeName: "level of detail", label: "Octree level", kind: "builtin" });
  return out;
}

function buildAttributePanel() {
  const host = $("#attribute-list");
  host.innerHTML = "";
  const options = attributeOptions();
  $("#attr-count").textContent = `${options.length}`;

  for (const opt of options) {
    const item = el("div", "attr");
    item.dataset.key = opt.potreeName;

    const swatch = el("div", "swatch");
    swatch.style.background = swatchFor(opt);
    item.appendChild(swatch);

    const txt = el("div", "txt");
    txt.appendChild(el("div", "l", opt.label ?? opt.name));
    const sub = opt.kind === "builtin" ? "derived"
      : `${opt.source ?? opt.name}${opt.numClasses ? ` · ${opt.numClasses} classes` : ""}`;
    txt.appendChild(el("div", "k", sub));
    item.appendChild(txt);

    if (opt.kind === "classification") item.appendChild(el("span", "badge lut", "classes"));
    else if (opt.kind === "categorical") item.appendChild(el("span", "badge seg", "categorical"));
    else if (opt.kind === "continuous") item.appendChild(el("span", "badge", "scalar"));
    else if (opt.kind === "color") item.appendChild(el("span", "badge", "rgb"));

    item.addEventListener("click", () => selectAttribute(opt));
    host.appendChild(item);
  }
}

function swatchFor(opt) {
  if (opt.kind === "color") return "linear-gradient(135deg,#e05c68,#45c78a,#4e9eff)";
  if (opt.kind === "classification" || opt.kind === "categorical") {
    const classes = opt.classes ?? state.info?.classification?.classes ?? [];
    const cols = classes.slice(0, 4).map((c) => rgbCss(c.color));
    if (cols.length === 0) return "#555";
    const step = 100 / cols.length;
    const stops = cols.map((c, i) => `${c} ${i * step}%, ${c} ${(i + 1) * step}%`);
    return `linear-gradient(135deg, ${stops.join(",")})`;
  }
  return gradientCss(Potree.Gradients.SPECTRAL);
}

/**
 * Applies the viewer's current look to one point cloud.
 *
 * This is the only place that decides how a cloud is drawn, so the scene and
 * every placed object stay identical. It matters most for the value ranges: an
 * instance octree carries its own per-object min/max, so left alone a spawned
 * car would stretch the colour ramp across just its own values and render the
 * same number as a different colour than the scene does.
 */
/**
 * The colour ramp for the active attribute, built once and shared by every
 * cloud. Sharing the array matters: Potree regenerates the gradient texture
 * whenever a different object is assigned, and per-class colour edits have to
 * reach the scene and every placed object alike.
 */
let gradientCache = { key: null, value: null };

function currentGradient(opt, st) {
  if (opt.kind !== "categorical") return Potree.Gradients[st.gradient];

  const cs = classStateFor(opt);
  const classes = classesFor(opt).map((c) => ({ ...c, color: cs.get(c.index)?.color ?? c.color }));
  const key = `${opt.potreeName}|${opt.min}|${opt.max}|${classes.map((c) => c.color.join(",")).join("|")}`;
  if (gradientCache.key !== key) {
    gradientCache = { key, value: categoricalGradient(classes, opt.min, opt.max) };
  }
  return gradientCache.value;
}

function styleMaterial(pointcloud) {
  const material = pointcloud.material;
  const r = state.render;

  // Potree's getPointSize() multiplies projFactor by the model matrix's scale,
  // so scaling an object up would fatten its points along with it. Scale is
  // meant to resize the object, not restyle it, so cancel that term out and keep
  // point size frozen to the scene's. (Fixed mode is already in screen pixels
  // and is unaffected by the model matrix, so it needs no compensation.)
  const modelScale = pointcloud.scale?.x ?? 1;
  const sizeCompensation = r.sizeType === "FIXED" ? 1 : (modelScale || 1);
  material.size = r.size / sizeCompensation;
  material.pointSizeType = Potree.PointSizeType[r.sizeType];
  material.shape = Potree.PointShape[r.shape];
  material.opacity = pointcloud.__pcitPreview ? 0.55 : r.opacity;
  pointcloud.showBoundingBox = r.bbox;

  // An object may have been cut from a different scene, whose point density is
  // nothing like this one's. `spacing` drives adaptive point size, so it has to
  // follow the scene being viewed, not the scene the object came from -- a tree
  // taken from a 80 m street into a 8 m indoor scan would otherwise render its
  // points ~10x too big. Potree re-reads this from the geometry every frame, so
  // the override belongs there rather than on the material.
  if (pointcloud !== state.pointcloud && pointcloud.pcoGeometry && state.info) {
    const finest = state.info.spacing / Math.pow(2, state.info.depth ?? 0);
    if (Number.isFinite(finest) && finest > 0) pointcloud.pcoGeometry.spacing = finest;
  }

  // Elevation is a world-space height ramp: every cloud must use the scene's
  // range, not its own, or a spawned object is coloured on its own few metres.
  if (state.info?.tightBoundingBox) {
    material.elevationRange = [
      state.info.tightBoundingBox.min[2],
      state.info.tightBoundingBox.max[2],
    ];
  }

  const opt = state.attribute;
  if (!opt) return;

  // An object borrowed from another scene may not carry this attribute.
  const available = pointcloud.pcoGeometry?.pointAttributes?.attributes?.map((a) => a.name) ?? [];
  const derived = ["elevation", "level of detail"];
  const usable = derived.includes(opt.potreeName)
    || available.includes(opt.potreeName)
    || (opt.potreeName === "intensity gradient" && available.includes("intensity"));
  material.activeAttributeName = usable
    ? opt.potreeName
    : (available.includes("rgba") ? "rgba" : "elevation");

  const st = attrStateFor(opt);
  if (opt.kind === "categorical" || opt.kind === "continuous") {
    material.gradient = currentGradient(opt, st);
    material.setRange(opt.potreeName, st.range.slice());
  } else if (opt.kind === "intensity") {
    material.gradient = currentGradient(opt, st);
    material.intensityRange = st.range.slice();
  }
}

/** Re-styles everything on screen: the scene and every placed object. */
function applyToAll() {
  for (const pointcloud of viewer.scene.pointclouds) styleMaterial(pointcloud);
}

function selectAttribute(opt) {
  if (!state.pointcloud) return;
  state.attribute = opt;

  for (const node of document.querySelectorAll(".attr")) {
    node.classList.toggle("active", node.dataset.key === opt.potreeName);
  }

  // The class lookup table lives on the viewer and is shared by every cloud.
  if (opt.kind === "classification") applyClassification(opt);

  applyToAll();
  buildMappingPanel(opt);
}

function attrStateFor(opt) {
  const key = opt.potreeName;
  if (!state.attrState.has(key)) {
    state.attrState.set(key, {
      gradient: opt.kind === "categorical" ? null : "SPECTRAL",
      range: [opt.min ?? 0, opt.max ?? 1],
    });
  }
  return state.attrState.get(key);
}

function classesFor(opt) {
  if (opt.kind === "classification") return state.info.classification?.classes ?? [];
  return opt.classes ?? [];
}

/** Per-class colour + visibility, remembered while the scene stays loaded. */
function classStateFor(opt) {
  const key = opt.potreeName;
  if (!state.classState.has(key)) {
    const m = new Map();
    for (const c of classesFor(opt)) m.set(c.index, { visible: true, color: c.color.slice() });
    state.classState.set(key, m);
  }
  return state.classState.get(key);
}

/**
 * Pushes the class table into Potree's classification LUT.
 *
 * This has to go through the *viewer*, not the material: viewer.update() copies
 * `viewer.classifications` onto every point cloud material each frame, so
 * anything written straight to the material is overwritten within one frame.
 */
function applyClassification(opt) {
  const cs = classStateFor(opt);
  const table = {};
  for (const c of classesFor(opt)) {
    const st = cs.get(c.index);
    table[c.index] = {
      visible: st.visible,
      name: c.name,
      color: [st.color[0] / 255, st.color[1] / 255, st.color[2] / 255, 1.0],
    };
  }
  table.DEFAULT = { visible: true, name: "other", color: [0.35, 0.38, 0.42, 1.0] };

  viewer.setClassifications(table);
  state.pointcloud.material.recomputeClassification();
}

// ---------------------------------------------------------- mapping panel
function buildMappingPanel(opt) {
  const host = $("#mapping-content");
  host.innerHTML = "";
  $("#mapping-title").textContent =
    opt.kind === "classification" || opt.kind === "categorical" ? "Classes" : "Mapping";

  if (opt.kind === "color") {
    host.appendChild(el("div", "note",
      "Points are drawn with the RGB values stored in the PCD file. Switch to a scalar field to inspect it."));
    return;
  }
  if (opt.kind === "builtin") {
    host.appendChild(el("div", "note", opt.potreeName === "elevation"
      ? "Height above the scene's lowest point, mapped through Potree's elevation ramp."
      : "Colour by octree level, which shows how the viewer is streaming detail."));
    return;
  }
  if (opt.kind === "classification" || opt.kind === "categorical") {
    buildClassPanel(host, opt);
    return;
  }
  buildScalarPanel(host, opt);
}

function buildClassPanel(host, opt) {
  const classes = classesFor(opt);
  const cs = classStateFor(opt);
  const isLut = opt.kind === "classification";
  const total = classes.reduce((a, c) => a + c.count, 0) || 1;

  // tools
  const tools = el("div", "legend-tools");
  const search = el("input");
  search.placeholder = "Filter classes…";
  tools.appendChild(search);
  if (isLut) {
    const allBtn = el("button", "btn small", "All");
    allBtn.title = "Show every class";
    allBtn.addEventListener("click", () => {
      for (const st of cs.values()) st.visible = true;
      applyClassification(opt);
      buildMappingPanel(opt);
    });
    tools.appendChild(allBtn);
  }
  host.appendChild(tools);

  const legend = el("div", "legend");
  host.appendChild(legend);

  const draw = () => {
    legend.innerHTML = "";
    const needle = search.value.trim().toLowerCase();
    const shown = classes.filter((c) =>
      !needle || c.name.toLowerCase().includes(needle) || String(c.value).includes(needle));

    if (shown.length === 0) {
      legend.appendChild(el("div", "empty-note", "No class matches."));
      return;
    }

    for (const c of shown) {
      const st = cs.get(c.index);
      const row = el("div", "legend-item");
      if (!st.visible) row.classList.add("hidden-class");

      // colour swatch doubles as a colour picker
      const sw = el("div", "sw");
      sw.style.background = rgbCss(st.color);
      const picker = el("input");
      picker.type = "color";
      picker.value = rgbHex(st.color);
      picker.addEventListener("input", () => {
        st.color = hexRgb(picker.value);
        sw.style.background = rgbCss(st.color);
        if (isLut) applyClassification(opt);
        else refreshCategoricalGradient();
      });
      sw.appendChild(picker);
      row.appendChild(sw);

      const name = el("div", "nm", c.name);
      name.title = `value ${c.value} · index ${c.index}`;
      row.appendChild(name);

      const pct = (100 * c.count) / total;
      const cnt = el("div", "cnt", pct >= 0.1 ? `${pct.toFixed(1)}%` : "<0.1%");
      cnt.title = `${fmtInt(c.count)} points`;
      row.appendChild(cnt);

      if (isLut) {
        const solo = el("div", "solo", "only");
        solo.title = "Show only this class";
        solo.addEventListener("click", () => {
          for (const [idx, s] of cs) s.visible = idx === c.index;
          applyClassification(opt);
          draw();
        });
        row.appendChild(solo);

        const eye = el("div", "eye", st.visible ? "◉" : "○");
        eye.title = st.visible ? "Hide this class" : "Show this class";
        eye.addEventListener("click", () => {
          st.visible = !st.visible;
          applyClassification(opt);
          draw();
        });
        row.appendChild(eye);
      }
      legend.appendChild(row);
    }
  };
  search.addEventListener("input", draw);
  draw();

  if (isLut) {
    host.appendChild(el("div", "note",
      "This field drives Potree's class lookup table, so classes can be recoloured and hidden live."));
  } else {
    const note = el("div", "note");
    note.textContent = "Shown as flat colour bands. Only the class field chosen at conversion time " +
      "supports hiding individual classes — reconvert with this field as the class field to get that.";
    host.appendChild(note);

    const btn = el("button", "btn", `Make "${opt.source}" the class field`);
    btn.style.marginTop = "8px";
    btn.addEventListener("click", () => reconvertWithPrimary(opt.source));
    host.appendChild(btn);
  }
}

function refreshCategoricalGradient() {
  gradientCache = { key: null, value: null };
  applyToAll();
}

async function reconvertWithPrimary(field) {
  const scene = state.scenes.find((s) => s.id === state.currentId);
  if (!scene) return;
  try {
    setStage(`Reconverting ${scene.name}`, `Using "${field}" as the class field…`, null);
    const jobScene = await new Promise((resolve, reject) => {
      fetchJson(`/api/scenes/${encId(scene.id)}/convert?force=1&primary=${encodeURIComponent(field)}`,
        { method: "POST" })
        .then(({ jobId }) => {
          const es = new EventSource(`/api/jobs/${jobId}/events`);
          es.onmessage = (ev) => {
            const msg = JSON.parse(ev.data);
            setStage(`Reconverting ${scene.name}`, msg.message ?? "", null);
            if (msg.state === "done") { es.close(); resolve(msg.scene); }
            else if (msg.state === "error") { es.close(); reject(new Error(msg.error)); }
          };
          es.onerror = () => { es.close(); reject(new Error("lost contact with the conversion job")); };
        }).catch(reject);
    });
    await refreshScenes();
    await loadOctree(jobScene);
    toast(`"${field}" is now the class field`);
  } catch (err) {
    setStage("Reconversion failed", err.message, null);
    toast(err.message, true);
  }
}

function buildScalarPanel(host, opt) {
  const st = attrStateFor(opt);
  const material = state.pointcloud.material;
  const isIntensity = opt.kind === "intensity";
  const showGradient = !(isIntensity && opt.potreeName === "intensity");

  if (showGradient) {
    const ctl = el("div", "ctl");
    ctl.appendChild(Object.assign(el("label"), { innerHTML: "<span>Colour ramp</span>" }));
    const preview = el("div", "gradient-preview");
    preview.style.background = gradientCss(Potree.Gradients[st.gradient]);
    ctl.appendChild(preview);

    const select = el("select");
    for (const g of GRADIENTS) {
      const o = el("option", null, g.toLowerCase().replace("_", " "));
      o.value = g;
      if (g === st.gradient) o.selected = true;
      select.appendChild(o);
    }
    select.addEventListener("change", () => {
      st.gradient = select.value;
      preview.style.background = gradientCss(Potree.Gradients[st.gradient]);
      applyToAll();
    });
    ctl.appendChild(select);
    host.appendChild(ctl);
  } else {
    host.appendChild(el("div", "note",
      "Greyscale intensity. Pick “Intensity (gradient)” to run it through a colour ramp instead."));
  }

  // value range
  const lo = opt.min ?? 0, hi = opt.max ?? 1;
  const ctl = el("div", "ctl");
  const label = el("label");
  label.appendChild(el("span", null, "Value range"));
  const valLabel = el("span", "val");
  label.appendChild(valLabel);
  ctl.appendChild(label);

  const dual = makeDualRange(lo, hi, st.range, (range) => {
    st.range = range;
    applyToAll();
    valLabel.textContent = `${fmtNum(range[0])} – ${fmtNum(range[1])}`;
  });
  valLabel.textContent = `${fmtNum(st.range[0])} – ${fmtNum(st.range[1])}`;
  ctl.appendChild(dual);

  const readout = el("div", "range-readout");
  readout.appendChild(el("span", null, fmtNum(lo)));
  readout.appendChild(el("span", null, fmtNum(hi)));
  ctl.appendChild(readout);

  const reset = el("button", "btn small", "Reset range");
  reset.addEventListener("click", () => {
    st.range = [lo, hi];
    buildMappingPanel(opt);
    applyToAll();
  });
  ctl.appendChild(reset);
  host.appendChild(ctl);

  const stats = el("dl", "kv");
  for (const [k, v] of [["field", opt.source ?? opt.name], ["min", fmtNum(opt.min)], ["max", fmtNum(opt.max)]]) {
    stats.appendChild(el("dt", null, k));
    stats.appendChild(el("dd", null, String(v)));
  }
  host.appendChild(stats);
}

/** Two overlaid range inputs acting as one min/max control. */
function makeDualRange(min, max, value, onChange) {
  const wrap = el("div", "dual");
  wrap.appendChild(el("div", "track"));
  const fill = el("div", "fill");
  wrap.appendChild(fill);

  const STEPS = 1000;
  const toSlider = (v) => Math.round(((v - min) / ((max - min) || 1)) * STEPS);
  const toValue = (s) => min + (s / STEPS) * (max - min);

  const lo = el("input"), hi = el("input");
  for (const [inp, v] of [[lo, value[0]], [hi, value[1]]]) {
    inp.type = "range";
    inp.min = 0; inp.max = STEPS; inp.step = 1;
    inp.value = toSlider(v);
    wrap.appendChild(inp);
  }

  const sync = () => {
    let a = Number(lo.value), b = Number(hi.value);
    if (a > b) { [a, b] = [b, a]; }
    fill.style.left = `${(a / STEPS) * 100}%`;
    fill.style.width = `${((b - a) / STEPS) * 100}%`;
    onChange([toValue(a), toValue(b)]);
  };
  lo.addEventListener("input", sync);
  hi.addEventListener("input", sync);

  fill.style.left = `${(toSlider(value[0]) / STEPS) * 100}%`;
  fill.style.width = `${((toSlider(value[1]) - toSlider(value[0])) / STEPS) * 100}%`;
  return wrap;
}

// --------------------------------------------------------- rendering panel
function buildRenderPanel() {
  const host = $("#render-content");
  host.innerHTML = "";
  const r = state.render;

  // Every control writes to the shared settings and then re-styles all clouds,
  // so placed objects never drift away from the scene's look.
  const change = (key, value) => { r[key] = value; applyToAll(); };

  host.appendChild(slider("Point budget", 0.1, 10, 0.1, r.budget,
    (v) => { r.budget = v; viewer.setPointBudget(v * 1e6); }, (v) => `${v.toFixed(1)}M`));

  host.appendChild(slider("Point size", 0.1, 3, 0.05, r.size,
    (v) => change("size", v), (v) => v.toFixed(2)));

  host.appendChild(dropdown("Size mode", {
    ADAPTIVE: "adaptive", FIXED: "fixed", ATTENUATED: "attenuated",
  }, r.sizeType, (k) => change("sizeType", k)));

  host.appendChild(dropdown("Point shape", {
    SQUARE: "square", CIRCLE: "circle", PARABOLOID: "paraboloid",
  }, r.shape, (k) => change("shape", k)));

  host.appendChild(slider("Opacity", 0.05, 1, 0.05, r.opacity,
    (v) => change("opacity", v), (v) => v.toFixed(2)));

  host.appendChild(dropdown("Background", {
    gradient: "gradient", skybox: "skybox", black: "black", white: "white", none: "none",
  }, r.background, (k) => { r.background = k; viewer.setBackground(k); }));

  host.appendChild(toggle("Eye-dome lighting", r.edl,
    (on) => { r.edl = on; viewer.setEDLEnabled(on); }));
  host.appendChild(toggle("Bounding box", r.bbox,
    (on) => change("bbox", on)));
}

function slider(labelText, min, max, step, value, onInput, fmt = String) {
  const ctl = el("div", "ctl");
  const label = el("label");
  label.appendChild(el("span", null, labelText));
  const val = el("span", "val", fmt(value));
  label.appendChild(val);
  ctl.appendChild(label);

  const input = el("input");
  input.type = "range";
  input.min = min; input.max = max; input.step = step; input.value = value;
  input.addEventListener("input", () => {
    const v = Number(input.value);
    val.textContent = fmt(v);
    onInput(v);
  });
  ctl.appendChild(input);
  return ctl;
}

function dropdown(labelText, options, initial, onChange) {
  const ctl = el("div", "ctl");
  const label = el("label");
  label.appendChild(el("span", null, labelText));
  ctl.appendChild(label);

  const select = el("select");
  for (const [k, text] of Object.entries(options)) {
    const o = el("option", null, text);
    o.value = k;
    if (k === initial) o.selected = true;
    select.appendChild(o);
  }
  select.addEventListener("change", () => onChange(select.value));
  ctl.appendChild(select);
  return ctl;
}

function toggle(labelText, initial, onChange) {
  const row = el("div", "toggle-row");
  row.appendChild(el("span", null, labelText));
  const sw = el("div", `switch${initial ? " on" : ""}`);
  sw.addEventListener("click", () => {
    const on = !sw.classList.contains("on");
    sw.classList.toggle("on", on);
    onChange(on);
  });
  row.appendChild(sw);
  return row;
}

// -------------------------------------------------------------- info panel
function buildInfoPanel() {
  const info = state.info;
  const host = $("#info-content");
  host.innerHTML = "";

  const rows = [
    ["points", fmtInt(info.numPoints)],
    ["skipped", fmtInt(info.numSkipped)],
    ["source", info.encoding],
    ["nodes", fmtInt(info.numNodes)],
    ["depth", info.depth],
    ["spacing", `${info.spacing.toFixed(3)} m`],
    ["bytes/pt", info.bytesPerPoint],
    ["octree", fmtBytes(info.octreeBytes)],
  ];
  const dl = el("dl", "kv");
  for (const [k, v] of rows) {
    dl.appendChild(el("dt", null, k));
    dl.appendChild(el("dd", null, String(v)));
  }
  host.appendChild(dl);

  const size = [0, 1, 2].map((i) => info.tightBoundingBox.max[i] - info.tightBoundingBox.min[i]);
  const dims = el("dl", "kv");
  dims.style.marginTop = "10px";
  dims.appendChild(el("dt", null, "extent"));
  dims.appendChild(el("dd", null, size.map((v) => v.toFixed(1)).join(" × ") + " m"));
  host.appendChild(dims);

  if (info.numSkipped > 0) {
    host.appendChild(el("div", "note",
      `${fmtInt(info.numSkipped)} point(s) had a non-finite coordinate and were left out.`));
  }
}

// ---------------------------------------------------------- stats overlay
let statsVisible = false;
let frames = 0, lastFpsUpdate = performance.now(), fps = 0;

function tickStats() {
  frames++;
  const now = performance.now();
  if (now - lastFpsUpdate > 500) {
    fps = Math.round((frames * 1000) / (now - lastFpsUpdate));
    frames = 0;
    lastFpsUpdate = now;

    if (statsVisible) {
      const pc = state.pointcloud;
      const visible = pc?.numVisiblePoints ?? 0;
      const nodes = pc?.visibleNodes?.length ?? 0;
      $("#stats-overlay").innerHTML =
        `<div><b>fps</b> ${fps}</div>` +
        `<div><b>visible</b> ${fmtInt(visible)} pts</div>` +
        `<div><b>nodes</b> ${nodes}</div>` +
        `<div><b>loading</b> ${Potree.numNodesLoading}</div>`;
    }
  }
  requestAnimationFrame(tickStats);
}
requestAnimationFrame(tickStats);

// ------------------------------------------------------------------ wiring
$("#scene-search").addEventListener("input", (e) => {
  state.filter = e.target.value;
  renderSceneList();
});
$("#refresh-scenes").addEventListener("click", refreshScenes);
$("#toggle-scenes").addEventListener("click", () => $("#app").classList.toggle("hide-scenes"));
$("#toggle-controls").addEventListener("click", () => $("#app").classList.toggle("hide-controls"));
$("#btn-fit").addEventListener("click", () => { if (state.pointcloud) viewer.fitToScreen(1.0); });
$("#btn-stats").addEventListener("click", (e) => {
  statsVisible = !statsVisible;
  $("#stats-overlay").classList.toggle("visible", statsVisible);
  e.currentTarget.classList.toggle("on", statsVisible);
});

for (const btn of document.querySelectorAll("#left-tabs button")) {
  btn.addEventListener("click", () => {
    for (const b of document.querySelectorAll("#left-tabs button")) b.classList.toggle("on", b === btn);
    for (const page of document.querySelectorAll(".tab-page")) {
      page.classList.toggle("on", page.dataset.page === btn.dataset.tab);
    }
    if (btn.dataset.tab === "segmentation") segmentation.loadPreview();
  });
}

for (const section of document.querySelectorAll(".section")) {
  section.querySelector("header").addEventListener("click", () => section.classList.toggle("collapsed"));
}

refreshScenes();
