/**
 * Instance library: browse the objects extracted from segmented scenes, spawn
 * copies into the current scene, and move them around.
 *
 * A spawned instance is a real Potree PointCloudOctree added to
 * viewer.scene.pointclouds, so it goes through the same renderer as the scene
 * itself and keeps eye-dome lighting, adaptive point size and the shared point
 * budget. Placement uses Potree's own GPU picking, so objects land on whatever
 * surface is under the cursor.
 */
import {
  $, el, fmtInt, fmtNum, rgbCss, drawSilhouette, fetchJson, toast, confirmDialog, encId,
} from "./util.js";

export function createInstanceManager({
  viewer, getSceneInfo, getActiveAttributeName, reloadScene, styleMaterial,
}) {
  // Potree bundles three.js privately; borrow the constructors from live objects.
  const V3 = viewer.scene.view.position.constructor;
  // Box3Helper extends THREE.LineSegments, so its prototype chain hands us the
  // classes needed to build a custom outline.
  const LineSegments = Object.getPrototypeOf(Potree.Box3Helper);
  let BufferGeometry = null, BufferAttribute = null, LineMaterial = null;
  {
    const probe = new Potree.Box3Helper(
      { min: new V3(0, 0, 0), max: new V3(1, 1, 1) }, 0xffffff);
    BufferGeometry = probe.geometry.constructor;
    BufferAttribute = probe.geometry.attributes.position.constructor;
    LineMaterial = probe.material.constructor;
    probe.geometry.dispose();
    probe.material.dispose();
  }

  /**
   * Outlines live in their own scene, rendered after Potree has finished.
   *
   * viewer.scene.scene is drawn *before* the eye-dome-lighting pass, and that
   * pass composites the point cloud over the whole frame -- so anything added
   * there is painted over by the points. Every render path dispatches
   * "render.pass.end" once it is done, which is the one place an overlay is
   * guaranteed to survive.
   */
  const overlayScene = new (viewer.scene.scene.constructor)();
  viewer.addEventListener("render.pass.end", () => {
    if (overlayScene.children.length === 0) return;
    viewer.renderer.render(overlayScene, viewer.scene.getActiveCamera());
  });

  const api = {};
  const state = {
    library: null,
    groupBy: "class",       // "class" | "scene"
    filter: "",
    placed: [],             // { uid, entry, pointcloud, pos, yaw, scale, size, anchorLocal, helper }
    selected: null,
    placing: null,          // { entry, pointcloud, committed:false }
    uid: 1,
    dragging: null,
    highlightStyle: "contour",  // "box" | "contour"
    resampleOnScale: true,      // rescale by resampling rather than stretching
    inspecting: null,           // { entry, helper } -- a scene instance, not a placed one
  };
  api.state = state;

  // ---------------------------------------------------------------- maths
  const area = $("#render-area");

  function mouseRay(event) {
    const rect = viewer.renderer.domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    const camera = viewer.scene.getActiveCamera();
    const target = new V3(x, y, 0.5).unproject(camera);
    const origin = camera.position.clone();
    return { origin, dir: target.sub(origin).normalize() };
  }

  function mouseCanvasXY(event) {
    const rect = viewer.renderer.domElement.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  /** Ray vs the oriented box of a placed instance; returns distance or null. */
  function rayHit(ray, item) {
    const c = Math.cos(-item.yaw), s = Math.sin(-item.yaw);
    const ox = ray.origin.x - item.pos[0];
    const oy = ray.origin.y - item.pos[1];
    const oz = ray.origin.z - item.pos[2];
    const o = [ox * c - oy * s, ox * s + oy * c, oz];
    const d = [ray.dir.x * c - ray.dir.y * s, ray.dir.x * s + ray.dir.y * c, ray.dir.z];

    const sc = item.scale;
    // Pad a little so small objects stay easy to grab.
    const pad = Math.max(0.12, 0.05 * Math.max(...item.size) * sc);
    const min = [-item.size[0] / 2 * sc - pad, -item.size[1] / 2 * sc - pad, -pad];
    const max = [item.size[0] / 2 * sc + pad, item.size[1] / 2 * sc + pad, item.size[2] * sc + pad];

    let tmin = -Infinity, tmax = Infinity;
    for (let a = 0; a < 3; a++) {
      if (Math.abs(d[a]) < 1e-9) {
        if (o[a] < min[a] || o[a] > max[a]) return null;
        continue;
      }
      let t1 = (min[a] - o[a]) / d[a];
      let t2 = (max[a] - o[a]) / d[a];
      if (t1 > t2) { const t = t1; t1 = t2; t2 = t; }
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) return null;
    }
    return tmax < 0 ? null : (tmin >= 0 ? tmin : tmax);
  }

  /** Intersection with a horizontal plane, used when nothing is under the cursor. */
  function planePoint(ray, z) {
    if (Math.abs(ray.dir.z) < 1e-6) return null;
    const t = (z - ray.origin.z) / ray.dir.z;
    if (t < 0) return null;
    return [
      ray.origin.x + ray.dir.x * t,
      ray.origin.y + ray.dir.y * t,
      z,
    ];
  }

  /**
   * Where should an object go for this cursor position?
   * Prefers the real surface under the cursor (Potree's GPU picking), and falls
   * back to a horizontal plane through the scene's floor.
   */
  function dropPoint(event, exclude = null) {
    // Everything except the object being moved, so it cannot land on itself.
    // The ghost is a preview, not geometry to drop things onto.
    const clouds = viewer.scene.pointclouds.filter((pc) => pc !== exclude && !isGhost(pc));

    if (clouds.length > 0) {
      try {
        const hit = Potree.Utils.getMousePointCloudIntersection(
          mouseCanvasXY(event), viewer.scene.getActiveCamera(), viewer, clouds, { pickClipped: false });
        if (hit && hit.location) return [hit.location.x, hit.location.y, hit.location.z];
      } catch { /* picking can fail while nodes stream in */ }
    }

    const info = getSceneInfo();
    const floor = info ? info.tightBoundingBox.min[2] : 0;
    return planePoint(mouseRay(event), floor);
  }

  // ------------------------------------------------------------ spawning
  function loadInstance(url, name) {
    return new Promise((resolve, reject) => {
      let settled = false;
      Potree.loadPointCloud(url, name, (e) => {
        if (settled) return;              // Potree's loader can fire twice
        settled = true;
        resolve(e.pointcloud);
      });
      setTimeout(() => { if (!settled) { settled = true; reject(new Error("instance failed to load")); } }, 30000);
    });
  }

  /**
   * A spawned object is drawn exactly like the scene: same colour mode, ramp,
   * value ranges, point size and shape. The viewer owns that definition, so
   * this just hands the cloud over to it.
   */
  function styleInstance(pc, { preview = false } = {}) {
    pc.__pcitPreview = preview;
    styleMaterial(pc);
  }

  function applyTransform(item) {
    const pc = item.pointcloud;
    // When an object has been resampled, its geometry already carries part of
    // the scale, so only the remainder goes on the transform.
    const effective = item.scale / (item.bakedScale ?? 1);
    pc.position.set(0, 0, 0);
    pc.rotation.set(0, 0, item.yaw);
    pc.scale.set(effective, effective, effective);
    pc.updateMatrix();

    // The octree's origin is the corner of its bounding cube. Compensate with
    // the anchor so the object rotates and scales about its own footprint.
    const v = new V3(item.anchorLocal[0], item.anchorLocal[1], item.anchorLocal[2]).applyMatrix4(pc.matrix);
    pc.position.set(item.pos[0] - v.x, item.pos[1] - v.y, item.pos[2] - v.z);
    pc.updateMatrix();
    pc.updateMatrixWorld(true);

    // Point size is compensated for the object's scale, so re-style whenever the
    // transform changes.
    styleMaterial(pc);

    if (item.helper) {
      item.helper.material.color.setHex(item.locked ? HIGHLIGHT_LOCKED : HIGHLIGHT_COLOR);
      item.helper.position.set(item.pos[0], item.pos[1], item.pos[2]);
      item.helper.rotation.set(0, 0, item.yaw);
      item.helper.scale.set(item.scale, item.scale, item.scale);
      item.helper.updateMatrixWorld(true);
    }
  }

  /** Outline styling is a viewer-wide preference, shared by every highlight. */
  // WebGL ignores line width, so an outline is always one pixel: contrast is the
  // only thing that makes it readable. These deliberately avoid the class
  // palette, or a blue outline would vanish against a blue car.
  const HIGHLIGHT_COLOR = 0xffffff;    // selected, movable
  const HIGHLIGHT_LOCKED = 0xffb020;   // selected, locked
  const HIGHLIGHT_INSPECT = 0x22e6ff;  // inspecting something in the scene
  const HIGHLIGHT_GHOST = 0xa88bff;    // a translucent stand-in at another position

  function styleOutline(line, color) {
    // A selection outline should never be hidden by the points it surrounds.
    line.material.color.setHex(color);
    line.material.depthTest = false;
    line.material.depthWrite = false;
    line.material.transparent = true;
    line.renderOrder = 10;
    return line;
  }

  /** Axis-aligned box around the object's extents. */
  function buildBox(size, color) {
    const [sx, sy, sz] = size;
    const Box3Ctor = viewer.scene.pointclouds[0]?.boundingBox?.constructor;
    if (!Box3Ctor) return null;
    const box = new Box3Ctor(new V3(-sx / 2, -sy / 2, 0), new V3(sx / 2, sy / 2, sz));
    return styleOutline(new Potree.Box3Helper(box, color), color);
  }

  /**
   * Outline that follows the object's actual footprint: the convex hull of its
   * points, drawn at the base and at full height with verticals between. Much
   * closer to the object than a bounding box, especially for anything not
   * aligned to the axes.
   */
  function buildContour(size, footprint, color) {
    if (!footprint || footprint.length < 3) return buildBox(size, color);
    const h = size[2];
    const n = footprint.length;
    const verts = [];
    const push = (a, b) => verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);

    for (let i = 0; i < n; i++) {
      const p = footprint[i], q = footprint[(i + 1) % n];
      push([p[0], p[1], 0], [q[0], q[1], 0]);       // base loop
      push([p[0], p[1], h], [q[0], q[1], h]);       // top loop
      push([p[0], p[1], 0], [p[0], p[1], h]);       // vertical edge
    }

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(verts), 3));
    const line = new LineSegments(geometry, new LineMaterial({ color }));
    return styleOutline(line, color);
  }

  function makeHelper(item, color = HIGHLIGHT_COLOR) {
    if (!viewer.scene.pointclouds[0]) return null;
    return state.highlightStyle === "contour"
      ? buildContour(item.size, item.footprint ?? item.entry?.footprint, color)
      : buildBox(item.size, color);
  }

  /** Swaps every existing outline over when the style preference changes. */
  function rebuildHighlights() {
    for (const item of state.placed) {
      if (!item.helper) continue;
      const visible = item.helper.visible;
      overlayScene.remove(item.helper);
      item.helper.geometry.dispose();
      item.helper.material.dispose();
      item.helper = makeHelper(item);
      if (item.helper) {
        item.helper.visible = visible;
        overlayScene.add(item.helper);
      }
      applyTransform(item);
    }
    if (state.inspecting) inspectInstance(state.inspecting.entry, { zoom: false });
  }

  async function spawn(entry, pos, { select = true } = {}) {
    const pc = await loadInstance(entry.url, `${entry.className} #${entry.id}`);
    styleInstance(pc);

    const item = {
      uid: state.uid++,
      entry,
      pointcloud: pc,
      pos: pos.slice(),
      yaw: 0,
      scale: 1,
      size: entry.size,
      footprint: entry.footprint,
      anchorLocal: entry.anchorLocal,
      locked: false,
      resample: state.resampleOnScale,
      bakedScale: 1,
      octreeDir: null,
      helper: null,
    };
    viewer.scene.addPointCloud(pc);
    item.helper = makeHelper(item);
    if (item.helper) overlayScene.add(item.helper);
    applyTransform(item);
    state.placed.push(item);
    renderPlacedList();
    if (select) selectItem(item);
    return item;
  }

  function toggleLock(item) {
    item.locked = !item.locked;
    applyTransform(item);
    renderPlacedList();
    renderTransformPanel();
  }
  api.toggleLock = toggleLock;

  function removeItem(item) {
    if (item.locked) { toast(`${item.entry.className} #${item.entry.id} is locked`, true); return; }
    viewer.scene.scenePointCloud.remove(item.pointcloud);
    const i = viewer.scene.pointclouds.indexOf(item.pointcloud);
    if (i !== -1) viewer.scene.pointclouds.splice(i, 1);
    if (item.helper) overlayScene.remove(item.helper);
    const j = state.placed.indexOf(item);
    if (j !== -1) state.placed.splice(j, 1);
    if (state.selected === item) selectItem(null);
    renderPlacedList();
  }

  api.clearPlaced = ({ force = false } = {}) => {
    for (const item of state.placed.slice()) {
      if (item.locked && !force) continue;
      item.locked = false;
      removeItem(item);
    }
    cancelPlacement();
    clearInspection();
    disposeGhost();
  };

  /** Drops everything, locked included -- used when a new scene is loaded. */
  api.resetPlaced = () => api.clearPlaced({ force: true });

  function selectItem(item) {
    if (item && state.inspecting) clearInspection();
    state.selected = item;
    for (const p of state.placed) {
      if (p.helper) p.helper.visible = p === item;
    }
    renderPlacedList();
    renderTransformPanel();

    if (item) {
      // The controls live below the (often long) class legend, so bring them up.
      const section = $("#sec-placed");
      section?.classList.remove("collapsed");
      section?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
  api.selectItem = selectItem;

  // ---------------------------------------------------------- placement
  async function beginPlacement(entry) {
    if (!entry.url) {
      toast(`${entry.className} #${entry.id} is a backdrop — there is no object to place`, true);
      return;
    }
    cancelPlacement();
    const session = { entry, pointcloud: null, cancelled: false };
    state.placing = session;
    $("#app").classList.add("placing");
    setHint(`Click to place ${entry.className} #${entry.id} · Shift-click to place several · Esc to cancel`);

    try {
      const pc = await loadInstance(entry.url, `preview ${entry.id}`);
      if (session.cancelled) return;
      styleInstance(pc, { preview: true });
      viewer.scene.addPointCloud(pc);
      session.pointcloud = pc;
      session.item = {
        pos: [0, 0, 0], yaw: 0, scale: 1,
        size: entry.size, anchorLocal: entry.anchorLocal, pointcloud: pc, helper: null,
      };
      applyTransform(session.item);
    } catch (err) {
      toast(err.message, true);
      cancelPlacement();
    }
  }

  function cancelPlacement() {
    const s = state.placing;
    state.placing = null;
    $("#app").classList.remove("placing");
    setHint(null);
    if (!s) return;
    s.cancelled = true;
    if (s.pointcloud) {
      viewer.scene.scenePointCloud.remove(s.pointcloud);
      const i = viewer.scene.pointclouds.indexOf(s.pointcloud);
      if (i !== -1) viewer.scene.pointclouds.splice(i, 1);
    }
  }
  api.cancelPlacement = cancelPlacement;

  function commitPlacement(pos) {
    const s = state.placing;
    if (!s || !s.pointcloud) return;
    const pc = s.pointcloud;
    styleInstance(pc);          // no longer a preview: full opacity

    const item = {
      uid: state.uid++,
      entry: s.entry,
      pointcloud: pc,
      pos: pos.slice(),
      yaw: s.item.yaw,
      scale: s.item.scale,
      size: s.entry.size,
      footprint: s.entry.footprint,
      anchorLocal: s.entry.anchorLocal,
      locked: false,
      resample: state.resampleOnScale,
      bakedScale: 1,
      octreeDir: null,
      helper: null,
    };
    item.helper = makeHelper(item);
    if (item.helper) overlayScene.add(item.helper);
    applyTransform(item);
    state.placed.push(item);

    // Keep the tool armed so several copies can be dropped in a row.
    state.placing = null;
    renderPlacedList();
    selectItem(item);
    if (state.keepPlacing) beginPlacement(s.entry);
    else { $("#app").classList.remove("placing"); setHint(null); }
  }

  function setHint(text) {
    let node = $("#place-hint");
    if (!text) { node?.remove(); return; }
    if (!node) {
      node = el("div", null, "");
      node.id = "place-hint";
      $("#viewport").appendChild(node);
    }
    node.textContent = text;
  }

  // ------------------------------------------------------------- input
  // Listeners sit on #render-area (the canvas's parent) in the capture phase,
  // so stopping propagation here keeps Potree's own camera controls from also
  // reacting to the same drag.
  let downAt = null;

  area.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    downAt = { x: event.clientX, y: event.clientY, moved: 0 };

    if (state.placing) return;          // placement commits on mouseup

    const ray = mouseRay(event);
    let best = null, bestT = Infinity;
    for (const item of state.placed) {
      const t = rayHit(ray, item);
      if (t != null && t < bestT) { bestT = t; best = item; }
    }
    if (best) {
      selectItem(best);
      // A locked object can still be selected (so it can be unlocked), but it is
      // not grabbed -- and the event is left alone so the camera still orbits.
      if (!best.locked) {
        state.dragging = { item: best, vertical: event.shiftKey, start: best.pos.slice() };
        event.stopPropagation();
        event.preventDefault();
      }
    }
  }, true);

  area.addEventListener("mousemove", (event) => {
    if (downAt) downAt.moved += Math.abs(event.movementX) + Math.abs(event.movementY);

    if (state.dragging) {
      const item = state.dragging.item;
      if (state.dragging.vertical || event.shiftKey) {
        // Vertical drag: slide along z on the plane facing the camera.
        const ray = mouseRay(event);
        const cam = viewer.scene.getActiveCamera();
        const nx = cam.position.x - item.pos[0], ny = cam.position.y - item.pos[1];
        const len = Math.hypot(nx, ny) || 1;
        const denom = ray.dir.x * (nx / len) + ray.dir.y * (ny / len);
        if (Math.abs(denom) > 1e-6) {
          const t = ((item.pos[0] - ray.origin.x) * (nx / len) + (item.pos[1] - ray.origin.y) * (ny / len)) / denom;
          if (t > 0) item.pos[2] = ray.origin.z + ray.dir.z * t;
        }
      } else {
        const p = dropPoint(event, item.pointcloud);
        if (p) { item.pos[0] = p[0]; item.pos[1] = p[1]; item.pos[2] = p[2]; }
      }
      applyTransform(item);
      renderTransformPanel();
      event.stopPropagation();
      event.preventDefault();
      return;
    }

    if (state.placing?.item) {
      const p = dropPoint(event, state.placing.pointcloud);
      if (p) {
        state.placing.item.pos = p;
        applyTransform(state.placing.item);
      }
    }
  }, true);

  area.addEventListener("mouseup", (event) => {
    if (event.button !== 0) return;
    const click = downAt && downAt.moved < 5;
    downAt = null;

    if (state.dragging) {
      state.dragging = null;
      event.stopPropagation();
      return;
    }

    if (state.placing && click) {
      // Shift keeps the tool armed so a row of objects can be dropped quickly.
      state.keepPlacing = event.shiftKey;
      const p = dropPoint(event, state.placing.pointcloud) ?? state.placing.item?.pos;
      if (p) commitPlacement(p);
      event.stopPropagation();
      event.preventDefault();
      return;
    }

    // A click that misses every placed object falls through to the scene's own
    // instances, so any object in the cloud can be inspected by clicking it.
    if (click) {
      const ray = mouseRay(event);
      if (state.placed.some((item) => rayHit(ray, item) != null)) return;

      const entry = pickSceneInstance(event);
      if (entry) {
        inspectInstance(entry, { zoom: false });
        if (pickHandler) {
          const handler = pickHandler;
          api.disarmInstancePick();
          handler(entry);            // hand the caller what was clicked
        }
      } else if (!pickHandler) {
        selectItem(null); clearInspection(); renderTransformPanel();
      }
    }
  }, true);

  window.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName ?? "");
    if (typing) return;

    if (event.key === "Escape") {
      if (pickHandler) api.disarmInstancePick({ notify: true });
      else if (state.placing) cancelPlacement();
      else selectItem(null);
      return;
    }
    if (!state.selected) return;
    const item = state.selected;

    if (event.key === "l" && !event.ctrlKey && !event.metaKey) {
      toggleLock(item);
      return;
    }
    if (item.locked) {
      // Everything below moves or removes the object.
      if (/^(Delete|Backspace|\[|\]|Arrow)/.test(event.key)) {
        toast(`${item.entry.className} #${item.entry.id} is locked`, true);
      }
      return;
    }

    if (event.key === "Delete" || event.key === "Backspace") {
      removeItem(item);
      event.preventDefault();
    } else if (event.key === "d" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      duplicate(item);
    } else if (event.key === "[" || event.key === "]") {
      item.yaw += (event.key === "[" ? -1 : 1) * (event.shiftKey ? Math.PI / 36 : Math.PI / 12);
      applyTransform(item);
      renderTransformPanel();
    } else if (event.key.startsWith("Arrow")) {
      const step = event.shiftKey ? 0.05 : 0.25;
      if (event.key === "ArrowLeft") item.pos[0] -= step;
      if (event.key === "ArrowRight") item.pos[0] += step;
      if (event.key === "ArrowUp") item.pos[1] += step;
      if (event.key === "ArrowDown") item.pos[1] -= step;
      applyTransform(item);
      renderTransformPanel();
      event.preventDefault();
    }
  });

  async function duplicate(item) {
    const copy = await spawn(item.entry, [item.pos[0] + item.size[0] * 1.2, item.pos[1], item.pos[2]]);
    copy.yaw = item.yaw;
    copy.scale = item.scale;
    applyTransform(copy);
    renderTransformPanel();
  }

  // ------------------------------------------------------------- library UI
  api.loadLibrary = async function loadLibrary() {
    try {
      state.library = await fetchJson("/api/library");
      renderLibrary();
    } catch (err) {
      toast(`Could not load the instance library: ${err.message}`, true);
    }
  };

  function matches(entry, needle) {
    if (!needle) return true;
    return `${entry.className} ${entry.sceneName} ${entry.id}`.toLowerCase().includes(needle);
  }

  function renderLibrary() {
    const host = $("#library-list");
    host.innerHTML = "";
    const lib = state.library;

    if (!lib || lib.count === 0) {
      const note = el("div", "empty-note");
      note.innerHTML = "No instances extracted yet.<br><br>They are built automatically when a scene " +
        "with an instance field is converted.";
      host.appendChild(note);
      return;
    }

    const needle = state.filter.trim().toLowerCase();
    const items = lib.instances.filter((e) => matches(e, needle));
    if (items.length === 0) {
      host.appendChild(el("div", "empty-note", "Nothing matches that filter."));
      return;
    }

    const groups = new Map();
    for (const entry of items) {
      const key = state.groupBy === "class" ? entry.className : entry.sceneName;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(entry);
    }

    for (const [key, entries] of [...groups].sort((a, b) => b[1].length - a[1].length)) {
      const head = el("div", "lib-group");
      const dot = el("span", "lib-dot");
      dot.style.background = state.groupBy === "class" ? rgbCss(entries[0].classColor) : "#54606e";
      head.appendChild(dot);
      head.appendChild(el("span", "lib-group-name", key));
      head.appendChild(el("span", "lib-group-count", String(entries.length)));
      host.appendChild(head);

      for (const entry of entries) host.appendChild(renderLibraryItem(entry));
    }
  }

  function renderLibraryItem(entry) {
    const info = getSceneInfo();
    const inScene = info && entry.sceneId === info.id && !entry.detached;

    const node = el("div", "lib-item");
    if (inScene) node.classList.add("in-scene");
    if (entry.detached) node.classList.add("detached");
    if (entry.background) node.classList.add("backdrop");

    const canvas = el("canvas", "lib-thumb");
    canvas.width = 60; canvas.height = 60;
    drawSilhouette(canvas, entry.silhouette, state.library.silhouetteSize ?? 32, entry.classColor);
    node.appendChild(canvas);

    const txt = el("div", "lib-txt");
    txt.appendChild(el("div", "lib-name",
      state.groupBy === "class" ? `#${entry.id} · ${entry.sceneName}` : `${entry.className} #${entry.id}`));
    const dims = entry.size.map((v) => v.toFixed(1)).join(" × ");
    txt.appendChild(el("div", "lib-meta", `${fmtInt(entry.points)} pts`));
    txt.appendChild(el("div", "lib-meta",
      `${dims} m${entry.detached ? " · detached" : ""}${entry.background ? " · backdrop" : ""}`));

    const actions = el("div", "lib-actions");
    if (!entry.background) {
      const addBtn = el("button", "chip-btn", "＋ Place");
      addBtn.title = "Drop a copy into the scene";
      addBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!getSceneInfo()) { toast("Load a scene first", true); return; }
        beginPlacement(entry);
      });
      actions.appendChild(addBtn);
    }

    if (inScene && entry.background) {
      const eyeBtn = el("button", "chip-btn", "◎ Inspect");
      eyeBtn.title = "Highlight it in the scene";
      eyeBtn.addEventListener("click", (e) => { e.stopPropagation(); inspectInstance(entry); });
      actions.appendChild(eyeBtn);
    } else if (inScene) {
      const eyeBtn = el("button", "chip-btn", "◎ Inspect");
      eyeBtn.title = "Highlight this object where it sits in the scene";
      eyeBtn.addEventListener("click", (e) => { e.stopPropagation(); inspectInstance(entry); });
      actions.appendChild(eyeBtn);

      const cutBtn = el("button", "chip-btn", "✂ Detach");
      cutBtn.title = "Lift this object out of the scene so it can be moved";
      cutBtn.addEventListener("click", (e) => { e.stopPropagation(); detachInstance(entry); });
      actions.appendChild(cutBtn);
    }
    txt.appendChild(actions);
    node.appendChild(txt);

    // Clicking the row does the most likely thing: inspect what is already in
    // the scene, place a copy of anything else.
    node.title = entry.background
      ? "Backdrop: click to inspect it in the scene"
      : (inScene ? "Click to inspect it in the scene" : "Click to place a copy");
    node.addEventListener("click", () => {
      if (!getSceneInfo()) { toast("Load a scene first", true); return; }
      if (inScene || entry.background) inspectInstance(entry);
      else beginPlacement(entry);
    });
    return node;
  }

  // -------------------------------------------------------- placed objects
  function renderPlacedList() {
    const host = $("#placed-list");
    if (!host) return;
    host.innerHTML = "";
    $("#placed-count").textContent = state.placed.length ? String(state.placed.length) : "";
    const integrateBtn = $("#integrate-placed");
    if (integrateBtn) integrateBtn.disabled = state.placed.length === 0;
    const clearBtn = $("#clear-placed");
    if (clearBtn) clearBtn.disabled = state.placed.length === 0;

    if (state.placed.length === 0) {
      host.appendChild(el("div", "empty-note", "Nothing placed yet. Pick an object from the Library tab."));
      return;
    }

    for (const item of state.placed) {
      const row = el("div", "placed-item");
      if (item === state.selected) row.classList.add("selected");

      const dot = el("span", "lib-dot");
      dot.style.background = rgbCss(item.entry.classColor);
      row.appendChild(dot);

      const txt = el("div", "lib-txt");
      txt.appendChild(el("div", "lib-name", `${item.entry.className} #${item.entry.id}`));
      txt.appendChild(el("div", "lib-meta",
        `${item.pos.map((v) => v.toFixed(1)).join(", ")}${item.locked ? " · locked" : ""}`));
      row.appendChild(txt);

      const lock = el("div", `eye lock${item.locked ? " on" : ""}`, item.locked ? "🔒" : "🔓");
      lock.title = item.locked ? "Unlock (L)" : "Lock in place (L)";
      lock.addEventListener("click", (e) => { e.stopPropagation(); toggleLock(item); });
      row.appendChild(lock);

      const del = el("div", "eye", "✕");
      del.title = item.locked ? "Locked" : "Remove";
      if (item.locked) del.classList.add("disabled");
      del.addEventListener("click", (e) => { e.stopPropagation(); removeItem(item); });
      row.appendChild(del);

      row.addEventListener("click", () => selectItem(item));
      host.appendChild(row);
    }
  }
  api.renderPlacedList = renderPlacedList;
  /** Re-renders the panels that depend on which scene is open. */
  api.refreshView = () => { renderLibrary(); renderPlacedList(); renderTransformPanel(); };

  function renderTransformPanel() {
    const host = $("#transform-content");
    if (!host) return;
    host.innerHTML = "";
    const item = state.selected;

    if (!item) {
      if (state.inspecting) {
        const entry = state.inspecting.entry;
        const head = el("div", "transform-head");
        const dot = el("span", "lib-dot");
        dot.style.background = rgbCss(entry.classColor);
        head.appendChild(dot);
        const nm = el("span", "lib-name", `${entry.className} #${entry.id}`);
        nm.style.flex = "1";
        head.appendChild(nm);
        host.appendChild(head);

        const dl = el("dl", "kv");
        for (const [k, v] of [
          ["points", fmtInt(entry.points)],
          ["size", entry.size.map((n) => n.toFixed(2)).join(" × ") + " m"],
          ["position", entry.anchor.map((n) => n.toFixed(1)).join(", ")],
          ["class", entry.className],
        ]) {
          dl.appendChild(el("dt", null, k));
          dl.appendChild(el("dd", null, String(v)));
        }
        host.appendChild(dl);

        const row = el("div", "btn-row");
        row.style.marginTop = "10px";
        if (!entry.background) {
          const detach = el("button", "btn small", "✂ Detach from scene");
          detach.addEventListener("click", () => detachInstance(entry));
          row.appendChild(detach);
        }
        const stop = el("button", "btn small", "Clear");
        stop.addEventListener("click", () => { clearInspection(); renderTransformPanel(); });
        row.appendChild(stop);
        host.appendChild(row);

        host.appendChild(el("div", "note", entry.background
          ? "This is a backdrop — it spans most of the scene, so it is kept for " +
            "inspection only and cannot be placed or detached."
          : "This object is part of the scene. Detaching rebuilds the scene without it " +
            "and hands you a movable copy."));
        return;
      }
      host.appendChild(el("div", "empty-note",
        "Select a placed object to move it, or click one in the viewport."));
      return;
    }

    const title = el("div", "transform-head");
    const dot = el("span", "lib-dot");
    dot.style.background = rgbCss(item.entry.classColor);
    title.appendChild(dot);
    const name = el("span", "lib-name", `${item.entry.className} #${item.entry.id}`);
    name.style.flex = "1";
    title.appendChild(name);

    const lockBtn = el("button", `btn small lock-btn${item.locked ? " on" : ""}`,
      item.locked ? "🔒 Locked" : "🔓 Unlocked");
    lockBtn.title = "Lock stops this object being moved or removed (L)";
    lockBtn.addEventListener("click", () => toggleLock(item));
    title.appendChild(lockBtn);
    host.appendChild(title);

    if (item.locked) {
      host.appendChild(el("div", "note locked-note",
        "Locked: this object cannot be dragged, nudged or removed. Unlock to edit it."));
    }

    // position
    const posRow = el("div", "xyz");
    ["x", "y", "z"].forEach((axis, i) => {
      const cell = el("label", "xyz-cell");
      cell.appendChild(el("span", null, axis));
      const input = el("input");
      input.type = "number";
      input.step = "0.1";
      input.value = item.pos[i].toFixed(2);
      input.disabled = item.locked;
      input.addEventListener("change", () => {
        const v = Number(input.value);
        if (Number.isFinite(v)) { item.pos[i] = v; applyTransform(item); renderPlacedList(); }
      });
      cell.appendChild(input);
      posRow.appendChild(cell);
    });
    host.appendChild(posRow);

    host.appendChild(makeSlider("Rotation", -180, 180, 1, (item.yaw * 180) / Math.PI,
      (v) => { item.yaw = (v * Math.PI) / 180; applyTransform(item); },
      (v) => `${v.toFixed(0)}°`, item.locked));
    host.appendChild(makeSlider("Scale", 0.1, 4, 0.05, item.scale,
      (v) => { item.scale = v; applyTransform(item); },
      (v) => `${v.toFixed(2)}×`, item.locked,
      () => { if (item.resample && !item.locked) resampleItem(item); }));

    const resRow = el("div", "toggle-row");
    resRow.appendChild(el("span", null, "Resample when scaling"));
    const resSwitch = el("div", `switch${item.resample ? " on" : ""}`);
    resSwitch.title = "Rebuild the object at its new size with the scene's point density, " +
      "instead of only spreading its points apart";
    resSwitch.addEventListener("click", () => {
      if (item.locked) return;
      item.resample = !item.resample;
      state.resampleOnScale = item.resample;
      resSwitch.classList.toggle("on", item.resample);
      if (item.resample) resampleItem(item);
    });
    resRow.appendChild(resSwitch);
    host.appendChild(resRow);

    if (item.bakedScale && Math.abs(item.bakedScale - 1) > 0.02) {
      const note = el("div", "lib-meta",
        `resampled at ${item.bakedScale.toFixed(2)}× · ${fmtInt(item.pointcloud.pcoGeometry?.root?.numPoints ?? 0)} pts`);
      note.style.marginTop = "-6px";
      host.appendChild(note);
    }

    const row = el("div", "btn-row");
    row.style.marginTop = "8px";
    const dropBtn = el("button", "btn small", "Drop to surface");
    dropBtn.title = "Sit the object on whatever is directly beneath it";
    dropBtn.disabled = item.locked;
    dropBtn.addEventListener("click", () => dropToSurface(item));
    row.appendChild(dropBtn);

    const dupBtn = el("button", "btn small", "Duplicate");
    dupBtn.addEventListener("click", () => duplicate(item));
    row.appendChild(dupBtn);

    const delBtn = el("button", "btn small", "Remove");
    delBtn.disabled = item.locked;
    delBtn.addEventListener("click", () => removeItem(item));
    row.appendChild(delBtn);
    host.appendChild(row);

    host.appendChild(el("div", "note",
      "Drag in the viewport to move · Shift-drag for height · [ and ] rotate · " +
      "L locks · Ctrl+D duplicates · Delete removes."));
  }
  api.renderTransformPanel = renderTransformPanel;

  /** Casts straight down from just above the object and rests it on the first hit. */
  function dropToSurface(item) {
    const camera = viewer.scene.getActiveCamera();
    const clouds = viewer.scene.pointclouds.filter((pc) => pc !== item.pointcloud);
    if (clouds.length === 0) return;

    // Project the object's anchor to the screen and pick there: the nearest
    // surface along that ray is what it should rest on.
    const p = new V3(item.pos[0], item.pos[1], item.pos[2]).project(camera);
    const rect = viewer.renderer.domElement.getBoundingClientRect();
    const mouse = { x: ((p.x + 1) / 2) * rect.width, y: ((1 - p.y) / 2) * rect.height };
    try {
      const hit = Potree.Utils.getMousePointCloudIntersection(mouse, camera, viewer, clouds, {});
      if (hit && hit.location) {
        item.pos = [hit.location.x, hit.location.y, hit.location.z];
        applyTransform(item);
        renderPlacedList();
        renderTransformPanel();
        return;
      }
    } catch { /* fall through */ }
    toast("Nothing underneath to rest on", true);
  }

  function makeSlider(label, min, max, step, value, onInput, fmt, disabled = false, onCommit = null) {
    const ctl = el("div", "ctl");
    const lab = el("label");
    lab.appendChild(el("span", null, label));
    const val = el("span", "val", fmt(value));
    lab.appendChild(val);
    ctl.appendChild(lab);
    const input = el("input");
    input.type = "range";
    input.min = min; input.max = max; input.step = step; input.value = value;
    input.disabled = disabled;
    input.addEventListener("input", () => {
      const v = Number(input.value);
      val.textContent = fmt(v);
      onInput(v);
    });
    // "change" fires when the drag ends, which is when an expensive follow-up
    // like resampling should run.
    if (onCommit) input.addEventListener("change", () => onCommit(Number(input.value)));
    ctl.appendChild(input);
    return ctl;
  }

  /**
   * Identifies the scene instance under the cursor.
   *
   * Potree's GPU pick returns every attribute of the point it hit, so the
   * instance id comes straight from the data. That beats testing bounding
   * boxes: a long thin wall has a box that swallows half the scene, and a ray
   * aimed at a car parked in front of it would hit the wall's box first.
   */
  function pickSceneInstance(event) {
    const info = getSceneInfo();
    if (!info) return null;
    const scenePc = viewer.scene.pointclouds.find(
      (pc) => !state.placed.some((i) => i.pointcloud === pc)
        && pc !== state.placing?.pointcloud && !isGhost(pc));
    if (!scenePc) return null;

    let hit = null;
    try {
      hit = Potree.Utils.getMousePointCloudIntersection(
        mouseCanvasXY(event), viewer.scene.getActiveCamera(), viewer, [scenePc], {});
    } catch { return null; }

    const raw = hit?.point?.instance;
    if (raw == null) return null;
    const id = Math.round(raw.length ? raw[0] : raw);
    return sceneInstances().find((e) => e.id === id && !e.detached) ?? null;
  }

  /**
   * Rebuilds a placed object at its current scale with the scene's own point
   * density, then swaps the new octree in.
   *
   * Without this, scaling only spreads the existing points: enlarge an object
   * and it turns sparse, shrink it and the points clot together. Here the object
   * is genuinely resampled, so it keeps the density of the cloud around it.
   */
  async function resampleItem(item) {
    const info = getSceneInfo();
    if (!info || !item.entry.url) return;
    const wanted = item.scale;
    if (Math.abs(wanted - (item.bakedScale ?? 1)) < 0.02) return;

    const finest = info.spacing / Math.pow(2, info.depth ?? 0);
    item.busy = true;
    setHint(`Resampling ${item.entry.className} #${item.entry.id} at ${wanted.toFixed(2)}x …`);
    try {
      const q = `scale=${wanted}&spacing=${finest}`;
      const started = await fetchJson(
        `/api/scenes/${encId(item.entry.sceneId)}/instances/${item.entry.id}/resample?${q}`,
        { method: "POST" });

      if (!started.cached) await waitForJob(started.jobId);
      if (item.scale !== wanted) { item.busy = false; setHint(null); return; }  // moved on

      const pc = await loadInstance(started.url, `${item.entry.className} #${item.entry.id}`);
      // Swap the geometry, keeping position, rotation and the logical scale.
      viewer.scene.scenePointCloud.remove(item.pointcloud);
      const idx = viewer.scene.pointclouds.indexOf(item.pointcloud);
      if (idx !== -1) viewer.scene.pointclouds.splice(idx, 1);

      item.pointcloud = pc;
      item.bakedScale = wanted;
      item.octreeDir = started.dir;
      item.anchorLocal = started.anchorLocal
        ?? pc.pcoGeometry.offset.clone().negate().toArray();
      viewer.scene.addPointCloud(pc);
      styleInstance(pc);
      applyTransform(item);
      renderPlacedList();
      renderTransformPanel();
    } catch (err) {
      toast(`Resample failed: ${err.message}`, true);
    } finally {
      item.busy = false;
      setHint(null);
    }
  }

  function waitForJob(jobId) {
    return new Promise((resolve, reject) => {
      const es = new EventSource(`/api/jobs/${jobId}/events`);
      es.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.state === "done") { es.close(); resolve(msg.scene); }
        else if (msg.state === "error") { es.close(); reject(new Error(msg.error)); }
      };
      es.onerror = () => { es.close(); reject(new Error("lost contact with the resample job")); };
    });
  }

  // --------------------------------------------------------------- ghost
  /**
   * A translucent stand-in for an object at some other position.
   *
   * Used by the ceteris paribus chart: hovering a point on the curve shows where
   * the object actually was when the model scored it, so the number and the
   * geometry can be read together.
   *
   * The loaded cloud is kept between hovers -- only its presence in the scene is
   * toggled -- so moving along the curve does not reload anything. It is
   * deliberately not a placed object: it cannot be selected, moved or
   * integrated, and it is excluded from picking so it never becomes a surface to
   * drop things onto or a candidate for inspection.
   */
  let ghost = null;   // { key, entry, pointcloud, helper, shown, token }
  let ghostToken = 0;

  function disposeGhost() {
    if (!ghost) return;
    detachGhost();
    if (ghost.helper) {
      ghost.helper.geometry.dispose();
      ghost.helper.material.dispose();
    }
    ghost = null;
  }

  function detachGhost() {
    if (!ghost?.shown) return;
    if (ghost.pointcloud) {
      viewer.scene.scenePointCloud.remove(ghost.pointcloud);
      const i = viewer.scene.pointclouds.indexOf(ghost.pointcloud);
      if (i !== -1) viewer.scene.pointclouds.splice(i, 1);
    }
    if (ghost.helper) overlayScene.remove(ghost.helper);
    ghost.shown = false;
  }

  function placeGhost(pos) {
    if (!ghost?.pointcloud) return;
    const pc = ghost.pointcloud;
    const [ax, ay, az] = ghost.entry.anchorLocal ?? [0, 0, 0];
    pc.position.set(0, 0, 0);
    pc.rotation.set(0, 0, 0);
    pc.scale.set(1, 1, 1);
    pc.updateMatrix();
    // Same anchor compensation as a placed object: the octree's origin is the
    // corner of its bounding cube, not the object's footprint.
    const v = new V3(ax, ay, az).applyMatrix4(pc.matrix);
    pc.position.set(pos[0] - v.x, pos[1] - v.y, pos[2] - v.z);
    pc.updateMatrix();
    pc.updateMatrixWorld(true);

    if (ghost.helper) {
      ghost.helper.position.set(pos[0], pos[1], pos[2]);
      ghost.helper.updateMatrixWorld(true);
    }
  }

  /** Loads the object's cloud without showing it, so the first hover is instant. */
  api.preloadGhost = async function preloadGhost(entry) {
    if (!entry?.url) return;
    const key = `${entry.sceneId}#${entry.id}`;
    if (ghost && ghost.key === key) return;

    disposeGhost();
    const token = ++ghostToken;
    ghost = { key, entry, pointcloud: null, helper: null, shown: false, token };

    let pc;
    try {
      pc = await loadInstance(entry.url, `ghost ${entry.id}`);
    } catch {
      if (ghost?.token === token) ghost = null;
      return;
    }
    if (!ghost || ghost.token !== token) return;   // superseded while loading

    ghost.pointcloud = pc;
    ghost.helper = makeHelper(
      { size: entry.size, footprint: entry.footprint }, HIGHLIGHT_GHOST);
    if (ghost.pending) api.showGhost(entry, ghost.pending);
  };

  api.showGhost = async function showGhost(entry, pos) {
    if (!entry?.url) return;
    const key = `${entry.sceneId}#${entry.id}`;

    if (!ghost || ghost.key !== key) {
      const loading = api.preloadGhost(entry);
      if (ghost) ghost.pending = pos;
      await loading;
      if (!ghost || ghost.key !== key || !ghost.pointcloud) return;
    }
    ghost.pending = pos;
    if (!ghost.pointcloud) return;                // still loading; it will land

    if (!ghost.shown) {
      styleInstance(ghost.pointcloud, { preview: true });
      viewer.scene.addPointCloud(ghost.pointcloud);
      if (ghost.helper) overlayScene.add(ghost.helper);
      ghost.shown = true;
    }
    placeGhost(pos);
  };

  /** Takes the ghost out of the scene but keeps it loaded for the next hover. */
  api.hideGhost = function hideGhost() {
    if (ghost) ghost.pending = null;
    detachGhost();
  };

  /** The ghost must never be mistaken for the scene or a placement target. */
  const isGhost = (pc) => Boolean(ghost && ghost.pointcloud === pc);

  // -------------------------------------------- inspecting scene instances
  /** Library entries that belong to the scene currently open. */
  function sceneInstances() {
    const info = getSceneInfo();
    if (!info || !state.library) return [];
    return state.library.instances.filter((e) => e.sceneId === info.id);
  }
  api.sceneInstances = sceneInstances;

  function clearInspection() {
    if (state.inspecting?.helper) {
      overlayScene.remove(state.inspecting.helper);
      state.inspecting.helper.geometry.dispose();
      state.inspecting.helper.material.dispose();
    }
    state.inspecting = null;
  }
  api.clearInspection = clearInspection;

  /**
   * One-shot "click an object in the viewport" mode.
   *
   * Explicit rather than implicit: clicking around the scene to look at things
   * should not quietly retarget an analysis, so the caller arms this and the
   * next instance click resolves it.
   */
  let pickHandler = null;

  api.armInstancePick = function armInstancePick(callback) {
    pickHandler = callback;
    $("#app").classList.add("picking-object");
    setHint("Click an object in the scene · Esc to cancel");
  };

  api.disarmInstancePick = function disarmInstancePick({ notify = false } = {}) {
    const handler = pickHandler;
    pickHandler = null;
    $("#app").classList.remove("picking-object");
    setHint(null);
    if (notify && handler) handler(null);
  };

  api.isPickingInstance = () => Boolean(pickHandler);

  /**
   * Highlights an instance where it sits in the scene and optionally frames it.
   * Nothing is moved or copied -- this is purely for looking at one object.
   */
  function inspectInstance(entry, { zoom = true } = {}) {
    clearInspection();
    selectItem(null);

    const helper = makeHelper(
      { size: entry.size, footprint: entry.footprint }, HIGHLIGHT_INSPECT);
    if (!helper) return;

    helper.position.set(entry.anchor[0], entry.anchor[1], entry.anchor[2]);
    helper.updateMatrixWorld(true);
    overlayScene.add(helper);
    state.inspecting = { entry, helper };

    if (zoom) {
      const [w, d, h] = entry.size;
      const reach = Math.max(w, d, h) * 2.2 + 4;
      const target = new V3(entry.anchor[0], entry.anchor[1], entry.anchor[2] + h / 2);
      viewer.scene.view.position.set(
        target.x + reach * 0.55, target.y - reach * 0.75, target.z + reach * 0.5);
      viewer.scene.view.lookAt(target);
    }

    renderTransformPanel();
    renderPlacedList();
  }
  api.inspectInstance = inspectInstance;

  /**
   * Lifts an instance out of the scene: the scene octree is rebuilt without its
   * points, and a movable copy is dropped in at the same spot, already selected.
   */
  async function detachInstance(entry) {
    if (entry.background) {
      toast(`${entry.className} #${entry.id} is a backdrop — it cannot be detached`, true);
      return;
    }
    const info = getSceneInfo();
    if (!info || entry.sceneId !== info.id) {
      toast("Only objects from the open scene can be detached", true);
      return;
    }

    const body = el("div");
    body.appendChild(el("p", "modal-lead",
      `${entry.className} #${entry.id} (${fmtInt(entry.points)} points) will be removed from ` +
      `${info.name} and handed to you as a movable object.`));
    body.appendChild(el("p", "modal-lead",
      "Move it wherever you like, then use Integrate into scene to put it back. " +
      "Removing it without integrating leaves the scene without it."));

    const ok = await confirmDialog({
      title: "Detach object from the scene",
      bodyNode: body,
      note: "This rebuilds the scene's cached octree. Re-converting the scene from its PCD " +
            "restores every detached object.",
      confirmText: "Detach",
    });
    if (!ok) return;

    clearInspection();
    setHint(`Detaching ${entry.className} #${entry.id}…`);
    try {
      const scene = await runBake(info.id, [], [entry.id]);
      if (reloadScene) await reloadScene(scene);
      setHint(null);
      // The scene reload clears placed objects, so put the copy in afterwards.
      const item = await spawn(entry, entry.anchor.slice());
      item.detachedFrom = entry.id;
      renderPlacedList();
      toast(`Detached ${entry.className} #${entry.id} — drag to move it`);
    } catch (err) {
      setHint(null);
      toast(`Detach failed: ${err.message}`, true);
    }
  }
  api.detachInstance = detachInstance;

  // ------------------------------------------------- integrate into scene
  /**
   * Merges every placed object into the scene's own octree, permanently.
   * Asks first, because it rewrites the cached octree and cannot be undone
   * from the viewer.
   */
  async function integrate() {
    const info = getSceneInfo();
    if (!info) { toast("Load a scene first", true); return; }
    if (state.placed.length === 0) { toast("Nothing placed to integrate", true); return; }

    // Summarise what is about to be merged.
    const byClass = new Map();
    let points = 0;
    for (const item of state.placed) {
      const k = item.entry.className;
      byClass.set(k, (byClass.get(k) ?? 0) + 1);
      points += item.entry.points;
    }

    const body = el("div");
    body.appendChild(el("p", "modal-lead",
      `${state.placed.length} object${state.placed.length === 1 ? "" : "s"} ` +
      `(${fmtInt(points)} points) will become part of ${info.name}.`));

    const list = el("div", "modal-list");
    for (const [name, n] of [...byClass].sort((a, b) => b[1] - a[1])) {
      const row = el("div", "modal-row");
      const dot = el("span", "lib-dot");
      const sample = state.placed.find((i) => i.entry.className === name);
      dot.style.background = rgbCss(sample.entry.classColor);
      row.appendChild(dot);
      row.appendChild(el("span", "lib-name", name));
      row.appendChild(el("span", "lib-meta", `× ${n}`));
      list.appendChild(row);
    }
    body.appendChild(list);

    const locked = state.placed.filter((i) => i.locked).length;
    body.appendChild(el("p", "modal-lead",
      "Each one keeps its class and is given a new instance id, so it stays a " +
      "distinct object in the segmentation." + (locked ? ` Locked objects (${locked}) are included.` : "")));

    const ok = await confirmDialog({
      title: "Integrate objects into the scene",
      bodyNode: body,
      note: "This rewrites the scene's cached octree. It cannot be undone here, and " +
            "re-converting this scene from its PCD would discard the merged objects.",
      confirmText: "Integrate",
    });
    if (!ok) return;

    const placements = state.placed.map((item) => ({
      sourceSceneId: item.entry.sceneId,
      instanceId: item.entry.id,
      // A resampled object lives in its own octree and already bakes in part of
      // the scale; merge that geometry with only the remainder applied.
      octreeDir: item.octreeDir ?? null,
      // If this object was detached from this scene, hand back the id it had so
      // its library entry is refreshed rather than orphaned.
      restoreInstanceId: item.detachedFrom ?? null,
      className: item.entry.className,
      pos: item.pos,
      yaw: item.yaw,
      scale: item.scale / (item.bakedScale ?? 1),
    }));

    setHint(`Integrating ${placements.length} object(s)…`);
    try {
      const scene = await runBake(info.id, placements);
      api.resetPlaced();
      setHint(null);
      toast(`${placements.length} object(s) merged into ${scene.name}`);
      if (reloadScene) await reloadScene(scene);
    } catch (err) {
      setHint(null);
      toast(`Integration failed: ${err.message}`, true);
    }
  }
  api.integrate = integrate;

  function runBake(sceneId, placements, exclude = []) {
    return new Promise((resolve, reject) => {
      fetchJson(`/api/scenes/${encId(sceneId)}/bake`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ placements, exclude }),
      }).then(({ jobId }) => {
        const es = new EventSource(`/api/jobs/${jobId}/events`);
        es.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          setHint(msg.message ? `${msg.message}…` : "Integrating…");
          if (msg.state === "done") { es.close(); resolve(msg.scene); }
          else if (msg.state === "error") { es.close(); reject(new Error(msg.error)); }
        };
        es.onerror = () => { es.close(); reject(new Error("lost contact with the merge job")); };
      }).catch(reject);
    });
  }

  /** Re-styles placed objects when the scene's colour mode changes. */
  api.syncAppearance = () => {
    for (const item of state.placed) styleInstance(item.pointcloud);
    if (state.placing?.pointcloud) styleInstance(state.placing.pointcloud, { preview: true });
  };

  // ------------------------------------------------------------- wiring
  $("#library-search").addEventListener("input", (e) => {
    state.filter = e.target.value;
    renderLibrary();
  });
  for (const btn of document.querySelectorAll("#library-groupby button")) {
    btn.addEventListener("click", () => {
      state.groupBy = btn.dataset.group;
      for (const b of document.querySelectorAll("#library-groupby button")) {
        b.classList.toggle("on", b === btn);
      }
      renderLibrary();
    });
  }
  for (const btn of document.querySelectorAll("#highlight-style button")) {
    btn.addEventListener("click", () => {
      state.highlightStyle = btn.dataset.style;
      for (const b of document.querySelectorAll("#highlight-style button")) {
        b.classList.toggle("on", b === btn);
      }
      rebuildHighlights();
    });
  }
  $("#clear-placed")?.addEventListener("click", () => api.clearPlaced());
  $("#integrate-placed")?.addEventListener("click", () => integrate());

  renderPlacedList();
  renderTransformPanel();
  return api;
}
