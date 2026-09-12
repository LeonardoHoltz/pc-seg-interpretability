/**
 * Segmentation inference tab.
 *
 * Sends the open scene's points to an external service and folds the labels it
 * returns back into the scene as a `prediction` attribute, which then behaves
 * like any other categorical field: colour bands, a legend, per-class counts.
 *
 * The points never pass through the browser. The viewer only asks the server to
 * run the request, so the payload goes straight from the cached octree to the
 * service as one binary body.
 */
import { $, el, fmtInt, fmtBytes, fetchJson, toast, confirmDialog, encId } from "./util.js";

const ENDPOINT_KEY = "pcit.segmentation.endpoint";

/**
 * Chart colour. Validated against this panel's surface (#14181d) with the
 * dataviz palette validator: lightness band, chroma floor and contrast all pass.
 */
const SERIES = "#3987e5";

export function createSegmentation({
  getSceneInfo, reloadScene, onPrediction, onHoverStep, onHoverEnd, onPreload,
  onSelectAttribute, onPickObject, onCancelPick,
}) {
  const state = {
    preview: null, endpoint: "", running: false, excluded: new Set(),
    // Ceteris paribus: move one object through a range of heights and watch one
    // class's probability respond.
    cp: {
      instances: null, instanceId: null, classValue: null,
      min: null, max: null, steps: 9,
      running: false, status: "", result: null, view: "chart",
    },
    // Saliency: one scalar per point, aggregated however the service sees fit.
    sal: { running: false, status: "" },
  };

  try {
    state.endpoint = localStorage.getItem(ENDPOINT_KEY) ?? "";
  } catch { /* private browsing */ }

  const api = { state };

  // The default endpoint lives in config/app.json, not here, so the deployment
  // decides where the service is. A value the user typed is theirs and always
  // wins; this only fills an empty field, and re-renders if the tab is already
  // on screen by the time the config arrives.
  fetchJson("/api/config").then((cfg) => {
    if (!state.endpoint && cfg?.inference?.endpoint) {
      state.endpoint = cfg.inference.endpoint;
      if ($("#segmentation-body")?.childElementCount) render();
    }
  }).catch(() => { /* the field is editable; a default is a convenience */ });

  // ---------------------------------------------------------------- render
  function render() {
    const host = $("#segmentation-body");
    if (!host) return;
    host.innerHTML = "";

    const info = getSceneInfo();
    if (!info) {
      host.appendChild(el("div", "empty-note",
        "Load a scene first. Its points are what gets sent for inference."));
      return;
    }

    // --- endpoint ---
    const epCtl = el("div", "ctl");
    const epLabel = el("label");
    epLabel.appendChild(el("span", null, "Inference endpoint"));
    epCtl.appendChild(epLabel);
    const ep = el("input");
    ep.type = "text";
    ep.className = "text-input";
    ep.placeholder = "http://host:port/segment";
    ep.value = state.endpoint;
    ep.spellcheck = false;
    ep.addEventListener("change", () => {
      state.endpoint = ep.value.trim();
      try { localStorage.setItem(ENDPOINT_KEY, state.endpoint); } catch { /* ignore */ }
    });
    epCtl.appendChild(ep);
    host.appendChild(epCtl);
    host.appendChild(el("div", "lib-meta",
      "The server posts the points there and waits for the labels."));

    // --- what will be sent ---
    const head = el("div", "seg-head");
    head.appendChild(el("span", null, "Payload"));
    const refresh = el("button", "btn small", "Refresh");
    refresh.addEventListener("click", () => loadPreview(true));
    head.appendChild(refresh);
    host.appendChild(head);

    if (!state.preview) {
      host.appendChild(el("div", "empty-note", "Working out what would be sent…"));
    } else {
      const p = state.preview;
      const included = p.arrays.filter((a) => !state.excluded.has(a.name));
      const bytes = included.reduce((n, a) => n + a.nbytes, 0);

      const summary = el("dl", "kv");
      for (const [k, v] of [
        ["points", fmtInt(p.numPoints)],
        ["arrays", String(included.length)],
        ["payload", fmtBytes(bytes)],
      ]) {
        summary.appendChild(el("dt", null, k));
        summary.appendChild(el("dd", null, v));
      }
      host.appendChild(summary);

      const list = el("div", "seg-arrays");
      for (const a of p.arrays) {
        const row = el("label", "seg-array");
        const box = el("input");
        box.type = "checkbox";
        box.checked = !state.excluded.has(a.name);
        const fixed = a.role === "coordinates" || a.role === "colour";
        box.disabled = fixed;
        box.addEventListener("change", () => {
          if (box.checked) state.excluded.delete(a.name); else state.excluded.add(a.name);
          render();
        });
        row.appendChild(box);

        const txt = el("div", "lib-txt");
        txt.appendChild(el("div", "lib-name", a.name));
        txt.appendChild(el("div", "lib-meta",
          `${a.dtype}  (${a.shape.join(", ")})  ·  ${fmtBytes(a.nbytes)}`));
        row.appendChild(txt);
        list.appendChild(row);
      }
      host.appendChild(list);

      host.appendChild(el("div", "note",
        "Coordinates go as float32 (3, N) and colour as uint8 (3, N) — three contiguous " +
        "per-axis arrays, which is already a C-contiguous [3, N] numpy array. Each scalar " +
        "field goes as its own (N,) array. Nothing is transposed or interleaved."));
    }

    // --- run ---
    const run = el("button", "btn primary", "Run segmentation");
    run.style.marginTop = "12px";
    run.disabled = state.running || !state.preview;
    run.addEventListener("click", () => confirmAndRun());
    host.appendChild(run);

    // --- last result ---
    if (info.prediction) {
      const res = el("div", "seg-result");
      res.appendChild(el("div", "seg-head-plain", "Last result"));
      const dl = el("dl", "kv");
      for (const [k, v] of [
        ["classes", String(info.prediction.numClasses)],
        ["points", fmtInt(info.prediction.numPoints)],
        ["scores", info.prediction.hasScores ? "yes" : "no"],
        ["when", new Date(info.prediction.at).toLocaleTimeString()],
      ]) {
        dl.appendChild(el("dt", null, k));
        dl.appendChild(el("dd", null, v));
      }
      res.appendChild(dl);

      const total = info.prediction.classes.reduce((n, c) => n + c.count, 0) || 1;
      const legend = el("div", "legend");
      legend.style.marginTop = "8px";
      for (const c of info.prediction.classes) {
        const row = el("div", "legend-item");
        const attr = info.attributes.find((a) => a.name === "prediction");
        const colour = attr?.classes?.find((x) => x.value === c.value)?.color ?? [140, 140, 140];
        const sw = el("div", "sw");
        sw.style.background = `rgb(${colour[0]},${colour[1]},${colour[2]})`;
        row.appendChild(sw);
        row.appendChild(el("div", "nm", c.name));
        row.appendChild(el("div", "cnt", `${((100 * c.count) / total).toFixed(1)}%`));
        legend.appendChild(row);
      }
      res.appendChild(legend);

      const show = el("button", "btn small", "Colour by prediction");
      show.style.marginTop = "8px";
      show.addEventListener("click", () => onPrediction?.());
      res.appendChild(show);
      host.appendChild(res);
    }

    const interp = el("div", "seg-interp");
    host.appendChild(interp);
    renderCeteris(interp);
  }
  api.render = render;

  // ------------------------------------------------- ceteris paribus panel
  /**
   * Draws the sweep as a line chart.
   *
   * One measure against one continuous variable, so a line with markers is the
   * right form; a single series needs no legend because the title names it. The
   * grid and axes stay recessive, only the peak is labelled directly, and a
   * table view carries the same numbers for anyone the colour does not reach.
   */
  function drawChart(host, result) {
    const steps = result.steps;
    const W = Math.max(240, host.clientWidth || 292);
    const H = 190;
    const m = { top: 14, right: 12, bottom: 30, left: 40 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;

    const xs = steps.map((s) => s.height);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const xSpan = (xMax - xMin) || 1;
    const px = (h) => m.left + ((h - xMin) / xSpan) * iw;
    const py = (v) => m.top + (1 - Math.max(0, Math.min(1, v))) * ih;

    const svg = (tag, attrs, text) => {
      const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      if (text != null) n.textContent = text;
      return n;
    };

    const root = svg("svg", {
      viewBox: `0 0 ${W} ${H}`, width: "100%", height: String(H),
      class: "cp-chart", role: "img",
      "aria-label": `Mean probability of the selected class against object height`,
    });

    // gridlines + y ticks
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      root.appendChild(svg("line", {
        x1: m.left, x2: m.left + iw, y1: py(v), y2: py(v),
        stroke: "var(--line-soft)", "stroke-width": 1,
      }));
      root.appendChild(svg("text", {
        x: m.left - 7, y: py(v) + 3.5, "text-anchor": "end",
        class: "cp-tick",
      }, v.toFixed(2)));
    }

    // x ticks: at most four, so they never collide in a narrow panel
    const tickCount = Math.min(4, steps.length);
    for (let i = 0; i < tickCount; i++) {
      const h = xMin + (xSpan * i) / (tickCount - 1 || 1);
      root.appendChild(svg("text", {
        x: px(h), y: H - 10, "text-anchor": i === 0 ? "start" : (i === tickCount - 1 ? "end" : "middle"),
        class: "cp-tick",
      }, h.toFixed(1)));
    }
    root.appendChild(svg("text", {
      x: m.left + iw / 2, y: H - 0.5, "text-anchor": "middle", class: "cp-axis",
    }, "object height (m)"));

    // baseline
    root.appendChild(svg("line", {
      x1: m.left, x2: m.left + iw, y1: py(0), y2: py(0),
      stroke: "var(--line)", "stroke-width": 1,
    }));

    // the series
    const d = steps.map((s, i) => `${i ? "L" : "M"}${px(s.height).toFixed(1)},${py(s.value).toFixed(1)}`).join("");
    root.appendChild(svg("path", {
      d, fill: "none", stroke: SERIES, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));

    for (const s of steps) {
      // A surface ring keeps neighbouring markers apart where the line is steep.
      root.appendChild(svg("circle", {
        cx: px(s.height), cy: py(s.value), r: 4,
        fill: SERIES, stroke: "var(--bg-panel)", "stroke-width": 2,
      }));
    }

    // one direct label, on the peak
    const peak = steps.reduce((a, b) => (b.value > a.value ? b : a), steps[0]);
    root.appendChild(svg("text", {
      x: Math.min(px(peak.height) + 8, m.left + iw),
      y: Math.max(py(peak.value) - 8, m.top + 8),
      "text-anchor": px(peak.height) > m.left + iw * 0.7 ? "end" : "start",
      class: "cp-peak",
    }, peak.value.toFixed(3)));

    // hover: crosshair + tooltip
    const crosshair = svg("line", {
      y1: m.top, y2: m.top + ih, stroke: "var(--text-faint)",
      "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0,
    });
    root.appendChild(crosshair);
    const hit = svg("rect", {
      x: m.left, y: m.top, width: iw, height: ih, fill: "transparent",
    });
    root.appendChild(hit);

    const tip = el("div", "cp-tip");
    tip.style.display = "none";
    let hovered = null;

    hit.addEventListener("mousemove", (event) => {
      const box = root.getBoundingClientRect();
      const xPix = ((event.clientX - box.left) / box.width) * W;
      let best = steps[0], bestD = Infinity;
      for (const s of steps) {
        const dd = Math.abs(px(s.height) - xPix);
        if (dd < bestD) { bestD = dd; best = s; }
      }
      crosshair.setAttribute("x1", px(best.height));
      crosshair.setAttribute("x2", px(best.height));
      crosshair.setAttribute("opacity", 1);
      tip.style.display = "";
      tip.textContent = `${best.height.toFixed(2)} m · ${best.value.toFixed(3)}`;
      tip.style.left = `${Math.min(Math.max((px(best.height) / W) * box.width - 40, 0), box.width - 90)}px`;
      tip.style.top = `${(py(best.value) / H) * box.height - 30}px`;

      // Show the object where it actually was when the model scored it.
      if (best !== hovered) {
        hovered = best;
        onHoverStep?.({ instanceId: result.instanceId, height: best.height });
      }
    });
    hit.addEventListener("mouseleave", () => {
      crosshair.setAttribute("opacity", 0);
      tip.style.display = "none";
      hovered = null;
      onHoverEnd?.();
    });

    const wrap = el("div", "cp-chart-wrap");
    wrap.appendChild(root);
    wrap.appendChild(tip);
    host.appendChild(wrap);
  }

  function drawTable(host, result) {
    const table = el("table", "cp-table");
    const head = el("tr");
    head.appendChild(el("th", null, "height (m)"));
    head.appendChild(el("th", null, "mean probability"));
    table.appendChild(head);
    for (const s of result.steps) {
      const tr = el("tr");
      tr.appendChild(el("td", null, s.height.toFixed(2)));
      tr.appendChild(el("td", null, s.value.toFixed(4)));
      tr.addEventListener("mouseenter", () =>
        onHoverStep?.({ instanceId: result.instanceId, height: s.height }));
      table.appendChild(tr);
    }
    table.addEventListener("mouseleave", () => onHoverEnd?.());
    host.appendChild(table);
  }

  /** The Interpretability section: pick an object, a class, a height range. */
  function renderCeteris(host) {
    const info = getSceneInfo();
    const cp = state.cp;

    host.appendChild(el("div", "seg-head-plain", "Interpretability"));
    host.appendChild(el("div", "lib-meta",
      "Probe the model around one object. Both analyses below use this object " +
      "and class."));

    if (!cp.instances) {
      host.appendChild(el("div", "empty-note", "Looking for objects…"));
      return;
    }
    if (cp.instances.length === 0) {
      host.appendChild(el("div", "note",
        "This scene has no instance field, so individual objects cannot be isolated."));
      return;
    }

    // --- object ---
    const objCtl = el("div", "ctl");
    const objLabel = el("label");
    objLabel.appendChild(el("span", null, "Focus object"));
    objCtl.appendChild(objLabel);
    const objRow = el("div", "pick-row");
    const objSel = el("select");
    for (const inst of cp.instances) {
      const o = el("option", null,
        `${inst.class?.name ?? "object"} #${inst.id} · ${fmtInt(inst.points)} pts`);
      o.value = String(inst.id);
      if (inst.id === cp.instanceId) o.selected = true;
      objSel.appendChild(o);
    }
    objSel.addEventListener("change", () => { cp.instanceId = Number(objSel.value); });
    objRow.appendChild(objSel);

    // ...or just click the thing in the scene.
    const pick = el("button", `btn small pick-btn${cp.picking ? " on" : ""}`,
      cp.picking ? "Click an object…" : "◎ Pick");
    pick.title = cp.picking
      ? "Click an object in the viewport, or Esc to cancel"
      : "Choose the object by clicking it in the viewport";
    pick.addEventListener("click", () => {
      if (cp.picking) { cp.picking = false; onCancelPick?.(); render(); return; }
      cp.picking = true;
      render();
      onPickObject?.((entry) => {
        cp.picking = false;
        if (entry) {
          // A backdrop is not in the dropdown, but it is still a valid target.
          if (!cp.instances.some((i) => i.id === entry.id)) {
            cp.instances = [{
              id: entry.id, points: entry.points,
              class: { name: entry.className, color: entry.classColor },
            }, ...cp.instances];
          }
          cp.instanceId = entry.id;
        }
        render();
      });
    });
    objRow.appendChild(pick);
    objCtl.appendChild(objRow);
    host.appendChild(objCtl);

    // --- class ---
    const classes = info?.prediction?.classes ?? null;
    const clsCtl = el("div", "ctl");
    const clsLabel = el("label");
    clsLabel.appendChild(el("span", null, "Class to track"));
    clsCtl.appendChild(clsLabel);
    if (classes && classes.length) {
      const clsSel = el("select");
      for (const c of classes) {
        const o = el("option", null, `${c.name} (${c.value})`);
        o.value = String(c.value);
        if (c.value === cp.classValue) o.selected = true;
        clsSel.appendChild(o);
      }
      if (cp.classValue == null) cp.classValue = classes[0].value;
      clsSel.addEventListener("change", () => { cp.classValue = Number(clsSel.value); });
      clsCtl.appendChild(clsSel);
    } else {
      const num = el("input");
      num.type = "number"; num.step = "1"; num.min = "0";
      num.value = String(cp.classValue ?? 0);
      num.className = "text-input";
      num.addEventListener("change", () => { cp.classValue = Number(num.value); });
      clsCtl.appendChild(num);
      clsCtl.appendChild(el("div", "lib-meta",
        "Run a segmentation once and the class names appear here."));
    }
    host.appendChild(clsCtl);

    // --- height range ---
    const cpBlock = el("div", "seg-block");
    cpBlock.appendChild(el("div", "seg-head-plain", "Ceteris paribus"));
    cpBlock.appendChild(el("div", "lib-meta",
      "Hold the scene fixed, move the object through a range of heights, and " +
      "watch the class's probability respond."));
    host.appendChild(cpBlock);

    const rangeCtl = el("div", "ctl");
    const rangeLabel = el("label");
    rangeLabel.appendChild(el("span", null, "Height sweep (m)"));
    rangeCtl.appendChild(rangeLabel);
    const row = el("div", "xyz cp-range");
    for (const [key, caption] of [["min", "from"], ["max", "to"], ["steps", "steps"]]) {
      const cell = el("label", "xyz-cell");
      cell.appendChild(el("span", null, caption));
      const input = el("input");
      input.type = "number";
      input.step = key === "steps" ? "1" : "0.1";
      input.value = String(key === "steps" ? cp.steps : (cp[key] ?? 0).toFixed(2));
      input.addEventListener("change", () => {
        const v = Number(input.value);
        if (Number.isFinite(v)) cp[key] = key === "steps" ? Math.max(2, Math.round(v)) : v;
      });
      cell.appendChild(input);
      row.appendChild(cell);
    }
    rangeCtl.appendChild(row);
    host.appendChild(rangeCtl);

    const run = el("button", "btn primary", "Run ceteris paribus");
    run.disabled = cp.running || state.running || state.sal.running;
    run.addEventListener("click", () => runCeteris());
    host.appendChild(run);

    if (cp.status) host.appendChild(el("div", "seg-status", cp.status));

    // --- result ---
    if (cp.result) {
      const r = cp.result;
      const name = classes?.find((c) => c.value === r.classValue)?.name
        ?? r.classNames?.[r.classValue] ?? `class ${r.classValue}`;

      const title = el("div", "cp-title");
      title.appendChild(el("div", "cp-title-main", `Mean P(${name}) vs height`));
      title.appendChild(el("div", "lib-meta",
        `object #${r.instanceId} · ${fmtInt(r.objectPoints)} points · ` +
        `one request, ${fmtBytes(r.requestBytes)} out / ${fmtBytes(r.responseBytes)} back`));
      host.appendChild(title);

      const toggle = el("div", "segmented");
      for (const [key, caption] of [["chart", "Chart"], ["table", "Table"]]) {
        const b = el("button", cp.view === key ? "on" : "", caption);
        b.addEventListener("click", () => { cp.view = key; render(); });
        toggle.appendChild(b);
      }
      host.appendChild(toggle);

      const plot = el("div", "cp-plot");
      host.appendChild(plot);
      if (cp.view === "chart") drawChart(plot, r); else drawTable(plot, r);
      // Fetch the object's cloud now so the first hover has nothing to wait for.
      onPreload?.({ instanceId: r.instanceId });
      host.appendChild(el("div", "lib-meta",
        "Hover a point to see the object at that height, drawn translucent in the scene."));
    }

    renderSaliency(host);
  }

  /**
   * Saliency: ask the service for one scalar per point about the focused object.
   * How that number is arrived at -- gradients, occlusion, attention, whatever --
   * and how it is aggregated is the service's business; the viewer just turns it
   * into a scalar field you can colour by.
   */
  function renderSaliency(host) {
    const info = getSceneInfo();
    const cp = state.cp;
    const sal = state.sal;

    const block = el("div", "seg-block");
    block.appendChild(el("div", "seg-head-plain", "Saliency"));
    block.appendChild(el("div", "lib-meta",
      "One scalar per point for the focused object. The service decides what it " +
      "measures and how it is aggregated; it arrives as a scalar field in Colour by."));

    const run = el("button", "btn", "Compute saliency");
    run.style.marginTop = "8px";
    run.disabled = sal.running || cp.running || state.running || cp.instanceId == null;
    run.addEventListener("click", () => runSaliency());
    block.appendChild(run);

    if (sal.status) block.appendChild(el("div", "seg-status", sal.status));

    if (info?.saliency) {
      const dl = el("dl", "kv");
      dl.style.marginTop = "10px";
      for (const [k, v] of [
        ["object", `#${info.saliency.instanceId}`],
        ["covers", info.saliency.scope === "object" ? "the object" : "whole scene"],
        ["range", `${info.saliency.range[0].toFixed(3)} – ${info.saliency.range[1].toFixed(3)}`],
        ["when", new Date(info.saliency.at).toLocaleTimeString()],
      ]) {
        dl.appendChild(el("dt", null, k));
        dl.appendChild(el("dd", null, v));
      }
      block.appendChild(dl);

      const show = el("button", "btn small", "Colour by saliency");
      show.style.marginTop = "8px";
      show.addEventListener("click", () => onSelectAttribute?.("saliency"));
      block.appendChild(show);
    }
    host.appendChild(block);
  }

  async function runSaliency() {
    const info = getSceneInfo();
    const cp = state.cp;
    const sal = state.sal;
    if (!info) return;
    if (!/^https?:\/\//i.test(state.endpoint)) { toast("Set an http(s) endpoint first", true); return; }
    if (cp.instanceId == null) { toast("Pick an object first", true); return; }

    onHoverEnd?.();
    sal.running = true;
    sal.status = "Starting…";
    render();
    try {
      const { jobId } = await fetchJson(`/api/scenes/${encId(info.id)}/saliency`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          endpoint: state.endpoint,
          instanceId: cp.instanceId,
          classValue: cp.classValue,
        }),
      });

      const scene = await new Promise((resolve, reject) => {
        const es = new EventSource(`/api/jobs/${jobId}/events`);
        es.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          sal.status = msg.message ?? "Working…";
          const box = [...document.querySelectorAll("#segmentation-body .seg-status")].pop();
          if (box) box.textContent = sal.status;
          if (msg.state === "done") { es.close(); resolve(msg.scene); }
          else if (msg.state === "error") { es.close(); reject(new Error(msg.error)); }
        };
        es.onerror = () => { es.close(); reject(new Error("lost contact with the saliency job")); };
      });

      sal.running = false;
      sal.status = "";
      toast(`Saliency computed for object #${cp.instanceId}`);
      if (reloadScene) await reloadScene(scene);
      onSelectAttribute?.("saliency");
      await loadPreview(true);
    } catch (err) {
      sal.running = false;
      sal.status = "";
      toast(`Saliency failed: ${err.message}`, true);
      render();
    }
  }

  async function loadInstances() {
    const info = getSceneInfo();
    const cp = state.cp;
    if (!info) { cp.instances = null; return; }
    try {
      const lib = await fetchJson(`/api/scenes/${encId(info.id)}/instances`);
      cp.instances = (lib.instances ?? [])
        .filter((i) => !i.background)
        .sort((a, b) => b.points - a.points);
    } catch {
      cp.instances = [];
    }
    if (cp.instanceId == null && cp.instances.length) cp.instanceId = cp.instances[0].id;
    if (cp.min == null && info.tightBoundingBox) {
      cp.min = info.tightBoundingBox.min[2];
      cp.max = info.tightBoundingBox.max[2];
    }
    if (cp.classValue == null) cp.classValue = info.prediction?.classes?.[0]?.value ?? 0;
  }

  async function runCeteris() {
    const info = getSceneInfo();
    const cp = state.cp;
    if (!info) return;
    onHoverEnd?.();
    if (!/^https?:\/\//i.test(state.endpoint)) { toast("Set an http(s) endpoint first", true); return; }
    if (cp.instanceId == null) { toast("Pick an object first", true); return; }

    cp.running = true;
    cp.status = "Starting…";
    render();
    try {
      const { jobId } = await fetchJson(`/api/scenes/${encId(info.id)}/ceteris`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          endpoint: state.endpoint,
          instanceId: cp.instanceId,
          classValue: cp.classValue,
          range: { min: cp.min, max: cp.max, steps: cp.steps },
        }),
      });

      cp.result = await new Promise((resolve, reject) => {
        const es = new EventSource(`/api/jobs/${jobId}/events`);
        es.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          cp.status = msg.message ?? "Working…";
          const box = $("#segmentation-body .seg-status");
          if (box) box.textContent = cp.status;
          if (msg.state === "done") { es.close(); resolve(msg.scene); }
          else if (msg.state === "error") { es.close(); reject(new Error(msg.error)); }
        };
        es.onerror = () => { es.close(); reject(new Error("lost contact with the analysis job")); };
      });
      cp.status = "";
    } catch (err) {
      cp.status = "";
      toast(`Ceteris paribus failed: ${err.message}`, true);
    }
    cp.running = false;
    render();
  }

  // --------------------------------------------------------------- actions
  async function loadPreview(force = false) {
    const info = getSceneInfo();
    if (!info) { state.preview = null; render(); return; }
    if (state.preview && state.preview.scene === info.id && !force) { render(); return; }
    if (state.cp.scene !== info.id) {
      Object.assign(state.cp, {
        scene: info.id, instances: null, instanceId: null,
        min: null, max: null, result: null, status: "",
      });
    }
    state.preview = null;
    render();
    try {
      const p = await fetchJson(`/api/scenes/${encId(info.id)}/segment/preview`);
      p.scene = info.id;
      state.preview = p;
    } catch (err) {
      toast(`Could not inspect the scene: ${err.message}`, true);
    }
    await loadInstances();
    render();
  }
  api.loadPreview = loadPreview;

  async function confirmAndRun() {
    const info = getSceneInfo();
    if (!info || !state.preview) return;
    if (!/^https?:\/\//i.test(state.endpoint)) {
      toast("Set an http(s) endpoint first", true);
      return;
    }

    const included = state.preview.arrays.filter((a) => !state.excluded.has(a.name));
    const bytes = included.reduce((n, a) => n + a.nbytes, 0);

    const body = el("div");
    body.appendChild(el("p", "modal-lead",
      `${fmtInt(state.preview.numPoints)} points (${fmtBytes(bytes)}) will be sent to ` +
      `${state.endpoint} and the labels it returns written into ${info.name} ` +
      `as a "prediction" attribute.`));
    const list = el("div", "modal-list");
    for (const a of included) {
      const row = el("div", "modal-row");
      row.appendChild(el("span", "lib-name", a.name));
      row.appendChild(el("span", "lib-meta", `${a.dtype} (${a.shape.join(", ")})`));
      list.appendChild(row);
    }
    body.appendChild(list);

    const ok = await confirmDialog({
      title: "Send this scene for segmentation",
      bodyNode: body,
      note: "This rewrites the scene's cached octree to add the prediction. Re-converting " +
            "the scene from its PCD discards it, and running again replaces it.",
      confirmText: "Send",
    });
    if (!ok) return;

    state.running = true;
    render();
    const status = el("div", "seg-status", "Starting…");
    $("#segmentation-body").appendChild(status);

    try {
      const fields = state.preview.arrays
        .filter((a) => a.role !== "coordinates" && a.role !== "colour" && !state.excluded.has(a.name))
        .map((a) => a.name);

      const { jobId } = await fetchJson(`/api/scenes/${encId(info.id)}/segment`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ endpoint: state.endpoint, fields }),
      });

      const scene = await new Promise((resolve, reject) => {
        const es = new EventSource(`/api/jobs/${jobId}/events`);
        es.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          status.textContent = msg.message ?? "Working…";
          if (msg.state === "done") { es.close(); resolve(msg.scene); }
          else if (msg.state === "error") { es.close(); reject(new Error(msg.error)); }
        };
        es.onerror = () => { es.close(); reject(new Error("lost contact with the inference job")); };
      });

      state.running = false;
      toast(`Segmentation returned ${scene.prediction?.numClasses ?? 0} classes`);
      if (reloadScene) await reloadScene(scene);
      onPrediction?.();
      await loadPreview(true);
    } catch (err) {
      state.running = false;
      toast(`Segmentation failed: ${err.message}`, true);
      render();
    }
  }

  return api;
}
