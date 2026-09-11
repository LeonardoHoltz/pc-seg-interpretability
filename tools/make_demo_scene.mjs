#!/usr/bin/env node
/**
 * Generates synthetic demo scenes under scenes/ so the viewer has something to
 * open before you drop your own data in. Each scene carries RGB colour, two
 * continuous scalars (intensity, confidence) and two categorical fields
 * (label = semantic class, instance = object id).
 *
 *   npm run demo
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { writePcd } from "./pcd_write.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SCENES = join(ROOT, "scenes");
const CONFIG = join(ROOT, "config");

// Deterministic PRNG so regenerating a scene gives the same points.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const CLASSES = [
  { id: 0, name: "unlabeled", color: [90, 90, 95] },
  { id: 1, name: "road", color: [128, 64, 128] },
  { id: 2, name: "sidewalk", color: [244, 35, 232] },
  { id: 3, name: "building", color: [180, 120, 70] },
  { id: 4, name: "vegetation", color: [107, 182, 70] },
  { id: 5, name: "pole", color: [153, 153, 153] },
  { id: 6, name: "car", color: [0, 96, 220] },
  { id: 7, name: "person", color: [220, 20, 60] },
];

function packRgb(r, g, b) {
  // PCL packs RGB into the mantissa of a float32.
  const u = new Uint32Array(1);
  u[0] = ((r & 255) << 16) | ((g & 255) << 8) | (b & 255);
  return new Float32Array(u.buffer)[0];
}

function buildScene(seed, opts) {
  const rand = rng(seed);
  const X = [], Y = [], Z = [], I = [], RGB = [], L = [], INST = [], CONF = [];
  let instanceCounter = 1;

  const push = (x, y, z, cls, inst, intensity) => {
    const c = CLASSES[cls].color;
    // Slight per-point colour jitter so RGB mode looks like real data.
    const j = () => (rand() - 0.5) * 24;
    X.push(x); Y.push(y); Z.push(z);
    I.push(intensity);
    RGB.push(packRgb(
      Math.max(0, Math.min(255, c[0] + j())),
      Math.max(0, Math.min(255, c[1] + j())),
      Math.max(0, Math.min(255, c[2] + j()))));
    L.push(cls);
    INST.push(inst);
    // Confidence: high in the middle of an object, lower at the edges.
    CONF.push(Math.max(0.05, Math.min(1, 0.65 + 0.35 * (rand() - 0.2))));
  };

  const { length, width, nGround } = opts;

  // Road + sidewalks
  for (let i = 0; i < nGround; i++) {
    const x = (rand() - 0.5) * length;
    const y = (rand() - 0.5) * width;
    const onRoad = Math.abs(y) < width * 0.3;
    const z = (rand() - 0.5) * 0.04 + (onRoad ? 0 : 0.12);
    push(x, y, z, onRoad ? 1 : 2, onRoad ? 0 : 0, onRoad ? 0.25 + rand() * 0.15 : 0.4 + rand() * 0.2);
  }

  // Buildings along both sides
  for (const side of [-1, 1]) {
    const inst = instanceCounter++;
    const n = Math.floor(nGround * 0.55);
    for (let i = 0; i < n; i++) {
      const x = (rand() - 0.5) * length;
      const y = side * (width * 0.5 + rand() * 0.3);
      const z = rand() * 9;
      push(x, y, z, 3, inst, 0.55 + rand() * 0.25);
    }
  }

  // Cars parked along the kerb
  const nCars = opts.nCars;
  for (let c = 0; c < nCars; c++) {
    const inst = instanceCounter++;
    const cx = (rand() - 0.5) * length * 0.9;
    const cy = (c % 2 === 0 ? 1 : -1) * width * 0.34;
    for (let i = 0; i < opts.nPerCar; i++) {
      const x = cx + (rand() - 0.5) * 4.3;
      const y = cy + (rand() - 0.5) * 1.8;
      const z = 0.15 + rand() * 1.35;
      push(x, y, z, 6, inst, 0.7 + rand() * 0.3);
    }
  }

  // Trees
  for (let t = 0; t < opts.nTrees; t++) {
    const inst = instanceCounter++;
    const cx = (rand() - 0.5) * length * 0.95;
    const cy = (t % 2 === 0 ? 1 : -1) * (width * 0.42);
    for (let i = 0; i < opts.nPerTree; i++) {
      // Spherical canopy over a thin trunk.
      const trunk = rand() < 0.12;
      if (trunk) {
        push(cx + (rand() - 0.5) * 0.25, cy + (rand() - 0.5) * 0.25, rand() * 2.4, 4, inst, 0.3 + rand() * 0.2);
      } else {
        const th = rand() * Math.PI * 2;
        const ph = Math.acos(2 * rand() - 1);
        const r = 1.4 + rand() * 0.9;
        push(cx + r * Math.sin(ph) * Math.cos(th), cy + r * Math.sin(ph) * Math.sin(th),
             3.4 + r * Math.cos(ph), 4, inst, 0.45 + rand() * 0.35);
      }
    }
  }

  // Light poles
  for (let p = 0; p < opts.nPoles; p++) {
    const inst = instanceCounter++;
    const cx = (rand() - 0.5) * length * 0.9;
    const cy = (p % 2 === 0 ? 1 : -1) * width * 0.46;
    for (let i = 0; i < opts.nPerPole; i++) {
      push(cx + (rand() - 0.5) * 0.14, cy + (rand() - 0.5) * 0.14, rand() * 6.5, 5, inst, 0.8 + rand() * 0.2);
    }
  }

  // A few pedestrians
  for (let p = 0; p < opts.nPeople; p++) {
    const inst = instanceCounter++;
    const cx = (rand() - 0.5) * length * 0.8;
    const cy = (rand() - 0.5) * width * 0.8;
    for (let i = 0; i < opts.nPerPerson; i++) {
      push(cx + (rand() - 0.5) * 0.5, cy + (rand() - 0.5) * 0.4, rand() * 1.8, 7, inst, 0.6 + rand() * 0.3);
    }
  }

  return { X, Y, Z, I, RGB, L, INST, CONF };
}

function writeScene(name, encoding, seed, opts, description) {
  const s = buildScene(seed, opts);
  const columns = [
    { name: "x", type: "F", size: 4, values: s.X },
    { name: "y", type: "F", size: 4, values: s.Y },
    { name: "z", type: "F", size: 4, values: s.Z },
    { name: "rgb", type: "F", size: 4, values: s.RGB },
    { name: "intensity", type: "F", size: 4, values: s.I },
    { name: "label", type: "U", size: 4, values: s.L },
    { name: "instance", type: "U", size: 4, values: s.INST },
    { name: "confidence", type: "F", size: 4, values: s.CONF },
  ];
  const file = join(SCENES, `${name}.pcd`);
  writePcd(file, columns, encoding);

  // Sidecar giving human-readable names + colours for the segmentation field.
  // It describes the dataset rather than the points, so it goes to config/,
  // which is tracked, while the .pcd stays in the untracked scenes/.
  mkdirSync(CONFIG, { recursive: true });
  writeFileSync(join(CONFIG, `${name}.classes.json`), JSON.stringify({
    description,
    fields: {
      label: {
        name: "Semantic class",
        classes: Object.fromEntries(CLASSES.map((c) => [c.id, { name: c.name, color: c.color }])),
      },
    },
  }, null, 2) + "\n");

  console.log(`  ${name}.pcd  ${s.X.length.toLocaleString()} points  (${encoding})`);
}

mkdirSync(SCENES, { recursive: true });
console.log("Writing demo scenes to scenes/ ...");
writeScene("demo_street_binary", "binary", 12345,
  { length: 80, width: 14, nGround: 90000, nCars: 8, nPerCar: 4500, nTrees: 8, nPerTree: 5200,
    nPoles: 6, nPerPole: 1200, nPeople: 5, nPerPerson: 900 },
  "Synthetic street scene with semantic + instance segmentation");
writeScene("demo_street_ascii", "ascii", 999,
  { length: 40, width: 12, nGround: 9000, nCars: 3, nPerCar: 900, nTrees: 3, nPerTree: 1100,
    nPoles: 2, nPerPole: 300, nPeople: 2, nPerPerson: 250 },
  "Small ASCII-encoded variant, handy for eyeballing the raw file");
console.log("Done.");
