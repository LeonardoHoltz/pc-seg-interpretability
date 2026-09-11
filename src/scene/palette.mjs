/**
 * Categorical colours for segmentation classes.
 *
 * Colours are assigned once, at conversion time, and stored in scene.json so
 * that the legend, the Potree classification LUT and the categorical gradient
 * all agree on what a class looks like.
 */

// Well-separated hues that stay legible on Potree's dark background.
const BASE = [
  [ 78, 158, 255], [255, 138,  59], [ 82, 199, 118], [232,  86,  99],
  [176, 126, 232], [214, 168,  95], [242, 124, 196], [140, 152, 166],
  [206, 214,  74], [ 76, 208, 224], [ 45, 106, 200], [200,  90,  40],
  [ 40, 140,  80], [170,  40,  60], [120,  80, 180], [150, 100,  50],
  [200,  80, 150], [ 90, 100, 110], [150, 160,  40], [ 40, 150, 165],
  [130, 190, 255], [255, 190, 140], [150, 225, 175], [255, 155, 165],
];

/** Golden-angle hues keep even large class sets distinguishable. */
function generated(index) {
  const hue = (index * 137.508) % 360;
  const sat = 0.55 + 0.2 * ((index % 3) / 2);
  const light = 0.48 + 0.14 * ((index % 4) / 3);
  return hslToRgb(hue / 360, sat, light);
}

function hslToRgb(h, s, l) {
  const f = (n) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)))));
  };
  return [f(0), f(8), f(4)];
}

export function classColor(index) {
  return index < BASE.length ? BASE[index] : generated(index - BASE.length);
}

/** Conventional colour for an "unlabeled"/"noise" class, whatever its id. */
export const UNLABELED_COLOR = [95, 99, 108];
export const UNLABELED_NAMES = new Set([
  "unlabeled", "unlabelled", "unclassified", "none", "noise", "ignore", "void", "background",
]);
