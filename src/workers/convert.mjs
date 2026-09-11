/**
 * Runs a conversion off the main thread.
 *
 * Building an octree is CPU-bound and would otherwise block the HTTP server for
 * the whole job, which also means no progress could reach the browser.
 */
import { parentPort, workerData } from "node:worker_threads";
import { convertScene } from "../scene/convert.mjs";

try {
  const scene = convertScene(workerData.id, {
    force: workerData.force,
    gridSize: workerData.gridSize,
    primaryField: workerData.primaryField,
    instances: workerData.instances,
    instanceField: workerData.instanceField,
    onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
  });
  parentPort.postMessage({ type: "done", scene });
} catch (err) {
  parentPort.postMessage({ type: "error", message: err.message, stack: err.stack });
}
