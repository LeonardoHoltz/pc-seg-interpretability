/** Runs a merge off the main thread, like the conversion worker. */
import { parentPort, workerData } from "node:worker_threads";
import { bakeInstances } from "../objects/bake.mjs";

try {
  const scene = bakeInstances(workerData.id, workerData.placements, {
    exclude: workerData.exclude,
    onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
  });
  parentPort.postMessage({ type: "done", scene });
} catch (err) {
  parentPort.postMessage({ type: "error", message: err.message, stack: err.stack });
}
