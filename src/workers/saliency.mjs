/** Runs a saliency request off the main thread. */
import { parentPort, workerData } from "node:worker_threads";
import { runSaliency } from "../inference/saliency.mjs";

runSaliency(workerData.id, {
  ...workerData,
  onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
}).then(
  (scene) => parentPort.postMessage({ type: "done", scene }),
  (err) => parentPort.postMessage({ type: "error", message: err.message, stack: err.stack }),
);
