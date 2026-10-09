/** Runs an ablation experiment off the main thread. */
import { parentPort, workerData } from "node:worker_threads";
import { runAblation } from "../inference/ablation.mjs";

runAblation(workerData.id, {
  ...workerData,
  onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
}).then(
  (result) => parentPort.postMessage({ type: "done", scene: result }),
  (err) => parentPort.postMessage({ type: "error", message: err.message, stack: err.stack }),
);
