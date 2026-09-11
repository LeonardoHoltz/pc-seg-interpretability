/** Runs a rescale off the main thread. */
import { parentPort, workerData } from "node:worker_threads";
import { resampleInstance } from "../objects/resample.mjs";

try {
  parentPort.postMessage({ type: "progress", progress: 0.1, message: "Resampling" });
  const result = resampleInstance(workerData);
  parentPort.postMessage({ type: "done", scene: result });
} catch (err) {
  parentPort.postMessage({ type: "error", message: err.message, stack: err.stack });
}
