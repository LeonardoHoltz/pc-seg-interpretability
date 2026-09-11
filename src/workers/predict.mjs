/** Runs a segmentation request off the main thread. */
import { parentPort, workerData } from "node:worker_threads";
import { runSegmentation } from "../inference/predict.mjs";

runSegmentation(workerData.id, workerData.endpoint, {
  fields: workerData.fields ?? null,
  onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
}).then(
  (scene) => parentPort.postMessage({ type: "done", scene }),
  (err) => parentPort.postMessage({ type: "error", message: err.message, stack: err.stack }),
);
