/** Runs a ceteris paribus sweep off the main thread. */
import { parentPort, workerData } from "node:worker_threads";
import { runCeterisParibus } from "../inference/ceteris.mjs";

runCeterisParibus(workerData.id, {
  ...workerData,
  onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
}).then(
  (result) => parentPort.postMessage({ type: "done", scene: result }),
  (err) => parentPort.postMessage({ type: "error", message: err.message, stack: err.stack }),
);
