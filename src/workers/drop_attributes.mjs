/** Takes analysis fields back off a scene, off the main thread. */
import { parentPort, workerData } from "node:worker_threads";
import { dropAnalysisAttributes } from "../scene/attributes.mjs";
import { cacheDirFor } from "../scene/registry.mjs";

try {
  const { scene, removed } = dropAnalysisAttributes(workerData.id, {
    cacheDir: cacheDirFor(workerData.id),
    names: workerData.names ?? null,
    onProgress: (p) => parentPort.postMessage({ type: "progress", ...p }),
  });
  parentPort.postMessage({ type: "done", scene: { ...scene, removedAttributes: removed } });
} catch (err) {
  parentPort.postMessage({ type: "error", message: err.message, stack: err.stack });
}
