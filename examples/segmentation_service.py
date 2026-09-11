#!/usr/bin/env python3
"""
Reference receiver for the viewer's segmentation inference API.

The viewer POSTs the scene's points as one binary payload and expects the same
framing back with a `labels` array. Nothing here is imported by the app -- it
documents the contract and is runnable as a stand-in service:

    python examples/segmentation_service.py --port 8500

Payload framing (both directions):

    [8]  uint64 little-endian   length of the JSON header
    [H]  UTF-8 JSON header
    [..] raw array bytes, back to back, each starting on an 8-byte boundary

Arrays arrive as numpy views over the request body -- no copying, no per-point
parsing:

    xyz          float32  (3, N)     metres, scene coordinates
    rgb          uint8    (3, N)     present only if the scene has colour
    <field>      float32  (N,)       one per continuous scalar field
    <field>      int32    (N,)       one per categorical field (label, instance, ...)

Reply with at least:

    labels       int32    (N,)       one class id per point, same order as sent

Optionally add:

    scores       float32  (N,)       confidence in the predicted label
    probs        float32  (C, N)     per-class probability -- needed for the
                                     ceteris paribus plot, which averages one
                                     class's probability over an object

and in the header, `class_names` as a {"<id>": "<name>"} mapping and `classes`
as the list of class values in the order of the rows of `probs`.

Saliency
--------
When the header carries `request: "saliency"`, a `mask` uint8 (N,) marks the
object of interest and the header's `saliency` block names it (and a
`target_class`, if one was chosen). Reply with

    saliency     float32  (N,)   one scalar per scene point
                 float32  (M,)   or just the masked points, in ascending index
                                 order, and the rest of the scene reads as 0

What the scalar means and how it is aggregated is entirely up to you -- the
viewer only carries it back and exposes it as a scalar field to colour by.

Ceteris paribus sweeps
----------------------
When the header carries `request: "ceteris_paribus"`, the whole sweep arrives as
one request. Alongside the usual arrays you get

    mask         uint8    (N,)       1 for the points of the object being moved

and a `ceteris_paribus` block in the header with `direction` (a unit vector),
`offsets` (how far to move along it) and the absolute `heights` they correspond
to. Move the masked points yourself and reply with

    logits       float32  (S, C, M)  sweep position, class, masked point

where M is `mask.sum()` and the point order is the masked points in ascending
index order -- exactly `xyz[:, mask.astype(bool)]`. Send `probs` with the same
shape instead if the model already normalises. Sending the scene once rather
than once per position is what keeps this cheap.
"""
import argparse
import json
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

DTYPES = {"|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<f4", "<f8"}

# Whether saliency is returned for the whole scene or only the masked object.
SCOPE = "scene"


def decode(payload: bytes):
    """Returns (header, {name: np.ndarray}). The arrays are views over `payload`."""
    (header_len,) = struct.unpack_from("<Q", payload, 0)
    header = json.loads(payload[8:8 + header_len].decode("utf-8"))
    base = 8 + header_len

    arrays = {}
    for spec in header["arrays"]:
        dtype = np.dtype(spec["dtype"])
        shape = tuple(spec["shape"])
        count = int(np.prod(shape)) if shape else 0
        arrays[spec["name"]] = np.frombuffer(
            payload, dtype=dtype, count=count, offset=base + spec["offset"]
        ).reshape(shape)
    return header, arrays


def encode(arrays: dict, **meta) -> bytes:
    """Inverse of decode(). `arrays` maps name -> np.ndarray."""
    specs, blobs, offset = [], [], 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        specs.append({
            "name": name,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": len(raw),
        })
        blobs.append(raw)
        offset = (offset + len(raw) + 7) & ~7
        blobs.append(b"\x00" * (offset - (specs[-1]["offset"] + len(raw))))

    header = json.dumps({"format": "pcit-arrays/1", **meta, "arrays": specs}).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"".join(blobs)


def segment(header, arrays):
    """
    Stand-in for a real model.

    Assigns a class from the point's height, softly, so that moving an object up
    or down genuinely changes its predicted class. That is what makes the
    viewer's ceteris paribus sweep show a curve rather than a step.
    """
    xyz = arrays["xyz"]                     # (3, N)
    z = xyz[2]
    lo, hi = float(z.min()), float(z.max())
    span = (hi - lo) or 1.0

    n_classes = 5
    # Band centres evenly spaced through the scene's height range.
    centres = (np.arange(n_classes) + 0.5) / n_classes
    t = (z - lo) / span                                     # (N,)

    # Soft assignment: a point belongs to the band whose centre it is nearest.
    sharpness = 12.0
    logits = -sharpness * np.abs(t[None, :] - centres[:, None])   # (C, N)
    logits -= logits.max(axis=0, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=0, keepdims=True)               # (C, N)

    labels = probs.argmax(axis=0).astype(np.int32)
    scores = probs.max(axis=0).astype(np.float32)
    names = {"0": "floor", "1": "low", "2": "mid", "3": "high", "4": "ceiling"}
    return labels, scores, probs.astype(np.float32), names


def sweep(header, arrays):
    """
    Ceteris paribus: move the masked object along `direction` by each offset and
    return the object's logits at every position.

    The scene arrived once, so only the masked coordinates are touched between
    positions -- everything else is genuinely held constant.
    """
    spec = header["ceteris_paribus"]
    xyz = arrays["xyz"]                                  # (3, N)
    sel = arrays["mask"].astype(bool)                    # (N,)
    direction = np.asarray(spec["direction"], dtype=np.float32).reshape(3, 1)
    offsets = np.asarray(spec["offsets"], dtype=np.float32)

    z_all = xyz[2]
    lo, hi = float(z_all.min()), float(z_all.max())
    span = (hi - lo) or 1.0
    n_classes, sharpness = 5, 12.0
    centres = (np.arange(n_classes) + 0.5) / n_classes

    base = xyz[:, sel]                                   # (3, M)
    out = np.empty((offsets.size, n_classes, base.shape[1]), dtype=np.float32)

    for s, off in enumerate(offsets):
        moved = base + direction * off                   # (3, M)
        t = (moved[2] - lo) / span
        out[s] = -sharpness * np.abs(t[None, :] - centres[:, None])

    names = {"0": "floor", "1": "low", "2": "mid", "3": "high", "4": "ceiling"}
    return out, list(range(n_classes)), names


def saliency(header, arrays):
    """
    Stand-in for a real attribution method.

    Scores each point by how close it is to the object under study, which is not
    a real explanation but does produce a field that is obviously *about* that
    object -- enough to check the plumbing and the colour ramp end to end.
    """
    xyz = arrays["xyz"]                                  # (3, N)
    sel = arrays["mask"].astype(bool)
    centre = xyz[:, sel].mean(axis=1, keepdims=True)     # (3, 1)

    d = np.linalg.norm(xyz - centre, axis=0)             # (N,)
    scale = float(np.percentile(d[sel], 90)) or 1.0
    out = np.exp(-(d / (2.0 * scale)) ** 2).astype(np.float32)
    out[sel] = 1.0                                       # the object itself

    # Either shape is valid. With --scope object only the masked points are sent
    # back and the viewer leaves the rest of the scene at zero.
    return out[sel] if SCOPE == "object" else out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        payload = self.rfile.read(length)
        try:
            header, arrays = decode(payload)
            if header.get("request") == "saliency":
                sp = header["saliency"]
                print(f"  saliency: instance {sp['instance']}, "
                      f"{sp['object_points']} masked points, "
                      f"target class {sp.get('target_class')}", flush=True)
                values = saliency(header, arrays)
                body = encode({"saliency": values}, num_points=int(values.shape[0]))
            elif header.get("request") == "ceteris_paribus":
                cp = header["ceteris_paribus"]
                print(f"  ceteris paribus: instance {cp['instance']}, "
                      f"{cp['object_points']} masked points, "
                      f"{len(cp['offsets'])} positions, one request", flush=True)
                logits, classes, names = sweep(header, arrays)
                body = encode({"logits": logits},
                              classes=classes, class_names=names,
                              num_points=int(header.get("num_points", 0)))
            else:
                print(f"  received {header.get('num_points')} points, "
                      f"arrays: {', '.join(f'{k}{v.shape}:{v.dtype}' for k, v in arrays.items())}",
                      flush=True)
                labels, scores, probs, names = segment(header, arrays)
                body = encode({"labels": labels, "scores": scores, "probs": probs},
                              num_points=int(labels.shape[0]),
                              class_names=names,
                              classes=list(range(probs.shape[0])))
            self.send_response(200)
            self.send_header("content-type", "application/octet-stream")
        except Exception as err:                      # noqa: BLE001
            body = json.dumps({"error": str(err)}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--scope", choices=("scene", "object"), default="scene",
                    help="return saliency for every point, or only the masked object")
    args = ap.parse_args()
    SCOPE = args.scope
    print(f"segmentation service listening on http://{args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()
