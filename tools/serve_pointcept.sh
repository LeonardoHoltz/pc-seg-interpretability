#!/usr/bin/env bash
#
# Runs the Pointcept inference service inside the Pointcept docker image and
# publishes it where the viewer expects to find it.
#
#   tools/serve_pointcept.sh                      # random LitePT-small, no weights
#   tools/serve_pointcept.sh --weight model_best.pth   # from pointcept/weights/
#   tools/serve_pointcept.sh --config configs/scannet/semseg-pt-v3m1-0-base.py
#
# Then set the Segmentation tab's endpoint to http://127.0.0.1:8500/.
#
# The repo is mounted read-only and PYTHONPATH points at *this* checkout of the
# fork, so the image's own bundled /workspace/Pointcept is shadowed and the code
# actually served is the code in pointcept/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT=/workspace/repo

# Prefer the locally built image (tools/build_pointcept_image.sh), which adds the
# CUDA extensions the published one leaves out, and fall back to the published
# image when it has not been built yet.
if [[ -n "${PCIT_POINTCEPT_IMAGE:-}" ]]; then
  IMAGE="$PCIT_POINTCEPT_IMAGE"
elif docker image inspect pcit-pointcept:latest >/dev/null 2>&1; then
  IMAGE="pcit-pointcept:latest"
else
  IMAGE="pointcept/pointcept:v1.6.0"
fi
CONFIG="configs/scannet/semseg-litept-v1m1-0-small.py"
CLASS_NAMES="config/scannet_subset/classes.json"
WEIGHT=""
PORT=8500
DEVICE=""
NAME="pcit-pointcept"
SHUFFLE=0          # LitePT permutes its serialization orders at random; see below
EXTRA=()

usage() {
  sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options:
  --config PATH        Pointcept config, relative to pointcept/ (default: LitePT-v1 small)
  --weight NAME        checkpoint in pointcept/weights/ (a bare filename is enough);
                       omit for a randomly initialised model
  --class-names PATH   class-name JSON, relative to the repo root
  --port N             host port to publish on (default: 8500)
  --device D           cuda | cuda:N | cpu (default: cuda if the GPU is reachable)
  --image REF          docker image (default: pointcept/pointcept:v1.6.0)
  --name NAME          container name (default: pcit-pointcept)
  --allow-shuffle      keep the backbone's random order shuffling (see below)
  --                   everything after this goes straight to serve_inference.py

Determinism: LitePT and PTv3 call `serialization(shuffle_orders=True)`, which draws a
`torch.randperm` on every forward -- at eval too, not just training, so each position
of a ceteris paribus sweep would carry a different permutation. This script therefore
passes `model.backbone.shuffle_orders=False` unless you ask for --allow-shuffle.

That removes the deliberate random draw but does not make runs bit-identical: the
sparse kernels accumulate with atomics, so repeat requests still differ by ~1e-3 on
logits spanning roughly [-1.3, 1.3]. Negligible for colouring, worth remembering if
you are reading small differences between sweep positions.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG="$2"; shift 2 ;;
    --weight)       WEIGHT="$2"; shift 2 ;;
    --class-names)  CLASS_NAMES="$2"; shift 2 ;;
    --port)         PORT="$2"; shift 2 ;;
    --device)       DEVICE="$2"; shift 2 ;;
    --image)        IMAGE="$2"; shift 2 ;;
    --name)         NAME="$2"; shift 2 ;;
    --allow-shuffle) SHUFFLE=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; EXTRA+=("$@"); break ;;
    *)              echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ---- how (or whether) the GPU can be reached --------------------------------
# Docker 29 routes --gpus through CDI, which is not configured on every host, so
# fall back to the nvidia runtime and then to the CPU rather than just failing.
gpu_args=()
probe() { docker run --rm "$@" "$IMAGE" true >/dev/null 2>&1; }

if [[ "$DEVICE" == cpu ]]; then
  gpu_mode="cpu (asked for)"
elif probe --gpus all; then
  gpu_args=(--gpus all); gpu_mode="--gpus all"
elif probe --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all; then
  gpu_args=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
  gpu_mode="--runtime=nvidia"
else
  # Last resort: hand the container the device nodes and the host's driver
  # libraries by hand. This is what the toolkit would do properly, and it works
  # without root -- but the library list is hand-maintained, so prefer the real
  # fix below when you can.
  manual=(--device /dev/nvidia0 --device /dev/nvidiactl
          --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools)
  for lib in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
    path="/usr/lib/x86_64-linux-gnu/$lib"
    [[ -e "$path" ]] && manual+=(-v "$path:$path:ro")
  done

  if [[ -e /dev/nvidia0 ]] && probe "${manual[@]}"; then
    gpu_args=("${manual[@]}")
    gpu_mode="manual device + driver injection"
    cat >&2 <<'WARN'
!! Neither `--gpus all` nor the nvidia runtime works here, so the GPU is being
!! passed through by hand. It works, but it is a stopgap. The real fix:
!!     sudo apt install -y nvidia-container-toolkit
!!     sudo nvidia-ctk runtime configure --runtime=docker
!!     sudo systemctl restart docker
WARN
  else
    gpu_mode="cpu (no GPU passthrough)"
    DEVICE="${DEVICE:-cpu}"
    cat >&2 <<'WARN'
!! No GPU passthrough available, falling back to the CPU. Note that spconv-based
!! backbones (LitePT, PTv3, SpUNet) have no CPU kernels and will fail outright.
!! To fix it:
!!     sudo apt install -y nvidia-container-toolkit
!!     sudo nvidia-ctk runtime configure --runtime=docker
!!     sudo systemctl restart docker
WARN
  fi
fi

# ---- the weight ---------------------------------------------------------------
# Checkpoints live in pointcept/weights/, which is mounted with the rest of the
# repo and ignored by git. A bare filename is resolved against it, so
# `--weight model_best.pth` is enough; a path is taken relative to pointcept/.
weight_in_container="$WEIGHT"
if [[ -n "$WEIGHT" ]]; then
  if [[ "$WEIGHT" != */* ]]; then
    WEIGHT="weights/$WEIGHT"
  fi
  if [[ ! -f "$REPO/pointcept/$WEIGHT" ]]; then
    echo "no such weight: pointcept/$WEIGHT" >&2
    echo "put checkpoints in pointcept/weights/ -- available:" >&2
    ls -1 "$REPO/pointcept/weights" 2>/dev/null | sed 's/^/  /' >&2 || echo "  (none yet)" >&2
    exit 1
  fi
  weight_in_container="$WEIGHT"
fi

# ---- the command inside the container ---------------------------------------
serve=(python tools/serve_inference.py --config-file "$CONFIG" --host 0.0.0.0 --port "$PORT")
[[ -n "$WEIGHT"      ]] && serve+=(--weight "$weight_in_container")
[[ -n "$DEVICE"      ]] && serve+=(--device "$DEVICE")
[[ -n "$CLASS_NAMES" && -f "$REPO/$CLASS_NAMES" ]] && serve+=(--class-names "$MOUNT/$CLASS_NAMES")
[[ "$SHUFFLE" -eq 0  ]] && serve+=(--options model.backbone.shuffle_orders=False)
serve+=("${EXTRA[@]+"${EXTRA[@]}"}")

docker rm -f "$NAME" >/dev/null 2>&1 || true

# Only ask for a TTY when there is one, so the script also works from a pipe,
# a CI job or `nohup ... &`.
tty_args=()
[[ -t 0 && -t 1 ]] && tty_args=(-it)

echo "image     $IMAGE"
echo "gpu       $gpu_mode"
echo "config    $CONFIG"
echo "weight    ${WEIGHT:-(none -- randomly initialised)}"
echo "endpoint  http://127.0.0.1:$PORT/"
echo

exec docker run --rm --name "$NAME" \
  "${tty_args[@]+"${tty_args[@]}"}" \
  "${gpu_args[@]+"${gpu_args[@]}"}" \
  --shm-size=8g \
  -v "$REPO:$MOUNT:ro" \
  -w "$MOUNT/pointcept" \
  -e PYTHONPATH="$MOUNT/pointcept" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -p "127.0.0.1:$PORT:$PORT" \
  "$IMAGE" \
  "${serve[@]}"
