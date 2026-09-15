#!/usr/bin/env bash
#
# Builds the Pointcept image the viewer serves from: the published image, plus
# the CUDA extensions it leaves out (pointrope, which LitePT wants for its
# rotary position embedding and otherwise falls back to plain PyTorch for), plus
# the Python packages it ships without (peft).
#
#   docker/build_pointcept_image.sh                 # pointrope, tagged pcit-pointcept:latest
#   docker/build_pointcept_image.sh --libs "pointrope pointops2 pointseg"
#
# tools/serve_pointcept.sh picks this image up automatically once it exists.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE="${PCIT_POINTCEPT_BASE_IMAGE:-pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel}"
TAG="${PCIT_POINTCEPT_IMAGE:-pcit-pointcept:latest}"
LIBS="pointrope"
CUDA_ARCH="8.6+PTX"
PIP_PACKAGES="peft"
NO_CACHE=()

usage() {
  sed -n '3,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options:
  --libs "A B"      which pointcept/libs/* to compile (default: pointrope)
  --tag REF         image tag to produce (default: pcit-pointcept:latest)
  --base REF        base image (default: pointcept/pointcept:v1.6.0)
  --cuda-arch A     TORCH_CUDA_ARCH_LIST for libs that honour it (default: 8.6+PTX,
                    i.e. Ampere / RTX 30-series). pointrope ignores it and builds
                    for every architecture torch itself supports.
  --pip "A B"       extra pip packages to install into the image (default: peft).
                    Pin them here -- torch is constrained to the image's build, so
                    a dependency cannot drag a different one in.
  --no-cache        rebuild from scratch
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libs)       LIBS="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    --base)       BASE="$2"; shift 2 ;;
    --cuda-arch)  CUDA_ARCH="$2"; shift 2 ;;
    --pip)        PIP_PACKAGES="$2"; shift 2 ;;
    --no-cache)   NO_CACHE=(--no-cache); shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  echo "base image $BASE is not present locally; pulling..."
  docker pull "$BASE"
fi

echo "base      $BASE"
echo "tag       $TAG"
echo "libs      $LIBS"
echo "arch      $CUDA_ARCH"
echo "pip       ${PIP_PACKAGES:-(none)}"
echo "context   $REPO/pointcept/libs"
echo
echo "Compiling CUDA extensions takes a few minutes; nvcc builds one object per"
echo "architecture torch supports."
echo

docker build "${NO_CACHE[@]+"${NO_CACHE[@]}"}" \
  -f "$REPO/docker/Dockerfile" \
  --build-arg BASE_IMAGE="$BASE" \
  --build-arg LIBS="$LIBS" \
  --build-arg CUDA_ARCH="$CUDA_ARCH" \
  --build-arg PIP_PACKAGES="$PIP_PACKAGES" \
  -t "$TAG" \
  "$REPO/pointcept/libs"

echo
echo "built $TAG -- tools/serve_pointcept.sh will use it automatically"
