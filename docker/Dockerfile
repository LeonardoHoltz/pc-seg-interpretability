# Pointcept image for the interpretability viewer.
#
# The published pointcept image ships pointops and pointgroup_ops compiled, but
# not pointrope. Without it LitePT logs
#
#   [PointROPE] CUDA implementation unavailable (ModuleNotFoundError: No module
#   named 'pointrope'). Using slower Pytorch fallback.
#
# and silently runs its rotary position embedding in plain PyTorch. This image
# adds the compiled extension so the CUDA path is taken.
#
# The build context is `pointcept/libs/` -- the sources compiled here are the
# ones in this checkout, not the image's bundled copy -- and keeping the context
# that narrow also keeps the 146 MB of weights/ out of it:
#
#   docker build -f docker/Dockerfile -t pcit-pointcept:latest pointcept/libs
#
# or just `tools/build_pointcept_image.sh`.

ARG BASE_IMAGE=pointcept/pointcept:v1.6.0
FROM ${BASE_IMAGE}

# Which of pointcept/libs/* to compile. Space separated; pointops2 and pointseg
# are the other two the base image leaves out, if you ever need them.
ARG LIBS="pointrope"

# No GPU is visible during `docker build`, so nothing may probe the device to
# decide what to target. pointrope's setup.py calls
# torch.cuda.get_gencode_flags(), which reports the architectures *torch itself*
# was compiled for -- a GPU-independent list -- and passes its own -gencode
# flags, so the build is reproducible without a device and the result runs on
# any card torch supports. TORCH_CUDA_ARCH_LIST is set for libraries that honour
# it instead; pointrope's explicit flags take precedence.
ARG CUDA_ARCH="8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    FORCE_CUDA=1 \
    MAX_JOBS=4

COPY . /tmp/libs

RUN set -eux; \
    for lib in ${LIBS}; do \
        test -f "/tmp/libs/$lib/setup.py" || { echo "no such lib: $lib" >&2; exit 1; }; \
        echo "==> building $lib"; \
        # --no-build-isolation: an isolated build would pull a fresh torch from
        # PyPI and compile against the wrong ABI.
        pip install --no-cache-dir --no-build-isolation "/tmp/libs/$lib"; \
    done; \
    rm -rf /tmp/libs

# Fail the build rather than the first request if an extension did not land.
# torch has to be imported first: these extensions link against libc10.so, which
# is only on the loader path once torch itself has pulled it in. That is the
# order Pointcept uses them in, so it is the order worth testing.
RUN set -eux; \
    for lib in ${LIBS}; do \
        python -c "import torch, importlib, sys; m = importlib.import_module(sys.argv[1]); print('  ' + sys.argv[1] + ': ok -> ' + m.__file__)" "$lib"; \
    done
