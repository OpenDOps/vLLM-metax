ARG BUILD_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi:9.6
ARG PYTHON_VERSION=3.10
ARG UV_EXTRA_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
ARG UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG UV_TRUSTED_HOST=repos.metax-tech.com

# may need passing a particular vllm version during build
ARG VLLM_VERSION
ARG MACA_VERSION
ARG CU_BRIDGE_VERSION=${MACA_VERSION}

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base

ARG PYTHON_VERSION

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"
RUN dnf -y install python3-pip && \
    dnf clean all

# --seed puts pip in the venv. Plain `uv venv` leaves `python -m pip` broken
# (/opt/venv/bin/python: No module named pip) even though `uv pip` works.
RUN python3 -m pip install --no-cache uv && \
    uv venv /opt/venv --python=${PYTHON_VERSION} --seed

RUN python3 --version && \
    uv self version


ENV UV_INDEX_STRATEGY="unsafe-best-match"

# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Must be on base so wheel_build (and final) can resolve MetaX torch/mcoplib wheels.
ARG UV_EXTRA_INDEX_URL
ARG UV_INDEX_URL
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}
ENV UV_INDEX_URL=${UV_INDEX_URL}

WORKDIR /workspace

#################### BASE BUILD IMAGE ####################

#################### Install MACA SDK and Metax-Driver ####################
FROM base AS full_maca

RUN yum makecache && yum install -y \
    unzip vim git openblas-devel make cmake \
    ninja-build gcc g++ procps-ng \
    libibverbs librdmacm libibumad \
    numactl-libs \
    && yum clean all

ARG MACA_VERSION

# Installing Metax-Driver
RUN printf "[metax-centos]\n\
name=Maca Driver Yum Repository\n\
baseurl=https://repos.metax-tech.com/r/metax-driver-centos-$(uname -m)/\n\
enabled=1\n\
gpgcheck=0" > /etc/yum.repos.d/metax-driver-centos.repo

# would install the newest 3.1.0.x release
# Metax-Driver mainly contains vbios and kmd file, which are not needed in a container.
# Here we want to get the mx-smi management tool. 
# kernel version mismatch errors are ignored
RUN yum makecache && \
    yum install -y metax-driver-${MACA_VERSION}* mxgvm && \
    yum clean all && rm -rf /var/cache/yum /tmp/*

# Installing MACA SDK from Softnova offline drop (public yum has no maca_sdk 3.8.2).
# Workflow copies maca-sdk-3.8.2.6-deb-x86_64.tar.xz into the build context from the
# self-hosted runner vendor dir. Archive is .deb-only; mxmaca-sdk-install.sh selects
# TYPE=rpm on yum hosts and cannot install these packages, so unpack data.tar.* here.
# cu-bridge ships in the SDK; do not curl gitee.
COPY maca-sdk-3.8.2.6-deb-x86_64.tar.xz /tmp/maca-sdk.tar.xz
# Softnova 3.8.2.6 debs use data.tar.gz (not data.tar.xz). Detect the member
# with `ar t` so we do not feed gzip payloads to xz.
RUN yum install -y binutils gzip xz \
    && tar -xJf /tmp/maca-sdk.tar.xz -C /tmp \
    && cd /tmp/maca-sdk-3.8.2.6 \
    && set -eu \
    && for f in deb/*.deb; do \
         echo "unpack $f"; \
         member="$(ar t "$f" | grep -E '^data\.tar(\.|$)' | head -n1)"; \
         test -n "$member" || { echo "no data.tar* in $f" >&2; ar t "$f" >&2; exit 1; }; \
         case "$member" in \
           data.tar.gz)  ar p "$f" "$member" | tar -xz -C / ;; \
           data.tar.xz)  ar p "$f" "$member" | tar -xJ -C / ;; \
           data.tar.bz2) ar p "$f" "$member" | tar -xj -C / ;; \
           data.tar)     ar p "$f" "$member" | tar -x  -C / ;; \
           *) echo "unsupported $member in $f" >&2; exit 1 ;; \
         esac; \
       done \
    && MACA_ROOT="$(ls -d /opt/maca-* 2>/dev/null | head -n1)" \
    && test -n "$MACA_ROOT" \
    && test -d "$MACA_ROOT/tools/cu-bridge" \
    && ln -sfn "$MACA_ROOT" /opt/maca \
    && test -d /opt/maca/tools/cu-bridge \
    && test -x /opt/maca/tools/cu-bridge/bin/cucc \
    && echo "/opt/maca/lib" > /etc/ld.so.conf.d/maca.conf \
    && echo "/opt/maca/mxgpu_llvm/lib" >> /etc/ld.so.conf.d/maca.conf \
    && ldconfig \
    && rm -rf /tmp/maca-sdk.tar.xz /tmp/maca-sdk-3.8.2.6 \
    && yum clean all && rm -rf /var/cache/yum

#################### Install MACA SDK and Metax-Driver ####################


#################### Build vllm-metax wheel ####################
FROM full_maca AS wheel_build

## Update environment variables
# setup MACA path
ENV MACA_PATH=/opt/maca
ENV MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin 
# cu-bridge
ENV CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
ENV CUDA_PATH=/root/cu-bridge/CUDA_DIR
ENV CUCC_CMAKE_ENTRY=2
# update PATH
ENV PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/tools/cu-bridge/bin:${PATH} 
ENV LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# install build and runtime dependencies and cache them
WORKDIR /workspace

# install vllm-metax build dependencies
COPY requirements/build.txt requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26.4 /opt/maca/share/mxsml/pymxsml-*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,src=.,target=/workspace/vllm-metax,rw \
    cd /workspace/vllm-metax && \
    uv build --wheel --out-dir=/workspace/vllm_metax_wheel_dist
#################### Build vllm-metax wheel ####################


#################### CLEANUP ####################
FROM full_maca AS clean_maca

ARG MACA_VERSION

# Do NOT strip mcflashattn_*: MetaX torch._C is linked against libmcFlashAttn.so.
# Softnova .deb SDK may not register RPMs; ignore missing packages.
RUN (rpm -e --nodeps \
        mcflashinfer_${MACA_VERSION} \
        mcflashinfer_${MACA_VERSION}* \
        mxreport-${MACA_VERSION} \
        mxreport-${MACA_VERSION}* \
        mccltests-${MACA_VERSION} \
        mccltests-${MACA_VERSION}* \
        2>/dev/null || true) \
    && find /opt/maca/ -type f -name "*.a" -delete \
    && yum clean all && rm -rf /var/cache/yum /tmp/*

#################### CLEANUP ####################

#################### FINAL IMAGE ####################
FROM base AS final
ARG MACA_VERSION

ENV MACA_PATH=/opt/maca
ENV PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/tools/cu-bridge/bin:${PATH} 
ENV LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN yum makecache && yum install -y \
    gcc \
    binutils \
    procps-ng \
    libibverbs \
    librdmacm \
    libibumad \
    openblas \
    numactl-libs \
    && yum clean all && rm -rf /var/cache/yum /tmp/*

COPY --from=clean_maca /opt/maca /opt/maca
COPY --from=clean_maca /opt/mxdriver /opt/mxdriver

RUN find /opt/maca -name 'libmcFlashAttn.so*' | grep -q . \
    || (echo "libmcFlashAttn.so missing under /opt/maca — do not strip mcflashattn" >&2; exit 1)

WORKDIR /workspace
ARG UV_EXTRA_INDEX_URL
ARG UV_INDEX_URL
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}
ENV UV_INDEX_URL=${UV_INDEX_URL}

# install vllm-metax from built wheels
# Override OpenCV: MetaX common.txt wants >=4.13 (numpy>=2) which conflicts with mcoplib's numpy<2.
COPY --from=wheel_build /workspace/vllm_metax_wheel_dist /tmp/wheels
COPY requirements/opencv_override.txt requirements/opencv_override.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_OVERRIDE=requirements/opencv_override.txt \
    uv pip install /tmp/wheels/* 

# install empty vllm
ARG VLLM_VERSION
COPY requirements/maca_private.txt requirements/maca_private.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_TARGET_DEVICE=empty \
    UV_OVERRIDE=requirements/maca_private.txt \
    uv pip install --no-binary=vllm vllm==${VLLM_VERSION} 

# Fix(hank): don't know why vllm installation also brings in flashinfer-python, remove it here.
RUN uv pip uninstall flashinfer-python cupy-cuda12x

# Fix(hank): torch-metax still use numpy<2?
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26
#################### FINAL IMAGE ####################