# syntax=docker/dockerfile:1.7

FROM mcr.microsoft.com/devcontainers/python:3.10

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        nvidia-cuda-toolkit \
        gcc-13 \
        g++-13 \
        ocl-icd-libopencl1 \
        clinfo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m venv /opt/venv

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt \
    && CC=/usr/bin/gcc-13 \
       CXX=/usr/bin/g++-13 \
       CUDACXX=/usr/bin/nvcc \
       /opt/venv/bin/pip install --force-reinstall --no-binary lightgbm lightgbm \
         --config-settings=cmake.define.USE_CUDA=ON \
         --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/gcc-13 \
         --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
         --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13

ENV PATH=/opt/venv/bin:${PATH}
