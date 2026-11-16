# SPDX-FileCopyrightText: 2026 The vfhe authors
# SPDX-License-Identifier: Apache-2.0
# One definition for both self-hosted images: BASE is Ubuntu for the
# avx512ifma runner and NVIDIA's CUDA toolkit image for the cuda runner.
ARG BASE=ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea
FROM ${BASE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libbz2-1.0 \
    libffi8 \
    liblzma5 \
    libncursesw6 \
    libreadline8 \
    libsqlite3-0 \
    libssl3 \
    libuuid1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*
