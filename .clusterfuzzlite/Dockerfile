# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:5d8ad2d0945830c9cae00441202025e0609eb2d846e351a3217cb5355c1f2109
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe

RUN git submodule update --init external/blake3/blake3

RUN python3 -m pip install --no-cache-dir --constraint ci-constraints-requirements.txt meson ninja setuptools-scm

COPY .clusterfuzzlite/build.sh $SRC/
