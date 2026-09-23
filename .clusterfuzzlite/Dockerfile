# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:47bcf5c42c4863af02843d16315014011571e31ece4ac9110a77a2e43949c5d2
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe

RUN git submodule update --init external/blake3/blake3

RUN python3 -m pip install --no-cache-dir --constraint ci-constraints-requirements.txt meson ninja setuptools-scm

COPY .clusterfuzzlite/build.sh $SRC/
