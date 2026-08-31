# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:8b4a73d83374b298a0a771eeec9a1d44ceff3416a678b7d7946303389d022aca
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe

RUN git submodule update --init external/blake3/blake3

RUN python3 -m pip install --no-cache-dir --constraint ci-constraints-requirements.txt meson ninja setuptools-scm

COPY .clusterfuzzlite/build.sh $SRC/
