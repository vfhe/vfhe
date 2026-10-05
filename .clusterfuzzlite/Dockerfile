# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:5add54bcdde77ae3427025db959a44f379ca64b6332bcddcc2766924582e47d9
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe

RUN git submodule update --init external/blake3/blake3

RUN python3 -m pip install --no-cache-dir --constraint ci-constraints-requirements.txt meson ninja setuptools-scm

COPY .clusterfuzzlite/build.sh $SRC/
