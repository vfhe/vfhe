# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:1f9cf74bc3d7029b528494542f9d9851b256dceed38d3f420989434dc5ae0760
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe

RUN git submodule update --init external/blake3/blake3 external/unity/unity

RUN python3 -m pip install --no-cache-dir --constraint ci-constraints-requirements.txt meson ninja

COPY .clusterfuzzlite/build.sh $SRC/
