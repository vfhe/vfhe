# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:3cf598908df31d5bcc8013f9040e7689daef8cbeedd0401789158fb1abd42cf4
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe

RUN git submodule update --init external/blake3/blake3 external/unity/unity

RUN python3 -m pip install --no-cache-dir --constraint ci-constraints-requirements.txt meson ninja

COPY .clusterfuzzlite/build.sh $SRC/
