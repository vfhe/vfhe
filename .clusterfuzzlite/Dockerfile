# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:8aa36c8f128e4afa83ae566f413e944094a4923969b26b5cd85be6803d5eacf3
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe
# BLAKE3 is compiled into every harness; ensure the submodule is present.
RUN git submodule update --init external/blake3
COPY .clusterfuzzlite/build.sh $SRC/
