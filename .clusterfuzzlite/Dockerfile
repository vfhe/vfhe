# SPDX-License-Identifier: Apache-2.0
FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:8b4a73d83374b298a0a771eeec9a1d44ceff3416a678b7d7946303389d022aca
COPY . $SRC/vfhe
WORKDIR $SRC/vfhe
# BLAKE3 is compiled into every harness; ensure the submodule is present.
RUN git submodule update --init external/blake3
COPY .clusterfuzzlite/build.sh $SRC/
