#!/bin/bash -eu
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

tree="$SRC/vfhe"

# shellcheck source=.env
. "$tree/.clusterfuzzlite/.env"

# An x86 engine's name is the /proc/cpuinfo flag it needs; without it every
# input looks like a crash. Advanced SIMD is arm64 baseline; portable needs nothing.
if [ "$VFHE_ENGINE" != portable ] && [ "$(uname -m)" = x86_64 ] && ! grep -qw "$VFHE_ENGINE" /proc/cpuinfo; then
    echo "this machine cannot run $VFHE_ENGINE" >&2
    exit 1
fi

meson setup "$WORK" "$tree" \
    --buildtype=plain \
    -Dfuzz="$VFHE_ENGINE" \
    -Dfuzz_link_args="$LIB_FUZZING_ENGINE"

meson compile -C "$WORK" fuzzers

# meson mirrors the source layout, so the fuzzers land beside their meson.build.
fuzzers="$WORK/test"
find "$fuzzers" -maxdepth 1 -type f -perm -u+x -exec cp {} "$OUT/" \;

# ClusterFuzzLite accepts an empty $OUT, so a wrong path here would fuzz nothing
# every night and still report success.
if [ -z "$(ls -A "$OUT")" ]; then
    echo "no fuzzers found in $fuzzers" >&2
    exit 1
fi
