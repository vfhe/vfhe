#!/bin/bash -eu
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

TREE="$SRC/vfhe"

set -a
# shellcheck source=/dev/null
. "$TREE/.clusterfuzzlite/.env"
set +a

meson setup "$WORK" "$TREE" \
    --buildtype=plain \
    -Dfuzz=true \
    -Dfuzz_engine="$VFHE_ENGINE" \
    -Dfuzz_link_args="$LIB_FUZZING_ENGINE"

meson compile -C "$WORK"

for test in "$WORK"/*; do
    if [ -f "$test" ] && [ -x "$test" ]; then
        cp "$test" "$OUT/"
    fi
done

ls "$OUT"
