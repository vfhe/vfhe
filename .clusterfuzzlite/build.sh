#!/bin/bash -eu
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

BUILD="$WORK/meson"
HARNESSES="$SRC/vfhe/.clusterfuzzlite/build_c_fuzz_tests.py"

ENGINE="$(python3 "$HARNESSES" --engine)"

# The container has the tree but no git history, so freeze a version for
# meson's project() the way an sdist does.
[ -f "$SRC/vfhe/.version" ] || echo "0.0.0+fuzz" > "$SRC/vfhe/.version"

meson setup "$BUILD" "$SRC/vfhe" -Dfuzz=true
meson compile -C "$BUILD" "vfhe_$ENGINE"
python3 "$HARNESSES" "$BUILD/libvfhe_$ENGINE.a"
