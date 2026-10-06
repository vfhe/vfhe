#!/bin/sh
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
#
# The C that vfhe.kernels compiles against, as one tarball.
#   sources.sh SOURCE_ROOT OUTPUT SOURCE...
set -eu

root=$1
out=$2
shift 2

{
    cd "$root" && find modules external/blake3/blake3/c \
        \( -name '*.h' -o -name '*.inc' \) \
        ! -name '._*' ! -path '*/c/test/*' | sort
    printf '%s\n' "$@"
} | COPYFILE_DISABLE=1 tar czf "$out" -C "$root" -T -
