#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -eu

here=$(cd "$(dirname "$0")" && pwd)
version=$(sed -n 's/.*BLAKE3_VERSION_STRING "\([^"]*\)".*/\1/p' "$here/blake3/c/blake3.h")

if ! commit=$(git -C "$here/blake3" rev-parse HEAD 2>/dev/null); then
    echo "no git history: skipping"
    exit 0
fi

if grep -q "\"version\": \"$version\"" "$here/blake3.cdx.json" &&
    grep -q "$commit" "$here/blake3.cdx.json"; then
    echo "blake3.cdx.json: BLAKE3 $version at $commit"
else
    echo "blake3.cdx.json does not describe BLAKE3 $version at $commit" >&2
    exit 1
fi
