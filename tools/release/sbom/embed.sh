#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Puts each vendored dependency's CycloneDX fragment in every wheel, where PEP 770
# has readers look. `wheel pack` rewrites RECORD, so each ends up listed.
#   embed.sh WHEEL...
set -eu

if [ $# -eq 0 ]; then
    echo "usage: $0 WHEEL..." >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/../../.." && pwd)
python=${PYTHON:-python3}

# Every vendored or adapted dependency's CycloneDX fragment.
fragments="
external/blake3/blake3.cdx.json
external/adapted.cdx.json
"

for wheel in "$@"; do
    into=$(dirname "$wheel")
    staging=$into/.sbom
    rm -rf "$staging"
    $python -m wheel unpack -d "$staging" "$wheel" >/dev/null

    sboms=$(echo "$staging"/*/*.dist-info)/sboms
    mkdir -p "$sboms"

    for fragment in $fragments; do
        cp "$root/$fragment" "$sboms/$(basename "$fragment")"
    done

    $python -m wheel pack -d "$into" "$staging"/* >/dev/null
    rm -rf "$staging"
    echo "embedded SBOMs in $(basename "$wheel")"
done
