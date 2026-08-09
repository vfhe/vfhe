#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Verify the build provenance of every file an index serves for a release:
#
#   REPOSITORY=owner/repo SIGNER_WORKFLOW=owner/repo/.github/workflows/x.yml \
#     provenance.sh vfhe 1.2.3
#
# The files are enumerated from the index, never resolved by an installer: pip
# would fetch the one wheel this host matches and leave the sdist and every
# other wheel unverified.
set -euo pipefail

PACKAGE=${1:?usage: provenance.sh <package> <version>}
VERSION=${2:?usage: provenance.sh <package> <version>}
INDEX_API=${INDEX_API:-https://pypi.org/pypi}
REPOSITORY=${REPOSITORY:?REPOSITORY must name the repository the attestation carries}
SIGNER_WORKFLOW=${SIGNER_WORKFLOW:?SIGNER_WORKFLOW must name the workflow allowed to sign}
ATTEMPTS=${ATTEMPTS:-5}
DELAY=${DELAY:-10}
DEST=${DEST:-published}

urls=""
for attempt in $(seq 1 "$ATTEMPTS"); do
    urls=$(curl -sS "$INDEX_API/$PACKAGE/$VERSION/json" | jq -r '.urls[].url' || true)
    [ -n "$urls" ] && break
    echo "attempt $attempt/$ATTEMPTS: $PACKAGE $VERSION not served yet"
    sleep "$DELAY"
done
[ -n "$urls" ] || {
    echo "::error::$PACKAGE $VERSION is not on $INDEX_API"
    exit 1
}

mkdir -p "$DEST"
while IFS= read -r url; do
    [ -n "$url" ] && curl -sSL --output-dir "$DEST" -O "$url"
done <<<"$urls"

echo "$PACKAGE $VERSION: $(printf '%s\n' "$urls" | wc -l | tr -d ' ') files served"
for file in "$DEST"/*; do
    echo "::group::$(basename "$file")"
    gh attestation verify "$file" --repo "$REPOSITORY" --signer-workflow "$SIGNER_WORKFLOW"
    echo "::endgroup::"
done
