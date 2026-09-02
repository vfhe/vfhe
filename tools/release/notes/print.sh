#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Prints one version's section of a Keep a Changelog file, compare link included.
#   print.sh VERSION [CHANGELOG]
set -eu

# Exit 2 is a usage error; exit 1 below means the version is undocumented.
if [ -z "${1:-}" ] || [ $# -gt 2 ]; then
    echo "usage: $0 VERSION [CHANGELOG]" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/../../.." && pwd)
version=$1
changelog=${2:-$root/CHANGELOG.md}

# index() matches literally, so 1.1 never answers for 1.1.0 and a version's dots
# are not wildcards. Command substitution drops the trailing blank lines, and the
# `seen` guard drops the leading ones.
section=$(awk -v heading="## [$version]" '
    index($0, heading) == 1              { inside = 1; next }
    inside && index($0, "## [") == 1     { exit }
    inside && /^\[[^]]+\]: /             { exit }
    inside && (NF || seen)               { print; seen = 1 }
' "$changelog")
if [ -z "$section" ]; then
    echo "$changelog has no notes for $version" >&2
    exit 1
fi
printf '%s\n' "$section"

link=$(awk -v prefix="[$version]: " '
    index($0, prefix) == 1 { print substr($0, length(prefix) + 1); exit }
' "$changelog")
if [ -n "$link" ]; then
    printf '\n**Full Changelog**: %s\n' "$link"
fi
