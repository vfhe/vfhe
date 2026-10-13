#!/bin/sh
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs smoke cases against an installed distribution.
#   run.sh [NAME...]     default: every case
set -eu

cases=$(cd "$(dirname "$0")/../../.." && pwd)/test/smoke
available=$(cd "$cases" && echo [!_]*.py | sed 's/\.py//g')
export PYTHONDONTWRITEBYTECODE=1

# tox joins its posargs into one argument, so the names split on whitespace.
# shellcheck disable=SC2048,SC2086  # deliberate: each name is its own argument
set -- ${*:-$available}

failed=''
for name in "$@"; do
    printf '\n--- smoke-%s\n' "$name"
    python3 "$cases/$name.py" || failed="$failed $name"
done

if [ -n "$failed" ]; then
    printf '\nFAILED:%s\n' "$failed"
    exit 1
fi
printf '\nPASSED: %s\n' "$*"
