#!/bin/sh
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs smoke cases against an installed distribution.
#   run.sh [NAME...]
set -eu

here=$(cd "$(dirname "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1

available=''
for path in "$here"/cases/*.py; do
    name=$(basename "$path" .py)
    if [ "${name#_}" = "$name" ]; then
        available="$available $name"
    fi
done
available=${available# }

# tox joins its posargs into one argument, so split on whitespace.
selected=$*
if [ -z "$selected" ]; then
    selected=$available
fi

# Reject a typo before running anything.
for name in $selected; do
    if ! echo "$available" | grep -qwF "$name"; then
        echo "no such smoke case: $name" >&2
        echo "have: $available" >&2
        exit 2
    fi
done

failed=''
for name in $selected; do
    printf '\n--- smoke-%s\n' "$name"
    if ! python3 "$here/cases/$name.py"; then
        failed="$failed $name"
    fi
done

if [ -n "$failed" ]; then
    printf '\nFAILED: %s\n' "${failed# }"
    exit 1
fi
printf '\nPASSED: %s\n' "$selected"
