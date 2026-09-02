#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Runs the unit tests meson defines, selecting by engine and suite.
#   run.sh [BUILD_DIR]     default: build
# Env: ENGINE=all|<name>  SUITES=c,fast,complete  EMULATE=1  VFHE_COVERAGE=true
#
# -f keeps globbing off throughout: the selectors are meson's to match, not the
# shell's, and an unlucky filename would otherwise consume one.
set -euf

build=${1:-build}
engine=${ENGINE:-all}
suites=${SUITES:-c,complete}
here=$(cd "$(dirname "$0")" && pwd)

# A test's name carries both axes, so a selector is a glob and a typo in either
# exits 1. --suite would union instead, and pass when it selects nothing.
selectors=''
for suite in $(echo "$suites" | tr ',' ' '); do
    case $suite in
        c) kind='c-*' ;;
        fast | complete) kind="pytest-$suite" ;;
        *)
            echo "unknown suite '$suite'; pick from c, fast, complete" >&2
            exit 2
            ;;
    esac
    if [ "$engine" = all ]; then
        selectors="$selectors *-$kind"
    else
        selectors="$selectors $engine-$kind"
    fi
done

if [ "${VFHE_COVERAGE:-}" = true ] && [ "$engine" = all ]; then
    echo "measure one engine at a time: ENGINE=<name> VFHE_COVERAGE=true" >&2
    exit 2
fi

if [ "${EMULATE:-}" = 1 ]; then
    if [ "$engine" = all ]; then
        echo "EMULATE=1 emulates one engine: pass ENGINE=<name>" >&2
        exit 2
    fi
    # The guard reads the launcher from the environment.
    VFHE_EMULATOR=$("$here/../../tools/sde/fetch.sh")
    export VFHE_EMULATOR
    set -- --setup "${engine}_emulated"
else
    set --
fi

# shellcheck disable=SC2086  # deliberate: each selector is its own argument
meson test -C "$build" -v "$@" $selectors

if [ "${VFHE_COVERAGE:-}" = true ]; then
    meson compile -C "$build" --ninja-args=coverage
fi
