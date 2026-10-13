#!/bin/sh
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs the unit tests meson defines, selecting by engine and suite.
#   run.sh [BUILD_DIR]     default: build
# Env: ENGINE=all|<name>  SUITES=c,fast,complete  EMULATE=1  VFHE_COVERAGE=true
# ENGINE=all means every built engine this CPU runs; a named engine must be one
# of them unless EMULATE=1.
set -eu

build=${1:-build}
engine=${ENGINE:-all}
here=$(cd "$(dirname "$0")" && pwd)

# An x86 engine's name is the /proc/cpuinfo flag it needs; the others are baseline.
runs() { [ "$1" = portable ] || [ "$(uname -m)" != x86_64 ] || grep -qw "$1" /proc/cpuinfo 2>/dev/null; }

if [ "$engine" = all ]; then
    # A test's name starts with its engine, so the built engines fall out of the list.
    engines=''
    for each in $(meson test -C "$build" --list | sed 's/^vfhe://; s/-.*//' | sort -u); do
        runs "$each" && engines="$engines $each"
    done
elif [ "${EMULATE:-}" = 1 ] || runs "$engine"; then
    engines=$engine
else
    echo "this CPU cannot run $engine; rerun with EMULATE=1 to emulate it" >&2
    exit 1
fi

# A test's name carries both axes, so a selector is a glob and a typo in either
# exits 1. --suite unions its arguments and passes when nothing matches.
set --
for suite in $(echo "${SUITES:-c,complete}" | tr ',' ' '); do
    for each in $engines; do
        case $suite in
            c) set -- "$@" "$each-c-*" ;;
            *) set -- "$@" "$each-py-$suite" ;;
        esac
    done
done

if [ "${EMULATE:-}" = 1 ]; then
    # meson defines the emulated setup once it finds the launcher fetch.sh fetched.
    "$here/../sde/fetch.sh"
    meson setup --reconfigure "$build" >/dev/null
    set -- --setup "${engine}_emulated" "$@"
fi

meson test -C "$build" -v "$@"

if [ "${VFHE_COVERAGE:-}" = true ]; then
    meson compile -C "$build" --ninja-args=coverage
fi
