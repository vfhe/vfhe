#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Runs a command only when this CPU has the capability it needs, else exits 77, meson's SKIP.
#   require_cpu.sh PROBE CAPABILITY -- CMD...
set -eu

if [ $# -lt 4 ] || [ "$3" != -- ]; then
    echo "usage: $0 PROBE CAPABILITY -- CMD..." >&2
    exit 2
fi
probe=$1
requires=$2
shift 3

# SDE instruments the program it launches, not what that program execs, so the
# launcher goes here: emulating this script instead would run the kernels on the
# bare CPU. The emulator reports the ISA, so it needs no probe.
if [ -n "${VFHE_SDE_FLAGS-}" ]; then
    : "${VFHE_EMULATOR:?no emulator: set it from tools/sde/fetch.sh}"
    # shellcheck disable=SC2086  # deliberate: each flag is its own argument
    exec "$VFHE_EMULATOR" $VFHE_SDE_FLAGS -no-follow-child -- "$@"
fi

if [ -z "$requires" ]; then
    exec "$@"
fi

rc=0
"$probe" "$requires" || rc=$?
case $rc in
    0) exec "$@" ;;
    1) echo "no $requires on this CPU; rerun with EMULATE=1 to emulate it" >&2; exit 77 ;;
    *) echo "the probe cannot judge '$requires' (exit $rc)" >&2; exit 1 ;;
esac
