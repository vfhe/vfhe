#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

set -eu
here=$(cd "$(dirname "$0")" && pwd)

notes=$here/print.sh
fixture=$here/fixtures/changelog.md
fail=0

check() {  # check <label> <expected> <actual>
    if [ "$2" = "$3" ]; then
        printf '  %-48s [ok]\n' "$1"
    else
        printf '  %-48s [FAIL] wanted %s, got %s\n' "$1" "$2" "$3"
        fail=1
    fi
}
status() {  # status <command>...: its exit code, 0 on success
    if "$@" >/dev/null 2>&1; then echo 0; else echo $?; fi
}
flat() { "$notes" "$@" 2>/dev/null | tr '\n' '/'; }

check "stops at the next heading"        "### Added//- a thing//**Full Changelog**: https://example.invalid/compare/1.1.0...1.2.0/" "$(flat 1.2.0 "$fixture")"
check "stops at the link definitions"    "- an older thing/"  "$(flat 1.1.0 "$fixture")"
check "trims blank edges"                "nothing yet/"       "$(flat Unreleased "$fixture")"
check "1.1 does not answer for 1.1.0"    1 "$(status "$notes" 1.1 "$fixture")"
check "refuses an undocumented version"  1 "$(status "$notes" 9.9.9 "$fixture")"
check "no version is a usage error"      2 "$(status "$notes" '' "$fixture")"
check "the real changelog still answers" 0 "$(status "$notes" 0.0.2)"

if [ "$fail" = 0 ]; then
    echo "OK: print.sh behaves."
else
    echo "FAILED"
fi
exit "$fail"
