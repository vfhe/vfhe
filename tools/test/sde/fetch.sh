#!/bin/sh
# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0
#
# Fetches and checksums the Intel SDE that .env pins into .cache/sde once, and
# links its launcher at .cache/sde/sde64, where meson looks.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=.env
. "$here/.env"

cache=$(cd "$here/../../.." && pwd)/.cache/sde
launcher=$cache/${tarball%.tar.xz}/sde64
archive=$cache/$tarball

if [ ! -x "$launcher" ]; then
    mkdir -p "$cache"
    trap 'rm -f "$archive"' EXIT  # the archive is needed only while this runs
    echo "[sde] $url" >&2
    curl -sSLf --max-time 300 -o "$archive" "$url"
    echo "$sha256  $archive" | shasum -a 256 -c -
    tar -xJf "$archive" -C "$cache"
    ln -sf "$launcher" "$cache/sde64"
fi

# SDE traces a child process, which Linux blocks unless ptrace is unrestricted.
scope=/proc/sys/kernel/yama/ptrace_scope
if [ -r "$scope" ] && [ "$(cat "$scope")" != 0 ]; then
    echo "[sde] $scope is on: tests that spawn compilers may fail. Fix: sudo sysctl -w kernel.yama.ptrace_scope=0" >&2
fi
