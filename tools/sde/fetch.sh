#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Prints the path to Intel SDE's launcher, fetching and checksumming it first if it is absent.
set -eu

# The sha256 is the integrity guarantee, so one authoritative URL suffices.
tarball=sde-external-10.8.0-2026-03-15-lin.tar.xz
sha256=50b320cd226acef7a491f5b321fc1be3c3c7984f9e27a456e64894b5b0979dd3
url=https://downloadmirror.intel.com/915934/$tarball

cache=$(cd "$(dirname "$0")/../.." && pwd)/.cache/sde
launcher=$cache/${tarball%.tar.xz}/sde64
archive=$cache/$tarball

if [ ! -x "$launcher" ]; then
    mkdir -p "$cache"
    trap 'rm -f "$archive"' EXIT  # an unverified download is never left behind
    echo "[sde] $url" >&2
    curl -sSLf --max-time 300 -o "$archive" "$url"
    echo "$sha256  $archive" | shasum -a 256 -c -
    tar -xJf "$archive" -C "$cache"
fi

# SDE traces a child process, which Linux blocks unless ptrace is unrestricted.
scope=/proc/sys/kernel/yama/ptrace_scope
if [ -r "$scope" ] && [ "$(cat "$scope")" != 0 ]; then
    echo "[sde] $scope is on: tests that spawn compilers may fail. Fix: sudo sysctl -w kernel.yama.ptrace_scope=0" >&2
fi

echo "$launcher"
