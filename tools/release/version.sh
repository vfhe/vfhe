#!/bin/sh
# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Prints the version a build would carry, from .version if a dist script wrote one.
# Takes no arguments, and runs from any directory.
set -eu

cd "$(dirname "$0")/../.."
cat .version 2>/dev/null || python3 -m setuptools_scm
