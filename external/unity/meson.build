# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

if not fs.exists('unity/src/unity.c')
    error('Unity submodule missing. Run: git submodule update --init --recursive')
endif

unity_includes = include_directories('unity/src')
unity_dep = declare_dependency(
    include_directories: unity_includes,
    link_with: static_library(
        'unity',
        files('unity/src/unity.c'),
        include_directories: unity_includes,
        build_by_default: false,
    ),
)

meson.add_dist_script(
    'sh',
    '-c',
    'find "${MESON_DIST_ROOT:?}/external/unity/unity" -mindepth 1 -maxdepth 1 ! -name src ! -name LICENSE.txt -exec rm -rf {} +',
)
