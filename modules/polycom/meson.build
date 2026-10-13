# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

vfhe_c_sources += files(
    'c/src/rscode_rns.c',
    'c/src/rscode_field.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
)
vfhe_c_public_headers += files(
    'c/include/rscode.h',
)
vfhe_python_cdefs += files(
    'python/cdef/polycom.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)
