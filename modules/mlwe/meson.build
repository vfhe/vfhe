# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

vfhe_c_sources += files(
    'c/src/gadget_rns.c',
    'c/src/lwe.c',
    'c/src/mgsw.c',
    'c/src/mlwe.c',
    'c/src/mlwe_rns.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
)
vfhe_c_public_headers += files(
    'c/include/mlwe.h',
    'c/include/rns-rlwe.h',
)
vfhe_python_cdefs += files(
    'python/cdef/mlwe.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)
