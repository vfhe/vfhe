# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

# GP25's symbols are prototyped in mlwe.h; the rest in fhe.h.
vfhe_c_sources += files(
    'c/src/bfv.c',
    'c/src/cggi16.c',
    'c/src/gp25.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
)
vfhe_c_public_headers += files(
    'c/include/fhe.h',
)
vfhe_python_cdefs += files(
    'python/cdef/fhe.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)
