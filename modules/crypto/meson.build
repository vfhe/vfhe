# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

vfhe_c_sources += files(
    'c/src/expand.c',
    'c/src/hash.c',
    'c/src/merkle.c',
    'c/src/normal.c',
    'c/src/prng.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
)
vfhe_c_public_headers += files(
    'c/include/crypto.h',
    'c/include/merkle.h',
)
vfhe_c_unit_tests += files(
    'c/test/unit/test_expand.c',
    'c/test/unit/test_hash.c',
    'c/test/unit/test_prng.c',
)
vfhe_python_cdefs += files(
    'python/cdef/hash.h',
    'python/cdef/merkle.h',
    'python/cdef/prng.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)
