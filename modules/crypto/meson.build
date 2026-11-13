# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

vfhe_c_sources += files(
    'c/src/expand.c',
    'c/src/hash.c',
    'c/src/merkle.c',
    'c/src/normal.c',
    'c/src/random.c',
    'c/src/seeded.c',
    # one family of each kernel per engine; the other families compile to nothing
    'c/src/kernels/portable/aes128.c',
    'c/src/kernels/portable/filter.c',
    'c/src/kernels/portable/seed.c',
    'c/src/kernels/portable/stream.c',
    'c/src/kernels/x86_64/aes128.c',
    'c/src/kernels/x86_64/filter.c',
    'c/src/kernels/x86_64/seed.c',
    'c/src/kernels/x86_64/stream.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
)
vfhe_c_public_headers += files(
    'c/include/expand.h',
    'c/include/hash.h',
    'c/include/merkle.h',
    'c/include/normal.h',
    'c/include/random.h',
    'c/include/seeded.h',
)
vfhe_c_unit_tests += files(
    'c/test/unit/test_expand.c',
    'c/test/unit/test_hash.c',
    'c/test/unit/test_merkle.c',
    'c/test/unit/test_normal.c',
    'c/test/unit/test_random.c',
    'c/test/unit/test_seeded.c',
)
vfhe_python_cdefs += files(
    'python/cdef/hash.h',
    'python/cdef/merkle.h',
    'python/cdef/random.h',
    'python/cdef/seeded.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)

install_subdir(
    'python/bench',
    install_dir: py.get_install_dir() / 'vfhe' / 'crypto',
)
