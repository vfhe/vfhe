# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

vfhe_c_sources += files(
    'c/src/alloc.c',
    'c/src/memory_pool.c',
    'c/src/parallel.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
)
vfhe_c_public_headers += files(
    'c/include/alloc.h',
    'c/include/memory_pool.h',
    'c/include/parallel.h',
    'c/include/x86_64.h',
)
vfhe_c_unit_tests += files(
    'c/test/unit/test_alloc.c',
    'c/test/unit/test_memory_pool.c',
    'c/test/unit/test_parallel.c',
)
vfhe_c_benchmarks += files(
    'c/bench/bench_memory_pool.c',
    'c/bench/bench_parallel.c',
)
vfhe_python_cdefs += files(
    'python/cdef/memory_pool.h',
    'python/cdef/parallel.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)
