# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files(
    'c/src/complex.c',
    'c/src/field.c',
    'c/src/mod.c',
    'c/src/mod32.c',
    'c/src/mod50.c',
    'c/src/mod64.c',
    'c/src/mod_portable.c',
    'c/src/mp_polynomial.c',
    'c/src/ntt.c',
    'c/src/ntt32.c',
    'c/src/ntt50.c',
    'c/src/ntt64.c',
    'c/src/ntt_portable.c',
    'c/src/number_theory.c',
    'c/src/rns_polynomial.c',
    'c/src/rns_vector.c',
)
lib_includes += include_directories('c/include', 'c/src')
header_files += files('c/include/arith.h')
cdef_files += files('python/cdef/arith.cdef')
c_test_files += files(
    'c/test/test_mod.c',
    'c/test/test_ntt.c',
    'c/test/test_number_theory.c',
)

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('c/include', install_dir: source_dir / 'modules/arith/c')
install_subdir('python/cdef', install_dir: source_dir / 'modules/arith/python')
install_subdir('python/src/vfhe', install_dir: python_dir)
