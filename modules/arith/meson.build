# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files(
    # the interface, and what belongs to no single representation
    'c/src/arith_generic.c',
    'c/src/number_theory.c',
    # shared modular-arithmetic and transform kernels
    'c/src/kernels/mod.c',
    'c/src/kernels/mod32.c',
    'c/src/kernels/mod50.c',
    'c/src/kernels/mod64.c',
    'c/src/kernels/mod_portable.c',
    'c/src/kernels/mod_scalar.c',
    'c/src/kernels/ntt.c',
    'c/src/kernels/ntt32.c',
    'c/src/kernels/ntt50.c',
    'c/src/kernels/ntt64.c',
    'c/src/kernels/ntt_portable.c',
    'c/src/kernels/ntt_scalar.c',
    # one directory per implementation
    'c/src/complex/complex.c',
    'c/src/field/field.c',
    'c/src/field/field_vector.c',
    'c/src/mp/mp_polynomial.c',
    'c/src/pmf/pmf.c',
    'c/src/rns/arith_rns.c',
    'c/src/rns/rns_polynomial.c',
    'c/src/rns/rns_vector.c',
)
lib_includes += include_directories('c/include', 'c/src')
header_files += files(
    'c/include/arith_config.h',
    'c/include/arith.h',
    'c/include/arith_generic.h',
)
cdef_files += files('python/cdef/arith.cdef')
c_test_files += files(
    'c/test/test_arith_generic.c',
    'c/test/test_mod.c',
    'c/test/test_ntt.c',
    'c/test/test_number_theory.c',
)

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('c/include', install_dir: source_dir / 'modules/arith/c')
install_subdir('python/cdef', install_dir: source_dir / 'modules/arith/python')
install_subdir('python/src/vfhe', install_dir: python_dir)
