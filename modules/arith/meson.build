# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

vfhe_c_sources += files(
    # the interface, and what belongs to no single representation
    'c/src/arith_generic.c',
    'c/src/mle.c',
    'c/src/number_theory.c',
    'c/src/util.c',
    # shared modular-arithmetic and transform kernels
    'c/src/kernels/interleave.c',
    'c/src/kernels/mod.c',
    'c/src/kernels/mod_basecvt.c',
    'c/src/kernels/mod32.c',
    'c/src/kernels/mod50.c',
    'c/src/kernels/mod64.c',
    'c/src/kernels/mod_portable.c',
    'c/src/kernels/mod_scalar.c',
    'c/src/kernels/mod_w32.c',
    'c/src/kernels/ntt.c',
    'c/src/kernels/ntt32.c',
    'c/src/kernels/ntt50.c',
    'c/src/kernels/ntt64.c',
    'c/src/kernels/ntt_portable.c',
    'c/src/kernels/ntt_scalar.c',
    'c/src/kernels/ntt_w32.c',
    # one directory per implementation
    'c/src/complex/complex.c',
    'c/src/field/field.c',
    'c/src/field/field_fused.c',
    'c/src/field/field_ntt.c',
    'c/src/field/field_vector.c',
    'c/src/mp/mp_polynomial.c',
    'c/src/pmf/pmf.c',
    'c/src/pmf/pmf_ntt.c',
    'c/src/pmf/pmf_vector.c',
    'c/src/rns/arith_rns.c',
    'c/src/rns/rns_io.c',
    'c/src/rns/rns_polynomial.c',
    'c/src/rns/rns_vector.c',
)
vfhe_c_include_dirs += include_directories(
    'c/include',
    'c/src',
)
vfhe_c_public_headers += files(
    'c/include/arith.h',
    'c/include/arith_generic.h',
    'c/include/mle.h',
)
vfhe_c_unit_tests += files(
    'c/test/unit/test_arith_generic.c',
    'c/test/unit/test_mod.c',
    'c/test/unit/test_ntt.c',
    'c/test/unit/test_number_theory.c',
    'c/test/unit/test_pmf_ntt.c',
)
vfhe_c_fuzz_tests += files(
    'c/test/fuzz/fuzz_ntt.c',
    'c/test/fuzz/fuzz_field_ext.c',
)
vfhe_python_cdefs += files(
    'python/cdef/arith.h',
    'python/cdef/mle.h',
)

install_subdir(
    'python/src/vfhe',
    install_dir: py.get_install_dir(),
)
