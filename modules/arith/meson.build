# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

module = {
    'sources': files(
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
    ),
    'includes': [include_directories('c/include')],
    'headers': files('c/include/arith.h'),
    'cdefs': files('python/cdef/arith.cdef'),
    'fuzz': files('c/fuzz/fuzz_ntt.c'),
    'c_tests': files(
        'c/test/test_mod.c',
        'c/test/test_ntt.c',
        'c/test/test_number_theory.c',
    ),
}
