# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

module = {
    'sources': files(
        'c/src/expand.c',
        'c/src/hash.c',
        'c/src/merkle.c',
        'c/src/normal.c',
        'c/src/prng.c',
    ),
    'c_tests': files('c/test/test_expand.c', 'c/test/test_hash.c', 'c/test/test_prng.c'),
    'includes': [include_directories('c/include')],
    'headers': files('c/include/crypto.h', 'c/include/merkle.h'),
    'cdefs': files('python/cdef/hash.cdef', 'python/cdef/merkle.cdef', 'python/cdef/prng.cdef'),
}
