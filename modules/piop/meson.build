# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

module = {
    'sources': files('c/src/merkle.c', 'c/src/mle.c', 'c/src/sumcheck.c'),
    'includes': [include_directories('c/include')],
    'headers': files(
        'c/include/merkle.h',
        'c/include/mle.h',
        'c/include/sumcheck.h',
    ),
    'cdefs': files(
        'python/cdef/merkle.cdef',
        'python/cdef/piop.cdef',
        'python/cdef/sumcheck.cdef',
    ),
}
