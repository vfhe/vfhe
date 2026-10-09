# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

module = {
    'sources': files(
        'c/src/gadget_rns.c',
        'c/src/keyswitch.c',
        'c/src/keyswitch_rns.c',
        'c/src/lwe.c',
        'c/src/mgsw.c',
        'c/src/mlwe.c',
        'c/src/mlwe_rns.c',
        'c/src/rns_rows.c',
    ),
    'includes': [include_directories('c/include')],
    'headers': files('c/include/mlwe.h', 'c/include/rns-rlwe.h'),
    'cdefs': files('python/cdef/mlwe.cdef'),
}
