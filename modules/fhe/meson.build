# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# GP25's symbols are prototyped in mlwe.h; the rest in fhe.h.
module = {
    'sources': files('c/src/bfv.c', 'c/src/cggi16.c', 'c/src/gp25.c'),
    'includes': [include_directories('c/include')],
    'headers': files('c/include/fhe.h'),
    'cdefs': files('python/cdef/fhe.cdef'),
}
