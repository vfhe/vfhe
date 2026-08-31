# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

module = {
    'sources': files(
        'c/src/alloc.c',
        'c/src/build_info.c',
        'c/src/cpu_probe.c',
        'c/src/util.c',
    ),
    'includes': [include_directories('c/include')],
    'headers': files('c/include/util.h'),
    'cdefs': files('python/cdef/engine.cdef'),
    'c_tests': files('c/test/test_cpu_probe.c'),
    'packages': ['dynamic_extensions', 'engine', 'info'],
    # The engine picker's probe, built from just these so asking "what can this
    # CPU run?" never loads an engine.
    'probe': {
        'sources': files('c/src/cpu_probe.c'),
        'header': files('c/include/vfhe_cpu.h'),
        'cdef': files('python/cdef/vfhe_cpu.cdef'),
        'includes': include_directories('c/include'),
    },
}
