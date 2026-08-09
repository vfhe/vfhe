# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files(
    'c/src/aes_rng.c',
    'c/src/build_info.c',
    'c/src/cpu_probe.c',
    'c/src/misc.c',
    'c/src/misc_tp.c',
    'c/src/prng.c',
)
lib_includes += include_directories('c/include')
header_files += files('c/include/misc.h')
cdef_files += files(
    'python/cdef/misc.cdef',
    'python/cdef/vfhe_cpu.cdef',
)

# The engine picker's probe: its own extension, built from just these, so
# asking "what can this CPU run?" never loads an engine.
probe_sources = files('c/src/cpu_probe.c')
probe_header = files('c/include/vfhe_cpu.h')
probe_cdef = files('python/cdef/vfhe_cpu.cdef')
probe_includes = include_directories('c/include')

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('c/include', install_dir: source_dir / 'modules/misc/c')
install_subdir('python/cdef', install_dir: source_dir / 'modules/misc/python')
install_subdir('python/src/vfhe', install_dir: python_dir)

# The import-time engine picker; top-level, like the extensions it fronts.
# py_install is the only thing here a fuzzing configuration does not have.
if not fuzz
    py_install.install_sources('python/src/_vfhe_native.py', subdir: '')
endif
