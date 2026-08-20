# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files('c/src/mle.c', 'c/src/sumcheck.c')
lib_includes += include_directories('c/include')
header_files += files('c/include/mle.h', 'c/include/sumcheck.h')
cdef_files += files('python/cdef/piop.cdef', 'python/cdef/sumcheck.cdef')

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('c/include', install_dir: source_dir / 'modules/piop/c')
install_subdir('python/cdef', install_dir: source_dir / 'modules/piop/python')
install_subdir('python/src/vfhe', install_dir: python_dir)
