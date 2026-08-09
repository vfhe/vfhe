# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files(
    'c/src/bfv.c',
    'c/src/gp25.c',
)
cdef_files += files('python/cdef/fhe.cdef')

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('python/cdef', install_dir: source_dir / 'modules/fhe/python')
install_subdir('python/src/vfhe', install_dir: python_dir)
