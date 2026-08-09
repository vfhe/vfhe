# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files(
    'c/src/gadget.c',
    'c/src/lwe.c',
    'c/src/mgsw.c',
    'c/src/mlwe.c',
)
lib_includes += include_directories('c/include')
header_files += files(
    'c/include/mlwe.h',
    'c/include/rns-rlwe.h',
)
cdef_files += files('python/cdef/mlwe.cdef')

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('c/include', install_dir: source_dir / 'modules/mlwe/c')
install_subdir('python/cdef', install_dir: source_dir / 'modules/mlwe/python')
install_subdir('python/src/vfhe', install_dir: python_dir)
