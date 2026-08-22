# SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
# SPDX-License-Identifier: Apache-2.0

lib_sources += files('c/src/rscode.c')
lib_includes += include_directories('c/include')
header_files += files('c/include/rscode.h')
cdef_files += files('python/cdef/polycom.cdef')

python_source_dirs += meson.current_source_dir() / 'python/src'

install_subdir('c/include', install_dir: source_dir / 'modules/polycom/c')
install_subdir('python/cdef', install_dir: source_dir / 'modules/polycom/python')
install_subdir('python/src/vfhe', install_dir: python_dir)
