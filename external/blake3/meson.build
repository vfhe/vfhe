# SPDX-FileCopyrightText: 2026 The vFHE Authors
# SPDX-License-Identifier: Apache-2.0

if not fs.exists('blake3/c/blake3.c')
    error('BLAKE3 submodule missing. Run: git submodule update --init --recursive')
endif

vfhe_c_sources += files(
    'blake3/c/blake3.c',
    'blake3/c/blake3_dispatch.c',
    'blake3/c/blake3_portable.c',
)
vfhe_c_include_dirs += include_directories('blake3/c')

# --- Kernels ------------------------------------------------------------------

if host_machine.cpu_family() == 'x86_64'
    vfhe_c_sources += files(
        'blake3/c/blake3_sse2_x86-64_unix.S',
        'blake3/c/blake3_sse41_x86-64_unix.S',
        'blake3/c/blake3_avx2_x86-64_unix.S',
        'blake3/c/blake3_avx512_x86-64_unix.S',
    )
elif host_machine.cpu_family() == 'aarch64'
    vfhe_c_sources += files('blake3/c/blake3_neon.c')
endif
