# SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
# SPDX-License-Identifier: Apache-2.0

if not fs.exists('blake3/c/blake3.c')
    error('BLAKE3 submodule missing. Run: git submodule update --init --recursive')
endif

module = {
    'sources': files(
        'blake3/c/blake3.c',
        'blake3/c/blake3_dispatch.c',
        'blake3/c/blake3_portable.c',
    ),
    'includes': [include_directories('blake3/c')],
}

if host_machine.cpu_family() == 'x86_64'
    module += {
        # Pre-assembled kernels; BLAKE3 dispatches between them at runtime.
        'simd_sources': files(
            'blake3/c/blake3_sse2_x86-64_unix.S',
            'blake3/c/blake3_sse41_x86-64_unix.S',
            'blake3/c/blake3_avx2_x86-64_unix.S',
            'blake3/c/blake3_avx512_x86-64_unix.S',
        ),
        'simd_flags_off': [
            '-DBLAKE3_NO_SSE2',
            '-DBLAKE3_NO_SSE41',
            '-DBLAKE3_NO_AVX2',
            '-DBLAKE3_NO_AVX512',
        ],
    }
elif host_machine.cpu_family() == 'aarch64'
    module += {
        'simd_sources': files('blake3/c/blake3_neon.c'),
        'simd_flags_on': ['-DBLAKE3_USE_NEON=1'],
        'simd_flags_off': ['-DBLAKE3_USE_NEON=0'],
    }
endif
