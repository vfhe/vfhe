// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Interleaving two planes into one, the move every split-and-recombine layout
// makes: `out[2i] = even[i]`, `out[2i + 1] = odd[i]`.
//
// It carries no arithmetic, so it lives here rather than with a modulus
// family, and both the field and pseudo-Mersenne vectors call it.
#include <arith.h>
#include "arith_internal.h"

#if VFHE_HAVE_AVX512F
#include <immintrin.h>

void vec_interleave_u64(uint64_t *out, const uint64_t *even, const uint64_t *odd, uint64_t n)
{
    // Two permutes turn a lane of `even` and a lane of `odd` into the two
    // halves of the answer: each index below picks lane k of `even` (0-7) or
    // lane k-8 of `odd` (8-15).
    const __m512i first = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i second = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    uint64_t i = 0;

    for (; i + 8 <= n; i += 8)
    {
        const __m512i e = _mm512_loadu_si512((const void *)(even + i));
        const __m512i o = _mm512_loadu_si512((const void *)(odd + i));
        _mm512_storeu_si512((void *)(out + 2 * i), _mm512_permutex2var_epi64(e, first, o));
        _mm512_storeu_si512((void *)(out + 2 * i + 8), _mm512_permutex2var_epi64(e, second, o));
    }
    for (; i < n; i++)
    {
        out[2 * i] = even[i];
        out[2 * i + 1] = odd[i];
    }
}

#else

void vec_interleave_u64(uint64_t *out, const uint64_t *even, const uint64_t *odd, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        out[2 * i] = even[i];
        out[2 * i + 1] = odd[i];
    }
}

#endif
