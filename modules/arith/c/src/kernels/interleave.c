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

// --- k-way: deinterleaving and spreading ---------------------------------
//
// vec_deinterleave_*: out[j][m] = in[j + k * m] for j < k, m < n -- the
// transpose of `in` read as n rows of k. vec_interleave_k_*, its inverse:
// out[j + k * m] = in[j][m], a NULL in[j] reading as zeros. vec_spread_*:
// out[k * m] = in[m] and every other word of out's k * n zero, written in one
// pass -- the interleave of in alone, reading only it.
//
// The vector paths cover a power-of-two k up to a vector's lanes, and any k
// that is a multiple of them, at n a multiple of the lanes; every other shape
// takes the scalar loops, which are also what the other engines run.

#include <string.h>

#define VEC_DEINTERLEAVE_SCALAR(W)                                                                 \
    static void deinterleave_scalar_u##W(uint##W##_t *const *out, const uint##W##_t *in,           \
                                         uint64_t k, uint64_t n)                                   \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m++)                                                           \
            for (uint64_t j = 0; j < k; j++)                                                       \
                out[j][m] = in[j + k * m];                                                         \
    }                                                                                              \
    static void interleave_k_scalar_u##W(uint##W##_t *out, const uint##W##_t *const *in,           \
                                         uint64_t k, uint64_t n)                                   \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m++)                                                           \
            for (uint64_t j = 0; j < k; j++)                                                       \
                out[j + k * m] = in[j] != NULL ? in[j][m] : 0;                                     \
    }                                                                                              \
    static void spread_scalar_u##W(uint##W##_t *out, const uint##W##_t *in, uint64_t k,            \
                                   uint64_t n)                                                     \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m++)                                                           \
        {                                                                                          \
            out[k * m] = in[m];                                                                    \
            for (uint64_t j = 1; j < k; j++)                                                       \
                out[k * m + j] = 0;                                                                \
        }                                                                                          \
    }

VEC_DEINTERLEAVE_SCALAR(64)
VEC_DEINTERLEAVE_SCALAR(32)

#if VFHE_HAVE_AVX512F

static int is_power_of_two(uint64_t k) { return k != 0 && (k & (k - 1)) == 0; }

// One level of the perfect-shuffle network over `count` vectors: the pair
// (v[2i], v[2i + 1]) leaves its even lanes at i and its odd lanes at
// count / 2 + i. log2(count) levels over `count` consecutive vectors of an
// array leave vector j holding words j, j + count, j + 2 count, ... -- so for
// count equal to the lanes, the rows of a square tile become its columns.
static inline __attribute__((always_inline)) void shuffle_level_u64(__m512i *v, unsigned count)
{
    const __m512i even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512i t[8];
    for (unsigned i = 0; i < count / 2; i++)
    {
        t[i] = _mm512_permutex2var_epi64(v[2 * i], even, v[2 * i + 1]);
        t[count / 2 + i] = _mm512_permutex2var_epi64(v[2 * i], odd, v[2 * i + 1]);
    }
    for (unsigned i = 0; i < count; i++)
        v[i] = t[i];
}

static inline __attribute__((always_inline)) void shuffle_level_u32(__m512i *v, unsigned count)
{
    const __m512i even =
        _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd =
        _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    __m512i t[16];
    for (unsigned i = 0; i < count / 2; i++)
    {
        t[i] = _mm512_permutex2var_epi32(v[2 * i], even, v[2 * i + 1]);
        t[count / 2 + i] = _mm512_permutex2var_epi32(v[2 * i], odd, v[2 * i + 1]);
    }
    for (unsigned i = 0; i < count; i++)
        v[i] = t[i];
}

// The inverse of one shuffle level: the vectors at i and count / 2 + i
// interleave back into the pair at 2i, 2i + 1. The levels are all the same
// permutation, so log2(count) of these undo log2(count) shuffle levels.
static inline __attribute__((always_inline)) void unshuffle_level_u64(__m512i *v, unsigned count)
{
    const __m512i lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    __m512i t[8];
    for (unsigned i = 0; i < count / 2; i++)
    {
        t[2 * i] = _mm512_permutex2var_epi64(v[i], lo, v[count / 2 + i]);
        t[2 * i + 1] = _mm512_permutex2var_epi64(v[i], hi, v[count / 2 + i]);
    }
    for (unsigned i = 0; i < count; i++)
        v[i] = t[i];
}

static inline __attribute__((always_inline)) void unshuffle_level_u32(__m512i *v, unsigned count)
{
    const __m512i lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i hi =
        _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    __m512i t[16];
    for (unsigned i = 0; i < count / 2; i++)
    {
        t[2 * i] = _mm512_permutex2var_epi32(v[i], lo, v[count / 2 + i]);
        t[2 * i + 1] = _mm512_permutex2var_epi32(v[i], hi, v[count / 2 + i]);
    }
    for (unsigned i = 0; i < count; i++)
        v[i] = t[i];
}

// k <= lanes: each step loads the k vectors covering `lanes` rows of `in`
// and stores one vector of every output.
#define DEINTERLEAVE_SMALL(W, LANES)                                                               \
    static inline __attribute__((always_inline)) void deinterleave_small_u##W(                     \
        uint##W##_t *const *out, const uint##W##_t *in, const unsigned k, uint64_t n)              \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m += LANES)                                                    \
        {                                                                                          \
            __m512i v[LANES];                                                                      \
            for (unsigned i = 0; i < k; i++)                                                       \
                v[i] = _mm512_loadu_si512((const void *)(in + k * m + LANES * i));                 \
            for (unsigned c = 1; c < k; c *= 2)                                                    \
                shuffle_level_u##W(v, k);                                                          \
            for (unsigned j = 0; j < k; j++)                                                       \
                _mm512_storeu_si512((void *)(out[j] + m), v[j]);                                   \
        }                                                                                          \
    }

// k a multiple of the lanes: square tiles of `lanes` rows by `lanes` words,
// read as whole runs of a row and written as whole runs of an output.
#define DEINTERLEAVE_TILES(W, LANES)                                                               \
    static void deinterleave_tiles_u##W(uint##W##_t *const *out, const uint##W##_t *in,            \
                                        uint64_t k, uint64_t n)                                    \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m += LANES)                                                    \
            for (uint64_t j0 = 0; j0 < k; j0 += LANES)                                             \
            {                                                                                      \
                __m512i v[LANES];                                                                  \
                for (unsigned i = 0; i < LANES; i++)                                               \
                    v[i] = _mm512_loadu_si512((const void *)(in + (m + i) * k + j0));              \
                for (unsigned c = 1; c < LANES; c *= 2)                                            \
                    shuffle_level_u##W(v, LANES);                                                  \
                for (unsigned j = 0; j < LANES; j++)                                               \
                    _mm512_storeu_si512((void *)(out[j0 + j] + m), v[j]);                          \
            }                                                                                      \
    }

// The interleave's two shapes, mirroring the deinterleave's: the inverse
// network for k <= lanes, and the same square tiles (a transpose is its own
// inverse) for k a multiple of the lanes.
#define INTERLEAVE_K(W, LANES)                                                                     \
    static inline __attribute__((always_inline)) __m512i load_or_zero_u##W(const uint##W##_t *row, \
                                                                           uint64_t m)             \
    {                                                                                              \
        return row != NULL ? _mm512_loadu_si512((const void *)(row + m)) : _mm512_setzero_si512(); \
    }                                                                                              \
    static inline __attribute__((always_inline)) void interleave_small_u##W(                       \
        uint##W##_t *out, const uint##W##_t *const *in, const unsigned k, uint64_t n)              \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m += LANES)                                                    \
        {                                                                                          \
            __m512i v[LANES];                                                                      \
            for (unsigned j = 0; j < k; j++)                                                       \
                v[j] = load_or_zero_u##W(in[j], m);                                                \
            for (unsigned c = 1; c < k; c *= 2)                                                    \
                unshuffle_level_u##W(v, k);                                                        \
            for (unsigned i = 0; i < k; i++)                                                       \
                _mm512_storeu_si512((void *)(out + k * m + LANES * i), v[i]);                      \
        }                                                                                          \
    }                                                                                              \
    static void interleave_tiles_u##W(uint##W##_t *out, const uint##W##_t *const *in, uint64_t k,  \
                                      uint64_t n)                                                  \
    {                                                                                              \
        for (uint64_t m = 0; m < n; m += LANES)                                                    \
            for (uint64_t j0 = 0; j0 < k; j0 += LANES)                                             \
            {                                                                                      \
                __m512i v[LANES];                                                                  \
                for (unsigned j = 0; j < LANES; j++)                                               \
                    v[j] = load_or_zero_u##W(in[j0 + j], m);                                       \
                for (unsigned c = 1; c < LANES; c *= 2)                                            \
                    shuffle_level_u##W(v, LANES);                                                  \
                for (unsigned i = 0; i < LANES; i++)                                               \
                    _mm512_storeu_si512((void *)(out + (m + i) * k + j0), v[i]);                   \
            }                                                                                      \
    }

DEINTERLEAVE_SMALL(64, 8)
DEINTERLEAVE_SMALL(32, 16)
DEINTERLEAVE_TILES(64, 8)
DEINTERLEAVE_TILES(32, 16)
INTERLEAVE_K(64, 8)
INTERLEAVE_K(32, 16)

void vec_interleave_k_u64(uint64_t *out, const uint64_t *const *in, uint64_t k, uint64_t n)
{
    if (n % 8 != 0 || k == 1)
        interleave_k_scalar_u64(out, in, k, n);
    else if (k == 2)
        interleave_small_u64(out, in, 2, n);
    else if (k == 4)
        interleave_small_u64(out, in, 4, n);
    else if (k % 8 == 0)
        interleave_tiles_u64(out, in, k, n);
    else
        interleave_k_scalar_u64(out, in, k, n);
}

void vec_interleave_k_u32(uint32_t *out, const uint32_t *const *in, uint64_t k, uint64_t n)
{
    if (n % 16 != 0 || k == 1)
        interleave_k_scalar_u32(out, in, k, n);
    else if (k == 2)
        interleave_small_u32(out, in, 2, n);
    else if (k == 4)
        interleave_small_u32(out, in, 4, n);
    else if (k == 8)
        interleave_small_u32(out, in, 8, n);
    else if (k % 16 == 0)
        interleave_tiles_u32(out, in, k, n);
    else
        interleave_k_scalar_u32(out, in, k, n);
}

void vec_deinterleave_u64(uint64_t *const *out, const uint64_t *in, uint64_t k, uint64_t n)
{
    if (n % 8 != 0)
        deinterleave_scalar_u64(out, in, k, n);
    else if (k == 1)
        memcpy(out[0], in, n * sizeof(uint64_t));
    else if (k == 2)
        deinterleave_small_u64(out, in, 2, n);
    else if (k == 4)
        deinterleave_small_u64(out, in, 4, n);
    else if (k % 8 == 0)
        deinterleave_tiles_u64(out, in, k, n);
    else
        deinterleave_scalar_u64(out, in, k, n);
}

void vec_deinterleave_u32(uint32_t *const *out, const uint32_t *in, uint64_t k, uint64_t n)
{
    if (n % 16 != 0)
        deinterleave_scalar_u32(out, in, k, n);
    else if (k == 1)
        memcpy(out[0], in, n * sizeof(uint32_t));
    else if (k == 2)
        deinterleave_small_u32(out, in, 2, n);
    else if (k == 4)
        deinterleave_small_u32(out, in, 4, n);
    else if (k == 8)
        deinterleave_small_u32(out, in, 8, n);
    else if (k % 16 == 0)
        deinterleave_tiles_u32(out, in, k, n);
    else
        deinterleave_scalar_u32(out, in, k, n);
}

// Spreading, k <= lanes: each input vector gives k output vectors, output
// q holding input lanes q * lanes / k, ... at its lanes 0, k, 2k, ... and
// zeros elsewhere -- one masked permute each. k a multiple of the lanes: an
// input word gives one vector holding it at lane 0, then zero vectors.
#define SPREAD(W, LANES, MASK_T)                                                                   \
    void vec_spread_u##W(uint##W##_t *out, const uint##W##_t *in, uint64_t k, uint64_t n)          \
    {                                                                                              \
        if (k == 1)                                                                                \
        {                                                                                          \
            memcpy(out, in, n * sizeof(uint##W##_t));                                              \
            return;                                                                                \
        }                                                                                          \
        if (k % LANES == 0)                                                                        \
        {                                                                                          \
            const __m512i zero = _mm512_setzero_si512();                                           \
            for (uint64_t m = 0; m < n; m++)                                                       \
            {                                                                                      \
                _mm512_storeu_si512((void *)(out + k * m),                                         \
                                    _mm512_maskz_loadu_epi##W((MASK_T)1, in + m));                 \
                for (uint64_t p = LANES; p < k; p += LANES)                                        \
                    _mm512_storeu_si512((void *)(out + k * m + p), zero);                          \
            }                                                                                      \
            return;                                                                                \
        }                                                                                          \
        if (!is_power_of_two(k) || k > LANES || n % LANES != 0)                                    \
        {                                                                                          \
            spread_scalar_u##W(out, in, k, n);                                                     \
            return;                                                                                \
        }                                                                                          \
        MASK_T mask = 0;                                                                           \
        for (unsigned lane = 0; lane < LANES; lane += k)                                           \
            mask |= (MASK_T)(1u << lane);                                                          \
        __m512i idx[LANES];                                                                        \
        for (unsigned q = 0; q < k; q++)                                                           \
        {                                                                                          \
            uint##W##_t lanes[LANES];                                                              \
            for (unsigned lane = 0; lane < LANES; lane++)                                          \
                lanes[lane] = (uint##W##_t)(q * (LANES / k) + lane / k);                           \
            idx[q] = _mm512_loadu_si512((const void *)lanes);                                      \
        }                                                                                          \
        for (uint64_t m = 0; m < n; m += LANES)                                                    \
        {                                                                                          \
            const __m512i v = _mm512_loadu_si512((const void *)(in + m));                          \
            for (unsigned q = 0; q < k; q++)                                                       \
                _mm512_storeu_si512((void *)(out + k * m + LANES * q),                             \
                                    _mm512_maskz_permutexvar_epi##W(mask, idx[q], v));             \
        }                                                                                          \
    }

SPREAD(64, 8, __mmask8)
SPREAD(32, 16, __mmask16)

#else

void vec_deinterleave_u64(uint64_t *const *out, const uint64_t *in, uint64_t k, uint64_t n)
{
    deinterleave_scalar_u64(out, in, k, n);
}

void vec_deinterleave_u32(uint32_t *const *out, const uint32_t *in, uint64_t k, uint64_t n)
{
    deinterleave_scalar_u32(out, in, k, n);
}

void vec_interleave_k_u64(uint64_t *out, const uint64_t *const *in, uint64_t k, uint64_t n)
{
    interleave_k_scalar_u64(out, in, k, n);
}

void vec_interleave_k_u32(uint32_t *out, const uint32_t *const *in, uint64_t k, uint64_t n)
{
    interleave_k_scalar_u32(out, in, k, n);
}

void vec_spread_u64(uint64_t *out, const uint64_t *in, uint64_t k, uint64_t n)
{
    spread_scalar_u64(out, in, k, n);
}

void vec_spread_u32(uint32_t *out, const uint32_t *in, uint64_t k, uint64_t n)
{
    spread_scalar_u32(out, in, k, n);
}

#endif
