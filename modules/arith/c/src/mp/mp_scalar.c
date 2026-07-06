// SPDX-License-Identifier: Apache-2.0
/**
 * @file mp_scalar.c
 * @brief Big-integer scalars in base-2^52 digit columns (see arith/mp.h).
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <arith/mp.h>
#include <base.h>

#include "mp_internal.h"

int mp_vector_size(void)
{
#if VFHE_MP_SIMD
    return 8;
#else
    return 1;
#endif
}

mp_vector_t *mp_vector_load(uint64_t in)
{
#if VFHE_MP_SIMD
    __m512i *res = (__m512i *)safe_aligned_malloc(sizeof(__m512i));
    *res = _mm512_set1_epi64((long long)in);
    return res;
#else
    uint64_t *res = (uint64_t *)safe_malloc(sizeof(uint64_t));
    *res = in;
    return res;
#endif
}

uint64_t mp_bit_slice_52(const uint64_t *array, uint64_t start)
{
    uint64_t res = 0;
    const uint64_t word_idx = start / 32, word_shift = 52 - 32 + (start % 32),
                   mod_mask = (1ULL << 52) - 1;
    res |= array[word_idx] << word_shift;
    if (word_shift < 32)
    {
        res |= array[word_idx + 1] >> (32 - word_shift);
    }
    else
    {
        res |= array[word_idx + 1] << (word_shift - 32);
        res |= array[word_idx + 2] >> (64 - word_shift);
    }
    res &= mod_mask;
    return res;
}

mp_scalar_t mp_scalar_load(const uint64_t *in, uint64_t d)
{
    mp_scalar_t out = (mp_scalar_t)safe_malloc(sizeof(*out));
    out->d = d;
#if VFHE_MP_SIMD
    out->digits = (__m512i *)safe_aligned_malloc(d * sizeof(__m512i));
    for (uint64_t i = 0; i < d; i++)
        out->digits[i] = _mm512_set1_epi64((long long)in[i]);
#else
    out->digits = (uint64_t *)safe_malloc(d * sizeof(uint64_t));
    for (uint64_t i = 0; i < d; i++)
        out->digits[i] = in[i];
#endif
    return out;
}

void mp_scalar_free(mp_scalar_t s)
{
    if (s == NULL)
        return;
    free(s->digits);
    free(s);
}

uint64_t mp_scalar_digit_count(const mp_scalar_t s) { return s->d; }

uint64_t mp_scalar_digit(const mp_scalar_t s, uint64_t i)
{
#if VFHE_MP_SIMD
    // Every lane carries the same digit; read lane 0.
    return ((const uint64_t *)&s->digits[i])[0];
#else
    return s->digits[i];
#endif
}

void mp_scalar_scale(mp_scalar_t out, const mp_scalar_t in, const mp_vector_t *m)
{
    assert(in != out);
    assert(out->d >= in->d + 1);
#if VFHE_MP_SIMD
    __m512i mod_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i zero = _mm512_setzero_si512();
    out->digits[0] = zero;
    for (uint64_t j = 0; j < in->d; j++)
    {
        out->digits[j + 1] = _mm512_madd52hi_epu64(zero, in->digits[j], *m);
        out->digits[j] = _mm512_madd52lo_epu64(out->digits[j], in->digits[j], *m);
        out->digits[j + 1] =
            _mm512_add_epi64(out->digits[j + 1], _mm512_srli_epi64(out->digits[j], 52));
        out->digits[j] = _mm512_and_epi64(out->digits[j], mod_mask);
    }
#else
    out->digits[0] = 0;
    for (uint64_t j = 0; j < in->d; j++)
    {
        out->digits[j + 1] = madd52hi(0, in->digits[j], *m);
        out->digits[j] = madd52lo(out->digits[j], in->digits[j], *m);
        out->digits[j + 1] += out->digits[j] >> 52;
        out->digits[j] &= 0x000FFFFFFFFFFFFFULL;
    }
#endif
}

void mp_scalar_sub(mp_scalar_t out, const mp_scalar_t a, const mp_scalar_t b)
{
    assert(b != out);
#if VFHE_MP_SIMD
    __m512i mod_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i neg_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i one = _mm512_set1_epi64(1), tmp;
    for (uint64_t j = 0; j < out->d; j++)
    {
        // out = a + (~b + 1): radix-complement subtraction, digits then carries.
        if (j < b->d)
        {
            tmp = _mm512_xor_epi64(b->digits[j], neg_mask);
        }
        else
        {
            tmp = neg_mask;
        }
        if (j == 0)
            tmp = _mm512_add_epi64(tmp, one);
        out->digits[j] = _mm512_add_epi64(tmp, a->digits[j]);
    }
    for (uint64_t j = 0; j < out->d - 1; j++)
    {
        out->digits[j + 1] =
            _mm512_add_epi64(out->digits[j + 1], _mm512_srli_epi64(out->digits[j], 52));
        out->digits[j] = _mm512_and_epi64(out->digits[j], mod_mask);
    }
    out->digits[out->d - 1] = _mm512_and_epi64(out->digits[out->d - 1], mod_mask);
#else
    for (uint64_t j = 0; j < out->d; j++)
    {
        uint64_t tmp;
        if (j < b->d)
        {
            tmp = b->digits[j] ^ 0x000FFFFFFFFFFFFFULL;
        }
        else
        {
            tmp = 0x000FFFFFFFFFFFFFULL;
        }
        if (j == 0)
            tmp += 1;
        out->digits[j] = tmp + a->digits[j];
    }
    for (uint64_t j = 0; j < out->d - 1; j++)
    {
        out->digits[j + 1] += out->digits[j] >> 52;
        out->digits[j] &= 0x000FFFFFFFFFFFFFULL;
    }
    out->digits[out->d - 1] &= 0x000FFFFFFFFFFFFFULL;
#endif
}

void mp_scalar_print(const mp_scalar_t x)
{
#if VFHE_MP_SIMD
    const uint64_t *v_int = (const uint64_t *)x->digits;
    printf("0x");
    for (int64_t i = (int64_t)x->d - 1; i >= 0; i--)
    {
        printf("%013llx", (unsigned long long)v_int[i * 8]);
    }
    printf("\n");
#else
    printf("0x");
    for (int64_t i = (int64_t)x->d - 1; i >= 0; i--)
    {
        printf("%013llx", (unsigned long long)x->digits[i]);
    }
    printf("\n");
#endif
}
