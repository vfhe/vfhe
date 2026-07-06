// SPDX-License-Identifier: Apache-2.0
/**
 * @file mp_poly.c
 * @brief Big-integer polynomial arithmetic in base-2^52 digits (arith/mp.h).
 */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <arith/mp.h>
#include <base.h>
#include <rng.h>

#include "mp_internal.h"

mp_polynomial_t mp_polynomial_new(uint64_t N, uint64_t d)
{
    mp_polynomial_t res = (mp_polynomial_t)safe_malloc(sizeof(*res));
    res->coeffs = (uint64_t **)safe_malloc(sizeof(*res->coeffs) * d);
    res->d = d;
    res->N = N;
    for (uint64_t i = 0; i < d; i++)
    {
        res->coeffs[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * N);
        memset(res->coeffs[i], 0, sizeof(uint64_t) * N);
    }
    return res;
}

void mp_polynomial_free(mp_polynomial_t p)
{
    for (uint64_t i = 0; i < p->d; i++)
    {
        free(p->coeffs[i]);
    }
    free(p->coeffs);
    free(p);
}

void mp_polynomial_zero(mp_polynomial_t poly)
{
    for (uint64_t j = 0; j < poly->d; j++)
    {
        memset(poly->coeffs[j], 0, sizeof(uint64_t) * poly->N);
    }
}

void mp_polynomial_randomize(mp_polynomial_t poly)
{
#if VFHE_MP_SIMD
    __m512i and_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    for (uint64_t j = 0; j < poly->d; j++)
    {
        rng_random_bytes(sizeof(uint64_t) * poly->N, (uint8_t *)poly->coeffs[j]);
        __m512i *polyv = (__m512i *)poly->coeffs[j];
        for (uint64_t i = 0; i < poly->N / 8; i++)
        {
            polyv[i] = _mm512_and_epi64(polyv[i], and_mask);
        }
    }
#else
    for (uint64_t j = 0; j < poly->d; j++)
    {
        rng_random_bytes(sizeof(uint64_t) * poly->N, (uint8_t *)poly->coeffs[j]);
        for (uint64_t i = 0; i < poly->N; i++)
        {
            poly->coeffs[j][i] &= 0x000FFFFFFFFFFFFFULL;
        }
    }
#endif
}

void mp_polynomial_mul_by_xai(mp_polynomial_t out, const mp_polynomial_t in, uint64_t a)
{
    const int64_t N = (int64_t)out->N;
    assert(out->d == in->d);
    assert(out->N == in->N);
    a &= (uint64_t)((N << 1) - 1); // a mod 2N
    if (!a)
        return;
    int64_t sa = (int64_t)a;
    for (uint64_t j = 0; j < in->d; j++)
    {
        if (sa < N)
        {
            for (int64_t i = 0; i < sa; i++)
                out->coeffs[j][i] = -in->coeffs[j][i - sa + N];
            for (int64_t i = sa; i < N; i++)
                out->coeffs[j][i] = in->coeffs[j][i - sa];
        }
        else
        {
            for (int64_t i = 0; i < sa - N; i++)
                out->coeffs[j][i] = in->coeffs[j][i - sa + 2 * N];
            for (int64_t i = sa - N; i < N; i++)
                out->coeffs[j][i] = -in->coeffs[j][i - sa + N];
        }
    }
}

void mp_polynomial_negate(mp_polynomial_t out, const mp_polynomial_t in)
{
#if VFHE_MP_SIMD
    // Radix complement: flip every digit, +1 on the lowest digit row.
    __m512i neg_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i neg_mask2 = _mm512_set1_epi64(1);
    for (uint64_t j = 0; j < in->d; j++)
    {
        __m512i *inv = (__m512i *)in->coeffs[j];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < in->N / 8; i++)
        {
            outv[i] = _mm512_xor_epi64(inv[i], neg_mask);
            if (j == 0)
                outv[i] = _mm512_add_epi64(outv[i], neg_mask2);
        }
    }
#else
    for (uint64_t j = 0; j < in->d; j++)
    {
        for (uint64_t i = 0; i < in->N; i++)
        {
            out->coeffs[j][i] = in->coeffs[j][i] ^ 0x000FFFFFFFFFFFFFULL;
            if (j == 0)
                out->coeffs[j][i] += 1;
        }
    }
#endif
}

void mp_polynomial_add(mp_polynomial_t out, const mp_polynomial_t a, const mp_polynomial_t b)
{
#if VFHE_MP_SIMD
    for (uint64_t j = 0; j < a->d; j++)
    {
        __m512i *av = (__m512i *)a->coeffs[j];
        __m512i *bv = (__m512i *)b->coeffs[j];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < a->N / 8; i++)
        {
            outv[i] = _mm512_add_epi64(av[i], bv[i]);
        }
    }
#else
    for (uint64_t j = 0; j < a->d; j++)
    {
        for (uint64_t i = 0; i < a->N; i++)
        {
            out->coeffs[j][i] = a->coeffs[j][i] + b->coeffs[j][i];
        }
    }
#endif
}

void mp_polynomial_drop_digits(mp_polynomial_t p, uint64_t num_digits)
{
    for (uint64_t i = 1; i <= num_digits; i++)
    {
        free(p->coeffs[p->d - i]);
    }
    p->d -= num_digits;
}

void mp_polynomial_scale_fixedpoint(mp_polynomial_t out, const mp_polynomial_t in, uint64_t scale)
{
    // Input must be carry-free (all digits < 2^52).
#if VFHE_MP_SIMD
    __m512i zero = _mm512_setzero_si512();
    __m512i scalev = _mm512_set1_epi64((long long)scale);
    for (uint64_t j = 0; j < in->d; j++)
    {
        __m512i *inv = (__m512i *)in->coeffs[j];
        __m512i *outv_prev = (__m512i *)out->coeffs[j - 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < in->N / 8; i++)
        {
            if (j > 0)
                outv_prev[i] = _mm512_madd52hi_epu64(outv_prev[i], inv[i], scalev);
            outv[i] = _mm512_madd52lo_epu64(zero, inv[i], scalev);
        }
    }
#else
    for (uint64_t j = 0; j < in->d; j++)
    {
        for (uint64_t i = 0; i < in->N; i++)
        {
            if (j > 0)
                out->coeffs[j - 1][i] = madd52hi(out->coeffs[j - 1][i], in->coeffs[j][i], scale);
            out->coeffs[j][i] = madd52lo(0, in->coeffs[j][i], scale);
        }
    }
#endif
}

void mp_polynomial_scale_fixedpoint_addto(mp_polynomial_t out, const mp_polynomial_t in,
                                          uint64_t scale)
{
#if VFHE_MP_SIMD
    __m512i scalev = _mm512_set1_epi64((long long)scale);
    for (uint64_t j = 0; j < in->d; j++)
    {
        __m512i *inv = (__m512i *)in->coeffs[j];
        __m512i *outv_prev = (__m512i *)out->coeffs[j - 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < in->N / 8; i++)
        {
            if (j > 0)
                outv_prev[i] = _mm512_madd52hi_epu64(outv_prev[i], inv[i], scalev);
            outv[i] = _mm512_madd52lo_epu64(outv[i], inv[i], scalev);
        }
    }
#else
    for (uint64_t j = 0; j < in->d; j++)
    {
        for (uint64_t i = 0; i < in->N; i++)
        {
            if (j > 0)
                out->coeffs[j - 1][i] = madd52hi(out->coeffs[j - 1][i], in->coeffs[j][i], scale);
            out->coeffs[j][i] = madd52lo(out->coeffs[j][i], in->coeffs[j][i], scale);
        }
    }
#endif
}

void mp_polynomial_scale_limb_by_scalar(mp_polynomial_t out, const mp_polynomial_t in,
                                        const mp_vector_t *scale)
{
#if VFHE_MP_SIMD
    __m512i zero = _mm512_setzero_si512();
    for (uint64_t j = 0; j < in->d; j++)
    {
        __m512i *inv_sp = (__m512i *)in->coeffs[0];
        __m512i *outv_prev = (__m512i *)out->coeffs[j - 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < in->N / 8; i++)
        {
            if (j > 0)
                outv_prev[i] = _mm512_madd52hi_epu64(outv_prev[i], scale[j], inv_sp[i]);
            outv[i] = _mm512_madd52lo_epu64(zero, scale[j], inv_sp[i]);
        }
    }
#else
    for (uint64_t j = 0; j < in->d; j++)
    {
        for (uint64_t i = 0; i < in->N; i++)
        {
            if (j > 0)
                out->coeffs[j - 1][i] = madd52hi(out->coeffs[j - 1][i], scale[j], in->coeffs[0][i]);
            out->coeffs[j][i] = madd52lo(0, scale[j], in->coeffs[0][i]);
        }
    }
#endif
}

void mp_polynomial_scale_int_by_scalar(mp_polynomial_t out, const uint64_t *in,
                                       const mp_scalar_t scale)
{
#if VFHE_MP_SIMD
    __m512i zero = _mm512_setzero_si512();
    for (uint64_t j = 0; j < out->d - 1; j++)
    {
        __m512i *inv_sp = (__m512i *)in;
        __m512i *outv_next = (__m512i *)out->coeffs[j + 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < out->N / 8; i++)
        {
            outv_next[i] = _mm512_madd52hi_epu64(zero, scale->digits[j], inv_sp[i]);
            if (j == 0)
            {
                outv[i] = _mm512_madd52lo_epu64(zero, scale->digits[j], inv_sp[i]);
            }
            else
            {
                outv[i] = _mm512_madd52lo_epu64(outv[i], scale->digits[j], inv_sp[i]);
            }
        }
    }
#else
    for (uint64_t j = 0; j < out->d - 1; j++)
    {
        for (uint64_t i = 0; i < out->N; i++)
        {
            out->coeffs[j + 1][i] = madd52hi(0, scale->digits[j], in[i]);
            if (j == 0)
            {
                out->coeffs[j][i] = madd52lo(0, scale->digits[j], in[i]);
            }
            else
            {
                out->coeffs[j][i] = madd52lo(out->coeffs[j][i], scale->digits[j], in[i]);
            }
        }
    }
#endif
}

void mp_polynomial_scale_int_by_scalar_addto(mp_polynomial_t out, const uint64_t *in,
                                             const mp_scalar_t scale)
{
#if VFHE_MP_SIMD
    for (uint64_t j = 0; j < out->d - 1; j++)
    {
        __m512i *inv_sp = (__m512i *)in;
        __m512i *outv_next = (__m512i *)out->coeffs[j + 1];
        __m512i *outv = (__m512i *)out->coeffs[j];
        for (uint64_t i = 0; i < out->N / 8; i++)
        {
            outv_next[i] = _mm512_madd52hi_epu64(outv_next[i], scale->digits[j], inv_sp[i]);
            outv[i] = _mm512_madd52lo_epu64(outv[i], scale->digits[j], inv_sp[i]);
        }
    }
#else
    for (uint64_t j = 0; j < out->d - 1; j++)
    {
        for (uint64_t i = 0; i < out->N; i++)
        {
            out->coeffs[j + 1][i] = madd52hi(out->coeffs[j + 1][i], scale->digits[j], in[i]);
            out->coeffs[j][i] = madd52lo(out->coeffs[j][i], scale->digits[j], in[i]);
        }
    }
#endif
}

void mp_polynomial_propagate_carry(mp_polynomial_t p)
{
#if VFHE_MP_SIMD
    __m512i mod_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    for (uint64_t j = 0; j < p->d - 1; j++)
    {
        __m512i *pv = (__m512i *)p->coeffs[j];
        __m512i *pv_next = (__m512i *)p->coeffs[j + 1];
        for (uint64_t i = 0; i < p->N / 8; i++)
        {
            pv_next[i] = _mm512_add_epi64(pv_next[i], _mm512_srli_epi64(pv[i], 52));
            pv[i] = _mm512_and_epi64(pv[i], mod_mask);
        }
    }
    __m512i *pv = (__m512i *)p->coeffs[p->d - 1];
    for (uint64_t i = 0; i < p->N / 8; i++)
    {
        pv[i] = _mm512_and_epi64(pv[i], mod_mask);
    }
#else
    for (uint64_t j = 0; j < p->d - 1; j++)
    {
        for (uint64_t i = 0; i < p->N; i++)
        {
            p->coeffs[j + 1][i] += p->coeffs[j][i] >> 52;
            p->coeffs[j][i] &= 0x000FFFFFFFFFFFFFULL;
        }
    }
    for (uint64_t i = 0; i < p->N; i++)
    {
        p->coeffs[p->d - 1][i] &= 0x000FFFFFFFFFFFFFULL;
    }
#endif
}

void mp_polynomial_mul_addto_sparse(mp_polynomial_t out, const mp_polynomial_t a, const uint64_t *b,
                                    uint64_t size)
{
    const uint64_t N = a->N, N_mask = N - 1;
    // Precompute -a once; each monomial then adds either a or -a shifted.
    mp_polynomial_t a_neg = mp_polynomial_new(N, a->d);
    mp_polynomial_negate(a_neg, a);
    mp_polynomial_propagate_carry(a_neg);
    for (uint64_t i = 0; i < size; i++)
    {
        for (uint64_t j = 0; j < N; j++)
        {
            const uint64_t pos = (b[i] + j) & N_mask;
            const uint64_t sign = (((b[i] & N_mask) + j) >= N) ^ (b[i] >> 63);
            for (uint64_t k = 0; k < a->d; k++)
            {
                if (sign)
                    out->coeffs[k][j] += a_neg->coeffs[k][pos];
                else
                    out->coeffs[k][j] += a->coeffs[k][pos];
            }
        }
    }
    mp_polynomial_propagate_carry(out);
    mp_polynomial_free(a_neg);
}

#if VFHE_MP_SIMD
/* Stack-allocate an mp_scalar of `size` digit columns (SIMD path only). */
#define MPScalar_stack_alloc_(var, size, suffix)                                                   \
    struct mp_scalar __p##suffix;                                                                  \
    __p##suffix.d = (size);                                                                        \
    __m512i __tmp##suffix[(size)];                                                                 \
    __p##suffix.digits = __tmp##suffix;                                                            \
    var = &__p##suffix;

#define MPScalar_stack_alloc__(var, size, suffix) MPScalar_stack_alloc_(var, size, suffix)
#define MPScalar_stack_alloc(var, size) MPScalar_stack_alloc__(var, size, __LINE__)
#endif

void mp_polynomial_mod_reduce(mp_polynomial_t out, const mp_scalar_t q, const mp_vector_t *m,
                              uint64_t k)
{
#if VFHE_MP_SIMD
    __m512i zero = _mm512_setzero_si512();
    mp_scalar_t tmp1, tmp2;
    MPScalar_stack_alloc(tmp1, out->d + 1);
    MPScalar_stack_alloc(tmp2, out->d);

    for (uint64_t i = 0; i < out->N / 8; i++)
    {
        // Barrett per coefficient column: quo = (x * m) >> k; x -= quo * q;
        // final conditional subtract selected by the top digit of x - q.
        for (uint64_t j = 0; j < out->d; j++)
            tmp2->digits[j] = ((__m512i *)(out->coeffs[j]))[i];
        tmp1->d = out->d + 1;
        mp_scalar_scale(tmp1, tmp2, m);
        const uint64_t bit_length = 52 * out->d - k;
        __m512i quo = tmp1->digits[out->d] << bit_length;
        quo |= tmp1->digits[out->d - 1] >> (52 - bit_length);
        tmp1->d = out->d;
        mp_scalar_scale(tmp1, q, &quo);
        mp_scalar_sub(tmp2, tmp2, tmp1);
        mp_scalar_sub(tmp1, tmp2, q);
        __mmask8 select = _mm512_cmpeq_epi64_mask(tmp1->digits[out->d - 1], zero);
        for (uint64_t j = 0; j < out->d; j++)
        {
            ((__m512i *)(out->coeffs[j]))[i] =
                _mm512_mask_blend_epi64(select, tmp2->digits[j], tmp1->digits[j]);
        }
    }
#else
    mp_scalar_t tmp1, tmp2;
    uint64_t tmp1_digits[out->d + 1];
    uint64_t tmp2_digits[out->d];
    struct mp_scalar s_tmp1 = {tmp1_digits, out->d + 1};
    struct mp_scalar s_tmp2 = {tmp2_digits, out->d};
    tmp1 = &s_tmp1;
    tmp2 = &s_tmp2;

    for (uint64_t i = 0; i < out->N; i++)
    {
        for (uint64_t j = 0; j < out->d; j++)
            tmp2->digits[j] = out->coeffs[j][i];
        tmp1->d = out->d + 1;
        mp_scalar_scale(tmp1, tmp2, m);
        const uint64_t bit_length = 52 * out->d - k;
        uint64_t quo = tmp1->digits[out->d] << bit_length;
        quo |= tmp1->digits[out->d - 1] >> (52 - bit_length);
        tmp1->d = out->d;
        mp_scalar_scale(tmp1, q, &quo);
        mp_scalar_sub(tmp2, tmp2, tmp1);
        mp_scalar_sub(tmp1, tmp2, q);
        int select = (tmp1->digits[out->d - 1] == 0);
        for (uint64_t j = 0; j < out->d; j++)
        {
            out->coeffs[j][i] = select ? tmp1->digits[j] : tmp2->digits[j];
        }
    }
#endif
}

// Process-global digit schedule for 1/p, reused across mod-switch calls.
static mp_vector_t *g_delta = NULL;
static uint64_t g_p = 0, g_d = 0;

void mp_mod_switch_delta_setup(uint64_t d, uint64_t p)
{
    if (p == g_p && g_d >= d)
    {
        return;
    }
    free(g_delta);
#if VFHE_MP_SIMD
    g_delta = (__m512i *)safe_aligned_malloc(sizeof(__m512i) * d);
#else
    g_delta = (uint64_t *)safe_malloc(sizeof(uint64_t) * d);
#endif
    assert(p < (1ULL << 32));
    // Long division of 1 by p in base 2^32, then re-sliced into 52-bit digits.
    uint64_t delta32[2 * d];
    uint64_t rem = 1;
    memset(delta32, 0, sizeof(uint64_t) * 2 * d);
    for (uint64_t i = 0; i < 2 * d; i++)
    {
        delta32[i] = (rem << 32) / p;
        rem = (rem << 32) % p;
    }
    for (uint64_t i = 0; i < d; i++)
    {
#if VFHE_MP_SIMD
        g_delta[i] = _mm512_set1_epi64((long long)mp_bit_slice_52(delta32, 52 * i));
#else
        g_delta[i] = mp_bit_slice_52(delta32, 52 * i);
#endif
    }
    g_p = p;
    g_d = d;
}
