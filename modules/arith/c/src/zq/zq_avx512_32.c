// SPDX-License-Identifier: Apache-2.0
/**
 * @file zq_avx512_32.c
 * @brief AVX-512 element-wise kernels for primes q < 2^32.
 *
 * At this width a full 64x64->64 `mullo` holds the product exactly, so
 * reduction is a single generic Barrett step using the precomputed
 * (barr_lo, prod_right_shift) pair from the ::zq_ctx. Reduction keeps values
 * lazily in [0, 2q) inside a kernel and emits [0, q).
 */
#include <stddef.h>

#include <arith/config.h>
#include <arith/zq.h>

#include "zq_backends.h"

#if VFHE_MP_SIMD

#include <immintrin.h>

/** Approximate high 64 bits of a 64x64 product (error <= 1, corrected later). */
static inline __m512i mulhi_approx_64(__m512i x, __m512i y)
{
    __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
    __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
    __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
    __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);
    __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);
    __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);

    __m512i sum_lo = _mm512_and_si512(z_lo_hi, lo_mask);
    __m512i sum_mid = _mm512_srli_epi64(z_lo_hi, 32);
    __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
    __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
    __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
    return _mm512_add_epi64(sum_hi, sum_mid2_hi);
}

/** Barrett-reduce a full 64-bit product vector to [0, q). */
static inline __m512i reduce_prod_32(__m512i prod, __m512i barr_lo_vec, __m512i q_vec,
                                     uint64_t prod_right_shift)
{
    __m512i c1 = _mm512_srli_epi64(prod, (unsigned int)prod_right_shift);
    __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
    __m512i res = _mm512_sub_epi64(prod, _mm512_mullo_epi64(q_hat, q_vec));
    res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
    res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
    return res;
}

static void v_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i prod = _mm512_mullo_epi64(_mm512_loadu_si512(&av[i]), _mm512_loadu_si512(&bv[i]));
        _mm512_storeu_si512(&outv[i],
                            reduce_prod_32(prod, barr_lo_vec, q_vec, zq->prod_right_shift));
    }
}

static void v_mul_addto(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                        const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i prod = _mm512_mullo_epi64(_mm512_loadu_si512(&av[i]), _mm512_loadu_si512(&bv[i]));
        __m512i res = reduce_prod_32(prod, barr_lo_vec, q_vec, zq->prod_right_shift);
        // Product and accumulator both in [0, q): one conditional subtract.
        __m512i acc = _mm512_add_epi64(_mm512_loadu_si512(&outv[i]), res);
        acc = _mm512_min_epu64(acc, _mm512_sub_epi64(acc, q_vec));
        _mm512_storeu_si512(&outv[i], acc);
    }
}

static void v_mul_subto(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                        const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i prod = _mm512_mullo_epi64(_mm512_loadu_si512(&av[i]), _mm512_loadu_si512(&bv[i]));
        __m512i res = reduce_prod_32(prod, barr_lo_vec, q_vec, zq->prod_right_shift);
        __m512i diff = _mm512_sub_epi64(_mm512_loadu_si512(&outv[i]), res);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        diff = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
        _mm512_storeu_si512(&outv[i], diff);
    }
}

static void v_scale(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i s_vec = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i prod = _mm512_mullo_epi64(_mm512_loadu_si512(&av[i]), s_vec);
        _mm512_storeu_si512(&outv[i],
                            reduce_prod_32(prod, barr_lo_vec, q_vec, zq->prod_right_shift));
    }
}

static void v_scale_addto(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n,
                          const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i s_vec = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i prod = _mm512_mullo_epi64(_mm512_loadu_si512(&av[i]), s_vec);
        __m512i res = reduce_prod_32(prod, barr_lo_vec, q_vec, zq->prod_right_shift);
        __m512i sum = _mm512_add_epi64(_mm512_loadu_si512(&outv[i]), res);
        sum = _mm512_min_epu64(sum, _mm512_sub_epi64(sum, q_vec));
        _mm512_storeu_si512(&outv[i], sum);
    }
}

static void v_add(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(_mm512_loadu_si512(&av[i]), _mm512_loadu_si512(&bv[i]));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

static void v_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(_mm512_loadu_si512(&av[i]), _mm512_loadu_si512(&bv[i]));
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        diff = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
        _mm512_storeu_si512(&outv[i], diff);
    }
}

static void v_negate(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i zero = _mm512_setzero_si512();

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = _mm512_loadu_si512(&av[i]);
        __m512i res = _mm512_sub_epi64(q_vec, v);
        __mmask8 mask = _mm512_cmpeq_epu64_mask(v, zero);
        res = _mm512_mask_blend_epi64(mask, res, zero);
        _mm512_storeu_si512(&outv[i], res);
    }
}

static void v_add_scalar(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i s_vec = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(_mm512_loadu_si512(&av[i]), s_vec);
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

static void v_sub_scalar(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i s_vec = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(_mm512_loadu_si512(&av[i]), s_vec);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        diff = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
        _mm512_storeu_si512(&outv[i], diff);
    }
}

static void v_reduce(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = _mm512_loadu_si512(&av[i]);
        _mm512_storeu_si512(&outv[i], reduce_prod_32(v, barr_lo_vec, q_vec, zq->prod_right_shift));
    }
}

static void v_reduce_signed(uint64_t *out, const int64_t *a, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = _mm512_loadu_si512(&av[i]);
        __mmask8 is_negative = _mm512_movepi64_mask(v);
        __m512i v_abs = _mm512_mask_sub_epi64(v, is_negative, _mm512_setzero_si512(), v);

        __m512i res = reduce_prod_32(v_abs, barr_lo_vec, q_vec, zq->prod_right_shift);

        __mmask8 is_zero = _mm512_cmpeq_epu64_mask(res, _mm512_setzero_si512());
        __mmask8 adjust_mask = is_negative & ~is_zero;
        __m512i v_final = _mm512_mask_sub_epi64(res, adjust_mask, q_vec, res);

        _mm512_storeu_si512(&outv[i], v_final);
    }
}

static void v_reduce_wide(uint64_t *out, const uint64_t *hi, const uint64_t *lo, uint64_t n,
                          const zq_ctx *zq)
{
    const __m512i *hiv = (const __m512i *)hi;
    const __m512i *lov = (const __m512i *)lo;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i w64 = _mm512_set1_epi64((long long)zq->w64);
    const __m512i barr_lo_vec = _mm512_set1_epi64((long long)zq->barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        // (hi * 2^64 + lo) mod q: reduce hi, fold with 2^64 mod q, reduce again.
        __m512i val_hi = _mm512_loadu_si512(&hiv[i]);
        __m512i val_lo = _mm512_loadu_si512(&lov[i]);

        __m512i hi_red = reduce_prod_32(val_hi, barr_lo_vec, q_vec, zq->prod_right_shift);
        __m512i sum = _mm512_add_epi64(_mm512_mullo_epi64(hi_red, w64), val_lo);
        _mm512_storeu_si512(&outv[i],
                            reduce_prod_32(sum, barr_lo_vec, q_vec, zq->prod_right_shift));
    }
}

const zq_ops zq_ops_avx512_32 = {
    .mul = v_mul,
    .mul_addto = v_mul_addto,
    .mul_subto = v_mul_subto,
    .scale = v_scale,
    .scale_addto = v_scale_addto,
    .add = v_add,
    .sub = v_sub,
    .negate = v_negate,
    .add_scalar = v_add_scalar,
    .sub_scalar = v_sub_scalar,
    .reduce = v_reduce,
    .reduce_signed = v_reduce_signed,
    .reduce_wide = v_reduce_wide,
};

#endif // VFHE_MP_SIMD
