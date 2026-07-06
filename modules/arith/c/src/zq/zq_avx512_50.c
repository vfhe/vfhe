// SPDX-License-Identifier: Apache-2.0
/**
 * @file zq_avx512_50.c
 * @brief AVX-512 IFMA element-wise kernels for primes q < 2^50.
 *
 * Uses the native 52-bit fused multiply-add units (`madd52lo/hi`), the same
 * reduction structure as Intel HEXL: products are produced as 52-bit hi/lo
 * halves and Barrett-reduced with the (ifma_barr_lo, ifma_shift) pair from
 * the ::zq_ctx. Values stay lazily in [0, 2q) inside a kernel and are emitted
 * in [0, q).
 */
#include <stddef.h>

#include <arith/config.h>
#include <arith/zq.h>

#include "zq_backends.h"

#if VFHE_MP_SIMD

#include <immintrin.h>

/** Barrett-reduce a 52-bit-split product (hi, lo) to [0, q). */
static inline __m512i reduce_prod_50(__m512i v_prod_hi, __m512i v_prod_lo, const zq_ctx *zq)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)zq->q);
    const __m512i m_ifma = _mm512_set1_epi64((long long)zq->ifma_barr_lo);

    const uint64_t shift = zq->ifma_shift;
    const __m128i v_shift = _mm_cvtsi64_si128((long long)shift);
    const __m128i v_shift_rev = _mm_cvtsi64_si128((long long)(52 - shift));

    __m512i c1 = _mm512_or_si512(_mm512_srl_epi64(v_prod_lo, v_shift),
                                 _mm512_sll_epi64(v_prod_hi, v_shift_rev));

    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i v_result = _mm512_madd52lo_epu64(v_prod_lo, q_hat, neg_q_vec);
    v_result = _mm512_and_epi64(v_result, low52b_mask);

    __m512i r2 = _mm512_sub_epi64(v_result, q_vec);
    return _mm512_min_epu64(v_result, r2);
}

/** Barrett-reduce a full 64-bit value to [0, q). */
static inline __m512i reduce_64b_50(__m512i v, const zq_ctx *zq)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)zq->q);
    const __m512i m_ifma = _mm512_set1_epi64((long long)zq->ifma_barr_lo);

    const __m128i v_shift = _mm_cvtsi64_si128((long long)zq->ifma_shift);

    __m512i c1 = _mm512_srl_epi64(v, v_shift);

    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i v_result = _mm512_madd52lo_epu64(v, q_hat, neg_q_vec);
    v_result = _mm512_and_epi64(v_result, low52b_mask);

    __m512i r2 = _mm512_sub_epi64(v_result, q_vec);
    return _mm512_min_epu64(v_result, r2);
}

/** Reduce a two-digit (base-2^52) value (hi, lo), folding hi with 2^52 mod q. */
static inline __m512i reduce_digits_50(__m512i v_hi, __m512i v_lo, const zq_ctx *zq)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)zq->q);
    const __m512i m_ifma = _mm512_set1_epi64((long long)zq->ifma_barr_lo);
    const __m512i v_w1 = _mm512_set1_epi64((long long)zq->w52_1);

    __m512i v2_hi = _mm512_madd52hi_epu64(zero, v_hi, v_w1);
    __m512i v2_lo = _mm512_madd52lo_epu64(v_lo, v_hi, v_w1);
    __m512i v3_hi = _mm512_madd52hi_epu64(zero, v2_hi, v_w1);
    __m512i v3_lo = _mm512_madd52lo_epu64(v2_lo, v2_hi, v_w1);

    v3_hi = _mm512_add_epi64(v3_hi, _mm512_srli_epi64(v3_lo, 52));
    v3_lo = _mm512_and_epi64(v3_lo, low52b_mask);

    const uint64_t shift = zq->ifma_shift;
    const __m128i v_shift = _mm_cvtsi64_si128((long long)shift);
    const __m128i v_shift_rev = _mm_cvtsi64_si128((long long)(52 - shift));

    __m512i c1 =
        _mm512_or_si512(_mm512_srl_epi64(v3_lo, v_shift), _mm512_sll_epi64(v3_hi, v_shift_rev));

    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i v_result = _mm512_madd52lo_epu64(v3_lo, q_hat, neg_q_vec);
    v_result = _mm512_and_epi64(v_result, low52b_mask);

    __m512i r2 = _mm512_sub_epi64(v_result, q_vec);
    __m512i res = _mm512_min_epu64(v_result, r2);
    __m512i r3 = _mm512_sub_epi64(res, q_vec);
    return _mm512_min_epu64(res, r3);
}

static void v_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = av[i];
        __m512i v_op2 = bv[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        outv[i] = reduce_prod_50(v_prod_hi, v_prod_lo, zq);
    }
}

static void v_mul_addto(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                        const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = av[i];
        __m512i v_op2 = bv[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        __m512i prod = reduce_prod_50(v_prod_hi, v_prod_lo, zq);
        // Product in [0, q), accumulator in [0, q): one conditional subtract.
        __m512i res = _mm512_add_epi64(outv[i], prod);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

static void v_mul_subto(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n,
                        const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = av[i];
        __m512i v_op2 = bv[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        __m512i prod = reduce_prod_50(v_prod_hi, v_prod_lo, zq);
        __m512i diff = _mm512_sub_epi64(outv[i], prod);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

static void v_scale(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const __m512i v_s = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i zero = _mm512_setzero_si512();
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = av[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v, v_s);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v, v_s);
        outv[i] = reduce_prod_50(v_prod_hi, v_prod_lo, zq);
    }
}

static void v_scale_addto(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n,
                          const zq_ctx *zq)
{
    __m512i *outv = (__m512i *)out;
    const __m512i *av = (const __m512i *)a;
    const __m512i v_s = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = av[i];
        __m512i v_out = outv[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v, v_s);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v, v_s);
        __m512i v_prod_reduced = reduce_prod_50(v_prod_hi, v_prod_lo, zq);
        __m512i res = _mm512_add_epi64(v_out, v_prod_reduced);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

static void v_add(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(av[i], bv[i]);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

static void v_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    const __m512i *bv = (const __m512i *)b;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(av[i], bv[i]);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

static void v_negate(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    const __m512i zero = _mm512_setzero_si512();
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = av[i];
        __m512i res = _mm512_sub_epi64(q_vec, v);
        __mmask8 mask = _mm512_cmpeq_epu64_mask(v, zero);
        outv[i] = _mm512_mask_blend_epi64(mask, res, zero);
    }
}

static void v_add_scalar(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const __m512i v_scalar = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(av[i], v_scalar);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

static void v_sub_scalar(uint64_t *out, const uint64_t *a, uint64_t s, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const __m512i v_scalar = _mm512_set1_epi64((long long)(s % zq->q));
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(av[i], v_scalar);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

static void v_reduce(uint64_t *out, const uint64_t *a, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        outv[i] = reduce_64b_50(av[i], zq);
    }
}

static void v_reduce_signed(uint64_t *out, const int64_t *a, uint64_t n, const zq_ctx *zq)
{
    const __m512i *av = (const __m512i *)a;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64((long long)zq->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = _mm512_loadu_si512(&av[i]);
        __mmask8 is_negative = _mm512_movepi64_mask(v);
        __m512i v_abs = _mm512_mask_sub_epi64(v, is_negative, _mm512_setzero_si512(), v);

        __m512i res = reduce_64b_50(v_abs, zq);

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
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const size_t n_vec = n / 8;
    const __m512i v_w1 = _mm512_set1_epi64((long long)zq->w52_1);
    const __m512i v_w2 = _mm512_set1_epi64((long long)zq->w52_2);
    for (size_t i = 0; i < n_vec; i++)
    {
        // Split the 128-bit value into base-2^52 digits x0..x2, fold with the
        // radix powers mod q, then finish with the two-digit reduction.
        __m512i v_hi = hiv[i];
        __m512i v_lo = lov[i];
        __m512i x0 = _mm512_and_epi64(v_lo, low52b_mask);
        __m512i x1 = _mm512_or_si512(
            _mm512_srli_epi64(v_lo, 52),
            _mm512_slli_epi64(_mm512_and_epi64(v_hi, _mm512_set1_epi64((1ULL << 40) - 1)), 12));
        __m512i x2 = _mm512_srli_epi64(v_hi, 40);

        __m512i v_S_hi = _mm512_madd52hi_epu64(zero, x1, v_w1);
        __m512i v_S_lo = _mm512_madd52lo_epu64(x0, x1, v_w1);
        v_S_hi = _mm512_madd52hi_epu64(v_S_hi, x2, v_w2);
        v_S_lo = _mm512_madd52lo_epu64(v_S_lo, x2, v_w2);
        v_S_hi = _mm512_add_epi64(v_S_hi, _mm512_srli_epi64(v_S_lo, 52));
        v_S_lo = _mm512_and_epi64(v_S_lo, low52b_mask);

        outv[i] = reduce_digits_50(v_S_hi, v_S_lo, zq);
    }
}

const zq_ops zq_ops_avx512_50 = {
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
