// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include "arith_internal.h"

#if VFHE_HAVE_AVX512IFMA

static inline __m512i _mm512_hexl_reduce_prod_50(__m512i v_prod_hi, __m512i v_prod_lo, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)mod->q);
    const __m512i m_ifma = _mm512_set1_epi64(mod->ifma_barr_lo);

    const uint64_t shift = mod->ifma_prod_right_shift;
    const __m128i v_shift = _mm_cvtsi64_si128(shift);
    const __m128i v_shift_rev = _mm_cvtsi64_si128(52 - shift);

    __m512i c1 = _mm512_or_si512(_mm512_srl_epi64(v_prod_lo, v_shift),
                                 _mm512_sll_epi64(v_prod_hi, v_shift_rev));

    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i v_result = _mm512_madd52lo_epu64(v_prod_lo, q_hat, neg_q_vec);
    v_result = _mm512_and_epi64(v_result, low52b_mask);

    __m512i r2 = _mm512_sub_epi64(v_result, q_vec);
    return _mm512_min_epu64(v_result, r2);
}

static inline __m512i _mm512_hexl_reduce_64b_50(__m512i v, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)mod->q);
    const __m512i m_ifma = _mm512_set1_epi64(mod->ifma_barr_lo);

    const uint64_t shift = mod->ifma_prod_right_shift;
    const __m128i v_shift = _mm_cvtsi64_si128(shift);

    __m512i c1 = _mm512_srl_epi64(v, v_shift);

    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i v_result = _mm512_madd52lo_epu64(v, q_hat, neg_q_vec);
    v_result = _mm512_and_epi64(v_result, low52b_mask);

    __m512i r2 = _mm512_sub_epi64(v_result, q_vec);
    return _mm512_min_epu64(v_result, r2);
}

static inline __m512i _mm512_hexl_reduce_epi64(__m512i v_hi, __m512i v_lo, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)mod->q);
    const __m512i m_ifma = _mm512_set1_epi64(mod->ifma_barr_lo);
    const __m512i v_w1 = _mm512_set1_epi64(mod->mp_w1);

    __m512i v2_hi = _mm512_madd52hi_epu64(zero, v_hi, v_w1);
    __m512i v2_lo = _mm512_madd52lo_epu64(v_lo, v_hi, v_w1);
    __m512i v3_hi = _mm512_madd52hi_epu64(zero, v2_hi, v_w1);
    __m512i v3_lo = _mm512_madd52lo_epu64(v2_lo, v2_hi, v_w1);

    v3_hi = _mm512_add_epi64(v3_hi, _mm512_srli_epi64(v3_lo, 52));
    v3_lo = _mm512_and_epi64(v3_lo, low52b_mask);

    const uint64_t shift = mod->ifma_prod_right_shift;
    const __m128i v_shift = _mm_cvtsi64_si128(shift);
    const __m128i v_shift_rev = _mm_cvtsi64_si128(52 - shift);

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

void mod_eltwise_mul_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = in1v[i];
        __m512i v_op2 = in2v[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        outv[i] = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, mod);
    }
}

/* out += in1 * in2  (mod q), fused: avoids a temp buffer and a second pass over memory.
 * Each reduced product is in [0,q) and the accumulator stays in [0,q), so a single
 * conditional subtract finishes the add. */
void mod_eltwise_mul_addto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = in1v[i];
        __m512i v_op2 = in2v[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        __m512i prod = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, mod);
        __m512i res = _mm512_add_epi64(outv[i], prod);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

/* out -= in1 * in2  (mod q), fused. */
void mod_eltwise_mul_subto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = in1v[i];
        __m512i v_op2 = in2v[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        __m512i prod = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, mod);
        __m512i diff = _mm512_sub_epi64(outv[i], prod);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

/* out += sum_t a[t] * b[t] (mod q), with one reduction per element. A product of two values below
 * q < 2^50 has a low 52-bit half and a high half below q^2 / 2^52; madd52lo and madd52hi add them
 * to two 64-bit sums, which hold up to MOD_INNER_PRODUCT_MAX_TERMS terms. With the low sum's carry
 * moved up the total is H 2^52 + L, L < 2^52, reduced as (H mod q) (2^52 mod q) + (L mod q). Each
 * of those three reductions takes a value below (q - 1)^2, as _mm512_hexl_reduce_prod_50
 * requires; for L that needs q > 2^26, so the dispatcher keeps smaller moduli on per-term
 * products. */

// acc + (hi 2^52 + lo) mod q, for the two sums above.
static inline __m512i inner_product_reduce_50(__m512i lo, __m512i hi, __m512i acc, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i w1 = _mm512_set1_epi64(mod->mp_w1); // 2^52 mod q
    hi = _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52));
    lo = _mm512_and_epi64(lo, low52b_mask);
    const __m512i h = _mm512_hexl_reduce_prod_50(_mm512_srli_epi64(hi, 52),
                                                 _mm512_and_epi64(hi, low52b_mask), mod);
    const __m512i folded = _mm512_hexl_reduce_prod_50(_mm512_madd52hi_epu64(zero, h, w1),
                                                      _mm512_madd52lo_epu64(zero, h, w1), mod);
    __m512i res = _mm512_add_epi64(folded, _mm512_hexl_reduce_prod_50(zero, lo, mod));
    res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
    res = _mm512_add_epi64(res, acc);
    return _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
}

void mod_eltwise_inner_product_addto_50(uint64_t *out, uint64_t *const *a, uint64_t *const *b,
                                        uint64_t terms, uint64_t n, Modulus mod)
{
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i lo = zero, hi = zero;
        for (uint64_t t = 0; t < terms; t++)
        {
            const __m512i x = ((const __m512i *)a[t])[i], y = ((const __m512i *)b[t])[i];
            lo = _mm512_madd52lo_epu64(lo, x, y);
            hi = _mm512_madd52hi_epu64(hi, x, y);
        }
        outv[i] = inner_product_reduce_50(lo, hi, outv[i], mod);
    }
}

/* `scale` and `fma` multiply by a scalar that does not vary over the array,
   and a fixed operand is what Shoup's form needs: precompute
   s' = floor(s * 2^52 / q) once, and the reduction per element is a
   multiply-high, a multiply and a subtract. The general two-varying-operand
   Barrett that `mul` must use is not needed here. One 128-bit divide per call
   sets s' up.

   Both madd52 operands stay under 2^52 as the instruction requires: s and a
   are below q < 2^50, and s' below 2^52 because s < q. The 52-bit mask is
   exact because the value it truncates, s*a - q_hat*q, is in [0, 2q), which
   one conditional subtract then brings into [0, q). */
static inline __m512i shoup_mul_50(__m512i a, __m512i s, __m512i sp, __m512i q_vec,
                                   __m512i neg_q_vec)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_hat = _mm512_madd52hi_epu64(zero, sp, a);
    const __m512i res = _mm512_and_epi64(
        _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero, s, a), q_hat, neg_q_vec), low52b_mask);
    return _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
}

void mod_eltwise_scale_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const uint64_t s = modq(scale, mod);
    const __m512i v_s = _mm512_set1_epi64(s);
    const __m512i v_sp = _mm512_set1_epi64((uint64_t)(((unsigned __int128)s << 52) / mod->q));
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        outv[i] = shoup_mul_50(inv[i], v_s, v_sp, q_vec, neg_q_vec);
    }
}

void mod_eltwise_fma_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    __m512i *outv = (__m512i *)out;
    const __m512i *inv = (const __m512i *)in;
    const uint64_t s = modq(scale, mod);
    const __m512i v_s = _mm512_set1_epi64(s);
    const __m512i v_sp = _mm512_set1_epi64((uint64_t)(((unsigned __int128)s << 52) / mod->q));
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(outv[i], shoup_mul_50(inv[i], v_s, v_sp, q_vec, neg_q_vec));
        outv[i] = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
    }
}

void mod_eltwise_lifted_combination_50(uint64_t *out, const uint64_t *const *in,
                                       const uint64_t *q_in, const uint64_t *scale, uint64_t m,
                                       bool centered, uint64_t n, Modulus mod)
{
    const uint64_t p = mod->q;
    uint64_t s[MOD_COMBINATION_MAX_TERMS], sp[MOD_COMBINATION_MAX_TERMS];
    uint64_t half[MOD_COMBINATION_MAX_TERMS], h[MOD_COMBINATION_MAX_TERMS];
    assert(m <= MOD_COMBINATION_MAX_TERMS);
    for (uint64_t t = 0; t < m; t++)
    {
        assert(q_in[t] < (1ULL << 52));
        s[t] = modq(scale[t], mod);
        sp[t] = (uint64_t)(((unsigned __int128)s[t] << 52) / p);
        half[t] = centered ? q_in[t] >> 1 : UINT64_MAX;
        h[t] = negate_modq(mul_modq(s[t], modq(q_in[t], mod), mod), p);
    }
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i neg_p = _mm512_set1_epi64(-(int64_t)p);
    // A copy no store to `out` can alias, so its constants stay in registers.
    struct _Modulus m_copy = *mod;
    for (uint64_t j = 0; j < n; j += 8)
    {
        // Each term is a lazy Shoup product in [0, 2p) plus the centering
        // correction, below 3p < 2^52, so the 64-bit lanes sum them all
        // without reducing until the end.
        __m512i acc = zero;
        for (uint64_t t = 0; t < m; t++)
        {
            const __m512i v = _mm512_loadu_si512((const void *)(in[t] + j));
            const __m512i q_hat = _mm512_madd52hi_epu64(zero, _mm512_set1_epi64(sp[t]), v);
            const __m512i r = _mm512_and_epi64(
                _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero, _mm512_set1_epi64(s[t]), v),
                                      q_hat, neg_p),
                low52b_mask);
            const __mmask8 above = _mm512_cmpgt_epu64_mask(v, _mm512_set1_epi64(half[t]));
            acc = _mm512_add_epi64(acc, r);
            acc = _mm512_mask_add_epi64(acc, above, acc, _mm512_set1_epi64(h[t]));
        }
        _mm512_storeu_si512((void *)(out + j), _mm512_hexl_reduce_64b_50(acc, &m_copy));
    }
}

void mod_eltwise_add_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const uint64_t s = modq(scalar, mod);
    const __m512i v_scalar = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(inv[i], v_scalar);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

void mod_eltwise_sub_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const uint64_t s = modq(scalar, mod);
    const __m512i v_scalar = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(inv[i], v_scalar);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

void mod_eltwise_negate_50(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    const __m512i zero = _mm512_setzero_si512();
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = inv[i];
        __m512i res = _mm512_sub_epi64(q_vec, v);
        __mmask8 mask = _mm512_cmpeq_epu64_mask(v, zero);
        outv[i] = _mm512_mask_blend_epi64(mask, res, zero);
    }
}

void mod_eltwise_add_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v1 = in1v[i];
        __m512i v2 = in2v[i];
        __m512i res = _mm512_add_epi64(v1, v2);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

void mod_eltwise_sub_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v1 = in1v[i];
        __m512i v2 = in2v[i];
        __m512i diff = _mm512_sub_epi64(v1, v2);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

void mod_eltwise_reduce_50(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        outv[i] = _mm512_hexl_reduce_64b_50(inv[i], mod);
    }
}

void mod_eltwise_reduce_signed_50(uint64_t *out, int64_t *in, uint64_t n, Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(mod->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&inv[i]);
        __mmask8 is_negative = _mm512_movepi64_mask(a);
        __m512i v_abs = _mm512_mask_sub_epi64(a, is_negative, _mm512_setzero_si512(), a);

        __m512i res = _mm512_hexl_reduce_64b_50(v_abs, mod);

        __mmask8 is_zero = _mm512_cmpeq_epu64_mask(res, _mm512_setzero_si512());
        __mmask8 adjust_mask = is_negative & ~is_zero;
        __m512i v_final = _mm512_mask_sub_epi64(res, adjust_mask, q_vec, res);

        _mm512_storeu_si512(&outv[i], v_final);
    }
}

void mod_eltwise_reduce_centered_50(uint64_t *out, uint64_t *in, uint64_t q_in, uint64_t n,
                                    Modulus mod)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i half = _mm512_set1_epi64(q_in >> 1);
    const __m512i neg_q_in = _mm512_set1_epi64(negate_modq(modq(q_in, mod), mod->q));
    // A copy no store to `out` can alias, so its constants stay in registers.
    struct _Modulus m = *mod;

    for (size_t i = 0; i < n_vec; i++)
        outv[i] = _mm512_hexl_reduce_64b_50(centered_shift(inv[i], half, neg_q_in), &m);
}

void mod_reduce_array_mp_50(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            Modulus mod)
{
    const __m512i *in_hiv = (const __m512i *)in_high;
    const __m512i *in_lov = (const __m512i *)in_low;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const size_t n_vec = n / 8;
    const __m512i v_w1 = _mm512_set1_epi64(mod->mp_w1);
    const __m512i v_w2 = _mm512_set1_epi64(mod->mp_w2);
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_hi = in_hiv[i];
        __m512i v_lo = in_lov[i];
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

        outv[i] = _mm512_hexl_reduce_epi64(v_S_hi, v_S_lo, mod);
    }
}

#endif
