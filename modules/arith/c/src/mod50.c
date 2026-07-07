#include <arith.h>
#include "arith_internal.h"

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)

static inline __m512i _mm512_hexl_reduce_prod_50(__m512i v_prod_hi, __m512i v_prod_lo,
                                                 NTT_proc proc)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)proc->q);
    const __m512i m_ifma = _mm512_set1_epi64(proc->ifma_barr_lo);

    const uint64_t shift = proc->ifma_prod_right_shift;
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

static inline __m512i _mm512_hexl_reduce_64b_50(__m512i v, NTT_proc proc)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)proc->q);
    const __m512i m_ifma = _mm512_set1_epi64(proc->ifma_barr_lo);

    const uint64_t shift = proc->ifma_prod_right_shift;
    const __m128i v_shift = _mm_cvtsi64_si128(shift);

    __m512i c1 = _mm512_srl_epi64(v, v_shift);

    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i v_result = _mm512_madd52lo_epu64(v, q_hat, neg_q_vec);
    v_result = _mm512_and_epi64(v_result, low52b_mask);

    __m512i r2 = _mm512_sub_epi64(v_result, q_vec);
    return _mm512_min_epu64(v_result, r2);
}

static inline __m512i _mm512_hexl_reduce_epi64(__m512i v_hi, __m512i v_lo, NTT_proc proc)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const __m512i neg_q_vec = _mm512_set1_epi64(-(int64_t)proc->q);
    const __m512i m_ifma = _mm512_set1_epi64(proc->ifma_barr_lo);
    const __m512i v_w1 = _mm512_set1_epi64(proc->mp_w1);

    __m512i v2_hi = _mm512_madd52hi_epu64(zero, v_hi, v_w1);
    __m512i v2_lo = _mm512_madd52lo_epu64(v_lo, v_hi, v_w1);
    __m512i v3_hi = _mm512_madd52hi_epu64(zero, v2_hi, v_w1);
    __m512i v3_lo = _mm512_madd52lo_epu64(v2_lo, v2_hi, v_w1);

    v3_hi = _mm512_add_epi64(v3_hi, _mm512_srli_epi64(v3_lo, 52));
    v3_lo = _mm512_and_epi64(v3_lo, low52b_mask);

    const uint64_t shift = proc->ifma_prod_right_shift;
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

void mod_eltwise_mul_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
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
        outv[i] = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, proc);
    }
}

/* out += in1 * in2  (mod q), fused: avoids a temp buffer and a second pass over memory.
 * Each reduced product is in [0,q) and the accumulator stays in [0,q), so a single
 * conditional subtract finishes the add. */
void mod_eltwise_mul_addto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = in1v[i];
        __m512i v_op2 = in2v[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        __m512i prod = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, proc);
        __m512i res = _mm512_add_epi64(outv[i], prod);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

/* out -= in1 * in2  (mod q), fused. */
void mod_eltwise_mul_subto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v_op1 = in1v[i];
        __m512i v_op2 = in2v[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v_op1, v_op2);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v_op1, v_op2);
        __m512i prod = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, proc);
        __m512i diff = _mm512_sub_epi64(outv[i], prod);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

void mod_eltwise_scale_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const uint64_t s = scale % proc->q;
    const __m512i v_s = _mm512_set1_epi64(s);
    const __m512i zero = _mm512_setzero_si512();
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = inv[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v, v_s);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v, v_s);
        outv[i] = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, proc);
    }
}

void mod_eltwise_fma_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    __m512i *outv = (__m512i *)out;
    const __m512i *inv = (const __m512i *)in;
    const uint64_t s = scale % proc->q;
    const __m512i v_s = _mm512_set1_epi64(s);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = inv[i];
        __m512i v_out = outv[i];
        __m512i v_prod_hi = _mm512_madd52hi_epu64(zero, v, v_s);
        __m512i v_prod_lo = _mm512_madd52lo_epu64(zero, v, v_s);
        __m512i v_prod_reduced = _mm512_hexl_reduce_prod_50(v_prod_hi, v_prod_lo, proc);
        __m512i res = _mm512_add_epi64(v_out, v_prod_reduced);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

void mod_eltwise_add_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const uint64_t s = scalar % proc->q;
    const __m512i v_scalar = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(inv[i], v_scalar);
        __m512i r2 = _mm512_sub_epi64(res, q_vec);
        outv[i] = _mm512_min_epu64(res, r2);
    }
}

void mod_eltwise_sub_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const uint64_t s = scalar % proc->q;
    const __m512i v_scalar = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(inv[i], v_scalar);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        outv[i] = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
    }
}

void mod_eltwise_negate_50(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
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

void mod_eltwise_add_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
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

void mod_eltwise_sub_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
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

void mod_eltwise_reduce_50(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    for (size_t i = 0; i < n_vec; i++)
    {
        outv[i] = _mm512_hexl_reduce_64b_50(inv[i], proc);
    }
}

void mod_eltwise_reduce_signed_50(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&inv[i]);
        __mmask8 is_negative = _mm512_movepi64_mask(a);
        __m512i v_abs = _mm512_mask_sub_epi64(a, is_negative, _mm512_setzero_si512(), a);

        __m512i res = _mm512_hexl_reduce_64b_50(v_abs, proc);

        __mmask8 is_zero = _mm512_cmpeq_epu64_mask(res, _mm512_setzero_si512());
        __mmask8 adjust_mask = is_negative & ~is_zero;
        __m512i v_final = _mm512_mask_sub_epi64(res, adjust_mask, q_vec, res);

        _mm512_storeu_si512(&outv[i], v_final);
    }
}

void mod_reduce_array_mp_50(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            NTT_proc proc)
{
    const __m512i *in_hiv = (const __m512i *)in_high;
    const __m512i *in_lov = (const __m512i *)in_low;
    __m512i *outv = (__m512i *)out;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    const size_t n_vec = n / 8;
    const __m512i v_w1 = _mm512_set1_epi64(proc->mp_w1);
    const __m512i v_w2 = _mm512_set1_epi64(proc->mp_w2);
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

        outv[i] = _mm512_hexl_reduce_epi64(v_S_hi, v_S_lo, proc);
    }
}

#endif
