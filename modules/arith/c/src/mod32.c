#include <arith.h>
#include "arith_internal.h"

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)

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

void mod_eltwise_mul_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&in1v[i]);
        __m512i b = _mm512_loadu_si512(&in2v[i]);
        __m512i prod = _mm512_mullo_epi64(a, b);
        __m512i c1 = _mm512_srli_epi64(prod, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(prod, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

/* out += in1 * in2 (mod q), fused (see mod_eltwise_mul_addto_50). */
void mod_eltwise_mul_addto_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&in1v[i]);
        __m512i b = _mm512_loadu_si512(&in2v[i]);
        __m512i prod = _mm512_mullo_epi64(a, b);
        __m512i c1 = _mm512_srli_epi64(prod, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(prod, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        __m512i acc = _mm512_add_epi64(_mm512_loadu_si512(&outv[i]), res);
        acc = _mm512_min_epu64(acc, _mm512_sub_epi64(acc, q_vec));
        _mm512_storeu_si512(&outv[i], acc);
    }
}

/* out -= in1 * in2 (mod q), fused. */
void mod_eltwise_mul_subto_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                              NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&in1v[i]);
        __m512i b = _mm512_loadu_si512(&in2v[i]);
        __m512i prod = _mm512_mullo_epi64(a, b);
        __m512i c1 = _mm512_srli_epi64(prod, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(prod, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        __m512i diff = _mm512_sub_epi64(_mm512_loadu_si512(&outv[i]), res);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        diff = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
        _mm512_storeu_si512(&outv[i], diff);
    }
}

void mod_eltwise_scale_32(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const uint64_t s = scale % proc->q;
    const __m512i s_vec = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&inv[i]);
        __m512i prod = _mm512_mullo_epi64(a, s_vec);
        __m512i c1 = _mm512_srli_epi64(prod, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(prod, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

void mod_eltwise_fma_32(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const uint64_t s = scale % proc->q;
    const __m512i s_vec = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&inv[i]);
        __m512i prod = _mm512_mullo_epi64(a, s_vec);
        __m512i c1 = _mm512_srli_epi64(prod, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(prod, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));

        __m512i out_val = _mm512_loadu_si512(&outv[i]);
        __m512i sum = _mm512_add_epi64(out_val, res);
        sum = _mm512_min_epu64(sum, _mm512_sub_epi64(sum, q_vec));
        _mm512_storeu_si512(&outv[i], sum);
    }
}

void mod_eltwise_add_scalar_32(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const uint64_t s = scalar % proc->q;
    const __m512i s_vec = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(_mm512_loadu_si512(&inv[i]), s_vec);
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

void mod_eltwise_sub_scalar_32(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const uint64_t s = scalar % proc->q;
    const __m512i s_vec = _mm512_set1_epi64(s);
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(_mm512_loadu_si512(&inv[i]), s_vec);
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        diff = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
        _mm512_storeu_si512(&outv[i], diff);
    }
}

void mod_eltwise_negate_32(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const __m512i zero = _mm512_setzero_si512();

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i v = _mm512_loadu_si512(&inv[i]);
        __m512i res = _mm512_sub_epi64(q_vec, v);
        __mmask8 mask = _mm512_cmpeq_epu64_mask(v, zero);
        res = _mm512_mask_blend_epi64(mask, res, zero);
        _mm512_storeu_si512(&outv[i], res);
    }
}

void mod_eltwise_add_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i res = _mm512_add_epi64(_mm512_loadu_si512(&in1v[i]), _mm512_loadu_si512(&in2v[i]));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

void mod_eltwise_sub_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    const __m512i *in1v = (const __m512i *)in1;
    const __m512i *in2v = (const __m512i *)in2;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i diff = _mm512_sub_epi64(_mm512_loadu_si512(&in1v[i]), _mm512_loadu_si512(&in2v[i]));
        __mmask8 sign_bits = _mm512_movepi64_mask(diff);
        diff = _mm512_mask_add_epi64(diff, sign_bits, diff, q_vec);
        _mm512_storeu_si512(&outv[i], diff);
    }
}

void mod_eltwise_reduce_32(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&inv[i]);
        __m512i c1 = _mm512_srli_epi64(a, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(a, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        _mm512_storeu_si512(&outv[i], res);
    }
}

void mod_eltwise_reduce_signed_32(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc)
{
    const __m512i *inv = (const __m512i *)in;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i a = _mm512_loadu_si512(&inv[i]);
        __mmask8 is_negative = _mm512_movepi64_mask(a);
        __m512i v_abs = _mm512_mask_sub_epi64(a, is_negative, _mm512_setzero_si512(), a);

        __m512i c1 = _mm512_srli_epi64(v_abs, prod_right_shift);
        __m512i q_hat = mulhi_approx_64(c1, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(v_abs, _mm512_mullo_epi64(q_hat, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));

        __mmask8 is_zero = _mm512_cmpeq_epu64_mask(res, _mm512_setzero_si512());
        __mmask8 adjust_mask = is_negative & ~is_zero;
        __m512i v_final = _mm512_mask_sub_epi64(res, adjust_mask, q_vec, res);

        _mm512_storeu_si512(&outv[i], v_final);
    }
}

void mod_reduce_array_mp_32(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            NTT_proc proc)
{
    const __m512i *in_hiv = (const __m512i *)in_high;
    const __m512i *in_lov = (const __m512i *)in_low;
    __m512i *outv = (__m512i *)out;
    const size_t n_vec = n / 8;
    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    uint64_t w1_val = ((unsigned __int128)1 << 64) % proc->q;
    const __m512i w1 = _mm512_set1_epi64(w1_val);

    int q_bits = 0;
    uint64_t temp_q = proc->q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t prod_right_shift = q_bits - 2;
    unsigned __int128 dividend = ((unsigned __int128)1 << (q_bits + 62));
    uint64_t barr_lo = (uint64_t)(dividend / proc->q);
    const __m512i barr_lo_vec = _mm512_set1_epi64(barr_lo);

    for (size_t i = 0; i < n_vec; i++)
    {
        __m512i val_hi = _mm512_loadu_si512(&in_hiv[i]);
        __m512i val_lo = _mm512_loadu_si512(&in_lov[i]);

        __m512i c1_hi = _mm512_srli_epi64(val_hi, prod_right_shift);
        __m512i q_hat_hi = mulhi_approx_64(c1_hi, barr_lo_vec);
        __m512i val_hi_red = _mm512_sub_epi64(val_hi, _mm512_mullo_epi64(q_hat_hi, q_vec));
        val_hi_red = _mm512_min_epu64(val_hi_red, _mm512_sub_epi64(val_hi_red, q_vec));
        val_hi_red = _mm512_min_epu64(val_hi_red, _mm512_sub_epi64(val_hi_red, q_vec));

        __m512i prod = _mm512_mullo_epi64(val_hi_red, w1);
        __m512i sum = _mm512_add_epi64(prod, val_lo);

        __m512i c1_sum = _mm512_srli_epi64(sum, prod_right_shift);
        __m512i q_hat_sum = mulhi_approx_64(c1_sum, barr_lo_vec);
        __m512i res = _mm512_sub_epi64(sum, _mm512_mullo_epi64(q_hat_sum, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
        res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));

        _mm512_storeu_si512(&outv[i], res);
    }
}

#endif
