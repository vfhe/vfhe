#include <arith.h>
#include "arith_internal.h"

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)

static inline void FwdButterfly(__m512i *X, __m512i *Y, __m512i W, __m512i W_precon,
                                __m512i neg_modulus, __m512i twice_modulus)
{
    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, twice_modulus));
    const __m512i zero = _mm512_setzero_si512();
    __m512i Q = _mm512_madd52hi_epu64(zero, W_precon, *Y);
    __m512i W_Y = _mm512_madd52lo_epu64(zero, W, *Y);
    __m512i T = _mm512_madd52lo_epu64(W_Y, Q, neg_modulus);
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    T = _mm512_and_epi64(T, low52b_mask);

    __m512i twice_mod_minus_T = _mm512_sub_epi64(twice_modulus, T);
    *Y = _mm512_add_epi64(*X, twice_mod_minus_T);
    *X = _mm512_add_epi64(*X, T);
}

static inline void InvButterfly(__m512i *X, __m512i *Y, __m512i W, __m512i W_precon,
                                __m512i neg_modulus, __m512i twice_modulus)
{
    __m512i Y_minus_2q = _mm512_sub_epi64(*Y, twice_modulus);
    __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

    *X = _mm512_add_epi64(*X, Y_minus_2q);
    __mmask8 sign_bits = _mm512_movepi64_mask(*X);
    *X = _mm512_mask_add_epi64(*X, sign_bits, *X, twice_modulus);

    const __m512i zero = _mm512_setzero_si512();
    __m512i Q = _mm512_madd52hi_epu64(zero, W_precon, T);
    __m512i Q_p = _mm512_madd52lo_epu64(zero, Q, neg_modulus);
    *Y = _mm512_madd52lo_epu64(Q_p, W, T);

    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    *Y = _mm512_and_epi64(*Y, low52b_mask);
}

static inline void InvButterfly_Final_50(__m512i *X, __m512i *Y, __m512i v_inv_n,
                                         __m512i v_inv_n_prime, __m512i v_inv_n_w,
                                         __m512i v_inv_n_w_prime, __m512i minus_q, __m512i q2,
                                         __m512i q_vec)
{
    __m512i Y_minus_2q = _mm512_sub_epi64(*Y, q2);
    __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

    __m512i X_plus_Y = _mm512_add_epi64(*X, *Y);
    X_plus_Y = _mm512_min_epu64(X_plus_Y, _mm512_sub_epi64(X_plus_Y, q2));

    const __m512i zero = _mm512_setzero_si512();
    __m512i Q1 = _mm512_madd52hi_epu64(zero, v_inv_n_prime, X_plus_Y);
    __m512i Q1_p = _mm512_madd52lo_epu64(zero, Q1, minus_q);
    *X = _mm512_madd52lo_epu64(Q1_p, v_inv_n, X_plus_Y);

    __m512i Q2 = _mm512_madd52hi_epu64(zero, v_inv_n_w_prime, T);
    __m512i Q2_p = _mm512_madd52lo_epu64(zero, Q2, minus_q);
    *Y = _mm512_madd52lo_epu64(Q2_p, v_inv_n_w, T);

    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    *X = _mm512_and_epi64(*X, low52b_mask);
    *Y = _mm512_and_epi64(*Y, low52b_mask);

    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, q_vec));
    *Y = _mm512_min_epu64(*Y, _mm512_sub_epi64(*Y, q_vec));
}

static uint64_t barrett_factor_52(uint64_t val, uint64_t q)
{
    unsigned __int128 num = (unsigned __int128)val << 52;
    return (uint64_t)(num / q);
}

static inline void FwdInterleaveT4(__m512i *A, __m512i *B)
{
    const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i perm_hi = _mm512_permutexvar_epi64(vperm2_idx, *B);
    *B = _mm512_mask_blend_epi64(0xf0, perm_hi, *A);
    *A = _mm512_mask_blend_epi64(0x0f, perm_hi, *A);
    *B = _mm512_permutexvar_epi64(vperm2_idx, *B);
}

static inline void LoadInvInterleavedT2(__m512i *out1, __m512i *out2, __m512i *in)
{
    __m512i v1 = _mm512_loadu_si512(in++);
    __m512i v2 = _mm512_loadu_si512(in);
    const __m512i v1_perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    __m512i v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, v1);
    __m512i v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, v2);
    *out1 = _mm512_mask_blend_epi64(0xaa, v1, v2_perm);
    *out2 = _mm512_mask_blend_epi64(0xaa, v1_perm, v2);
}

static inline void FwdInterleaveT2(__m512i *A, __m512i *B)
{
    const __m512i v1_perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
    __m512i v1_perm = _mm512_permutexvar_epi64(v1_perm_idx, *A);
    __m512i v2_perm = _mm512_permutexvar_epi64(v1_perm_idx, *B);
    *A = _mm512_mask_blend_epi64(0xcc, *A, v2_perm);
    *B = _mm512_mask_blend_epi64(0xcc, v1_perm, *B);
}

static inline void FwdInterleaveT1(__m512i *A, __m512i *B)
{
    const __m512i perm_idx = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    __m512i v1_perm = _mm512_permutexvar_epi64(perm_idx, *A);
    __m512i v2_perm = _mm512_permutexvar_epi64(perm_idx, *B);
    *A = _mm512_mask_blend_epi64(0xaa, *A, v2_perm);
    *B = _mm512_mask_blend_epi64(0xaa, v1_perm, *B);
}

static inline void FwdReInterleaveT1(__m512i *A, __m512i *B)
{
    const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    const __m512i v_X_out_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
    const __m512i v_Y_out_idx = _mm512_set_epi64(3, 7, 2, 6, 1, 5, 0, 4);
    *B = _mm512_permutexvar_epi64(vperm2_idx, *B);
    __m512i perm_lo = _mm512_mask_blend_epi64(0x0f, *A, *B);
    __m512i perm_hi = _mm512_mask_blend_epi64(0xf0, *A, *B);
    *A = _mm512_permutexvar_epi64(v_X_out_idx, perm_hi);
    *B = _mm512_permutexvar_epi64(v_Y_out_idx, perm_lo);
}

static inline void InvInterleaveT1(__m512i *A, __m512i *B)
{
    const __m512i vperm_hi_idx = _mm512_set_epi64(6, 4, 2, 0, 7, 5, 3, 1);
    const __m512i vperm_lo_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    const __m512i vperm2_idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
    __m512i perm_lo = _mm512_permutexvar_epi64(vperm_lo_idx, *A);
    __m512i perm_hi = _mm512_permutexvar_epi64(vperm_hi_idx, *B);
    *A = _mm512_mask_blend_epi64(0x0f, perm_hi, perm_lo);
    *B = _mm512_mask_blend_epi64(0xf0, perm_hi, perm_lo);
    *B = _mm512_permutexvar_epi64(vperm2_idx, *B);
}

static inline void InvInterleaveT4(__m512i *A, __m512i *B)
{
    const __m512i perm_idx = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
    __m512i v1_perm = _mm512_permutexvar_epi64(perm_idx, *A);
    __m512i v2_perm = _mm512_permutexvar_epi64(perm_idx, *B);
    *A = _mm512_mask_blend_epi64(0xcc, *A, v2_perm);
    *B = _mm512_mask_blend_epi64(0xcc, v1_perm, *B);
}

static inline void InvReInterleaveT4(__m512i *A, __m512i *B)
{
    __m256i x0 = _mm512_extracti64x4_epi64(*A, 0);
    __m256i x1 = _mm512_extracti64x4_epi64(*A, 1);
    __m256i y0 = _mm512_extracti64x4_epi64(*B, 0);
    __m256i y1 = _mm512_extracti64x4_epi64(*B, 1);
    *A = _mm512_inserti64x4(_mm512_castsi256_si512(x0), y0, 1);
    *B = _mm512_inserti64x4(_mm512_castsi256_si512(x1), y1, 1);
}

static void ntt_CT_NR_internal_50(__m512i *x, __m512i **ws, __m512i **w_precon, uint64_t sub_n,
                                  uint64_t q, size_t level, size_t offset_i, NTT_proc proc)
{
    __m512i minus_q = _mm512_set1_epi64(-q);
    __m512i q2 = _mm512_set1_epi64(2 * q);

    if (sub_n <= 1024)
    {
        size_t l = level;
        for (; (1ULL << (l - level + 3)) < sub_n; l++)
        {
            const __m512i *wsv = ws[l];
            const __m512i *ws_precon_v = w_precon[l];
            const uint64_t t = sub_n >> (l - level + 4);
            size_t m = 1ULL << (l - level);
            size_t start_i = offset_i << (l - level);
            for (size_t i = 0; i < m; i++)
            {
                const uint64_t slice = 2 * i * t;
                const __m512i w_i = wsv[start_i + i];
                const __m512i wp_i = ws_precon_v[start_i + i];
                for (size_t j = slice; j < slice + t; j++)
                {
                    FwdButterfly(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
                }
            }
        }

        if (sub_n >= 16)
        {
            size_t w_offset = offset_i * (sub_n / 16);

            { // t = 4
                const __m512i *wsv = ws[l];
                const __m512i *ws_precon_v = w_precon[l];
                for (size_t i = 0; i < sub_n / 16; i++)
                {
                    const uint64_t slice = 2 * i;
                    const __m512i w_i = wsv[w_offset + i];
                    const __m512i wp_i = ws_precon_v[w_offset + i];
                    FwdInterleaveT4(&x[slice], &x[slice + 1]);
                    FwdButterfly(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
                }
            }
            l++;

            { // t = 2
                const __m512i *wsv = ws[l];
                const __m512i *ws_precon_v = w_precon[l];
                for (size_t i = 0; i < sub_n / 16; i++)
                {
                    const uint64_t slice = 2 * i;
                    const __m512i w_i = wsv[w_offset + i];
                    const __m512i wp_i = ws_precon_v[w_offset + i];
                    FwdInterleaveT2(&x[slice], &x[slice + 1]);
                    FwdButterfly(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
                }
            }
            l++;

            { // t = 1
                const __m512i *wsv = ws[l];
                const __m512i *ws_precon_v = w_precon[l];
                for (size_t i = 0; i < sub_n / 16; i++)
                {
                    const uint64_t slice = 2 * i;
                    const __m512i w_i = wsv[w_offset + i];
                    const __m512i wp_i = ws_precon_v[w_offset + i];
                    FwdInterleaveT1(&x[slice], &x[slice + 1]);
                    FwdButterfly(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
                    FwdReInterleaveT1(&x[slice], &x[slice + 1]);
                }
            }
        }
    }
    else
    {
        size_t t = sub_n >> 4;
        const __m512i w_i = ws[level][offset_i];
        const __m512i wp_i = w_precon[level][offset_i];
        for (size_t j = 0; j < t; j++)
        {
            FwdButterfly(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
        }
        ntt_CT_NR_internal_50(x, ws, w_precon, sub_n / 2, q, level + 1, 2 * offset_i, proc);
        ntt_CT_NR_internal_50(x + t, ws, w_precon, sub_n / 2, q, level + 1, 2 * offset_i + 1, proc);
    }
}

void ntt_forward_50(uint64_t *out, uint64_t *in, NTT_proc proc)
{
    if (out != in)
    {
        for (size_t i = 0; i < proc->n; i++)
        {
            out[i] = in[i];
        }
    }
    ntt_CT_NR_internal_50((__m512i *)out, (__m512i **)proc->ws_fwd, (__m512i **)proc->w_precon_fwd,
                          proc->n, proc->q, 0, 0, proc);

    const __m512i q_vec = _mm512_set1_epi64(proc->q);
    const __m512i q2_vec = _mm512_set1_epi64(2 * proc->q);
    for (size_t i = 0; i < proc->n / 8; i++)
    {
        __m512i v = ((__m512i *)out)[i];
        v = _mm512_min_epu64(v, _mm512_sub_epi64(v, q2_vec));
        v = _mm512_min_epu64(v, _mm512_sub_epi64(v, q_vec));
        ((__m512i *)out)[i] = v;
    }
}

static void ntt_GS_RN_internal_50(__m512i *x, __m512i **ws, __m512i **w_precon, uint64_t sub_n,
                                  uint64_t q, size_t l_merge, size_t offset_i, int is_top_level,
                                  uint64_t inv_n, NTT_proc proc)
{
    __m512i minus_q = _mm512_set1_epi64(-q);
    __m512i q2 = _mm512_set1_epi64(2 * q);

    if (sub_n <= 1024)
    {
        size_t l = 0;
        if (sub_n >= 16)
        {
            size_t w_offset = offset_i * (sub_n / 16);

            { // t = 1
                const __m512i *wsv = ws[l];
                const __m512i *ws_precon_v = w_precon[l];
                for (size_t i = 0; i < sub_n / 16; i++)
                {
                    const uint64_t slice = 2 * i;
                    const __m512i w_i = wsv[w_offset + i];
                    const __m512i wp_i = ws_precon_v[w_offset + i];
                    InvInterleaveT1(&x[slice], &x[slice + 1]);
                    InvButterfly(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
                }
            }
            l++;

            { // t = 2
                const __m512i *wsv = ws[l];
                const __m512i *ws_precon_v = w_precon[l];
                for (size_t i = 0; i < sub_n / 16; i++)
                {
                    const uint64_t slice = 2 * i;
                    const __m512i w_i = wsv[w_offset + i];
                    const __m512i wp_i = ws_precon_v[w_offset + i];
                    __m512i v_X, v_Y;
                    LoadInvInterleavedT2(&v_X, &v_Y, &x[slice]);
                    InvButterfly(&v_X, &v_Y, w_i, wp_i, minus_q, q2);
                    x[slice] = v_X;
                    x[slice + 1] = v_Y;
                }
            }
            l++;

            { // t = 4
                const __m512i *wsv = ws[l];
                const __m512i *ws_precon_v = w_precon[l];
                for (size_t i = 0; i < sub_n / 16; i++)
                {
                    const uint64_t slice = 2 * i;
                    const __m512i w_i = wsv[w_offset + i];
                    const __m512i wp_i = ws_precon_v[w_offset + i];
                    InvInterleaveT4(&x[slice], &x[slice + 1]);
                    InvButterfly(&x[slice], &x[slice + 1], w_i, wp_i, minus_q, q2);
                    InvReInterleaveT4(&x[slice], &x[slice + 1]);
                }
            }
            l++;
        }

        for (; (1ULL << (l + is_top_level)) < sub_n; l++)
        {
            const __m512i *wsv = ws[l];
            const __m512i *ws_precon_v = w_precon[l];
            const uint64_t t = 1ULL << (l - 3);
            const uint64_t m_sub = sub_n >> (l + 1);
            size_t start_i = offset_i * m_sub;
            for (size_t i = 0; i < m_sub; i++)
            {
                const uint64_t slice = 2 * i * t;
                const __m512i w_i = wsv[start_i + i];
                const __m512i wp_i = ws_precon_v[start_i + i];
                for (size_t j = slice; j < slice + t; j++)
                {
                    InvButterfly(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
                }
            }
        }

        if (is_top_level)
        {
            size_t l_top = l;
            const __m512i *wsv = ws[l_top];
            const uint64_t t = 1ULL << (l_top - 3);
            uint64_t inv_n_prime = barrett_factor_52(inv_n, q);
            __m512i v_inv_n = _mm512_set1_epi64(inv_n);
            __m512i v_inv_n_prime = _mm512_set1_epi64(inv_n_prime);
            __m512i q_vec = _mm512_set1_epi64(q);
            uint64_t w = _mm_cvtsi128_si64(_mm512_castsi512_si128(wsv[0]));
            uint64_t inv_n_w = (uint64_t)(((unsigned __int128)inv_n * w) % q);
            uint64_t inv_n_w_prime = barrett_factor_52(inv_n_w, q);
            __m512i v_inv_n_w = _mm512_set1_epi64(inv_n_w);
            __m512i v_inv_n_w_prime = _mm512_set1_epi64(inv_n_w_prime);
            for (size_t j = 0; j < t; j++)
            {
                InvButterfly_Final_50(&x[j], &x[j + t], v_inv_n, v_inv_n_prime, v_inv_n_w,
                                      v_inv_n_w_prime, minus_q, q2, q_vec);
            }
        }
    }
    else
    {
        size_t t = sub_n >> 4;
        ntt_GS_RN_internal_50(x, ws, w_precon, sub_n / 2, q, l_merge - 1, 2 * offset_i, 0, inv_n,
                              proc);
        ntt_GS_RN_internal_50(x + t, ws, w_precon, sub_n / 2, q, l_merge - 1, 2 * offset_i + 1, 0,
                              inv_n, proc);
        if (is_top_level)
        {
            uint64_t inv_n_prime = barrett_factor_52(inv_n, q);
            __m512i v_inv_n = _mm512_set1_epi64(inv_n);
            __m512i v_inv_n_prime = _mm512_set1_epi64(inv_n_prime);
            __m512i q_vec = _mm512_set1_epi64(q);
            const __m512i w_i = ws[l_merge][offset_i];
            uint64_t w = _mm_cvtsi128_si64(_mm512_castsi512_si128(w_i));
            uint64_t inv_n_w = (uint64_t)(((unsigned __int128)inv_n * w) % q);
            uint64_t inv_n_w_prime = barrett_factor_52(inv_n_w, q);
            __m512i v_inv_n_w = _mm512_set1_epi64(inv_n_w);
            __m512i v_inv_n_w_prime = _mm512_set1_epi64(inv_n_w_prime);
            for (size_t j = 0; j < t; j++)
            {
                InvButterfly_Final_50(&x[j], &x[j + t], v_inv_n, v_inv_n_prime, v_inv_n_w,
                                      v_inv_n_w_prime, minus_q, q2, q_vec);
            }
        }
        else
        {
            const __m512i w_i = ws[l_merge][offset_i];
            const __m512i wp_i = w_precon[l_merge][offset_i];
            for (size_t j = 0; j < t; j++)
            {
                InvButterfly(&x[j], &x[j + t], w_i, wp_i, minus_q, q2);
            }
        }
    }
}

void ntt_reverse_50(uint64_t *out, uint64_t *in, NTT_proc proc)
{
    if (out != in)
    {
        for (size_t i = 0; i < proc->n; i++)
        {
            out[i] = in[i];
        }
    }
    int logn = 0;
    while ((1ULL << logn) < proc->n)
        logn++;
    uint64_t inv_n = inverse_mod(proc->n, proc->q);
    ntt_GS_RN_internal_50((__m512i *)out, (__m512i **)proc->ws_inv, (__m512i **)proc->w_precon_inv,
                          proc->n, proc->q, logn - 1, 0, 1, inv_n, proc);
}

#endif
