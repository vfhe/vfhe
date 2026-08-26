// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include "arith_internal.h"

#if VFHE_HAVE_AVX512IFMA

static uint64_t reverse_bits(uint64_t x, int bits)
{
    uint64_t res = 0;
    for (int i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

// The per-twiddle Shoup constant floor(val * 2^shift / q). A real division on
// purpose: this is what a Barrett-style reduction is built *from*, so it cannot
// be computed with one -- same reason mod_new divides.
static uint64_t barrett_factor(uint64_t val, uint64_t q, uint64_t shift)
{
    unsigned __int128 num = (unsigned __int128)val << shift;
    return (uint64_t)(num / q);
}

void ntt_precompute_fwd(uint64_t n, Modulus mod, uint64_t root_of_unity, __m512i ***out_ws,
                        __m512i ***out_w_precon)
{
    const uint64_t q = mod->q;
    size_t logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t shift = (q < (1ULL << 32)) ? 32 : (q < (1ULL << 50)) ? 52 : 64;

    uint64_t *rou = (uint64_t *)malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    uint64_t idx = 0, prev_idx = 0;
    for (size_t i = 1; i < n; i++)
    {
        idx = reverse_bits(i, logn);
        rou[idx] = mul_modq(rou[prev_idx], root_of_unity, mod);
        prev_idx = idx;
    }
    __m512i **ws = (__m512i **)malloc(logn * sizeof(__m512i *));
    __m512i **w_precon = (__m512i **)malloc(logn * sizeof(__m512i *));
    size_t level = 0;
    size_t w_idx = 1;
    for (; (1ULL << (level + 3)) < n; level++)
    {
        size_t m = 1ULL << level;
        ws[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        for (size_t i = 0; i < m; i++)
        {
            uint64_t w = rou[w_idx + i];
            uint64_t wp = barrett_factor(w, q, shift);
            ws[level][i] = _mm512_set1_epi64(w);
            w_precon[level][i] = _mm512_set1_epi64(wp);
        }
        w_idx += m;
    }
    if (level < logn)
    {
        size_t m = 1ULL << level;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w_0 = rou[w_idx + 2 * i];
            uint64_t w_1 = rou[w_idx + 2 * i + 1];
            ws[level][i] = _mm512_set_epi64(w_1, w_1, w_1, w_1, w_0, w_0, w_0, w_0);
            w_precon[level][i] =
                _mm512_set_epi64(barrett_factor(w_1, q, shift), barrett_factor(w_1, q, shift),
                                 barrett_factor(w_1, q, shift), barrett_factor(w_1, q, shift),
                                 barrett_factor(w_0, q, shift), barrett_factor(w_0, q, shift),
                                 barrett_factor(w_0, q, shift), barrett_factor(w_0, q, shift));
        }
        w_idx += m;
        level++;
    }
    if (level < logn)
    {
        size_t m = 1ULL << level;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w_0 = rou[w_idx + 4 * i];
            uint64_t w_1 = rou[w_idx + 4 * i + 1];
            uint64_t w_2 = rou[w_idx + 4 * i + 2];
            uint64_t w_3 = rou[w_idx + 4 * i + 3];
            ws[level][i] = _mm512_set_epi64(w_3, w_3, w_2, w_2, w_1, w_1, w_0, w_0);
            w_precon[level][i] =
                _mm512_set_epi64(barrett_factor(w_3, q, shift), barrett_factor(w_3, q, shift),
                                 barrett_factor(w_2, q, shift), barrett_factor(w_2, q, shift),
                                 barrett_factor(w_1, q, shift), barrett_factor(w_1, q, shift),
                                 barrett_factor(w_0, q, shift), barrett_factor(w_0, q, shift));
        }
        w_idx += m;
        level++;
    }
    if (level < logn)
    {
        size_t m = 1ULL << level;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            ws[level][i] = _mm512_set_epi64(rou[w_idx + 8 * i + 7], rou[w_idx + 8 * i + 6],
                                            rou[w_idx + 8 * i + 5], rou[w_idx + 8 * i + 4],
                                            rou[w_idx + 8 * i + 3], rou[w_idx + 8 * i + 2],
                                            rou[w_idx + 8 * i + 1], rou[w_idx + 8 * i + 0]);
            w_precon[level][i] = _mm512_set_epi64(barrett_factor(rou[w_idx + 8 * i + 7], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 6], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 5], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 4], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 3], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 2], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 1], q, shift),
                                                  barrett_factor(rou[w_idx + 8 * i + 0], q, shift));
        }
        w_idx += m;
        level++;
    }
    free(rou);
    *out_ws = ws;
    *out_w_precon = w_precon;
}

void ntt_precompute_inv(uint64_t n, Modulus mod, uint64_t inv_root_of_unity, __m512i ***out_ws,
                        __m512i ***out_w_precon)
{
    const uint64_t q = mod->q;
    size_t logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t shift = (q < (1ULL << 32)) ? 32 : (q < (1ULL << 50)) ? 52 : 64;

    uint64_t *rou = (uint64_t *)malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    uint64_t idx = 0, prev_idx = 0;
    for (size_t i = 1; i < n; i++)
    {
        idx = reverse_bits(i, logn);
        rou[idx] = mul_modq(rou[prev_idx], inv_root_of_unity, mod);
        prev_idx = idx;
    }
    uint64_t *temp = (uint64_t *)malloc(n * sizeof(uint64_t));
    temp[0] = 1;
    idx = 1;
    for (size_t m = (n >> 1); m > 0; m >>= 1)
    {
        for (size_t i = 0; i < m; i++)
        {
            temp[idx++] = rou[m + i];
        }
    }
    __m512i **ws = (__m512i **)malloc(logn * sizeof(__m512i *));
    __m512i **w_precon = (__m512i **)malloc(logn * sizeof(__m512i *));
    size_t level = 0;
    size_t w_idx = 1;
    if (level < logn)
    {
        size_t m = n >> 1;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            ws[level][i] = _mm512_set_epi64(temp[w_idx + 8 * i + 7], temp[w_idx + 8 * i + 6],
                                            temp[w_idx + 8 * i + 5], temp[w_idx + 8 * i + 4],
                                            temp[w_idx + 8 * i + 3], temp[w_idx + 8 * i + 2],
                                            temp[w_idx + 8 * i + 1], temp[w_idx + 8 * i + 0]);
            w_precon[level][i] =
                _mm512_set_epi64(barrett_factor(temp[w_idx + 8 * i + 7], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 6], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 5], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 4], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 3], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 2], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 1], q, shift),
                                 barrett_factor(temp[w_idx + 8 * i + 0], q, shift));
        }
        w_idx += m;
        level++;
    }
    if (level < logn)
    {
        size_t m = n >> 2;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w_0 = temp[w_idx + 4 * i];
            uint64_t w_1 = temp[w_idx + 4 * i + 1];
            uint64_t w_2 = temp[w_idx + 4 * i + 2];
            uint64_t w_3 = temp[w_idx + 4 * i + 3];
            ws[level][i] = _mm512_set_epi64(w_3, w_3, w_2, w_2, w_1, w_1, w_0, w_0);
            w_precon[level][i] =
                _mm512_set_epi64(barrett_factor(w_3, q, shift), barrett_factor(w_3, q, shift),
                                 barrett_factor(w_2, q, shift), barrett_factor(w_2, q, shift),
                                 barrett_factor(w_1, q, shift), barrett_factor(w_1, q, shift),
                                 barrett_factor(w_0, q, shift), barrett_factor(w_0, q, shift));
        }
        w_idx += m;
        level++;
    }
    if (level < logn)
    {
        size_t m = n >> 3;
        size_t num_vectors = n / 16;
        ws[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(num_vectors * sizeof(__m512i), 64);
        for (size_t i = 0; i < num_vectors; i++)
        {
            uint64_t w_0 = temp[w_idx + 2 * i];
            uint64_t w_1 = temp[w_idx + 2 * i + 1];
            ws[level][i] = _mm512_set_epi64(w_1, w_1, w_1, w_1, w_0, w_0, w_0, w_0);
            w_precon[level][i] =
                _mm512_set_epi64(barrett_factor(w_1, q, shift), barrett_factor(w_1, q, shift),
                                 barrett_factor(w_1, q, shift), barrett_factor(w_1, q, shift),
                                 barrett_factor(w_0, q, shift), barrett_factor(w_0, q, shift),
                                 barrett_factor(w_0, q, shift), barrett_factor(w_0, q, shift));
        }
        w_idx += m;
        level++;
    }
    for (; level < logn; level++)
    {
        size_t m = n >> (level + 1);
        ws[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        w_precon[level] = (__m512i *)_mm_malloc(m * sizeof(__m512i), 64);
        for (size_t i = 0; i < m; i++)
        {
            uint64_t w = temp[w_idx + i];
            uint64_t wp = barrett_factor(w, q, shift);
            ws[level][i] = _mm512_set1_epi64(w);
            w_precon[level][i] = _mm512_set1_epi64(wp);
        }
        w_idx += m;
    }
    free(rou);
    free(temp);
    *out_ws = ws;
    *out_w_precon = w_precon;
}

void ntt_free_precompute(__m512i **ws, __m512i **w_precon, uint64_t n)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;
    for (int i = 0; i < logn; i++)
    {
        _mm_free(ws[i]);
        _mm_free(w_precon[i]);
    }
    free(ws);
    free(w_precon);
}

NTT_Plan ntt_new_plan(uint64_t n, Modulus mod)
{
    const uint64_t q = mod->q;

    // Deterministic search for a primitive 2n-th root of unity, identical to the
    // portable kernel's so both builds agree on the transform basis. Raise
    // successive candidates g = 2, 3, 4, ... to the (q-1)/2n power and keep the
    // first whose n-th power is -1 (order exactly 2n).
    uint64_t root_of_unity = 0;
    for (uint64_t g = 2; g < q; g++)
    {
        uint64_t candidate = power_mod(g, (q - 1) / (2 * n), q);
        if (power_mod(candidate, n, q) == q - 1)
        {
            root_of_unity = candidate;
            break;
        }
    }
    uint64_t inv_root_of_unity = inverse_mod(root_of_unity, q);

    NTT_Plan res = (NTT_Plan)malloc(sizeof(struct _NTT_Plan));
    res->mod = mod; // borrowed; ntt_free_plan leaves it alone
    res->n = n;
    res->root_of_unity = root_of_unity;
    res->inv_root_of_unity = inv_root_of_unity;

    if (n < NTT_MIN_VECTOR_LEN)
    {
        // No vectorized stage can run at this length, and the tables below would
        // be empty (they are sized n / 16). Build the scalar ones instead; the
        // transforms and the free path branch on plan->n the same way.
        ntt_scalar_precompute(n, mod, root_of_unity, (uint64_t ***)&res->ws_fwd);
        ntt_scalar_precompute(n, mod, inv_root_of_unity, (uint64_t ***)&res->ws_inv);
        res->w_precon_fwd = NULL;
        res->w_precon_inv = NULL;
        return res;
    }

    ntt_precompute_fwd(n, mod, root_of_unity, (__m512i ***)&res->ws_fwd,
                       (__m512i ***)&res->w_precon_fwd);
    ntt_precompute_inv(n, mod, inv_root_of_unity, (__m512i ***)&res->ws_inv,
                       (__m512i ***)&res->w_precon_inv);
    return res;
}

void ntt_forward(uint64_t *out, uint64_t *in, NTT_Plan plan)
{
    if (plan->n < NTT_MIN_VECTOR_LEN)
    {
        if (out != in)
        {
            for (size_t i = 0; i < plan->n; i++)
            {
                out[i] = in[i];
            }
        }
        ntt_CT_NR_gen(out, ((uint64_t **)plan->ws_fwd)[0], plan);
        return;
    }
    if (plan->mod->q < (1ULL << 32))
    {
        ntt_forward_32(out, in, plan);
    }
    else if (plan->mod->q < (1ULL << 50))
    {
        ntt_forward_50(out, in, plan);
    }
    else
    {
        ntt_forward_64(out, in, plan);
    }
}

void ntt_reverse(uint64_t *out, uint64_t *in, NTT_Plan plan)
{
    if (plan->n < NTT_MIN_VECTOR_LEN)
    {
        if (out != in)
        {
            for (size_t i = 0; i < plan->n; i++)
            {
                out[i] = in[i];
            }
        }
        ntt_GS_RN_gen(out, ((uint64_t **)plan->ws_inv)[0], plan);
        return;
    }
    if (plan->mod->q < (1ULL << 32))
    {
        ntt_reverse_32(out, in, plan);
    }
    else if (plan->mod->q < (1ULL << 50))
    {
        ntt_reverse_50(out, in, plan);
    }
    else
    {
        ntt_reverse_64(out, in, plan);
    }
}

void ntt_free_plan(NTT_Plan plan)
{
    if (plan->n < NTT_MIN_VECTOR_LEN)
    {
        ntt_scalar_free_precompute((uint64_t **)plan->ws_fwd);
        ntt_scalar_free_precompute((uint64_t **)plan->ws_inv);
        free(plan);
        return;
    }
    ntt_free_precompute((__m512i **)plan->ws_fwd, (__m512i **)plan->w_precon_fwd, plan->n);
    ntt_free_precompute((__m512i **)plan->ws_inv, (__m512i **)plan->w_precon_inv, plan->n);
    free(plan);
}

#endif
