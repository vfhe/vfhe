// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// Size-generic scalar NTT, compiled into *every* engine.
//
// They are the whole implementation on the portable engine, and on the
// vectorized ones they serve transform lengths below NTT_MIN_VECTOR_LEN, which
// the AVX512 transforms cannot handle: those need two lane groups per butterfly
// stage and size their twiddle tables `n / 16`.
//
// Twiddle layout: one flat table of `n` powers of the root of unity in
// bit-reversed order, indexed `ws[m + i]` -- the standard Cooley-Tukey
// (natural-to-reversed) forward and Gentleman-Sande (reversed-to-natural)
// inverse pair, which is the same transform basis the vectorized kernels use.
//
// Both kernels take their input reduced to [0, q) and leave it reduced.
#include <arith.h>
#include "arith_internal.h"

static uint64_t reverse_bits_gen(uint64_t x, int bits)
{
    uint64_t res = 0;
    for (int i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

void ntt_scalar_precompute(uint64_t n, Modulus mod, uint64_t root_of_unity, uint64_t ***out_ws)
{
    int logn = 0;
    while ((1ULL << logn) < n)
        logn++;

    uint64_t *rou = (uint64_t *)malloc(n * sizeof(uint64_t));
    rou[0] = 1;
    uint64_t idx = 0, prev_idx = 0;
    for (size_t i = 1; i < n; i++)
    {
        idx = reverse_bits_gen(i, logn);
        rou[idx] = mul_modq(rou[prev_idx], root_of_unity, mod);
        prev_idx = idx;
    }

    uint64_t **ws = (uint64_t **)malloc(1 * sizeof(uint64_t *));
    ws[0] = rou;
    *out_ws = ws;
}

void ntt_scalar_free_precompute(uint64_t **ws)
{
    if (ws)
    {
        free(ws[0]);
        free(ws);
    }
}

void ntt_CT_NR_gen(uint64_t *a, uint64_t *ws, NTT_Plan plan)
{
    const Modulus mod = plan->mod;
    const uint64_t n = plan->n, q = mod->q;
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
        for (size_t i = 0; i < m; i++)
        {
            const size_t j1 = 2 * i * t, j2 = j1 + t;
            const uint64_t w = ws[m + i];
            for (size_t j = j1; j < j2; j++)
            {
                const uint64_t u = a[j];
                const uint64_t v = mul_modq(a[j + t], w, mod);
                a[j] = add_modq(u, v, q);
                a[j + t] = sub_modq(u, v, q);
            }
        }
    }
}

void ntt_GS_RN_gen(uint64_t *a, uint64_t *ws, NTT_Plan plan)
{
    const Modulus mod = plan->mod;
    const uint64_t n = plan->n, q = mod->q;
    size_t t = 1;
    for (size_t m = n; m > 1; m >>= 1)
    {
        const size_t h = m >> 1;
        for (size_t i = 0; i < h; i++)
        {
            const size_t j1 = 2 * i * t, j2 = j1 + t;
            const uint64_t w = ws[h + i];
            for (size_t j = j1; j < j2; j++)
            {
                const uint64_t u = a[j], v = a[j + t];
                a[j] = add_modq(u, v, q);
                a[j + t] = mul_modq(sub_modq(u, v, q), w, mod);
            }
        }
        t <<= 1;
    }

    // Scale by n^-1. Extended-Euclid rather than a Modulus operation, and once
    // per transform rather than per coefficient.
    const uint64_t inv_n = inverse_mod(n, q);
    for (size_t i = 0; i < n; i++)
        a[i] = mul_modq(a[i], inv_n, mod);
}
