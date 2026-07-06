// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_galois.c
 * @brief Coefficient-domain ring maps: Galois automorphisms and negacyclic
 *        monomial shifts.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/poly.h>
#include <arith/zq.h>
#include <base.h>

int poly_permute(rns_poly_t out, const rns_poly_t in, uint64_t gen)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    if (out == in)
        return VFHE_ERR_ARG;
    const uint64_t N = out->ring->N;
    if (gen == 0 || gen >= 2 * N)
        return VFHE_ERR_ARG;

    const uint64_t mod_mask = N - 1;
    const uint64_t split_degree = out->ring->split_degree;
    const uint64_t split_degree_mod = split_degree - 1;
    int split_degree_log = 0;
    while ((1ULL << split_degree_log) < split_degree)
        split_degree_log++;
    const uint64_t ps = out->ring->poly_size;

    int64_t *temp_signed = (int64_t *)safe_aligned_malloc(N * sizeof(int64_t));
    out->mask = in->mask;
    out->domain = VFHE_COEFF;
    for (uint64_t j = 0; j < out->ring->l; j++)
    {
        if (out->mask & (1ULL << j))
        {
            // X^i -> X^(i*gen); the sign flips whenever i*gen wraps an odd number
            // of times around X^N = -1 (bit N of the product index).
            for (uint64_t i = 0; i < split_degree; i++)
            {
                for (uint64_t i2 = 0; i2 < ps; i2++)
                {
                    const uint64_t idx = (i + (i2 << split_degree_log)) * gen;
                    const uint64_t dst =
                        (idx & split_degree_mod) * ps + ((idx & mod_mask) >> split_degree_log);
                    int64_t val = (int64_t)in->limb[j][i * ps + i2];
                    temp_signed[dst] = (idx & N) ? -val : val;
                }
            }
            zq_arr_reduce_signed(&out->ring->limbs[j]->zq, out->limb[j], temp_signed, N);
        }
        else
        {
            memset(out->limb[j], 0, sizeof(uint64_t) * N);
        }
    }
    free(temp_signed);
    return VFHE_OK;
}

int poly_mul_xai(rns_poly_t out, const rns_poly_t in, uint64_t a)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    if (out == in)
        return VFHE_ERR_ARG;
    if (out->ring->split_degree != 1)
        return VFHE_ERR_UNSUPPORTED;
    const uint64_t N = out->ring->N;
    a &= ((N << 1) - 1); // a mod 2N; X^(a+N) = -X^a
    if (a == 0)
    {
        poly_copy(out, in);
        return VFHE_OK;
    }
    out->mask = in->mask;
    out->domain = VFHE_COEFF;
    for (uint64_t j = 0; j < out->ring->l; j++)
    {
        if (!(out->mask & (1ULL << j)))
            continue;
        const zq_ctx *zq = &in->ring->limbs[j]->zq;
        const uint64_t q = zq->q;
        if (a < N)
        {
            if (a % 8 == 0)
            {
                // Segment lengths are multiples of 8: vector kernels apply.
                zq_arr_negate(zq, out->limb[j], in->limb[j] + N - a, a);
                memcpy(out->limb[j] + a, in->limb[j], (N - a) * sizeof(uint64_t));
            }
            else
            {
                for (uint64_t i = 0; i < a; i++)
                {
                    out->limb[j][i] = zq_scalar_negate(in->limb[j][i - a + N], q);
                }
                for (uint64_t i = a; i < N; i++)
                {
                    out->limb[j][i] = in->limb[j][i - a];
                }
            }
        }
        else
        {
            if (a % 8 == 0)
            {
                memcpy(out->limb[j], in->limb[j] + 2 * N - a, (a - N) * sizeof(uint64_t));
                zq_arr_negate(zq, out->limb[j] + a - N, in->limb[j], 2 * N - a);
            }
            else
            {
                for (uint64_t i = 0; i < a - N; i++)
                {
                    out->limb[j][i] = in->limb[j][i - a + 2 * N];
                }
                for (uint64_t i = a - N; i < N; i++)
                {
                    out->limb[j][i] = zq_scalar_negate(in->limb[j][i - a + N], q);
                }
            }
        }
    }
    return VFHE_OK;
}

int poly_mul_xai_minus1(rns_poly_t out, const rns_poly_t in, uint64_t a)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    if (out == in)
        return VFHE_ERR_ARG;
    if (out->ring->split_degree != 1)
        return VFHE_ERR_UNSUPPORTED;
    const uint64_t N = out->ring->N;
    a &= ((N << 1) - 1);
    if (a == 0)
    {
        // X^0 - 1 = 0.
        out->mask = in->mask;
        out->domain = VFHE_COEFF;
        for (uint64_t j = 0; j < out->ring->l; j++)
        {
            memset(out->limb[j], 0, sizeof(uint64_t) * N);
        }
        return VFHE_OK;
    }
    out->mask = in->mask;
    out->domain = VFHE_COEFF;
    for (uint64_t j = 0; j < out->ring->l; j++)
    {
        if (!(out->mask & (1ULL << j)))
            continue;
        const zq_ctx *zq = &in->ring->limbs[j]->zq;
        const uint64_t q = zq->q;
        if (a < N)
        {
            if (a % 8 == 0)
            {
                zq_arr_negate(zq, out->limb[j], in->limb[j] + N - a, a);
                zq_arr_sub(zq, out->limb[j], out->limb[j], in->limb[j], a);
                zq_arr_sub(zq, out->limb[j] + a, in->limb[j], in->limb[j] + a, N - a);
            }
            else
            {
                for (uint64_t i = 0; i < a; i++)
                {
                    uint64_t term1 = zq_scalar_negate(in->limb[j][i - a + N], q);
                    out->limb[j][i] = zq_scalar_sub(term1, in->limb[j][i], q);
                }
                for (uint64_t i = a; i < N; i++)
                {
                    out->limb[j][i] = zq_scalar_sub(in->limb[j][i - a], in->limb[j][i], q);
                }
            }
        }
        else
        {
            if (a % 8 == 0)
            {
                zq_arr_sub(zq, out->limb[j], in->limb[j] + 2 * N - a, in->limb[j], a - N);
                zq_arr_negate(zq, out->limb[j] + a - N, in->limb[j], 2 * N - a);
                zq_arr_sub(zq, out->limb[j] + a - N, out->limb[j] + a - N, in->limb[j] + a - N,
                           2 * N - a);
            }
            else
            {
                for (uint64_t i = 0; i < a - N; i++)
                {
                    out->limb[j][i] = zq_scalar_sub(in->limb[j][i - a + 2 * N], in->limb[j][i], q);
                }
                for (uint64_t i = a - N; i < N; i++)
                {
                    uint64_t term1 = zq_scalar_negate(in->limb[j][i - a + N], q);
                    out->limb[j][i] = zq_scalar_sub(term1, in->limb[j][i], q);
                }
            }
        }
    }
    return VFHE_OK;
}
