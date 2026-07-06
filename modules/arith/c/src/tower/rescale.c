// SPDX-License-Identifier: Apache-2.0
/**
 * @file rescale.c
 * @brief Exact division by limb primes (floor/round) and tower projection.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/nt.h>
#include <arith/tower.h>
#include <arith/zq.h>
#include <base.h>

int poly_div_floor(rns_poly_t p, uint64_t divide_mask)
{
    if (p->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    uint64_t mask = divide_mask & p->mask;
    if (mask == 0)
        return VFHE_OK;

    const uint64_t N = p->ring->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    // Divide one limb prime at a time: for each remaining limb i,
    // x_i <- (x_i - (x mod p)) * p^-1 mod q_i, then drop limb p.
    for (uint64_t idx = 0; idx < p->ring->l; idx++)
    {
        if (mask & (1ULL << idx))
        {
            const uint64_t prime = p->ring->limbs[idx]->q;
            for (uint64_t i = 0; i < p->ring->l; i++)
            {
                if (p->mask & (1ULL << i))
                {
                    if (i == idx)
                        continue;
                    const zq_ctx *zq_i = &p->ring->limbs[i]->zq;
                    const uint64_t inv_p = nt_inverse_mod(prime, zq_i->q);
                    zq_arr_reduce(zq_i, tmp, p->limb[idx], N);
                    zq_arr_sub(zq_i, p->limb[i], p->limb[i], tmp, N);
                    zq_arr_scale(zq_i, p->limb[i], p->limb[i], inv_p, N);
                }
            }
            memset(p->limb[idx], 0, sizeof(uint64_t) * N);
            p->mask &= ~(1ULL << idx);
        }
    }
    free(tmp);
    return VFHE_OK;
}

int poly_div_round(rns_poly_t p, uint64_t divide_mask)
{
    if (p->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    uint64_t mask = divide_mask & p->mask;
    if (mask == 0)
        return VFHE_OK;

    const uint64_t N = p->ring->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    // As poly_div_floor, but bias by p/2 first so the truncation rounds.
    for (uint64_t idx = 0; idx < p->ring->l; idx++)
    {
        if (mask & (1ULL << idx))
        {
            const uint64_t prime = p->ring->limbs[idx]->q, half_p = prime / 2;
            const zq_ctx *zq_idx = &p->ring->limbs[idx]->zq;
            zq_arr_add_scalar(zq_idx, p->limb[idx], p->limb[idx], half_p, N);
            for (uint64_t i = 0; i < p->ring->l; i++)
            {
                if (p->mask & (1ULL << i))
                {
                    if (i == idx)
                        continue;
                    const zq_ctx *zq_i = &p->ring->limbs[i]->zq;
                    const uint64_t inv_p = nt_inverse_mod(prime, zq_i->q);
                    const uint64_t half_p_mod_q = half_p % zq_i->q;
                    zq_arr_reduce(zq_i, tmp, p->limb[idx], N);
                    zq_arr_add_scalar(zq_i, p->limb[i], p->limb[i], half_p_mod_q, N);
                    zq_arr_sub(zq_i, p->limb[i], p->limb[i], tmp, N);
                    zq_arr_scale(zq_i, p->limb[i], p->limb[i], inv_p, N);
                }
            }
            memset(p->limb[idx], 0, sizeof(uint64_t) * N);
            p->mask &= ~(1ULL << idx);
        }
    }
    free(tmp);
    return VFHE_OK;
}

int poly_div_floor_last(rns_poly_t p)
{
    int last_active = limbset_last(p->mask);
    if (last_active < 0)
        return VFHE_OK;
    return poly_div_floor(p, 1ULL << last_active);
}

int poly_div_round_last(rns_poly_t p)
{
    int last_active = limbset_last(p->mask);
    if (last_active < 0)
        return VFHE_OK;
    return poly_div_round(p, 1ULL << last_active);
}

int poly_mod_reduce(rns_poly_t out, const rns_poly_t in)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            memcpy(out->limb[i], in->limb[i], out->ring->N * sizeof(uint64_t));
        }
    }
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}

int poly_lift_residue(rns_poly_t out, const rns_poly_t in, uint64_t src_limb)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            if (i == src_limb)
            {
                // The row already holds residues < q_src; reducing mod the same
                // prime is the identity.
                if (out->limb[i] != in->limb[src_limb])
                    memcpy(out->limb[i], in->limb[src_limb], out->ring->N * sizeof(uint64_t));
            }
            else
            {
                zq_arr_reduce(&out->ring->limbs[i]->zq, out->limb[i], in->limb[src_limb],
                              out->ring->N);
            }
        }
    }
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}
