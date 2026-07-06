// SPDX-License-Identifier: Apache-2.0
/**
 * @file baseconv.c
 * @brief Fast RNS base extension (plans + per-polynomial conversion).
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/nt.h>
#include <arith/tower.h>
#include <arith/zq.h>
#include <base.h>

baseconv_plan_t baseconv_plan_new(const ring_ctx *ring, uint64_t in_mask, uint64_t out_mask)
{
    baseconv_plan_t plan = (baseconv_plan_t)safe_malloc(sizeof(*plan));
    plan->in_mask = in_mask;
    plan->out_mask = out_mask;

    plan->D = (uint32_t *)safe_malloc(sizeof(uint32_t) * ring->l);
    plan->P = (uint32_t *)safe_malloc(sizeof(uint32_t) * ring->l);
    plan->w = 0;
    plan->v = 0;

    for (uint64_t i = 0; i < ring->l; i++)
    {
        if (in_mask & (1ULL << i))
        {
            plan->D[plan->w++] = (uint32_t)i;
        }
    }

    for (uint64_t i = 0; i < ring->l; i++)
    {
        if ((out_mask & (1ULL << i)) && !(in_mask & (1ULL << i)))
        {
            plan->P[plan->v++] = (uint32_t)i;
        }
    }

    if (plan->v == 0 || plan->w == 0)
    {
        // Nothing to extend into (or from); conversion degenerates to a copy.
        plan->Dhat = NULL;
        plan->D_mod_p = NULL;
        return plan;
    }

    // Dhat[j] = (prod_{k != j} q_k)^-1 mod q_j -- the CRT interpolation weight
    // of source limb j.
    plan->Dhat = (uint64_t *)safe_malloc(sizeof(uint64_t) * plan->w);
    for (uint32_t j = 0; j < plan->w; j++)
    {
        uint64_t idx_j = plan->D[j];
        const zq_ctx *zq_j = &ring->limbs[idx_j]->zq;
        uint64_t prod = 1;
        for (uint32_t k = 0; k < plan->w; k++)
        {
            if (k == j)
                continue;
            uint64_t q_k = ring->limbs[plan->D[k]]->q;
            prod = zq_scalar_mul(prod, zq_reduce_u128(q_k, zq_j), zq_j);
        }
        plan->Dhat[j] = nt_inverse_mod(prod, zq_j->q);
    }

    // D_mod_p[i][j] = (prod_{k != j} q_k) mod p_i -- source limb j's CRT basis
    // element evaluated in target prime p_i.
    plan->D_mod_p = (uint64_t **)safe_malloc(sizeof(uint64_t *) * plan->v);
    for (uint32_t i = 0; i < plan->v; i++)
    {
        plan->D_mod_p[i] = (uint64_t *)safe_malloc(sizeof(uint64_t) * plan->w);
        const zq_ctx *zq_i = &ring->limbs[plan->P[i]]->zq;
        for (uint32_t j = 0; j < plan->w; j++)
        {
            uint64_t prod = 1;
            for (uint32_t k = 0; k < plan->w; k++)
            {
                if (k == j)
                    continue;
                uint64_t q_k = ring->limbs[plan->D[k]]->q;
                prod = zq_scalar_mul(prod, zq_reduce_u128(q_k, zq_i), zq_i);
            }
            plan->D_mod_p[i][j] = prod;
        }
    }

    return plan;
}

void baseconv_plan_free(baseconv_plan_t plan)
{
    if (plan == NULL)
        return;
    if (plan->D_mod_p != NULL)
    {
        for (uint32_t i = 0; i < plan->v; i++)
        {
            free(plan->D_mod_p[i]);
        }
        free(plan->D_mod_p);
    }
    free(plan->Dhat);
    free(plan->D);
    free(plan->P);
    free(plan);
}

int poly_base_convert(rns_poly_t out, const rns_poly_t in, const baseconv_plan_t plan)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;

    const uint64_t N = out->ring->N;
    uint64_t in_mask = in->mask;
    uint64_t out_mask = out->mask;

    // Shared limbs pass through unchanged.
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if ((in_mask & (1ULL << i)) && (out_mask & (1ULL << i)))
        {
            if (out != in)
            {
                memcpy(out->limb[i], in->limb[i], sizeof(uint64_t) * N);
            }
        }
    }

    baseconv_plan_t local = plan;
    if (local == NULL)
    {
        local = baseconv_plan_new(in->ring, in_mask, out_mask);
    }
    else if (local->in_mask != in_mask || local->out_mask != out_mask)
    {
        return VFHE_ERR_ARG;
    }

    const uint32_t w = local->w;
    const uint32_t v = local->v;

    if (v == 0)
    {
        out->mask = out_mask;
        out->domain = VFHE_COEFF;
        if (plan == NULL)
            baseconv_plan_free(local);
        return VFHE_OK;
    }

    for (uint32_t i = 0; i < v; i++)
    {
        memset(out->limb[local->P[i]], 0, sizeof(uint64_t) * N);
    }

    // Fast base extension: out_p = sum_j [ (x_j * Dhat_j mod q_j) * (D/q_j) ]
    // mod p, for every new target prime p.
    uint64_t *v_tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    uint64_t *v_tmp2 = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    for (uint32_t j = 0; j < w; j++)
    {
        uint64_t idx_j = local->D[j];
        const zq_ctx *zq_j = &in->ring->limbs[idx_j]->zq;
        zq_arr_scale(zq_j, v_tmp, in->limb[idx_j], local->Dhat[j], N);

        for (uint32_t i = 0; i < v; i++)
        {
            uint64_t idx_i = local->P[i];
            const zq_ctx *zq_i = &out->ring->limbs[idx_i]->zq;
            zq_arr_reduce(zq_i, v_tmp2, v_tmp, N);
            zq_arr_scale_addto(zq_i, out->limb[idx_i], v_tmp2, local->D_mod_p[i][j], N);
        }
    }

    free(v_tmp);
    free(v_tmp2);

    if (plan == NULL)
        baseconv_plan_free(local);

    out->mask = out_mask;
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}
