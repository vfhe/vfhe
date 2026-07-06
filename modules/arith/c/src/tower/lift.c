// SPDX-License-Identifier: Apache-2.0
/**
 * @file lift.c
 * @brief Scaled lifting into a larger limb set (modulus-ratio embedding).
 */
#include <string.h>

#include <arith/error.h>
#include <arith/tower.h>
#include <arith/zq.h>

void tower_scaling_factors(uint64_t *delta_out, const ring_ctx *ring, uint64_t in_mask,
                           uint64_t out_mask)
{
    uint64_t diff_mask = out_mask & ~in_mask;
    for (uint64_t i = 0; i < ring->l; i++)
    {
        if (out_mask & (1ULL << i))
        {
            const zq_ctx *zq_i = &ring->limbs[i]->zq;
            if (in_mask & (1ULL << i))
            {
                // delta = prod of the newly added primes, reduced mod q_i.
                uint64_t delta_i = 1;
                for (uint64_t j = 0; j < ring->l; j++)
                {
                    if (diff_mask & (1ULL << j))
                    {
                        uint64_t p_j = ring->limbs[j]->q;
                        delta_i = zq_scalar_mul(delta_i, zq_reduce_u128(p_j, zq_i), zq_i);
                    }
                }
                delta_out[i] = delta_i;
            }
            else
            {
                delta_out[i] = 0;
            }
        }
        else
        {
            delta_out[i] = 0;
        }
    }
}

int poly_scaled_lift(rns_poly_t out, const rns_poly_t in, const uint64_t *delta)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    const uint64_t N = out->ring->N;

    uint64_t local_delta[64];
    const uint64_t *actual_delta = delta;
    if (actual_delta == NULL)
    {
        tower_scaling_factors(local_delta, out->ring, in->mask, out->mask);
        actual_delta = local_delta;
    }

    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            const zq_ctx *zq_i = &out->ring->limbs[i]->zq;
            if (in->mask & (1ULL << i))
            {
                zq_arr_scale(zq_i, out->limb[i], in->limb[i], actual_delta[i], N);
            }
            else
            {
                // v * Delta is divisible by every new prime: those rows are zero.
                memset(out->limb[i], 0, N * sizeof(uint64_t));
            }
        }
    }
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}
