// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_sample.c
 * @brief Randomized polynomial generation (uniform / Gaussian) and noise
 *        addition. The only poly-layer file that touches the rng module.
 */
#include <math.h>
#include <stdlib.h>

#include <arith/error.h>
#include <arith/poly.h>
#include <arith/zq.h>
#include <base.h>
#include <rng.h>

void poly_sample_uniform(rns_poly_t p, vfhe_domain domain)
{
    for (uint64_t i = 0; i < p->ring->l; i++)
    {
        if (p->mask & (1ULL << i))
        {
            const uint64_t q = p->ring->limbs[i]->q;
            rng_random_bytes(sizeof(uint64_t) * p->ring->N, (uint8_t *)p->limb[i]);
            // Rejection-free: fold the uniform 2^k window onto [0, q) by rescaling.
            array_mod_switch_from_2k(p->limb[i], p->limb[i], q, q, p->ring->N);
        }
    }
    p->domain = domain;
}

void poly_sample_gaussian(rns_poly_t p, double sigma)
{
    const uint64_t N = p->ring->N;
    int64_t *noise = (int64_t *)safe_aligned_malloc(N * sizeof(int64_t));
    // One rounded normal per coefficient, shared across limbs (the same
    // integer, reduced into each prime).
    for (uint64_t j = 0; j < N; j++)
    {
        noise[j] = (int64_t)round(rng_gaussian(sigma));
    }
    for (uint64_t i = 0; i < p->ring->l; i++)
    {
        if (p->mask & (1ULL << i))
        {
            zq_arr_reduce_signed(&p->ring->limbs[i]->zq, p->limb[i], noise, N);
        }
    }
    free(noise);
    p->domain = VFHE_COEFF;
}

int poly_add_noise(rns_poly_t out, const rns_poly_t in, double sigma)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    const uint64_t N = out->ring->N;
    int64_t *noise = (int64_t *)safe_aligned_malloc(N * sizeof(int64_t));
    uint64_t *noise_reduced = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    for (uint64_t j = 0; j < N; j++)
    {
        noise[j] = (int64_t)round(rng_gaussian(sigma));
    }
    out->mask = in->mask;
    out->domain = VFHE_COEFF;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            const zq_ctx *zq = &out->ring->limbs[i]->zq;
            zq_arr_reduce_signed(zq, noise_reduced, noise, N);
            zq_arr_add(zq, out->limb[i], in->limb[i], noise_reduced, N);
        }
    }
    free(noise_reduced);
    free(noise);
    return VFHE_OK;
}
