// SPDX-License-Identifier: Apache-2.0
/**
 * @file ring.c
 * @brief Ring context construction and pool management (see arith/ring.h).
 */
#include <math.h>
#include <stdlib.h>

#include <arith/error.h>
#include <arith/ring.h>
#include <base.h>

#include "ring_internal.h"

/** Build one limb (zq ctx + NTT plan + twist table) for prime q. */
static ring_limb *limb_new(uint64_t q, uint64_t poly_size, uint64_t log_poly_size)
{
    ring_limb *limb = (ring_limb *)safe_malloc(sizeof(*limb));
    limb->q = q;
    zq_ctx_init(&limb->zq, q);
    ntt_plan_init(&limb->ntt, poly_size, &limb->zq);

    // Twist table: powers root^1 .. root^(2*poly_size) of the plan's 2n-th
    // root, bit-reversed over log2(poly_size)+1 bits and truncated to the
    // first poly_size entries -- the schedule the split-degree cross-term
    // kernel multiplies wrapped blocks by.
    uint64_t *w_p = (uint64_t *)safe_malloc(2 * poly_size * sizeof(uint64_t));
    limb->twist = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
    w_p[0] = limb->ntt.root;
    for (uint64_t j = 1; j < 2 * poly_size; j++)
    {
        w_p[j] = zq_scalar_mul(w_p[j - 1], limb->ntt.root, &limb->zq);
    }
    bit_rev(limb->twist, w_p, poly_size, log_poly_size + 1);
    free(w_p);
    return limb;
}

static void limb_free(ring_limb *limb)
{
    ntt_plan_clear(&limb->ntt);
    free(limb->twist);
    free(limb);
}

ring_t ring_new(const uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l)
{
    if (l == 0 || l > 64 || split_degree == 0 || N % split_degree != 0)
        return NULL;

    ring_ctx *r = (ring_ctx *)safe_malloc(sizeof(*r));
    r->N = N;
    r->split_degree = split_degree;
    r->poly_size = N / split_degree;
    r->l = l;
    r->limbs = (ring_limb **)safe_malloc(l * sizeof(ring_limb *));

    const uint64_t log_poly_size = (uint64_t)log2((double)r->poly_size);
    for (uint64_t i = 0; i < l; i++)
    {
        r->limbs[i] = limb_new(primes[i], r->poly_size, log_poly_size);
    }

    r->mul_limb = (split_degree == 1) ? ring_mul_split1 : ring_mul_splitk;
    return r;
}

int ring_extend(ring_t r, const uint64_t *primes, uint64_t count)
{
    if (count == 0)
        return VFHE_OK;
    if (r->l + count > 64)
        return VFHE_ERR_ARG;

    uint64_t new_l = r->l + count;
    // Only the pointer array moves; every existing limb keeps its address.
    r->limbs = (ring_limb **)realloc(r->limbs, new_l * sizeof(ring_limb *));

    const uint64_t log_poly_size = (uint64_t)log2((double)r->poly_size);
    for (uint64_t i = r->l; i < new_l; i++)
    {
        r->limbs[i] = limb_new(primes[i - r->l], r->poly_size, log_poly_size);
    }
    r->l = new_l;
    return VFHE_OK;
}

void ring_free(ring_t r)
{
    if (r == NULL)
        return;
    for (uint64_t i = 0; i < r->l; i++)
        limb_free(r->limbs[i]);
    free(r->limbs);
    free(r);
}

uint64_t ring_level_count(const ring_ctx *r) { return r->l; }

const ring_limb *ring_limb_at(const ring_ctx *r, uint64_t i) { return r->limbs[i]; }

const uint64_t *ring_twist(const ring_ctx *r, uint64_t i) { return r->limbs[i]->twist; }
