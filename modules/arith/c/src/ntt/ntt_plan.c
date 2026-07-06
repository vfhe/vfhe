// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_plan.c
 * @brief Plan construction: root search + backend selection (see arith/ntt.h).
 */
#include <stddef.h>

#include <arith/error.h>
#include <arith/nt.h>

#include "ntt_backends.h"

int ntt_plan_init(ntt_plan *plan, uint64_t n, const zq_ctx *zq)
{
    if (n == 0 || (n & (n - 1)) != 0)
        return VFHE_ERR_ARG;

    plan->n = n;
    uint32_t log_n = 0;
    while ((1ULL << log_n) < n)
        log_n++;
    plan->log_n = log_n;
    plan->zq = zq;

    // Deterministic primitive 2n-th root: the negacyclic transform needs
    // w^n == -1 mod q, i.e. a primitive root of the cyclic group of order 2n.
    plan->root = nt_gen_root_of_unity(zq->q, 2 * n);
    plan->inv_root = nt_inverse_mod(plan->root, zq->q);

#if VFHE_MP_SIMD
    if (zq->q < (1ULL << 32))
    {
        ntt_backend_avx512_32_init(plan);
    }
    else if (zq->q < (1ULL << 50))
    {
        ntt_backend_avx512_50_init(plan);
    }
    else
    {
        ntt_backend_avx512_64_init(plan);
    }
#else
    ntt_backend_portable_init(plan);
#endif

    return VFHE_OK;
}

void ntt_plan_clear(ntt_plan *plan)
{
    if (plan->free_tables)
        plan->free_tables(plan);
    plan->tw_fwd = plan->tw_fwd_pre = plan->tw_inv = plan->tw_inv_pre = NULL;
    plan->forward = plan->inverse = NULL;
    plan->free_tables = NULL;
}
