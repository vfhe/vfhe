// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_portable.c
 * @brief Portable scalar NTT backend (any prime, any target).
 *
 * Textbook Cooley-Tukey forward / Gentleman-Sande inverse over the raw
 * bit-reversed twiddle schedules; reduction goes through the scalar
 * ::zq_reduce_u128. The inverse folds in the 1/n factor with a final scaling
 * pass. Compiled on every build; it is the selected backend when
 * ::VFHE_MP_SIMD is 0.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/nt.h>
#include <base.h>

#include "ntt_backends.h"

static void fwd_ct_nr(uint64_t *a, uint64_t n, uint64_t q, const uint64_t *ws, const zq_ctx *zq)
{
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
        for (size_t i = 0; i < m; i++)
        {
            size_t j1 = 2 * i * t;
            size_t j2 = j1 + t;
            uint64_t w = ws[m + i];
            for (size_t j = j1; j < j2; j++)
            {
                uint64_t u = a[j];
                uint64_t v = zq_reduce_u128((unsigned __int128)a[j + t] * w, zq);
                a[j] = u + v;
                if (a[j] >= q)
                    a[j] -= q;
                a[j + t] = u + q - v;
                if (a[j + t] >= q)
                    a[j + t] -= q;
            }
        }
    }
}

static void inv_gs_rn(uint64_t *a, uint64_t n, uint64_t q, const uint64_t *ws, const zq_ctx *zq)
{
    size_t t = 1;
    for (size_t m = n; m > 1; m >>= 1)
    {
        size_t h = m >> 1;
        for (size_t i = 0; i < h; i++)
        {
            size_t j1 = 2 * i * t;
            size_t j2 = j1 + t;
            uint64_t w = ws[h + i];
            for (size_t j = j1; j < j2; j++)
            {
                uint64_t u = a[j];
                uint64_t v = a[j + t];
                a[j] = u + v;
                if (a[j] >= q)
                    a[j] -= q;
                uint64_t diff = u + q - v;
                if (diff >= q)
                    diff -= q;
                a[j + t] = zq_reduce_u128((unsigned __int128)diff * w, zq);
            }
        }
        t <<= 1;
    }

    // Fold in the 1/n factor.
    uint64_t inv_n = nt_inverse_mod(n, q);
    for (size_t i = 0; i < n; i++)
    {
        a[i] = zq_reduce_u128((unsigned __int128)a[i] * inv_n, zq);
    }
}

static void portable_forward(uint64_t *out, const uint64_t *in, const ntt_plan *plan)
{
    if (out != in)
        memcpy(out, in, plan->n * sizeof(uint64_t));
    fwd_ct_nr(out, plan->n, plan->zq->q, (const uint64_t *)plan->tw_fwd[0], plan->zq);
}

static void portable_inverse(uint64_t *out, const uint64_t *in, const ntt_plan *plan)
{
    if (out != in)
        memcpy(out, in, plan->n * sizeof(uint64_t));
    inv_gs_rn(out, plan->n, plan->zq->q, (const uint64_t *)plan->tw_inv[0], plan->zq);
}

static void portable_free_tables(ntt_plan *plan)
{
    if (plan->tw_fwd)
    {
        free(plan->tw_fwd[0]);
        free(plan->tw_fwd);
    }
    if (plan->tw_inv)
    {
        free(plan->tw_inv[0]);
        free(plan->tw_inv);
    }
}

void ntt_backend_portable_init(ntt_plan *plan)
{
    const uint64_t q = plan->zq->q;

    // Both schedules are the raw bit-reversed power tables (both passes index
    // them as ws[stage + i]); wrap each in a one-slot table handle.
    uint64_t *fwd = ntt_tables_bitrev_powers(plan->n, q, plan->root);
    uint64_t *inv = ntt_tables_bitrev_powers(plan->n, q, plan->inv_root);

    plan->tw_fwd = (void **)safe_malloc(sizeof(void *));
    plan->tw_fwd[0] = fwd;
    plan->tw_inv = (void **)safe_malloc(sizeof(void *));
    plan->tw_inv[0] = inv;
    plan->tw_fwd_pre = NULL;
    plan->tw_inv_pre = NULL;

    plan->forward = portable_forward;
    plan->inverse = portable_inverse;
    plan->free_tables = portable_free_tables;
}
