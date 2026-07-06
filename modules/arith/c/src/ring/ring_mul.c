// SPDX-License-Identifier: Apache-2.0
/**
 * @file ring_mul.c
 * @brief Negacyclic multiplication strategies (one kernel, three accumulation
 *        modes -- see ::vfhe_acc).
 *
 * This is the single home of the split-degree cross-term logic. The former
 * mul / mul_addto / mul_subto near-triplicates collapse into one parametrized
 * routine: the accumulation mode selects the fused element-wise kernel for
 * the aligned block products and the combine direction for the twisted
 * (wrapped) cross terms.
 */
#include <string.h>

#include <arith/zq.h>

#include "ring_internal.h"

void ring_mul_split1(uint64_t *out, const uint64_t *a, const uint64_t *b, const ring_limb *limb,
                     const ring_ctx *r, vfhe_acc acc, uint64_t *scratch)
{
    (void)scratch;
    const zq_ctx *zq = &limb->zq;
    switch (acc)
    {
    case VFHE_ACC_SET:
        zq_arr_mul(zq, out, a, b, r->N);
        break;
    case VFHE_ACC_ADD:
        zq_arr_mul_addto(zq, out, a, b, r->N);
        break;
    case VFHE_ACC_SUB:
        zq_arr_mul_subto(zq, out, a, b, r->N);
        break;
    }
}

void ring_mul_splitk(uint64_t *out, const uint64_t *a, const uint64_t *b, const ring_limb *limb,
                     const ring_ctx *r, vfhe_acc acc, uint64_t *scratch)
{
    const zq_ctx *zq = &limb->zq;
    const uint64_t sd = r->split_degree;
    const uint64_t ps = r->poly_size;

    // SET starts from a zeroed row and then accumulates like ADD.
    if (acc == VFHE_ACC_SET)
    {
        memset(out, 0, r->N * sizeof(uint64_t));
        acc = VFHE_ACC_ADD;
    }

    for (uint64_t j = 0; j < sd; j++)
    {
        const uint64_t *a_j = &a[j * ps];

        // Aligned products: block j x block k lands in block j + k. Fused
        // multiply-accumulate, one pass over memory.
        for (uint64_t k = 0; k < sd - j; k++)
        {
            uint64_t *dst = &out[(j + k) * ps];
            if (acc == VFHE_ACC_ADD)
            {
                zq_arr_mul_addto(zq, dst, a_j, &b[k * ps], ps);
            }
            else
            {
                zq_arr_mul_subto(zq, dst, a_j, &b[k * ps], ps);
            }
        }

        // Wrapped products: block j + k - sd, folded back through X^N = -1 via
        // the per-slot twist table.
        for (uint64_t k = sd - j; k < sd; k++)
        {
            uint64_t *dst = &out[(j + k - sd) * ps];
            zq_arr_mul(zq, scratch, a_j, &b[k * ps], ps);
            zq_arr_mul(zq, scratch, scratch, limb->twist, ps);
            if (acc == VFHE_ACC_ADD)
            {
                zq_arr_add(zq, dst, dst, scratch, ps);
            }
            else
            {
                zq_arr_sub(zq, dst, dst, scratch, ps);
            }
        }
    }
}
