// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_mul.c
 * @brief Polynomial multiplication entry points.
 *
 * All variants funnel into one helper that walks the active limbs and
 * delegates to the ring's multiplication strategy with the requested
 * accumulation mode -- the strategy (pointwise vs. split-degree schoolbook)
 * was fixed when the ring was created.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/poly.h>
#include <base.h>

/** Shared body of mul / mul_addto / mul_subto. */
static int mul_acc(rns_poly_t out, const rns_poly_t a, const rns_poly_t b, vfhe_acc acc)
{
    if (out == a || out == b)
        return VFHE_ERR_ARG;
    if (a->domain != VFHE_EVAL || b->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;
    if (acc != VFHE_ACC_SET && out->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;

    const ring_ctx *r = out->ring;
    out->mask = a->mask & b->mask;
    out->domain = VFHE_EVAL;

    uint64_t *scratch = (r->split_degree > 1)
                            ? (uint64_t *)safe_aligned_malloc(r->poly_size * sizeof(uint64_t))
                            : NULL;

    for (uint64_t i = 0; i < r->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            r->mul_limb(out->limb[i], a->limb[i], b->limb[i], r->limbs[i], r, acc, scratch);
        }
    }

    free(scratch);
    return VFHE_OK;
}

int poly_mul(rns_poly_t out, const rns_poly_t a, const rns_poly_t b)
{
    return mul_acc(out, a, b, VFHE_ACC_SET);
}

int poly_mul_addto(rns_poly_t out, const rns_poly_t a, const rns_poly_t b)
{
    return mul_acc(out, a, b, VFHE_ACC_ADD);
}

int poly_mul_subto(rns_poly_t out, const rns_poly_t a, const rns_poly_t b)
{
    return mul_acc(out, a, b, VFHE_ACC_SUB);
}

int poly_mul_into(rns_poly_t out, const rns_poly_t a)
{
    if (out->domain != VFHE_EVAL || a->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;

    const ring_ctx *r = out->ring;
    out->mask = out->mask & a->mask;

    // In-place: stage the current row aside, then run the strategy in SET mode.
    uint64_t *row_copy = (uint64_t *)safe_aligned_malloc(r->N * sizeof(uint64_t));
    uint64_t *scratch = (r->split_degree > 1)
                            ? (uint64_t *)safe_aligned_malloc(r->poly_size * sizeof(uint64_t))
                            : NULL;

    for (uint64_t i = 0; i < r->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            memcpy(row_copy, out->limb[i], r->N * sizeof(uint64_t));
            r->mul_limb(out->limb[i], a->limb[i], row_copy, r->limbs[i], r, VFHE_ACC_SET, scratch);
        }
    }

    free(scratch);
    free(row_copy);
    return VFHE_OK;
}
