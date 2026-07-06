// SPDX-License-Identifier: Apache-2.0
/**
 * @file zq_vec.c
 * @brief Vectors of RNS scalars (see arith/zqvec.h).
 *
 * Note the dependency: only ::zq_ctx contexts are borrowed from the ring --
 * this type never touches twiddles or transforms.
 */
#include <stdlib.h>

#include <arith/zqvec.h>
#include <base.h>

zqvec_t zqvec_new(const ring_ctx *ring, uint64_t n, uint64_t l)
{
    zqvec_t res = (zqvec_t)safe_malloc(sizeof(*res));
    res->n = n;
    res->l = l;
    const zq_ctx **zq = (const zq_ctx **)safe_malloc(sizeof(zq_ctx *) * l);
    res->rows = (uint64_t **)safe_malloc(sizeof(uint64_t *) * l);
    for (uint64_t i = 0; i < l; i++)
    {
        zq[i] = &ring->limbs[i]->zq;
        res->rows[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    }
    res->zq = zq;
    return res;
}

void zqvec_free(zqvec_t v)
{
    if (v == NULL)
        return;
    for (uint64_t i = 0; i < v->l; i++)
    {
        free(v->rows[i]);
    }
    free(v->rows);
    free((void *)v->zq);
    free(v);
}

void zqvec_add(zqvec_t out, const zqvec_t a, const zqvec_t b)
{
    for (uint64_t i = 0; i < out->l; i++)
    {
        zq_arr_add(out->zq[i], out->rows[i], a->rows[i], b->rows[i], out->n);
    }
}

void zqvec_sub(zqvec_t out, const zqvec_t a, const zqvec_t b)
{
    for (uint64_t i = 0; i < out->l; i++)
    {
        zq_arr_sub(out->zq[i], out->rows[i], a->rows[i], b->rows[i], out->n);
    }
}

void zqvec_scale(zqvec_t out, const zqvec_t a, uint64_t s)
{
    for (uint64_t i = 0; i < out->l; i++)
    {
        zq_arr_scale(out->zq[i], out->rows[i], a->rows[i], s, out->n);
    }
}
