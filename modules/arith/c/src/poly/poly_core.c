// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_core.c
 * @brief RNS polynomial lifecycle, copy/compare, and state accessors.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/poly.h>
#include <base.h>

rns_poly_t poly_new(const ring_ctx *ring, uint64_t mask)
{
    rns_poly_t p = (rns_poly_t)safe_malloc(sizeof(*p));
    p->limb = (uint64_t **)safe_malloc(sizeof(uint64_t *) * ring->l);
    // One aligned slab backs every row, so a polynomial costs two allocations
    // regardless of level count and the rows are contiguous. Every pool slot is
    // backed (not just the masked ones) so the mask can grow later -- base
    // extension into headroom limbs needs no reallocation. The per-row stride is
    // rounded up so each row keeps 64-byte alignment even for N < 8.
    const uint64_t row_words = (ring->N + 7) & ~(uint64_t)7;
    p->slab = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * row_words * ring->l);
    for (uint64_t i = 0; i < ring->l; i++)
    {
        p->limb[i] = p->slab + i * row_words;
    }
    p->ring = ring;
    p->mask = mask;
    p->alloc_l = ring->l;
    p->domain = VFHE_COEFF;
    return p;
}

void poly_free(void *p)
{
    if (p == NULL)
        return;
    rns_poly_t pp = (rns_poly_t)p;
    free(pp->slab);
    free(pp->limb);
    free(pp);
}

rns_poly_t *poly_array_new(uint64_t size, const ring_ctx *ring, uint64_t mask)
{
    rns_poly_t *res = (rns_poly_t *)safe_malloc(sizeof(rns_poly_t) * size);
    for (uint64_t i = 0; i < size; i++)
    {
        res[i] = poly_new(ring, mask);
    }
    return res;
}

void poly_array_free(uint64_t size, rns_poly_t *p)
{
    for (uint64_t i = 0; i < size; i++)
    {
        poly_free(p[i]);
    }
    free(p);
}

void poly_zero(rns_poly_t p)
{
    for (uint64_t i = 0; i < p->ring->l; i++)
    {
        if (p->mask & (1ULL << i))
        {
            memset(p->limb[i], 0, sizeof(uint64_t) * p->ring->N);
        }
    }
}

void poly_copy(rns_poly_t out, const rns_poly_t in)
{
    out->mask = in->mask;
    out->domain = in->domain;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            memcpy(out->limb[i], in->limb[i], sizeof(uint64_t) * out->ring->N);
        }
    }
}

bool poly_eq(const rns_poly_t a, const rns_poly_t b)
{
    const uint64_t N = a->ring->N;
    const uint64_t max_l = a->ring->l > b->ring->l ? a->ring->l : b->ring->l;
    for (uint64_t i = 0; i < max_l; i++)
    {
        bool active_a = (i < a->ring->l) && (a->mask & (1ULL << i));
        bool active_b = (i < b->ring->l) && (b->mask & (1ULL << i));
        if (active_a && active_b)
        {
            if (memcmp(a->limb[i], b->limb[i], N * sizeof(uint64_t)) != 0)
                return false;
        }
        else if (active_a)
        {
            // Limb only on one side: equal only if it carries the zero row there.
            for (uint64_t j = 0; j < N; j++)
            {
                if (a->limb[i][j])
                    return false;
            }
        }
        else if (active_b)
        {
            for (uint64_t j = 0; j < N; j++)
            {
                if (b->limb[i][j])
                    return false;
            }
        }
    }
    return true;
}

uint64_t poly_mask(const rns_poly_t p) { return p->mask; }

int poly_domain(const rns_poly_t p) { return (int)p->domain; }

void poly_assume_domain(rns_poly_t p, vfhe_domain domain) { p->domain = domain; }

uint64_t *poly_limb_data(const rns_poly_t p, uint64_t idx) { return p->limb[idx]; }
