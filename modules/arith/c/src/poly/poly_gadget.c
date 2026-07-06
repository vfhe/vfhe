// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_gadget.c
 * @brief Gadget (digit) decomposition of RNS polynomials.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/poly.h>
#include <arith/tower.h>
#include <base.h>

int poly_decompose_digit(rns_poly_t out, const rns_poly_t in, uint64_t log_base, uint64_t level)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(out->ring->N * sizeof(uint64_t));
    const uint64_t mask = (1ULL << log_base) - 1;
    const uint64_t shift = log_base * level;
    // The composite value rides on the highest active limb; slice its digit...
    int last_active = limbset_last(in->mask);
    if (last_active < 0)
    {
        free(tmp);
        return VFHE_ERR_ARG;
    }
    for (uint64_t i = 0; i < out->ring->N; i++)
    {
        tmp[i] = (in->limb[last_active][i] >> shift) & mask;
    }
    // ...and replicate it into every active output limb (digits are < base,
    // hence already reduced for every prime).
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            memcpy(out->limb[i], tmp, sizeof(uint64_t) * out->ring->N);
        }
    }
    free(tmp);
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}
