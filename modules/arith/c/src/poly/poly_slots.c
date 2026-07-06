// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_slots.c
 * @brief Evaluation-slot manipulation (broadcast / rotate / copy).
 *
 * These treat each split block of an EVAL-domain row as an array of
 * poly_size independent evaluation slots.
 */
#include <string.h>

#include <arith/error.h>
#include <arith/poly.h>

int poly_broadcast_slot(rns_poly_t out, const rns_poly_t in, uint64_t slot_idx)
{
    if (in->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;
    const uint64_t ps = out->ring->poly_size;
    out->mask = in->mask;
    out->domain = VFHE_EVAL;
    for (uint64_t i = 0; i < in->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            for (uint64_t j = 0; j < out->ring->split_degree; j++)
            {
                const uint64_t v = in->limb[i][j * ps + slot_idx];
                for (uint64_t k = 0; k < ps; k++)
                {
                    out->limb[i][j * ps + k] = v;
                }
            }
        }
    }
    return VFHE_OK;
}

int poly_broadcast_limb(rns_poly_t out, const rns_poly_t in, uint64_t src_limb)
{
    if (in->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;
    out->mask = in->mask;
    out->domain = VFHE_EVAL;
    for (uint64_t i = 0; i < in->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            memcpy(out->limb[i], in->limb[src_limb], out->ring->N * sizeof(uint64_t));
        }
    }
    return VFHE_OK;
}

int poly_rotate_slots(rns_poly_t out, const rns_poly_t in, uint64_t rot)
{
    if (in->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;
    if (out == in)
        return VFHE_ERR_ARG;
    const uint64_t ps = out->ring->poly_size;
    out->mask = in->mask;
    out->domain = VFHE_EVAL;
    for (uint64_t i = 0; i < in->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            for (uint64_t j = 0; j < out->ring->split_degree; j++)
            {
                for (uint64_t k = 0; k < ps - rot; k++)
                {
                    out->limb[i][j * ps + k] = in->limb[i][j * ps + k + rot];
                }
                for (uint64_t k = ps - rot; k < ps; k++)
                {
                    out->limb[i][j * ps + k] = in->limb[i][j * ps + k + rot - ps];
                }
            }
        }
    }
    return VFHE_OK;
}

int poly_copy_slot(rns_poly_t out, uint64_t dst, const rns_poly_t in, uint64_t src)
{
    if (in->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;
    const uint64_t ps = out->ring->poly_size;
    out->mask = in->mask;
    out->domain = VFHE_EVAL;
    for (uint64_t i = 0; i < in->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            for (uint64_t j = 0; j < out->ring->split_degree; j++)
            {
                out->limb[i][j * ps + dst] = in->limb[i][j * ps + src];
            }
        }
    }
    return VFHE_OK;
}
