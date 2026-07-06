// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_convert.c
 * @brief Domain conversion (COEFF <-> EVAL) and coefficient loading.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/ntt.h>
#include <arith/poly.h>
#include <base.h>

int poly_to_eval(rns_poly_t out, const rns_poly_t in)
{
    if (in->domain == VFHE_EVAL)
    {
        if (out != in)
            poly_copy(out, in);
        return VFHE_OK;
    }
    out->mask = in->mask;
    const uint64_t ps = out->ring->poly_size;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            const ntt_plan *plan = &out->ring->limbs[i]->ntt;
            // Each split block is one independent transform of size poly_size.
            for (uint64_t k = 0; k < out->ring->split_degree; k++)
            {
                ntt_forward(plan, &out->limb[i][k * ps], &in->limb[i][k * ps]);
            }
        }
    }
    out->domain = VFHE_EVAL;
    return VFHE_OK;
}

int poly_to_coeff(rns_poly_t out, const rns_poly_t in)
{
    if (in->domain == VFHE_COEFF)
    {
        if (out != in)
            poly_copy(out, in);
        return VFHE_OK;
    }
    out->mask = in->mask;
    const uint64_t ps = out->ring->poly_size;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            const ntt_plan *plan = &out->ring->limbs[i]->ntt;
            for (uint64_t k = 0; k < out->ring->split_degree; k++)
            {
                ntt_inverse(plan, &out->limb[i][k * ps], &in->limb[i][k * ps]);
            }
        }
    }
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}

/**
 * Scatter natural-order coefficients into the split block layout:
 * coefficient j -> row[(j % split_degree) * poly_size + j / split_degree].
 */
static void scatter_split(uint64_t *row, const uint64_t *coeffs, uint64_t N, uint64_t split_degree,
                          uint64_t poly_size)
{
    const uint64_t mod_mask = split_degree - 1;
    for (uint64_t j = 0; j < N; j++)
    {
        row[(j & mod_mask) * poly_size + j / split_degree] = coeffs[j];
    }
}

int poly_from_int_array(rns_poly_t out, const uint64_t *in)
{
    const uint64_t N = out->ring->N;
    const uint64_t ps = out->ring->poly_size;
    uint64_t *temp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            // Signed reduction into this limb, then the split-layout scatter.
            zq_arr_reduce_signed(&out->ring->limbs[i]->zq, temp, (const int64_t *)in, N);
            scatter_split(out->limb[i], temp, N, out->ring->split_degree, ps);
        }
    }
    free(temp);
    out->domain = VFHE_COEFF;
    return poly_to_eval(out, out);
}

int poly_from_residues(rns_poly_t out, uint64_t *const *in)
{
    const uint64_t N = out->ring->N;
    const uint64_t ps = out->ring->poly_size;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            scatter_split(out->limb[i], in[i], N, out->ring->split_degree, ps);
        }
    }
    out->domain = VFHE_COEFF;
    return poly_to_eval(out, out);
}

int poly_from_int_poly(rns_poly_t out, const int_poly_t in)
{
    return poly_from_int_array(out, in->coeffs);
}
