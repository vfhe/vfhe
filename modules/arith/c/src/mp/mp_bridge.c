// SPDX-License-Identifier: Apache-2.0
/**
 * @file mp_bridge.c
 * @brief CRT bridge between RNS polynomials and big-integer polynomials.
 *
 * The per-prime constants this bridge needs (2^52 mod q for the Horner
 * reduction) live in the limb's ::zq_ctx -- they are reduction constants of
 * the prime, not NTT state.
 */
#include <arith/error.h>
#include <arith/mp.h>
#include <arith/zq.h>

int mp_polynomial_from_poly(mp_polynomial_t out, const rns_poly_t in, const mp_scalar_t *PW,
                            const mp_scalar_t q, const mp_vector_t *m, uint64_t k)
{
    if (in->domain != VFHE_COEFF)
        return VFHE_ERR_DOMAIN;
    // out = sum_i x_i * PW_i, accumulated digit-wise with periodic carry
    // flushes, then one big Barrett reduction mod Q. PW is indexed by *level*
    // (position within the mask), matching the caller's prime list.
    uint64_t level = 0;
    for (uint64_t i = 0; i < in->ring->l; i++)
    {
        if (!(in->mask & (1ULL << i)))
            continue;
        if (level == 0)
        {
            mp_polynomial_scale_int_by_scalar(out, in->limb[i], PW[level]);
        }
        else
        {
            mp_polynomial_scale_int_by_scalar_addto(out, in->limb[i], PW[level]);
            if ((level & 0xFF) == 0)
            {
                mp_polynomial_propagate_carry(out);
            }
        }
        level++;
    }
    mp_polynomial_propagate_carry(out);
    mp_polynomial_mod_reduce(out, q, m, k);
    return VFHE_OK;
}

int mp_polynomial_to_poly(rns_poly_t out, const mp_polynomial_t in)
{
    const uint64_t N = in->N, D = in->d;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (!(out->mask & (1ULL << i)))
            continue;
        const zq_ctx *zq = &out->ring->limbs[i]->zq;
        const uint64_t w1 = zq->w52_1, q = zq->q;
        // Horner over the base-2^52 digits, top digit first.
        for (uint64_t c = 0; c < N; c++)
        {
            uint64_t res = 0;
            for (int64_t j = (int64_t)D - 1; j >= 0; j--)
            {
                res = zq_scalar_mul(res, w1, zq);
                res =
                    zq_scalar_add(res, zq_reduce_u128((unsigned __int128)in->coeffs[j][c], zq), q);
            }
            out->limb[i][c] = res;
        }
    }
    out->domain = VFHE_COEFF;
    return VFHE_OK;
}
