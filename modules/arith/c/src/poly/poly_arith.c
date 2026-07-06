// SPDX-License-Identifier: Apache-2.0
/**
 * @file poly_arith.c
 * @brief Domain-generic RNS polynomial arithmetic (add/sub/negate/scale/...)
 *        and slot-wise inversion.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/error.h>
#include <arith/poly.h>
#include <arith/zq.h>
#include <base.h>

int poly_add(rns_poly_t out, const rns_poly_t a, const rns_poly_t b)
{
    if (a->domain != b->domain)
        return VFHE_ERR_DOMAIN;
    out->mask = a->mask & b->mask;
    out->domain = a->domain;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            zq_arr_add(&out->ring->limbs[i]->zq, out->limb[i], a->limb[i], b->limb[i],
                       out->ring->N);
        }
    }
    return VFHE_OK;
}

int poly_sub(rns_poly_t out, const rns_poly_t a, const rns_poly_t b)
{
    if (a->domain != b->domain)
        return VFHE_ERR_DOMAIN;
    out->mask = a->mask & b->mask;
    out->domain = a->domain;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            zq_arr_sub(&out->ring->limbs[i]->zq, out->limb[i], a->limb[i], b->limb[i],
                       out->ring->N);
        }
    }
    return VFHE_OK;
}

void poly_negate(rns_poly_t out, const rns_poly_t a)
{
    out->mask = a->mask;
    out->domain = a->domain;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            zq_arr_negate(&out->ring->limbs[i]->zq, out->limb[i], a->limb[i], out->ring->N);
        }
    }
}

void poly_scale(rns_poly_t out, const rns_poly_t a, uint64_t s)
{
    out->mask = a->mask;
    out->domain = a->domain;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            zq_arr_scale(&out->ring->limbs[i]->zq, out->limb[i], a->limb[i], s, out->ring->N);
        }
    }
}

int poly_scale_addto(rns_poly_t out, const rns_poly_t a, uint64_t s)
{
    if (out->domain != a->domain)
        return VFHE_ERR_DOMAIN;
    out->mask = a->mask;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            zq_arr_scale_addto(&out->ring->limbs[i]->zq, out->limb[i], a->limb[i], s, out->ring->N);
        }
    }
    return VFHE_OK;
}

void poly_scale_vec(rns_poly_t out, const rns_poly_t a, const uint64_t *s)
{
    out->mask = a->mask;
    out->domain = a->domain;
    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (out->mask & (1ULL << i))
        {
            zq_arr_scale(&out->ring->limbs[i]->zq, out->limb[i], a->limb[i], s[i], out->ring->N);
        }
    }
}

int poly_add_scalar(rns_poly_t out, const rns_poly_t a, uint64_t c)
{
    const bool negative = (c >> 63) != 0;
    out->mask = a->mask;
    out->domain = a->domain;

    for (uint64_t i = 0; i < out->ring->l; i++)
    {
        if (!(out->mask & (1ULL << i)))
            continue;
        const zq_ctx *zq = &out->ring->limbs[i]->zq;
        const uint64_t q = zq->q;
        if (out != a)
            memcpy(out->limb[i], a->limb[i], out->ring->N * sizeof(uint64_t));

        // Interpret c as a signed two's-complement integer modulo q.
        uint64_t c_mod_q = negative
                               ? zq_scalar_negate(zq_reduce_u128((unsigned __int128)(-c), zq), q)
                               : zq_reduce_u128((unsigned __int128)c, zq);

        if (a->domain == VFHE_EVAL)
        {
            // The constant polynomial c evaluates to c in every slot; with the
            // split layout those slots are the first block of the row.
            zq_arr_add_scalar(zq, out->limb[i], out->limb[i], c_mod_q, out->ring->poly_size);
        }
        else
        {
            // Coefficient form: c lands on the constant coefficient only.
            out->limb[i][0] = zq_scalar_add(out->limb[i][0], c_mod_q, q);
        }
    }
    return VFHE_OK;
}

/**
 * Extended-Euclidean modular inverse (adapted from uecm_modinv_64).
 * Requires 0 < a < p, gcd(a, p) == 1, p odd. Faster than Fermat for the
 * single per-limb inversion the batched algorithm needs.
 */
static inline uint64_t modinv_ext_euclid_64(uint64_t a, uint64_t p)
{
    uint64_t ps1, ps2, parity, dividend, divisor, rem, q, t;

    q = 1;
    rem = a;
    dividend = p;
    divisor = a;
    ps1 = 1;
    ps2 = 0;
    parity = 0;

    while (divisor > 1)
    {
        rem = dividend - divisor;
        t = rem - divisor;
        if (rem >= divisor)
        {
            q += ps1;
            rem = t;
            t -= divisor;
            if (rem >= divisor)
            {
                q += ps1;
                rem = t;
                t -= divisor;
                if (rem >= divisor)
                {
                    q += ps1;
                    rem = t;
                    t -= divisor;
                    if (rem >= divisor)
                    {
                        q += ps1;
                        rem = t;
                        t -= divisor;
                        if (rem >= divisor)
                        {
                            q += ps1;
                            rem = t;
                            t -= divisor;
                            if (rem >= divisor)
                            {
                                q += ps1;
                                rem = t;
                                t -= divisor;
                                if (rem >= divisor)
                                {
                                    q += ps1;
                                    rem = t;
                                    t -= divisor;
                                    if (rem >= divisor)
                                    {
                                        q += ps1;
                                        rem = t;
                                        if (rem >= divisor)
                                        {
                                            q = dividend / divisor;
                                            rem = dividend % divisor;
                                            q *= ps1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        q += ps2;
        parity = ~parity;
        dividend = divisor;
        divisor = rem;
        ps2 = ps1;
        ps1 = q;
    }

    return (parity == 0) ? ps1 : p - ps1;
}

int poly_inverse(rns_poly_t out, const rns_poly_t in)
{
    if (in->domain != VFHE_EVAL)
        return VFHE_ERR_DOMAIN;
    if (in->ring->split_degree != 1)
        return VFHE_ERR_UNSUPPORTED;
    const uint64_t N = in->ring->N;
    out->mask = in->mask;
    out->domain = VFHE_EVAL;

    // Batched (Montgomery-trick) inversion per limb: one prefix-product pass,
    // one modular inverse, one suffix pass.
    uint64_t *prefix = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    for (uint64_t i = 0; i < in->ring->l; i++)
    {
        if (!(in->mask & (1ULL << i)))
            continue;
        const zq_ctx *zq = &in->ring->limbs[i]->zq;
        const uint64_t *a = in->limb[i];

        if (a[0] == 0)
        {
            free(prefix);
            return VFHE_ERR_NOT_INVERTIBLE;
        }
        prefix[0] = a[0];
        for (uint64_t k = 1; k < N; k++)
        {
            if (a[k] == 0)
            {
                free(prefix);
                return VFHE_ERR_NOT_INVERTIBLE;
            }
            prefix[k] = zq_scalar_mul(prefix[k - 1], a[k], zq);
        }

        uint64_t t = modinv_ext_euclid_64(prefix[N - 1], zq->q);

        uint64_t *o = out->limb[i];
        for (uint64_t k = N - 1; k > 0; k--)
        {
            o[k] = zq_scalar_mul(t, prefix[k - 1], zq);
            t = zq_scalar_mul(t, a[k], zq);
        }
        o[0] = t;
    }

    free(prefix);
    return VFHE_OK;
}
