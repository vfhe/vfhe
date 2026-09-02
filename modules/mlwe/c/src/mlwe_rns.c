// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "util.h"
#include <crypto.h>

// The MLWE operations that need the RNS representation itself.
//
// Everything else in this module works one whole ring element at a time and
// lives in mlwe.c. What is here cannot: it reaches into residues and
// coefficients, which no representation-independent interface exposes -- and
// deliberately so, since an interface fine enough to express them would cost
// a call per coefficient.
//
// mlwe_extract_LWE reads one coefficient per prime and negates residues to
// turn a ring element into the scalar LWE sample of one of its slots.
// mlwe_full_packing_keyswitch assembles ring elements coefficient by
// coefficient from a vector of scalar LWE samples. The key-switch key
// plumbing lives here because deriving a key's ring and relabeling its
// accumulator into it are mask operations, which is how RNS identifies a
// ring.

LWE mlwe_extract_LWE(RNSc_MLWE in, uint64_t idx)
{
    const uint64_t N = arith_rns_polynomial(&in->a[0])->base->N;
    const uint64_t l = rns_mask_to_l(arith_rns_polynomial(&in->a[0])->rns_mask);
    const uint64_t r = in->r;
    LWE res = lwe_alloc_sample(r * N, l, arith_rns_polynomial(&in->a[0])->base);

    for (size_t j = 0; j < l; j++)
    {
        int g_idx = rns_mask_get_active_index(arith_rns_polynomial(&in->a[0])->rns_mask, j);
        assert(g_idx >= 0);
        Modulus mod = arith_rns_polynomial(&in->a[0])->base->mods[g_idx];

        for (size_t k = 0; k < r; k++)
        {
            // Reverse and negate for negacyclic
            for (size_t i = 0; i <= idx; i++)
            {
                res->a[j][k * N + i] = arith_rns_polynomial(&in->a[k])->coeffs[g_idx][idx - i];
            }
            for (size_t i = idx + 1; i < N; i++)
            {
                res->a[j][k * N + i] = negate_modq(
                    arith_rns_polynomial(&in->a[k])->coeffs[g_idx][N + idx - i], mod->q);
            }
        }
        res->b[j] = arith_rns_polynomial(&in->b)->coeffs[g_idx][idx];
    }
    return res;
}

void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE_KS_Key key,
                                 uint64_t lvl)
{
    (void)lvl;
    const uint64_t N = arith_rns_polynomial(&out->b)->base->N;
    const uint64_t in_n = in[0]->n;
    const uint64_t lwe_l = in[0]->l;

    const uint64_t target_mask = arith_rns_polynomial(&out->b)->rns_mask;
    const uint64_t extended_mask = key->mask;
    const uint64_t divide_mask = extended_mask & ~target_mask;

    for (size_t i = 0; i < out->r; i++)
    {
        arith_rns_polynomial(&out->a[i])->rns_mask = extended_mask;
    }
    arith_rns_polynomial(&out->b)->rns_mask = extended_mask;

    mlwe_RNS_trivial_sample_of_zero(out);

    RNSc_Polynomial tmp_poly = (RNSc_Polynomial)polynomial_new_RNS_polynomial(
        N, target_mask, arith_rns_polynomial(&out->b)->base);
    RNSc_Polynomial tmp_poly_red = (RNSc_Polynomial)polynomial_new_RNS_polynomial(
        N, extended_mask, arith_rns_polynomial(&out->b)->base);
    RNS_Polynomial tmp_rns = (RNS_Polynomial)polynomial_new_RNS_polynomial(
        N, extended_mask, arith_rns_polynomial(&out->b)->base);

    for (size_t i = 0; i < in_n; i++)
    {
        for (size_t j = 0; j < tmp_poly->base->l; j++)
        {
            if (tmp_poly->rns_mask & (1ULL << j))
            {
                memset(tmp_poly->coeffs[j], 0, N * sizeof(uint64_t));
            }
        }

        for (size_t limb = 0; limb < lwe_l; limb++)
        {
            int g_idx = rns_mask_get_active_index(target_mask, limb);
            assert(g_idx >= 0);
            for (size_t k = 0; k < size; k++)
            {
                tmp_poly->coeffs[g_idx][k] = in[k]->a[limb][i];
            }
        }

        uint64_t ksk_idx = 0;
        for (size_t j = 0; j < tmp_poly->base->l; j++)
        {
            if (tmp_poly->rns_mask & (1ULL << j))
            {
                polynomial_RNSc_mod_reduce_lifted(tmp_poly_red, tmp_poly, j);
                polynomial_RNSc_to_RNS(tmp_rns, tmp_poly_red);
                ArithElement factor = {tmp_rns, ARITH_DOMAIN_MUL};
                mlwe_RNS_mul_subto_by_poly(out, key->s[i][ksk_idx++], &factor);
            }
        }
    }

    // body part: out->b += sum B_k X^k
    for (size_t j = 0; j < tmp_poly->base->l; j++)
    {
        if (tmp_poly->rns_mask & (1ULL << j))
        {
            memset(tmp_poly->coeffs[j], 0, N * sizeof(uint64_t));
        }
    }
    for (size_t limb = 0; limb < lwe_l; limb++)
    {
        int g_idx = rns_mask_get_active_index(target_mask, limb);
        assert(g_idx >= 0);
        for (size_t k = 0; k < size; k++)
        {
            tmp_poly->coeffs[g_idx][k] = in[k]->b[limb];
        }
    }

    mlwe_RNS_to_RNSc(out, out);
    if (divide_mask > 0)
    {
        for (size_t j = 0; j < out->r; j++)
        {
            polynomial_round_division_RNSc_wo_free(
                (RNSc_Polynomial)arith_rns_polynomial(&out->a[j]), divide_mask);
        }
        polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)arith_rns_polynomial(&out->b),
                                               divide_mask);
    }
    polynomial_add_RNSc_polynomial((RNSc_Polynomial)arith_rns_polynomial(&out->b),
                                   (RNSc_Polynomial)arith_rns_polynomial(&out->b), tmp_poly);
    mlwe_RNSc_to_RNS(out, out);

    free_RNS_polynomial(tmp_poly);
    free_RNS_polynomial(tmp_poly_red);
    free_RNS_polynomial(tmp_rns);
}

RNS_MLWE_KS_Key mlwe_new_RNS_ks_key(RNS_MLWE **s, uint64_t count)
{
    RNS_MLWE_KS_Key key = (RNS_MLWE_KS_Key)safe_malloc(sizeof(*key));
    key->s = (RNS_MLWE **)safe_malloc(count * sizeof(RNS_MLWE *));
    memcpy(key->s, s, count * sizeof(RNS_MLWE *));
    key->count = count;

    // The key's ring comes from its first real component; NULL components are
    // pass-throughs and carry no samples.
    RNS_MLWE sample = NULL;
    for (size_t i = 0; i < count && sample == NULL; i++)
    {
        sample = key->s[i] == NULL ? NULL : key->s[i][0];
    }
    assert(sample != NULL);
    key->mask = arith_rns_polynomial(&sample->b)->rns_mask;
    key->ring = sample->ring;
    return key;
}

// The component arrays are borrowed from their creator; only the key's own
// copy of the pointer array is released.
void free_mlwe_RNS_ks_key(RNS_MLWE_KS_Key key)
{
    free(key->s);
    free(key);
}

// --- RNS parameters resolved to a ring -----------------------------------
//
// These are the entry points that still speak in primes and bases, because
// that is how their callers name a ring. They translate, and the generic
// allocators in mlwe.c do the work.

// The ring a sample lives in is what identifies it; the RNS parameters only
// name that ring, so they are resolved to one here.
RNS_MLWE mlwe_alloc_RNS_sample(uint64_t N, uint64_t r, uint64_t mask, RNS_Base base)
{
    return mlwe_alloc_sample(arith_rns_ring_get(N, mask, base), r);
}

RNSc_MLWE mlwe_alloc_RNSc_sample(uint64_t N, uint64_t r, uint64_t mask, RNS_Base base)
{
    return mlwe_alloc_RNS_sample(N, r, mask, base);
}

// Every component of a sample is in the same domain, so the body answers for
// all of them.

RNS_MLWE *mlwe_alloc_RNS_sample_array(uint64_t size, uint64_t N, uint64_t r, uint64_t mask,
                                      RNS_Base base)
{
    RNS_MLWE *res;
    res = (RNS_MLWE *)safe_malloc(size * sizeof(*res));
    for (size_t i = 0; i < size; i++)
    {
        res[i] = mlwe_alloc_RNS_sample(N, r, mask, base);
    }
    return res;
}

RNS_MLWE mlwe_new_RNS_trivial_sample_of_zero(uint64_t N, uint64_t r, uint64_t mask, RNS_Base base)
{
    RNS_MLWE res = mlwe_alloc_RNS_sample(N, r, mask, base);
    mlwe_RNS_trivial_sample_of_zero(res);
    return res;
}

// The RNS_ spelling means the mul domain, and a zeroed sample is entitled to
// that label: the transform fixes zero.

RNS_MLWE_Key mlwe_alloc_RNS_key(uint64_t N, uint64_t r, uint64_t l, RNS_Base base, double sigma)
{
    return mlwe_alloc_key(arith_rns_ring_get(N, (1ULL << l) - 1, base), r, l, sigma);
}

RNS_MLWE_Key mlwe_new_RNS_key_from_array(uint64_t *array, uint64_t N, uint64_t r, uint64_t l,
                                         RNS_Base base, double sigma)
{
    RNS_MLWE_Key res = mlwe_alloc_key(arith_rns_ring_get(N, (1ULL << l) - 1, base), r, l, sigma);
    for (size_t i = 0; i < r; i++)
    {
        // from_int_array lands in the mul domain, which is where a key is used.
        arith_from_int_array(res->ring, &res->s[i], &array[i * N], N);
    }
    return res;
}

RNS_MLWE_Key mlwe_new_RNS_gaussian_key(uint64_t N, uint64_t r, uint64_t l, double key_sigma,
                                       RNS_Base base, double sigma)
{
    RNS_MLWE_Key res = mlwe_alloc_key(arith_rns_ring_get(N, (1ULL << l) - 1, base), r, l, sigma);
    // Sampling into a plain array and loading it keeps key generation
    // representation-independent: the noise is integers either way.
    uint64_t *coeffs = (uint64_t *)safe_malloc(N * sizeof(uint64_t));
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            coeffs[j] = (uint64_t)((int64_t)generate_normal_random(key_sigma));
        }
        arith_from_int_array(res->ring, &res->s[i], coeffs, N);
    }
    free(coeffs);
    return res;
}

RNS_MLWE_Key mlwe_get_RNS_key_from_array(uint64_t N, uint64_t r, uint64_t l, uint64_t *array,
                                         RNS_Base base, double sigma)
{
    RNS_MLWE_Key res = mlwe_alloc_key(arith_rns_ring_get(N, (1ULL << l) - 1, base), r, l, sigma);
    for (size_t j = 0; j < r; j++)
    {
        arith_from_int_array(res->ring, &res->s[j], &array[j * N], N);
    }
    return res;
}
