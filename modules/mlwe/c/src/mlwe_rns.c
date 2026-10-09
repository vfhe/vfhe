// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
// RNS row width and accessors: this file is the RNS backend, so it reaches
// into the representation on purpose.
#include "arith_internal.h"
#include "util.h"
#include <crypto.h>

// The MLWE operations that need the RNS representation itself.
//
// Everything else in this module works one whole ring element at a time and
// lives in mlwe.c (keyswitch.c for key switching). What is here cannot: it
// reaches into residues and coefficients, which no representation-independent
// interface exposes -- and deliberately so, since an interface fine enough to
// express them would cost a call per coefficient. The key-switching
// operations of that kind are in keyswitch_rns.c.
//
// mlwe_extract_LWE reads one coefficient per prime and negates residues to
// turn a ring element into the scalar LWE sample of one of its slots.

LWE mlwe_extract_LWE(RNSc_MLWE in, uint64_t idx)
{
    const uint64_t N = arith_rns_polynomial(&in->a[0])->base->N;
    const uint64_t mask = arith_rns_polynomial(&in->a[0])->rns_mask;
    const uint64_t l = rns_mask_to_l(mask);
    const uint64_t r = in->r;
    LWE res = lwe_alloc_sample(r * N, mask, arith_rns_polynomial(&in->a[0])->base);

    for (size_t j = 0; j < l; j++)
    {
        int g_idx = rns_mask_get_active_index(arith_rns_polynomial(&in->a[0])->rns_mask, j);
        assert(g_idx >= 0);
        Modulus mod = arith_rns_polynomial(&in->a[0])->base->mods[g_idx];

        // one width test for this prime; the extraction below is typed
        const bool narrow = rns_row_is_narrow(arith_rns_polynomial(&in->a[0])->base, g_idx);
        for (size_t k = 0; k < r; k++)
        {
            RNS_Polynomial ak = arith_rns_polynomial(&in->a[k]);
            // Reverse and negate for negacyclic
            if (narrow)
            {
                const uint32_t *row = ak->rows32[g_idx];
                for (size_t i = 0; i <= idx; i++)
                    res->a[j][k * N + i] = row[idx - i];
                for (size_t i = idx + 1; i < N; i++)
                    res->a[j][k * N + i] = negate_modq(row[N + idx - i], mod->q);
            }
            else
            {
                const uint64_t *row = ak->rows64[g_idx];
                for (size_t i = 0; i <= idx; i++)
                    res->a[j][k * N + i] = row[idx - i];
                for (size_t i = idx + 1; i < N; i++)
                    res->a[j][k * N + i] = negate_modq(row[N + idx - i], mod->q);
            }
        }
        {
            RNS_Polynomial bp = arith_rns_polynomial(&in->b);
            res->b[j] = rns_row_is_narrow(bp->base, g_idx) ? (uint64_t)bp->rows32[g_idx][idx]
                                                           : bp->rows64[g_idx][idx];
        }
    }
    return res;
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

// In RNS, reducing to a quotient keeps a subset of the residues (in either
// domain), so only the masks change. Dropped rows stay allocated until the
// sample is freed.
void mlwe_mod_reduce(MLWE c, ArithRing to)
{
    const uint64_t keep = arith_rns_ring_mask(to);
    for (size_t i = 0; i < c->r; i++)
    {
        RNS_Polynomial a = arith_rns_polynomial(&c->a[i]);
        assert((a->rns_mask & keep) == keep);
        a->rns_mask = keep;
    }
    RNS_Polynomial b = arith_rns_polynomial(&c->b);
    assert((b->rns_mask & keep) == keep);
    b->rns_mask = keep;
    c->ring = to;
}
