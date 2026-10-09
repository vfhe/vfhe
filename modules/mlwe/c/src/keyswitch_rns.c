// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
// RNS row width and accessors: this file is the RNS backend, so it reaches
// into the representation on purpose.
#include "arith_internal.h"
#include "util.h"

// The key-switching operations that need the RNS representation itself.
//
// mlwe_full_packing_keyswitch assembles ring elements coefficient by
// coefficient from a vector of scalar LWE samples. The key-switch key
// plumbing lives here because deriving a key's ring and relabeling its
// accumulator into it are mask operations, which is how RNS identifies a
// ring.

// Writes one value per sample, sample k at coefficient k of `poly`'s rows
// over `mask`; the other coefficients are zero. `row[limb]` is the row of
// `poly`'s base that an LWE limb's prime sits at.
static void spread_lwe_values(RNSc_Polynomial poly, uint64_t mask, LWE *in, uint64_t size,
                              const uint64_t *row, uint64_t l, const uint64_t *index)
{
    const uint64_t N = poly->base->N;
    for (size_t j = 0; j < poly->base->l; j++)
        if (mask & (1ULL << j))
            RNS_ROW_ZERO(poly, j, N);
    for (size_t limb = 0; limb < l; limb++)
    {
        const uint64_t g_idx = row[limb];
        const bool narrow = rns_row_is_narrow(poly->base, g_idx);
        for (size_t k = 0; k < size; k++)
        {
            const uint64_t v = index == NULL ? in[k]->b[limb] : in[k]->a[limb][*index];
            if (narrow)
                poly->rows32[g_idx][k] = (uint32_t)v;
            else
                poly->rows64[g_idx][k] = v;
        }
    }
}

void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE_KS_Key key)
{
    RNS_Base out_base = arith_rns_polynomial(&out->b)->base;
    const uint64_t N = out_base->N;
    const uint64_t in_n = in[0]->n;
    const uint64_t lwe_l = in[0]->l;
    assert(size <= N);

    const uint64_t target_mask = arith_rns_polynomial(&out->b)->rns_mask;
    const uint64_t extended_mask = key->mask;
    const uint64_t divide_mask = extended_mask & ~target_mask;

    // The samples may come from a ring of another dimension, over another
    // base: their primes are found in out's base by value.
    assert(lwe_l == rns_mask_to_l(target_mask));
    uint64_t *row = (uint64_t *)safe_malloc(lwe_l * sizeof(uint64_t));
    for (size_t limb = 0; limb < lwe_l; limb++)
    {
        const int idx = rns_mask_get_active_index(in[0]->mask, limb);
        assert(idx >= 0);
        const uint64_t q = in[0]->base->mods[idx]->q;
        row[limb] = UINT64_MAX;
        for (size_t j = 0; j < out_base->l; j++)
            if ((target_mask & (1ULL << j)) && out_base->mods[j]->q == q)
                row[limb] = j;
        assert(row[limb] != UINT64_MAX);
    }

    for (size_t i = 0; i < out->r; i++)
    {
        arith_rns_polynomial(&out->a[i])->rns_mask = extended_mask;
    }
    arith_rns_polynomial(&out->b)->rns_mask = extended_mask;

    mlwe_RNS_trivial_sample_of_zero(out);

    RNSc_Polynomial tmp_poly =
        (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, target_mask, out_base);
    ArithElement column = {tmp_poly, ARITH_DOMAIN_CANONICAL};

    // out -= sum_i (sum_k a_k[i] X^k) (x) KS(s_i)
    for (size_t i = 0; i < in_n; i++)
    {
        spread_lwe_values(tmp_poly, target_mask, in, size, row, lwe_l, &i);
        gadget_mul_subto_polynomial(out, key->s[i], &column, key->log_base, key->balanced);
    }

    // body part: out->b += sum B_k X^k
    spread_lwe_values(tmp_poly, target_mask, in, size, row, lwe_l, NULL);

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
    free(row);
}

RNS_MLWE_KS_Key mlwe_new_RNS_ks_key(RNS_MLWE **s, uint64_t count, uint64_t log_base, bool balanced)
{
    RNS_MLWE_KS_Key key = (RNS_MLWE_KS_Key)safe_malloc(sizeof(*key));
    key->s = (RNS_MLWE **)safe_malloc(count * sizeof(RNS_MLWE *));
    memcpy(key->s, s, count * sizeof(RNS_MLWE *));
    key->count = count;
    key->log_base = log_base;
    key->balanced = balanced;

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
