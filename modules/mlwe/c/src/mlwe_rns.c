// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "misc.h"

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
    const uint64_t N = in->a[0]->base->N;
    const uint64_t l = rns_mask_to_l(in->a[0]->rns_mask);
    const uint64_t r = in->r;
    LWE res = lwe_alloc_sample(r * N, l, in->a[0]->base);

    for (size_t j = 0; j < l; j++)
    {
        int g_idx = rns_mask_get_active_index(in->a[0]->rns_mask, j);
        assert(g_idx >= 0);
        Modulus mod = in->a[0]->base->mods[g_idx];

        for (size_t k = 0; k < r; k++)
        {
            // Reverse and negate for negacyclic
            for (size_t i = 0; i <= idx; i++)
            {
                res->a[j][k * N + i] = in->a[k]->coeffs[g_idx][idx - i];
            }
            for (size_t i = idx + 1; i < N; i++)
            {
                res->a[j][k * N + i] = negate_modq(in->a[k]->coeffs[g_idx][N + idx - i], mod->q);
            }
        }
        res->b[j] = in->b->coeffs[g_idx][idx];
    }
    return res;
}

void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE_KS_Key key,
                                 uint64_t lvl)
{
    (void)lvl;
    const uint64_t N = out->b->base->N;
    const uint64_t in_n = in[0]->n;
    const uint64_t lwe_l = in[0]->l;

    const uint64_t target_mask = out->b->rns_mask;
    const uint64_t extended_mask = key->mask;
    const uint64_t divide_mask = extended_mask & ~target_mask;

    for (size_t i = 0; i < out->r; i++)
    {
        out->a[i]->rns_mask = extended_mask;
    }
    out->b->rns_mask = extended_mask;

    mlwe_RNS_trivial_sample_of_zero(out);

    RNSc_Polynomial tmp_poly =
        (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, target_mask, out->b->base);
    RNSc_Polynomial tmp_poly_red =
        (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, extended_mask, out->b->base);
    RNS_Polynomial tmp_rns =
        (RNS_Polynomial)polynomial_new_RNS_polynomial(N, extended_mask, out->b->base);

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
                mlwe_RNS_mul_subto_by_poly(out, key->s[i][ksk_idx++], tmp_rns);
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

    mlwe_RNS_to_RNSc((RNSc_MLWE)out, out);
    if (divide_mask > 0)
    {
        for (size_t j = 0; j < out->r; j++)
        {
            polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)out->a[j], divide_mask);
        }
        polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)out->b, divide_mask);
    }
    polynomial_add_RNSc_polynomial((RNSc_Polynomial)out->b, (RNSc_Polynomial)out->b, tmp_poly);
    mlwe_RNSc_to_RNS(out, (RNSc_MLWE)out);

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
    key->mask = sample->b->rns_mask;
    key->acc = mlwe_alloc_RNSc_sample(sample->b->base->N, sample->r, key->mask, sample->b->base);
    return key;
}

void mlwe_rns_ks_key_reset_acc(RNS_MLWE_KS_Key key)
{
    for (size_t i = 0; i < key->acc->r; i++)
    {
        key->acc->a[i]->rns_mask = key->mask;
    }
    key->acc->b->rns_mask = key->mask;
}

// The component arrays are borrowed from their creator; only what the key
// itself allocated is released.
void free_mlwe_RNS_ks_key(RNS_MLWE_KS_Key key)
{
    free_RNS_mlwe_sample((RNS_MLWE)key->acc);
    free(key->s);
    free(key);
}
