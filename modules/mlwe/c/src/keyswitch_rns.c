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
// coefficient from a vector of scalar LWE samples, and the subring maps of
// ring switching move coefficients between dimensions. The key-switch key
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

// --- Subring maps --------------------------------------------------------
//
// R_n = Z[Y]/(Y^n + 1) sits in R_N = Z[X]/(X^N + 1) as Y = X^k, k = N / n.
// Both maps below only move coefficients: per row, the projection is a k-way
// deinterleave and the embedding a k-way spread (arith's vector kernels). The
// two rings have bases of their own, so a row is found in the other base by
// its prime. Every row of the output is written whole, so it is not zeroed
// first.

// row[i], for every base index i of `from` that `from_mask` selects: the
// index in `to`'s base of the same prime, which `to_mask` must select. The
// masks select the same primes. Indices outside `from_mask` are unused.
static uint64_t *rows_by_prime(RNS_Base from, uint64_t from_mask, RNS_Base to, uint64_t to_mask)
{
    assert(rns_mask_to_l(from_mask) == rns_mask_to_l(to_mask));
    uint64_t *row = (uint64_t *)safe_malloc(from->l * sizeof(uint64_t));
    for (size_t i = 0; i < from->l; i++)
    {
        row[i] = UINT64_MAX;
        if (!(from_mask & (1ULL << i)))
            continue;
        for (size_t j = 0; j < to->l; j++)
            if ((to_mask & (1ULL << j)) && to->mods[j]->q == from->mods[i]->q)
                row[i] = j;
        assert(row[i] != UINT64_MAX);
    }
    return row;
}

// Labels every element of `out` canonical over all of its ring's primes, the
// state the maps leave it in once they have written every row.
static void claim_rows(MLWE out, uint64_t mask)
{
    for (size_t e = 0; e <= out->r; e++)
    {
        ArithElement *x = e < out->r ? &out->a[e] : &out->b;
        arith_rns_polynomial(x)->rns_mask = mask;
        x->domain = ARITH_DOMAIN_CANONICAL;
    }
}

// The k outputs of a deinterleave: row `oi` of out->a[first + j], j < k.
#define DEINTERLEAVE_ROW(W, out, first, oi, in, ii, k, n)                                          \
    do                                                                                             \
    {                                                                                              \
        uint##W##_t **rows_ = (uint##W##_t **)safe_malloc((k) * sizeof(uint##W##_t *));            \
        for (size_t j_ = 0; j_ < (k); j_++)                                                        \
            rows_[j_] = rns_row##W(arith_rns_polynomial(&(out)->a[(first) + j_]), (oi));           \
        vec_deinterleave_u##W(rows_, rns_row##W((in), (ii)), (k), (n));                            \
        free(rows_);                                                                               \
    } while (0)

void mlwe_project_subring(MLWE out, MLWE in)
{
    assert(mlwe_domain(in) == ARITH_DOMAIN_CANONICAL);
    const uint64_t n = out->ring->N;
    assert(in->ring->N % n == 0);
    const uint64_t k = in->ring->N / n;
    assert(out->r == in->r * k);
    RNS_Polynomial in_b = arith_rns_polynomial(&in->b);
    RNS_Polynomial out_b = arith_rns_polynomial(&out->b);
    const uint64_t out_mask = arith_rns_ring_mask(out->ring);
    uint64_t *row = rows_by_prime(in_b->base, in_b->rns_mask, out_b->base, out_mask);
    for (size_t i = 0; i < in_b->base->l; i++)
    {
        if (!(in_b->rns_mask & (1ULL << i)))
            continue;
        const bool narrow = rns_row_is_narrow(in_b->base, i);
        for (size_t c = 0; c < in->r; c++)
        {
            RNS_Polynomial a = arith_rns_polynomial(&in->a[c]);
            if (narrow)
                DEINTERLEAVE_ROW(32, out, c * k, row[i], a, i, k, n);
            else
                DEINTERLEAVE_ROW(64, out, c * k, row[i], a, i, k, n);
        }
        // Only part 0 of the body is kept.
        if (narrow)
        {
            uint32_t *o = rns_row32(out_b, row[i]);
            const uint32_t *b = rns_row32(in_b, i);
            for (size_t m = 0; m < n; m++)
                o[m] = b[k * m];
        }
        else
        {
            uint64_t *o = rns_row64(out_b, row[i]);
            const uint64_t *b = rns_row64(in_b, i);
            for (size_t m = 0; m < n; m++)
                o[m] = b[k * m];
        }
    }
    free(row);
    claim_rows(out, out_mask);
}

// The k inputs of an interleave: row `ii` of each sample's element `e`, NULL
// past `count`.
#define INTERLEAVE_ROW(W, dst, oi, in, count, e, ii, k, n)                                         \
    do                                                                                             \
    {                                                                                              \
        const uint##W##_t **rows_ =                                                                \
            (const uint##W##_t **)safe_malloc((k) * sizeof(uint##W##_t *));                        \
        for (size_t j_ = 0; j_ < (k); j_++)                                                        \
            rows_[j_] = j_ < (count) ? rns_row##W(element_rows((in)[j_], (e)), (ii)) : NULL;       \
        vec_interleave_k_u##W(rns_row##W((dst), (oi)), rows_, (k), (n));                           \
        free(rows_);                                                                               \
    } while (0)

static RNS_Polynomial element_rows(MLWE c, size_t e)
{
    return arith_rns_polynomial(e < c->r ? &c->a[e] : &c->b);
}

void mlwe_embed_subring(MLWE out, MLWE *in, uint64_t count)
{
    const uint64_t n = in[0]->ring->N;
    assert(out->ring->N % n == 0);
    const uint64_t k = out->ring->N / n;
    assert(count >= 1 && count <= k);
    for (size_t i = 0; i < count; i++)
    {
        assert(mlwe_domain(in[i]) == ARITH_DOMAIN_CANONICAL);
        assert(in[i]->ring == in[0]->ring && in[i]->r == out->r);
    }
    RNS_Polynomial in_b = arith_rns_polynomial(&in[0]->b);
    const uint64_t out_mask = arith_rns_ring_mask(out->ring);
    uint64_t *row =
        rows_by_prime(in_b->base, in_b->rns_mask, arith_rns_polynomial(&out->b)->base, out_mask);
    for (size_t i = 0; i < in_b->base->l; i++)
    {
        if (!(in_b->rns_mask & (1ULL << i)))
            continue;
        const bool narrow = rns_row_is_narrow(in_b->base, i);
        for (size_t e = 0; e <= out->r; e++)
        {
            RNS_Polynomial dst = element_rows(out, e);
            if (count == 1 && narrow)
                vec_spread_u32(rns_row32(dst, row[i]), rns_row32(element_rows(in[0], e), i), k, n);
            else if (count == 1)
                vec_spread_u64(rns_row64(dst, row[i]), rns_row64(element_rows(in[0], e), i), k, n);
            else if (narrow)
                INTERLEAVE_ROW(32, dst, row[i], in, count, e, i, k, n);
            else
                INTERLEAVE_ROW(64, dst, row[i], in, count, e, i, k, n);
        }
    }
    free(row);
    claim_rows(out, out_mask);
}
