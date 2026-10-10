// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <string.h>

#include "mlwe.h"
#include "rns_rows.h"
#include "util.h"

// Gadget decompositions against the RNS base.
//
// The default gadget here *is* the prime factorisation: a value is split into
// its residues, one key-switch key per prime, and the products summed. That
// makes the whole file RNS-specific by nature -- another representation
// decomposes against a different gadget, or none. The signatures stay generic
// so the key switch in keyswitch.c can call it without knowing any of that.
//
// `balanced` picks the RNS gadget's digit: the centered residue, in
// (-p_j/2, p_j/2], or the residue as stored, in [0, p_j). Both are the same
// value mod p_j, so the sum is exact either way and the keys are the same; the
// centered one has a quarter of the second moment -- which is what the
// accumulated noise grows with -- at the same cost.
//
// `primes_per_digit` a > 1 groups the primes instead: the source's primes, in
// ascending base index, a at a time, and the digit of group J, modulus Q_J,
// is x mod Q_J lifted to every prime of the key by the fast base conversion.
// The residues of J are multiplied by (Q_J/p_k)^-1 once (element_decomposition_scale)
// and every row then takes their combination
// (polynomial_RNSc_lifted_combination), centered if `balanced`. The result is
// x plus a multiple of Q_J, which the gadget element P e_J (e_J the group's
// CRT idempotent) sends to a multiple of Q P, so the sum stays exact. What it
// costs is the digit's size: up to about a Q_J / 2, against p_j / 2, which the
// special modulus P has to make up for.
//
// A non-zero `log_base` asks for the radix gadget instead: every residue is
// further split into base-2^log_base digits, so the decomposition of x is
//
//     x = sum_j sum_k d_{j,k} * (2^(log_base*k) * e_j)   mod Q,
//
// with e_j the CRT idempotent of prime j. `balanced` applies here too: the
// digits are those of the centered residue, each in [-2^(log_base-1),
// 2^(log_base-1)) but the top one (which takes the carry, and is at most
// 2^(log_base-1) in absolute value), instead of the unsigned digits of the
// residue as stored, in [0, 2^log_base). As for the RNS gadget, the digit
// count and the keys are the same either way, and the second moment about a
// quarter. In base 2 the balanced digits are no smaller, so log_base 1 always
// takes the unsigned ones.
// The key array must hold one key per (prime, digit) pair, prime-major with
// the input's primes in ascending base index, and its keys must carry that
// gadget (`gadget_radix_digits` says how many digits a prime takes). What it
// buys is the bound on the products accumulated below: 2^log_base rather than
// the prime itself.
//
// Note what the digits are *not*: the base-2^log_base digits of the value x
// represents mod Q. Each residue is decomposed on its own, and it is the
// gadget that carries the idempotent, so nothing here ever reconstructs x.
// The cost of that is exactness: unlike a radix decomposition of x itself,
// this one cannot drop its low digits for a controlled error, because the
// error would be multiplied by an idempotent, which is not small mod Q --
// except over a single prime, where the idempotent is 1. There the gadget may
// keep only its top `digits`: the residue is rounded to the nearest multiple
// of 2^(log_base * dropped), dropped the digits left out, and the digits kept
// are those of the rounded residue (polynomial_RNSc_decompose_digit_rounded
// and polynomial_RNSc_decompose_digit_balanced, which round inside the digit
// extraction).

uint64_t gadget_radix_digits(uint64_t prime, uint64_t log_base)
{
    assert(log_base > 0 && prime > 1);
    const uint64_t bits = (uint64_t)(64 - __builtin_clzll(prime));
    return (bits + log_base - 1) / log_base;
}

uint64_t gadget_primes_per_digit(const GadgetParams *gadget_params)
{
    return gadget_params->primes_per_digit > 1 ? gadget_params->primes_per_digit : 1;
}

// How many of prime `j`'s lowest radix digits the parameters leave out.
static uint64_t dropped_of(RNS_Base base, size_t j, const GadgetParams *gadget_params)
{
    if (!gadget_params->log_base || !gadget_params->digits)
        return 0;
    const uint64_t all = gadget_radix_digits(base->mods[j]->q, gadget_params->log_base);
    return gadget_params->digits < all ? all - gadget_params->digits : 0;
}

// The digits of one element, in key order, and what computing them needs.
typedef struct
{
    RNSc_Polynomial source;
    const GadgetParams *gadget_params;
    uint64_t n;
    // Per digit: the primes of the source it covers (one, but for a group),
    // and for the radix gadget its level among that prime's digits.
    uint64_t *primes;
    uint64_t *level;
    // Groups of several primes only: (Q_J/p_k)^-1 mod p_k at k, the source
    // with residue k multiplied by it, and (Q_J/p_k) mod p_i at [k * l + i].
    uint64_t *inverse;
    RNSc_Polynomial scaled;
    uint64_t *factors;
} ElementDecomposition;

// The digits of `source` for a key over `key`'s primes. The scaled residues
// are allocated but not computed: element_decomposition_scale does that.
static void element_decomposition_init(ElementDecomposition *e, RNS_Polynomial source,
                                       RNS_Polynomial key, const GadgetParams *gadget_params)
{
    RNS_Base base = source->base;
    const uint64_t l = base->l, per_digit = gadget_primes_per_digit(gadget_params);
    assert(per_digit == 1 || !gadget_params->log_base);
    *e = (ElementDecomposition){
        (RNSc_Polynomial)source, gadget_params, 0, NULL, NULL, NULL, NULL, NULL};

    uint64_t members = 0;
    for (size_t j = 0; j < l; j++)
    {
        if (!(source->rns_mask & (1ULL << j)))
            continue;
        if (gadget_params->log_base)
            e->n += gadget_radix_digits(base->mods[j]->q, gadget_params->log_base) -
                    dropped_of(base, j, gadget_params);
        else if (members++ % per_digit == 0)
            e->n++;
    }
    e->primes = (uint64_t *)safe_malloc((e->n + 1) * sizeof(uint64_t));
    e->level = (uint64_t *)safe_malloc((e->n + 1) * sizeof(uint64_t));
    uint64_t t = 0;
    members = 0;
    for (size_t j = 0; j < l; j++)
    {
        if (!(source->rns_mask & (1ULL << j)))
            continue;
        if (gadget_params->log_base)
        {
            const uint64_t all = gadget_radix_digits(base->mods[j]->q, gadget_params->log_base);
            for (uint64_t d = dropped_of(base, j, gadget_params); d < all; d++, t++)
            {
                e->primes[t] = 1ULL << j;
                e->level[t] = d;
            }
        }
        else if (members++ % per_digit == 0)
            e->primes[t++] = 1ULL << j;
        else
            e->primes[t - 1] |= 1ULL << j;
    }

    uint64_t grouped = 0;
    for (t = 0; t < e->n; t++)
        if (e->primes[t] & (e->primes[t] - 1))
            grouped |= e->primes[t];
    if (!grouped)
        return;
    e->inverse = (uint64_t *)safe_malloc(l * sizeof(uint64_t));
    e->factors = (uint64_t *)safe_malloc(l * l * sizeof(uint64_t));
    memset(e->inverse, 0, l * sizeof(uint64_t));
    memset(e->factors, 0, l * l * sizeof(uint64_t));
    for (t = 0; t < e->n; t++)
    {
        const uint64_t group = e->primes[t];
        if (!(group & (group - 1)))
            continue;
        for (size_t k = 0; k < l; k++)
        {
            if (!(group & (1ULL << k)))
                continue;
            // Q_J/p_k modulo p_k, and modulo every prime of the key.
            const uint64_t rows = key->rns_mask | (1ULL << k);
            for (size_t i = 0; i < l; i++)
            {
                if (!(rows & (1ULL << i)))
                    continue;
                uint64_t f = 1;
                for (size_t m = 0; m < l; m++)
                    if (m != k && (group & (1ULL << m)))
                        f = mul_modq(f, modq(base->mods[m]->q, base->mods[i]), base->mods[i]);
                if (i == k)
                    e->inverse[k] = inverse_mod(f, base->mods[k]->q);
                if (key->rns_mask & (1ULL << i))
                    e->factors[k * l + i] = f;
            }
        }
    }
    e->scaled = (RNSc_Polynomial)polynomial_new_RNS_polynomial(base->N, grouped, base);
}

static void scale_source_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    const ElementDecomposition *e = (const ElementDecomposition *)ctx;
    struct _RNS_Polynomial out_storage, in_storage;
    polynomial_scale_RNS_polynomial_RNS(
        polynomial_RNS_view(&out_storage, (RNS_Polynomial)e->scaled, 1ULL << row),
        polynomial_RNS_view(&in_storage, (RNS_Polynomial)e->source, 1ULL << row), e->inverse);
}

// Computes the scaled residues of the primes in `mask` that groups take.
static void element_decomposition_scale(ElementDecomposition *e, uint64_t mask)
{
    if (e->scaled)
        rns_rows_for(e->source->base->N, e->scaled->rns_mask & mask, 1, scale_source_row, e);
}

static void element_decomposition_free(ElementDecomposition *e)
{
    free(e->primes);
    free(e->level);
    free(e->inverse);
    free(e->factors);
    if (e->scaled)
        free_RNS_polynomial((RNS_Polynomial)e->scaled);
}

// Digit `t` of the source, lifted to `tmp`'s ring: the whole residue for the
// RNS gadget (or the fast base conversion of a group's), one base-2^log_base
// digit of it for the radix one; centered or balanced if `balanced`.
static void gadget_digit(RNSc_Polynomial tmp, const ElementDecomposition *e, uint64_t t)
{
    const GadgetParams *gadget_params = e->gadget_params;
    const uint64_t primes = e->primes[t];
    const size_t j = (size_t)__builtin_ctzll(primes);
    if (gadget_params->log_base)
    {
        const uint64_t dropped = dropped_of(e->source->base, j, gadget_params);
        if (gadget_params->balanced && gadget_params->log_base > 1)
            polynomial_RNSc_decompose_digit_balanced(tmp, e->source, j, gadget_params->log_base,
                                                     e->level[t], dropped);
        else
            polynomial_RNSc_decompose_digit_rounded(tmp, e->source, j, gadget_params->log_base,
                                                    e->level[t], dropped);
    }
    else if (primes & (primes - 1))
        polynomial_RNSc_lifted_combination(tmp, e->scaled, primes, e->factors,
                                           gadget_params->balanced);
    else if (gadget_params->balanced)
        polynomial_RNSc_mod_reduce_lifted_centered(tmp, e->source, j);
    else
        polynomial_RNSc_mod_reduce_lifted(tmp, e->source, j);
}

// Digits are left out of a source over one prime only (see above).
static void check_dropped_digits(RNS_Polynomial source, const GadgetParams *gadget_params)
{
#ifndef NDEBUG
    for (size_t i = 0; i < source->base->l; i++)
        if ((source->rns_mask & (1ULL << i)) && dropped_of(source->base, i, gadget_params))
            assert(__builtin_popcountll(source->rns_mask) == 1);
#else
    (void)source;
    (void)gadget_params;
#endif
}

typedef struct
{
    RNS_MLWE out;
    RNS_MLWE *ksk;
    const ElementDecomposition *decomposition;
    RNS_Polynomial key;
    int subtract;
} KeyProduct;

static RNS_Polynomial sample_part(RNS_MLWE c, uint64_t k)
{
    return arith_rns_polynomial(k < c->r ? &c->a[k] : &c->b);
}

// Row `row` of the product: every digit lifted to that prime alone,
// transformed, and multiplied into the same row of every component of `out`.
static void key_product_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    const KeyProduct *p = (const KeyProduct *)ctx;
    RNS_Base base = p->key->base;
    const uint64_t row_mask = 1ULL << row, parts = p->out->r + 1;
    RNSc_Polynomial tmp = (RNSc_Polynomial)polynomial_new_RNS_polynomial(base->N, row_mask, base);
    struct _RNS_Polynomial out_storage, key_storage;

    for (uint64_t t = 0; t < p->decomposition->n; t++)
    {
        gadget_digit(tmp, p->decomposition, t);
        polynomial_RNSc_to_RNS((RNS_Polynomial)tmp, tmp);
        for (uint64_t k = 0; k < parts; k++)
        {
            RNS_Polynomial o = polynomial_RNS_view(&out_storage, sample_part(p->out, k), row_mask);
            RNS_Polynomial x =
                polynomial_RNS_view(&key_storage, sample_part(p->ksk[t], k), row_mask);
            if (p->subtract)
                polynomial_mul_subto_RNS_polynomial(o, x, (RNS_Polynomial)tmp);
            else
                polynomial_mul_addto_RNS_polynomial(o, x, (RNS_Polynomial)tmp);
        }
    }
    free_RNS_polynomial(tmp);
}

static void gadget_mul_accumulate(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                  int subtract, const GadgetParams *gadget_params)
{
    // This file knows the representation, so it calls the RNS entry points
    // rather than routing through the dispatcher: nothing here would gain from
    // a ring it cannot have. The product is formed prime by prime -- each row
    // of `out` takes every digit, lifted to that prime alone -- so the rows
    // are independent tasks.
    RNS_Polynomial source = arith_rns_polynomial(poly);
    RNS_Polynomial key = arith_rns_polynomial(&ksk[0]->b);
    if (source->rns_mask == 0)
        return;
    check_dropped_digits(source, gadget_params);
    ElementDecomposition e;
    element_decomposition_init(&e, source, key, gadget_params);
    element_decomposition_scale(&e, source->rns_mask);
    KeyProduct p = {out, ksk, &e, key, subtract};
    rns_rows_for(key->base->N, key->rns_mask, 1, key_product_row, &p);

    // Each row's last product set that row's mask in its view; the mask of
    // the whole element is what that product gives on every row.
    for (uint64_t k = 0; k <= out->r; k++)
        sample_part(out, k)->rns_mask = sample_part(ksk[e.n - 1], k)->rns_mask & key->rns_mask;
    element_decomposition_free(&e);
}

void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                 const GadgetParams *gadget_params)
{
    gadget_mul_accumulate(out, ksk, poly, 0, gadget_params);
}

void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                 const GadgetParams *gadget_params)
{
    gadget_mul_accumulate(out, ksk, poly, 1, gadget_params);
}

// Digits stay in the mul domain on a fully split ring, where an automorphism
// just permutes them; otherwise they stay canonical, and each product
// transforms its permuted copy.
typedef struct
{
    GadgetDigits *out;
    const ElementDecomposition *decomposition;
    int mul_domain;
} StoredDecomposition;

static void decompose_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    const StoredDecomposition *dec = (const StoredDecomposition *)ctx;
    struct _RNS_Polynomial storage;
    for (uint64_t t = 0; t < dec->decomposition->n; t++)
    {
        RNSc_Polynomial digit = (RNSc_Polynomial)polynomial_RNS_view(
            &storage, (RNS_Polynomial)dec->out->digit[t].handle, 1ULL << row);
        gadget_digit(digit, dec->decomposition, t);
        if (dec->mul_domain)
            polynomial_RNSc_to_RNS((RNS_Polynomial)digit, digit);
    }
}

void gadget_decompose(GadgetDigits *out, RNS_MLWE *ksk, const ArithElement *poly,
                      const GadgetParams *gadget_params)
{
    RNS_Polynomial source = arith_rns_polynomial(poly);
    RNS_Polynomial key = arith_rns_polynomial(&ksk[0]->b);
    RNS_Base base = key->base;
    const int mul_domain = base->split_degree == 1;

    check_dropped_digits(source, gadget_params);
    ElementDecomposition e;
    element_decomposition_init(&e, source, key, gadget_params);
    out->n = e.n;
    out->digit = (ArithElement *)safe_malloc(e.n * sizeof(ArithElement));
    for (uint64_t i = 0; i < e.n; i++)
    {
        out->digit[i].handle = polynomial_new_RNS_polynomial(base->N, key->rns_mask, base);
        out->digit[i].domain = mul_domain ? ARITH_DOMAIN_MUL : ARITH_DOMAIN_CANONICAL;
    }
    // Each digit row is the residue lifted to that prime alone, so the rows
    // are independent tasks.
    element_decomposition_scale(&e, source->rns_mask);
    StoredDecomposition dec = {out, &e, mul_domain};
    rns_rows_for(base->N, key->rns_mask, 1, decompose_row, &dec);
    element_decomposition_free(&e);
}

void gadget_decompose_digit(ArithElement *out, RNS_MLWE *ksk, const ArithElement *poly, uint64_t i,
                            const GadgetParams *gadget_params)
{
    RNS_Polynomial source = arith_rns_polynomial(poly);
    RNS_Polynomial key = arith_rns_polynomial(&ksk[0]->b);
    RNSc_Polynomial digit = (RNSc_Polynomial)arith_rns_polynomial(out);
    const int mul_domain = key->base->split_degree == 1;
    ElementDecomposition e;
    element_decomposition_init(&e, source, key, gadget_params);
    if (i < e.n)
    {
        check_dropped_digits(source, gadget_params);
        element_decomposition_scale(&e, e.primes[i]);
        gadget_digit(digit, &e, i);
        if (mul_domain)
            polynomial_RNSc_to_RNS((RNS_Polynomial)digit, digit);
        out->domain = mul_domain ? ARITH_DOMAIN_MUL : ARITH_DOMAIN_CANONICAL;
    }
    element_decomposition_free(&e);
}

void gadget_digits_free(GadgetDigits *digits)
{
    for (uint64_t i = 0; i < digits->n; i++)
        free_RNS_polynomial(digits->digit[i].handle);
    free(digits->digit);
    digits->digit = NULL;
    digits->n = 0;
}

typedef struct
{
    RNS_MLWE out;
    RNS_MLWE *ksk;
    const GadgetDigits *digits;
    uint64_t gen;
    const uint32_t *idx;
    int mul_domain;
} AutomorphismProduct;

// Row `row` of the product: every digit permuted on that prime alone, and
// multiplied into the same row of every component of `out`.
static void automorphism_product_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    const AutomorphismProduct *p = (const AutomorphismProduct *)ctx;
    RNS_Polynomial first = (RNS_Polynomial)p->digits->digit[0].handle;
    RNS_Base base = first->base;
    const uint64_t row_mask = 1ULL << row, parts = p->out->r + 1;
    RNS_Polynomial tmp = polynomial_new_RNS_polynomial(base->N, row_mask, base);
    struct _RNS_Polynomial out_storage, key_storage, digit_storage;

    for (uint64_t i = 0; i < p->digits->n; i++)
    {
        RNS_Polynomial digit = (RNS_Polynomial)p->digits->digit[i].handle;
        RNS_Polynomial factor = tmp;
        if (p->mul_domain && p->gen == 1)
            factor = polynomial_RNS_view(&digit_storage, digit, row_mask);
        else if (p->mul_domain)
            polynomial_RNS_permute_rows(tmp, digit, p->idx, row_mask);
        else
        {
            polynomial_RNSc_permute_rows((RNSc_Polynomial)tmp, (RNSc_Polynomial)digit, p->gen,
                                         row_mask);
            polynomial_RNSc_to_RNS(tmp, (RNSc_Polynomial)tmp);
        }
        for (uint64_t k = 0; k < parts; k++)
        {
            RNS_Polynomial o = polynomial_RNS_view(&out_storage, sample_part(p->out, k), row_mask);
            RNS_Polynomial x =
                polynomial_RNS_view(&key_storage, sample_part(p->ksk[i], k), row_mask);
            polynomial_mul_subto_RNS_polynomial(o, x, factor);
        }
    }
    free_RNS_polynomial(tmp);
}

void gadget_mul_subto_automorphism(RNS_MLWE out, RNS_MLWE *ksk, const GadgetDigits *digits,
                                   uint64_t gen)
{
    if (digits->n == 0)
        return;
    RNS_Polynomial first = (RNS_Polynomial)digits->digit[0].handle;
    RNS_Base base = first->base;
    const uint64_t N = base->N;
    const int mul_domain = digits->digit[0].domain == ARITH_DOMAIN_MUL;

    uint32_t *idx = NULL;
    if (mul_domain && gen != 1)
    {
        idx = (uint32_t *)safe_malloc(N * sizeof(uint32_t));
        polynomial_RNS_automorphism_index(idx, N, gen);
    }
    AutomorphismProduct p = {out, ksk, digits, gen, idx, mul_domain};
    rns_rows_for(N, first->rns_mask, 1, automorphism_product_row, &p);
    free(idx);

    // As in gadget_mul_accumulate: the mask the last product gives every row.
    for (uint64_t k = 0; k <= out->r; k++)
        sample_part(out, k)->rns_mask =
            sample_part(ksk[digits->n - 1], k)->rns_mask & first->rns_mask;
}
