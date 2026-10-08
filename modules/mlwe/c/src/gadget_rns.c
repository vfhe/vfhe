// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "rns_rows.h"
#include "util.h"

// Gadget decompositions against the RNS base.
//
// The default gadget here *is* the prime factorisation: a value is split into
// its residues, one key-switch key per prime, and the products summed. That
// makes the whole file RNS-specific by nature -- another representation
// decomposes against a different gadget, or none. The signatures stay generic
// so the key switch in mlwe.c can call it without knowing any of that.
//
// `balanced` picks the RNS gadget's digit: the centered residue, in
// (-p_j/2, p_j/2], or the residue as stored, in [0, p_j). Both are the same
// value mod p_j, so the sum is exact either way and the keys are the same; the
// centered one has a quarter of the second moment -- which is what the
// accumulated noise grows with -- at the same cost.
//
// A non-zero `log_base` asks for the radix gadget instead: every residue is
// further split into base-2^log_base digits, so the decomposition of x is
//
//     x = sum_j sum_k d_{j,k} * (2^(log_base*k) * e_j)   mod Q,
//
// with e_j the CRT idempotent of prime j and every digit below 2^log_base.
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
// error would be multiplied by an idempotent, which is not small mod Q.

uint64_t gadget_radix_digits(uint64_t prime, uint64_t log_base)
{
    assert(log_base > 0 && prime > 1);
    const uint64_t bits = (uint64_t)(64 - __builtin_clzll(prime));
    return (bits + log_base - 1) / log_base;
}

// Digit `d` of residue `j` of `source`, lifted to `tmp`'s ring: the whole
// residue for the RNS gadget, centered if `balanced`, one base-2^log_base digit
// of it for the radix one.
static void gadget_digit(RNSc_Polynomial tmp, RNSc_Polynomial source, size_t j, uint64_t log_base,
                         uint64_t d, bool balanced)
{
    if (log_base)
        polynomial_RNSc_decompose_digit(tmp, source, j, log_base, d);
    else if (balanced)
        polynomial_RNSc_mod_reduce_lifted_centered(tmp, source, j);
    else
        polynomial_RNSc_mod_reduce_lifted(tmp, source, j);
}

static uint64_t gadget_digits_of(RNS_Base base, size_t j, uint64_t log_base)
{
    return log_base ? gadget_radix_digits(base->mods[j]->q, log_base) : 1;
}

typedef struct
{
    RNS_MLWE out;
    RNS_MLWE *ksk;
    RNS_Polynomial source, key;
    uint64_t log_base;
    bool balanced;
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

    uint64_t ksk_idx = 0;
    for (size_t j = 0; j < base->l; j++)
    {
        if (!(p->source->rns_mask & (1ULL << j)))
            continue;
        const uint64_t digits = gadget_digits_of(base, j, p->log_base);
        for (uint64_t d = 0; d < digits; d++, ksk_idx++)
        {
            gadget_digit(tmp, (RNSc_Polynomial)p->source, j, p->log_base, d, p->balanced);
            polynomial_RNSc_to_RNS((RNS_Polynomial)tmp, tmp);
            for (uint64_t k = 0; k < parts; k++)
            {
                RNS_Polynomial o =
                    polynomial_RNS_view(&out_storage, sample_part(p->out, k), row_mask);
                RNS_Polynomial x =
                    polynomial_RNS_view(&key_storage, sample_part(p->ksk[ksk_idx], k), row_mask);
                if (p->subtract)
                    polynomial_mul_subto_RNS_polynomial(o, x, (RNS_Polynomial)tmp);
                else
                    polynomial_mul_addto_RNS_polynomial(o, x, (RNS_Polynomial)tmp);
            }
        }
    }
    free_RNS_polynomial(tmp);
}

static void gadget_mul_accumulate(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                  int subtract, uint64_t log_base, bool balanced)
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
    KeyProduct p = {out, ksk, source, key, log_base, balanced, subtract};
    rns_rows_for(key->base->N, key->rns_mask, 1, key_product_row, &p);

    // Each row's last product set that row's mask in its view; the mask of
    // the whole element is what that product gives on every row.
    uint64_t last = 0;
    for (size_t j = 0; j < key->base->l; j++)
    {
        if (source->rns_mask & (1ULL << j))
            last += gadget_digits_of(key->base, j, log_base);
    }
    for (uint64_t k = 0; k <= out->r; k++)
        sample_part(out, k)->rns_mask = sample_part(ksk[last - 1], k)->rns_mask & key->rns_mask;
}

void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                 uint64_t log_base, bool balanced)
{
    gadget_mul_accumulate(out, ksk, poly, 0, log_base, balanced);
}

void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                 uint64_t log_base, bool balanced)
{
    gadget_mul_accumulate(out, ksk, poly, 1, log_base, balanced);
}

// Digits stay in the mul domain on a fully split ring, where an automorphism
// just permutes them; otherwise they stay canonical, and each product
// transforms its permuted copy.
typedef struct
{
    GadgetDigits *out;
    RNS_Polynomial source;
    RNS_Base base;
    uint64_t log_base;
    bool balanced;
    int mul_domain;
} Decomposition;

static void decompose_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    const Decomposition *dec = (const Decomposition *)ctx;
    struct _RNS_Polynomial storage;
    uint64_t i = 0;
    for (size_t j = 0; j < dec->base->l; j++)
    {
        if (!(dec->source->rns_mask & (1ULL << j)))
            continue;
        const uint64_t digits = gadget_digits_of(dec->base, j, dec->log_base);
        for (uint64_t d = 0; d < digits; d++, i++)
        {
            RNSc_Polynomial digit = (RNSc_Polynomial)polynomial_RNS_view(
                &storage, (RNS_Polynomial)dec->out->digit[i].handle, 1ULL << row);
            gadget_digit(digit, (RNSc_Polynomial)dec->source, j, dec->log_base, d, dec->balanced);
            if (dec->mul_domain)
                polynomial_RNSc_to_RNS((RNS_Polynomial)digit, digit);
        }
    }
}

void gadget_decompose(GadgetDigits *out, RNS_MLWE *ksk, const ArithElement *poly, uint64_t log_base,
                      bool balanced)
{
    RNS_Polynomial source = arith_rns_polynomial(poly);
    RNS_Polynomial key = arith_rns_polynomial(&ksk[0]->b);
    const uint64_t mask = source->rns_mask;
    RNS_Base base = key->base;
    const int mul_domain = base->split_degree == 1;

    uint64_t n = 0;
    for (size_t j = 0; j < base->l; j++)
    {
        if (mask & (1ULL << j))
            n += gadget_digits_of(base, j, log_base);
    }
    out->n = n;
    out->digit = (ArithElement *)safe_malloc(n * sizeof(ArithElement));
    for (uint64_t i = 0; i < n; i++)
    {
        out->digit[i].handle = polynomial_new_RNS_polynomial(base->N, key->rns_mask, base);
        out->digit[i].domain = mul_domain ? ARITH_DOMAIN_MUL : ARITH_DOMAIN_CANONICAL;
    }
    // Each digit row is the residue lifted to that prime alone, so the rows
    // are independent tasks.
    Decomposition dec = {out, source, base, log_base, balanced, mul_domain};
    rns_rows_for(base->N, key->rns_mask, 1, decompose_row, &dec);
}

void gadget_decompose_digit(ArithElement *out, RNS_MLWE *ksk, const ArithElement *poly, uint64_t i,
                            uint64_t log_base, bool balanced)
{
    RNS_Polynomial source = arith_rns_polynomial(poly);
    RNS_Base base = arith_rns_polynomial(&ksk[0]->b)->base;
    RNSc_Polynomial digit = (RNSc_Polynomial)arith_rns_polynomial(out);
    const int mul_domain = base->split_degree == 1;
    for (size_t j = 0; j < base->l; j++)
    {
        if (!(source->rns_mask & (1ULL << j)))
            continue;
        const uint64_t digits = gadget_digits_of(base, j, log_base);
        if (i < digits)
        {
            gadget_digit(digit, (RNSc_Polynomial)source, j, log_base, i, balanced);
            if (mul_domain)
                polynomial_RNSc_to_RNS((RNS_Polynomial)digit, digit);
            out->domain = mul_domain ? ARITH_DOMAIN_MUL : ARITH_DOMAIN_CANONICAL;
            return;
        }
        i -= digits;
    }
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
