// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "util.h"

// Gadget decompositions against the RNS base.
//
// The default gadget here *is* the prime factorisation: a value is split into
// its residues, one key-switch key per prime, and the products summed. That
// makes the whole file RNS-specific by nature -- another representation
// decomposes against a different gadget, or none. The signatures stay generic
// so the key switch in mlwe.c can call it without knowing any of that.
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
// residue for the RNS gadget, one base-2^log_base digit of it for the radix one.
static void gadget_digit(RNSc_Polynomial tmp, RNSc_Polynomial source, size_t j, uint64_t log_base,
                         uint64_t d)
{
    if (log_base)
        polynomial_RNSc_decompose_digit(tmp, source, j, log_base, d);
    else
        polynomial_RNSc_mod_reduce_lifted(tmp, source, j);
}

static uint64_t gadget_digits_of(RNS_Base base, size_t j, uint64_t log_base)
{
    return log_base ? gadget_radix_digits(base->mods[j]->q, log_base) : 1;
}

static void gadget_mul_accumulate(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                  int subtract, uint64_t log_base)
{
    // This file knows the representation, so it calls the RNS entry points
    // rather than routing through the dispatcher: nothing here would gain from
    // a ring it cannot have. Only the per-ciphertext multiply below stays
    // generic -- that operation belongs to mlwe.c, over r+1 whole elements.
    RNS_Polynomial source = arith_rns_polynomial(poly);
    RNS_Polynomial key = arith_rns_polynomial(&ksk[0]->b);
    const uint64_t mask = source->rns_mask;

    RNSc_Polynomial tmp =
        (RNSc_Polynomial)polynomial_new_RNS_polynomial(key->base->N, key->rns_mask, key->base);
    ArithElement factor = {tmp, ARITH_DOMAIN_MUL};

    uint64_t ksk_idx = 0;
    for (size_t j = 0; j < key->base->l; j++)
    {
        if (!(mask & (1ULL << j)))
            continue;
        // The j-th residue lifted to the key's ring, then transformed so the
        // multiply below is pointwise -- as one piece for the RNS gadget, or
        // one digit at a time for the radix one.
        const uint64_t digits = gadget_digits_of(key->base, j, log_base);
        for (uint64_t d = 0; d < digits; d++)
        {
            gadget_digit(tmp, (RNSc_Polynomial)source, j, log_base, d);
            polynomial_RNSc_to_RNS((RNS_Polynomial)tmp, tmp);
            if (subtract)
            {
                mlwe_RNS_mul_subto_by_poly(out, ksk[ksk_idx++], &factor);
            }
            else
            {
                mlwe_RNS_mul_addto_by_poly(out, ksk[ksk_idx++], &factor);
            }
        }
    }
    free_RNS_polynomial(tmp);
}

void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                 uint64_t log_base)
{
    gadget_mul_accumulate(out, ksk, poly, 0, log_base);
}

void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                 uint64_t log_base)
{
    gadget_mul_accumulate(out, ksk, poly, 1, log_base);
}

// Digits stay in the mul domain on a fully split ring, where an automorphism
// just permutes them; otherwise they stay canonical, and each product
// transforms its permuted copy.
void gadget_decompose(GadgetDigits *out, RNS_MLWE *ksk, const ArithElement *poly, uint64_t log_base)
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

    uint64_t i = 0;
    for (size_t j = 0; j < base->l; j++)
    {
        if (!(mask & (1ULL << j)))
            continue;
        const uint64_t digits = gadget_digits_of(base, j, log_base);
        for (uint64_t d = 0; d < digits; d++, i++)
        {
            RNSc_Polynomial digit =
                (RNSc_Polynomial)polynomial_new_RNS_polynomial(base->N, key->rns_mask, base);
            gadget_digit(digit, (RNSc_Polynomial)source, j, log_base, d);
            if (mul_domain)
                polynomial_RNSc_to_RNS((RNS_Polynomial)digit, digit);
            out->digit[i].handle = digit;
            out->digit[i].domain = mul_domain ? ARITH_DOMAIN_MUL : ARITH_DOMAIN_CANONICAL;
        }
    }
}

uint64_t gadget_digit_count(RNS_MLWE *ksk, const ArithElement *poly, uint64_t log_base)
{
    const uint64_t mask = arith_rns_polynomial(poly)->rns_mask;
    RNS_Base base = arith_rns_polynomial(&ksk[0]->b)->base;
    uint64_t n = 0;
    for (size_t j = 0; j < base->l; j++)
        if (mask & (1ULL << j))
            n += gadget_digits_of(base, j, log_base);
    return n;
}

void gadget_decompose_digit(ArithElement *out, RNS_MLWE *ksk, const ArithElement *poly, uint64_t i,
                            uint64_t log_base)
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
            gadget_digit(digit, (RNSc_Polynomial)source, j, log_base, i);
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
    RNS_Polynomial tmp = polynomial_new_RNS_polynomial(N, first->rns_mask, base);
    ArithElement factor = {tmp, ARITH_DOMAIN_MUL};

    for (uint64_t i = 0; i < digits->n; i++)
    {
        RNS_Polynomial digit = (RNS_Polynomial)digits->digit[i].handle;
        if (mul_domain && gen == 1)
        {
            ArithElement unpermuted = {digit, ARITH_DOMAIN_MUL};
            mlwe_RNS_mul_subto_by_poly(out, ksk[i], &unpermuted);
            continue;
        }
        if (mul_domain)
        {
            polynomial_RNS_permute(tmp, digit, idx);
        }
        else
        {
            polynomial_RNSc_permute((RNSc_Polynomial)tmp, (RNSc_Polynomial)digit, gen);
            polynomial_RNSc_to_RNS(tmp, (RNSc_Polynomial)tmp);
        }
        mlwe_RNS_mul_subto_by_poly(out, ksk[i], &factor);
    }
    free_RNS_polynomial(tmp);
    free(idx);
}
