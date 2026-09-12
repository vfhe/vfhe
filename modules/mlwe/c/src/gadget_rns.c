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
// The key array must hold one key per (prime, digit) pair, prime-major, and
// its keys must carry that gadget (`gadget_radix_digits` says how many digits
// a prime takes). What it buys is the bound on the products accumulated
// below: 2^log_base rather than the prime itself.
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
        const uint64_t digits = log_base ? gadget_radix_digits(key->base->mods[j]->q, log_base) : 1;
        for (uint64_t d = 0; d < digits; d++)
        {
            if (log_base)
                polynomial_RNSc_decompose_digit(tmp, (RNSc_Polynomial)source, j, log_base, d);
            else
                polynomial_RNSc_mod_reduce_lifted(tmp, (RNSc_Polynomial)source, j);
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
