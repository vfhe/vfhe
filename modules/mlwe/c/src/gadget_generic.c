// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "gadget_rns.h"
#include "mlwe.h"
#include "rns_rows.h"
#include <arith_generic.h>

// The gadget products of gadget_rns.c through the generic interface, a
// whole element at a time: every digit is an element of its own, computed
// by arith_decompose_digit from the parameters the decomposition
// enumerates, and every product one arith_mul_addto or arith_mul_subto.
// Slower than the row kernels, which lift and multiply a row at a time and
// reduce once per sum; the same residues, since the sums are modular.

// Digit t of the decomposition, as an element over `ring`, canonical.
static void digit_of(ArithRing ring, ArithElement *digit, const ArithElement *poly,
                     const ElementDecomposition *e, uint64_t t)
{
    uint64_t primes, log_base, level, dropped;
    int centered;
    gadget_decomposition_parameters(e, t, &primes, &log_base, &level, &dropped, &centered);
    arith_decompose_digit(ring, digit, poly, primes, log_base, level, dropped, centered);
}

// out -= or += factor * ksk[t], component by component, in the mul domain.
static void products(RNS_MLWE out, RNS_MLWE key, const ArithElement *factor, int subtract)
{
    for (uint64_t k = 0; k <= out->r; k++)
    {
        if (subtract)
            arith_mul_subto(out->ring, mlwe_component(out, k), factor, mlwe_component(key, k));
        else
            arith_mul_addto(out->ring, mlwe_component(out, k), factor, mlwe_component(key, k));
    }
}

void gadget_mul_accumulate_generic(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                   int subtract, const GadgetParams *gadget_params)
{
    ArithRing ring = out->ring;
    ElementDecomposition *e = gadget_decomposition_new(ksk, poly, gadget_params);
    ArithElement digit;
    arith_new(ring, &digit);
    for (uint64_t t = 0; t < gadget_decomposition_digits(e); t++)
    {
        digit_of(ring, &digit, poly, e, t);
        arith_to_mul(ring, &digit);
        products(out, ksk[t], &digit, subtract);
    }
    arith_free(ring, &digit);
    gadget_decomposition_free(e);
}

void gadget_mul_automorphism_generic(RNS_MLWE out, RNS_MLWE *ksk, const GadgetDigits *digits,
                                     uint64_t gen, int subtract)
{
    ArithRing ring = out->ring;
    ArithElement digit, permuted;
    arith_new(ring, &digit);
    arith_new(ring, &permuted);
    for (uint64_t i = 0; i < digits->n; i++)
    {
        // Permuted in the canonical domain, multiplied in the mul domain.
        arith_copy(ring, &digit, &digits->digit[i]);
        ArithElement *factor = &digit;
        if (gen != 1)
        {
            arith_to_canonical(ring, &digit);
            arith_permute(ring, &permuted, &digit, gen);
            factor = &permuted;
        }
        arith_to_mul(ring, factor);
        products(out, ksk[i], factor, subtract);
    }
    arith_free(ring, &digit);
    arith_free(ring, &permuted);
}

void gadget_decompose_generic(GadgetDigits *out, RNS_MLWE *ksk, const ArithElement *poly,
                              const GadgetParams *gadget_params)
{
    ArithRing ring = ksk[0]->ring;
    RNS_Polynomial key = arith_rns_polynomial(&ksk[0]->b);
    ElementDecomposition *e = gadget_decomposition_new(ksk, poly, gadget_params);
    gadget_digits_new(out, gadget_decomposition_digits(e), key);
    for (uint64_t t = 0; t < out->n; t++)
    {
        const ArithDomain domain = out->digit[t].domain;
        digit_of(ring, &out->digit[t], poly, e, t);
        if (domain == ARITH_DOMAIN_MUL)
            arith_to_mul(ring, &out->digit[t]);
    }
    gadget_decomposition_free(e);
}
