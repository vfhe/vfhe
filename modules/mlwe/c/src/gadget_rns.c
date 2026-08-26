// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "misc.h"

// Gadget decomposition against the RNS base.
//
// The gadget here *is* the prime factorisation: a value is split into its
// residues, one key-switch key per prime, and the products summed. That makes
// the whole file RNS-specific by nature -- another representation decomposes
// against a different gadget, or none. The signatures stay generic so the
// key switch in mlwe.c can call it without knowing any of that.

static void gadget_mul_accumulate(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly,
                                  int subtract)
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
        if (mask & (1ULL << j))
        {
            // The j-th residue lifted to the key's ring, then transformed so
            // the multiply below is pointwise.
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

void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly)
{
    gadget_mul_accumulate(out, ksk, poly, 0);
}

void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly)
{
    gadget_mul_accumulate(out, ksk, poly, 1);
}
