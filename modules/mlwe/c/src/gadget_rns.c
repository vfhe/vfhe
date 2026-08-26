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
    ArithRing key_ring = ksk[0]->ring;
    const uint64_t mask = arith_rns_polynomial(poly)->rns_mask;

    ArithElement tmp;
    arith_new(key_ring, &tmp);
    uint64_t ksk_idx = 0;
    for (size_t j = 0; j < arith_rns_polynomial(&out->b)->base->l; j++)
    {
        if (mask & (1ULL << j))
        {
            // The j-th residue lifted to the key's ring, then transformed so
            // the multiply below is pointwise.
            polynomial_RNSc_mod_reduce_lifted((RNSc_Polynomial)arith_rns_polynomial(&tmp),
                                              (RNSc_Polynomial)arith_rns_polynomial(poly), j);
            tmp.domain = ARITH_DOMAIN_CANONICAL;
            arith_to_mul(key_ring, &tmp);
            if (subtract)
            {
                mlwe_RNS_mul_subto_by_poly(out, ksk[ksk_idx++], &tmp);
            }
            else
            {
                mlwe_RNS_mul_addto_by_poly(out, ksk[ksk_idx++], &tmp);
            }
        }
    }
    arith_free(key_ring, &tmp);
}

void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly)
{
    gadget_mul_accumulate(out, ksk, poly, 0);
}

void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, const ArithElement *poly)
{
    gadget_mul_accumulate(out, ksk, poly, 1);
}
