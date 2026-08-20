// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"

// -------------------------------------------------------------
// Sumcheck prover kernels over dense-MLE (RNS_Polynomial) tables
// -------------------------------------------------------------
// Libra-style evaluation-basis prover: per round, accumulate the round
// polynomial's evaluations from the table halves. Tables are compact with
// the round variable at the LSB: pairs (table[2i], table[2i+1]). Entries
// must be in NTT (RNS) form. Folding with the challenge is binding the
// round variable — mle_dense_poly_evaluate in mle.c, not duplicated here.

void sumcheck_round(RNS_Polynomial g0, RNS_Polynomial g1, RNS_Polynomial *table,
                    uint64_t size)
{
    polynomial_RNS_zero(g0);
    polynomial_RNS_zero(g1);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        polynomial_add_RNS_polynomial(g0, g0, table[2 * i]);
        polynomial_add_RNS_polynomial(g1, g1, table[2 * i + 1]);
    }
}

void sumcheck_prod2_round(RNS_Polynomial *g_out, RNS_Polynomial *tf,
                          RNS_Polynomial *tg, uint64_t size)
{
    incNTT ntt = tf[0]->ntt;
    RNS_Polynomial tmp1 = polynomial_new_RNS_polynomial(ntt->N, tf[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, tg[0]->rns_mask, ntt);

    polynomial_RNS_zero(g_out[0]);
    polynomial_RNS_zero(g_out[1]);
    polynomial_RNS_zero(g_out[2]);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        // t = 0 and t = 1: the halves themselves.
        polynomial_mul_addto_RNS_polynomial(g_out[0], tf[2 * i], tg[2 * i]);
        polynomial_mul_addto_RNS_polynomial(g_out[1], tf[2 * i + 1], tg[2 * i + 1]);
        // t = 2, extrapolated: 2·hi − lo for each factor.
        polynomial_scale_RNS_polynomial(tmp1, tf[2 * i + 1], 2);
        polynomial_sub_RNS_polynomial(tmp1, tmp1, tf[2 * i]);
        polynomial_scale_RNS_polynomial(tmp2, tg[2 * i + 1], 2);
        polynomial_sub_RNS_polynomial(tmp2, tmp2, tg[2 * i]);
        polynomial_mul_addto_RNS_polynomial(g_out[2], tmp1, tmp2);
    }

    free_RNS_polynomial(tmp1);
    free_RNS_polynomial(tmp2);
}
