// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Sumcheck prover kernels over dense-MLE tables of RNS_Polynomial entries
    // (Libra-style, evaluation basis). Tables are compact: `size` live entries
    // where the round variable is the LSB of the index, i.e. the pairs
    // (table[2i], table[2i+1]) are the two halves f|x=0, f|x=1. Entries must
    // be in the NTT (RNS) representation. Defined in piop/c/src/sumcheck.c.

    // Round message for one multilinear table: g0 = sum_i T[2i] = g(0) and
    // g1 = sum_i T[2i+1] = g(1) (the round polynomial has degree 1).
    void sumcheck_round(RNS_Polynomial g0, RNS_Polynomial g1, RNS_Polynomial *table,
                        uint64_t size);

    // Degree-2 round message for the product of two tables: g_out[t] =
    // sum_i (f at t)·(g at t) for t in {0, 1, 2}, with the evaluation at
    // t = 2 extrapolated division-free as 2·T[2i+1] − T[2i].
    //
    // Folding with the round challenge is NOT a sumcheck kernel: it is
    // exactly binding the round variable, i.e. mle_dense_poly_evaluate
    // (mle.h) — the Python protocol folds through the MLE API.
    void sumcheck_prod2_round(RNS_Polynomial *g_out, RNS_Polynomial *tf,
                              RNS_Polynomial *tg, uint64_t size);

#ifdef __cplusplus
}
#endif
