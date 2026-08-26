// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>
#include <arith_generic.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Sumcheck prover kernels over dense-MLE tables of const ArithElement *entries
    // (Libra-style, evaluation basis). Tables are compact: `size` live entries
    // (2^num_vars), in the ring's mul domain. The round variable may sit at
    // any position; each message has one entry point per pair layout —
    // adjacent entries for the LSB variable (_pairs), the two table halves for
    // the MSB variable (_halves), and stride-computed indices for any position
    // (the generic fallback taking eval_var_idx) — mirroring the mle.h binding
    // kernels. Defined in piop/c/src/sumcheck.c.

    // Round message for one multilinear table: g0 = sum over the pairs' low
    // entries = g(0) and g1 = sum over the high entries = g(1) (the round
    // polynomial has degree 1).
    void sumcheck_round_pairs(ArithRing ring, ArithElement *g0, ArithElement *g1,
                              ArithElement *table, uint64_t size);
    void sumcheck_round_halves(ArithRing ring, ArithElement *g0, ArithElement *g1,
                               ArithElement *table, uint64_t size);
    void sumcheck_round(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                        uint64_t size, uint64_t eval_var_idx);

    // Degree-2 round message for the product of two tables: g_out[t] =
    // sum (f at t)·(g at t) for t in {0, 1, 2}, with the evaluation at
    // t = 2 extrapolated division-free as 2·hi − lo per factor.
    //
    // Folding with the round challenge is NOT a sumcheck kernel: it is
    // exactly binding the round variable, i.e. mle_dense_poly_evaluate*
    // (mle.h) — the Python protocol folds through the MLE API.
    void sumcheck_prod2_round_pairs(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                    ArithElement *tg, uint64_t size);
    void sumcheck_prod2_round_halves(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                     ArithElement *tg, uint64_t size);
    void sumcheck_prod2_round(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                              ArithElement *tg, uint64_t size, uint64_t eval_var_idx);

#ifdef __cplusplus
}
#endif
