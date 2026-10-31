// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#ifndef VFHE_SUMCHECK_H
#define VFHE_SUMCHECK_H
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
    //
    // Every round runs on up to the library's thread limit (parallel.h) once
    // its tables are large enough, and on the calling thread while the
    // execution tape records (arith_trace.h). The message is the same at any
    // number of threads.

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

    // Degree-3 round message for the product of three tables: g_out[t] =
    // sum (f at t)·(g at t)·(h at t) for t in {0, 1, 2, 3}, with the
    // evaluations at t = 2, 3 extrapolated division-free as lo + t·(hi − lo)
    // per factor. The three tables share their variable layout.
    void sumcheck_prod3_round_pairs(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                    ArithElement *tg, ArithElement *th, uint64_t size);
    void sumcheck_prod3_round_halves(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                     ArithElement *tg, ArithElement *th, uint64_t size);
    void sumcheck_prod3_round(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                              ArithElement *tg, ArithElement *th, uint64_t size,
                              uint64_t eval_var_idx);

    // Round message for the product of any number of tables, at any number
    // of nodes: g_out[t] = sum over the pairs of prod_j (lo_j + t (hi_j -
    // lo_j)) for t in {0, ..., nodes - 1}, the values at t >= 2 extrapolated
    // division-free by adding hi_j - lo_j once per node. `tables` holds
    // `factors` >= 1 tables sharing their variable layout (a table may
    // appear more than once) and g_out `nodes` >= 1 elements.
    void sumcheck_prod_round_pairs(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                                   ArithElement *const *tables, uint64_t factors, uint64_t size);
    void sumcheck_prod_round_halves(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                                    ArithElement *const *tables, uint64_t factors, uint64_t size);
    void sumcheck_prod_round(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                             ArithElement *const *tables, uint64_t factors, uint64_t size,
                             uint64_t eval_var_idx);

#ifdef __cplusplus
}
#endif

#endif // VFHE_SUMCHECK_H
