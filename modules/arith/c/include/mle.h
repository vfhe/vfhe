// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#ifndef VFHE_MLE_H
#define VFHE_MLE_H
#include <arith.h>
#include <arith_generic.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Dense multilinear-extension (const ArithElement *vector) operations. Defined in
    // arith/c/src/mle.c; declared here so the CFFI preamble sees the prototypes.
    //
    // Each computes its output entries independently. A large table is computed on up to the
    // library's thread limit (parallel.h), and the result does not depend on the number of
    // threads; while the execution tape records (ARITH_GENERIC_PATH), everything runs on the
    // calling thread, in entry order.
    void mle_dense_poly_add(ArithRing ring, ArithElement *out, ArithElement *in1, ArithElement *in2,
                            uint64_t size);
    void mle_dense_poly_sub(ArithRing ring, ArithElement *out, ArithElement *in1, ArithElement *in2,
                            uint64_t size);
    void mle_dense_poly_scale(ArithRing ring, ArithElement *out, ArithElement *in,
                              const ArithElement *scale, uint64_t size);
    void mle_dense_poly_scale_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                     uint64_t scale, uint64_t size);
    // Variable binding, one entry point per pair layout: adjacent entries for
    // the LSB variable (pairs), the two table halves for the MSB variable
    // (halves), and stride-computed indices for any position (the generic
    // fallback). `size` is the output (folded) table size, 2^(num_vars - 1).
    void mle_dense_poly_evaluate_pairs(ArithRing ring, ArithElement *out, ArithElement *in,
                                       const ArithElement *a, uint64_t size);
    void mle_dense_poly_evaluate_pairs_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                              uint64_t a, uint64_t size);
    void mle_dense_poly_evaluate_halves(ArithRing ring, ArithElement *out, ArithElement *in,
                                        const ArithElement *a, uint64_t size);
    void mle_dense_poly_evaluate_halves_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                               uint64_t a, uint64_t size);
    void mle_dense_poly_evaluate(ArithRing ring, ArithElement *out, ArithElement *in,
                                 const ArithElement *a, uint64_t num_vars, uint64_t eval_var_idx);
    void mle_dense_poly_evaluate_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                        uint64_t a, uint64_t num_vars, uint64_t eval_var_idx);
    // out[j] = sum_(i < k) coeff[i] * tables[i][j] for j < size: a linear combination of k
    // tables of `size` entries each, with ring-element coefficients. Every coefficient and table
    // entry must be in the mul domain and over the same primes as out[j]; out[j] is left in the
    // mul domain. `out` aliases no table or coefficient. With k = 0,
    // every out[j] is zero. Returns ARITH_BAD_DOMAIN, writing nothing, when an operand is not in
    // the mul domain.
    ArithStatus mle_dense_poly_linear_combination(ArithRing ring, ArithElement *out,
                                                  ArithElement *const *tables,
                                                  const ArithElement *coeff, uint64_t k,
                                                  uint64_t size);

#ifdef __cplusplus
}
#endif

#endif // VFHE_MLE_H
