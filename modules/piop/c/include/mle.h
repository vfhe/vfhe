// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>
#include <arith_generic.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Dense multilinear-extension (const ArithElement *vector) operations. Defined in
    // piop/c/src/mle.c; declared here so the CFFI preamble sees the prototypes.
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

#ifdef __cplusplus
}
#endif
