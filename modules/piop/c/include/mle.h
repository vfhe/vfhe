// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Dense multilinear-extension (RNS_Polynomial vector) operations. Defined in
    // piop/c/src/mle.c; declared here so the CFFI preamble sees the prototypes.
    void mle_dense_poly_add(RNS_Polynomial *out, RNS_Polynomial *in1, RNS_Polynomial *in2,
                            uint64_t size);
    void mle_dense_poly_sub(RNS_Polynomial *out, RNS_Polynomial *in1, RNS_Polynomial *in2,
                            uint64_t size);
    void mle_dense_poly_scale(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial scale,
                              uint64_t size);
    void mle_dense_poly_scale_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t scale,
                                     uint64_t size);
    // Variable binding, one entry point per pair layout: adjacent entries for
    // the LSB variable (pairs), the two table halves for the MSB variable
    // (halves), and stride-computed indices for any position (the generic
    // fallback). `size` is the output (folded) table size, 2^(num_vars - 1).
    void mle_dense_poly_evaluate_pairs(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial a,
                                       uint64_t size);
    void mle_dense_poly_evaluate_pairs_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t a,
                                              uint64_t size);
    void mle_dense_poly_evaluate_halves(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial a,
                                        uint64_t size);
    void mle_dense_poly_evaluate_halves_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t a,
                                               uint64_t size);
    void mle_dense_poly_evaluate(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial a,
                                 uint64_t num_vars, uint64_t eval_var_idx);
    void mle_dense_poly_evaluate_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t a,
                                        uint64_t num_vars, uint64_t eval_var_idx);

#ifdef __cplusplus
}
#endif
