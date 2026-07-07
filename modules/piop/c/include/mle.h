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
    void mle_dense_poly_evaluate(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial a,
                                 uint64_t num_vars, uint64_t eval_var_idx);
    void mle_dense_poly_evaluate_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t a,
                                        uint64_t num_vars, uint64_t eval_var_idx);

#ifdef __cplusplus
}
#endif
