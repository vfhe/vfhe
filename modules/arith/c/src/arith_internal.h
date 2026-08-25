// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#ifndef __ARITH_INTERNAL_H__
#define __ARITH_INTERNAL_H__

#include <arith.h>

// Size-generic scalar declarations (mod_scalar.c / ntt_scalar.c). Compiled into
// every engine: the vectorized kernels below need n >= 8 (element-wise) or
// n >= 16 (NTT) to do any work at all, so the dispatchers in mod.c / base.c hand
// shorter lengths to these, and the portable engine is built entirely on them.
void mod_eltwise_mul_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_addto_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               Modulus mod);
void mod_eltwise_mul_subto_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n,
                               Modulus mod);
void mod_eltwise_scale_gen(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_fma_gen(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_add_scalar_gen(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod);
void mod_eltwise_sub_scalar_gen(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod);
void mod_eltwise_negate_gen(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_add_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_sub_gen(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_reduce_gen(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_reduce_signed_gen(uint64_t *out, int64_t *in, uint64_t n, Modulus mod);
void mod_reduce_array_mp_gen(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                             Modulus mod);

// The NTT length below which the vectorized transforms cannot run: they consume
// two AVX512 lane groups per butterfly stage, and their twiddle tables are
// sized `n / 16`, so nothing is precomputed and no stage executes under it.
#define NTT_MIN_VECTOR_LEN 16
// Likewise for the element-wise kernels: one lane group, `n / 8` iterations.
#define MOD_MIN_VECTOR_LEN 8

void ntt_scalar_precompute(uint64_t n, Modulus mod, uint64_t root_of_unity, uint64_t ***out_ws);
void ntt_scalar_free_precompute(uint64_t **ws);
// `ws` is the only thing the plan cannot supply: forward and inverse read
// different tables. Length and modulus come from the plan.
void ntt_CT_NR_gen(uint64_t *a, uint64_t *ws, NTT_Plan plan);
void ntt_GS_RN_gen(uint64_t *a, uint64_t *ws, NTT_Plan plan);

// 32-bit declarations
void ntt_forward_32(uint64_t *out, uint64_t *in, NTT_Plan plan);
void ntt_reverse_32(uint64_t *out, uint64_t *in, NTT_Plan plan);
void mod_eltwise_mul_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_addto_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_subto_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_scale_32(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_fma_32(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_add_scalar_32(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod);
void mod_eltwise_sub_scalar_32(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod);
void mod_eltwise_negate_32(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_add_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_sub_32(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_reduce_32(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_reduce_signed_32(uint64_t *out, int64_t *in, uint64_t n, Modulus mod);
void mod_reduce_array_mp_32(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            Modulus mod);

// 50-bit declarations
void ntt_forward_50(uint64_t *out, uint64_t *in, NTT_Plan plan);
void ntt_reverse_50(uint64_t *out, uint64_t *in, NTT_Plan plan);
void mod_eltwise_mul_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_addto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_subto_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_scale_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_fma_50(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_add_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod);
void mod_eltwise_sub_scalar_50(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod);
void mod_eltwise_negate_50(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_add_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_sub_50(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_reduce_50(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_reduce_signed_50(uint64_t *out, int64_t *in, uint64_t n, Modulus mod);
void mod_reduce_array_mp_50(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            Modulus mod);

// 64-bit declarations
void ntt_forward_64(uint64_t *out, uint64_t *in, NTT_Plan plan);
void ntt_reverse_64(uint64_t *out, uint64_t *in, NTT_Plan plan);
void mod_eltwise_mul_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_addto_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_subto_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_scale_64(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_fma_64(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_add_scalar_64(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod);
void mod_eltwise_sub_scalar_64(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod);
void mod_eltwise_negate_64(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_add_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_sub_64(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_reduce_64(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_reduce_signed_64(uint64_t *out, int64_t *in, uint64_t n, Modulus mod);
void mod_reduce_array_mp_64(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                            Modulus mod);

// Exact unsigned __int128 reference for the pseudo-Mersenne field (pmf.c). Like
// the _gen kernels above, these are compiled into every engine, and for a related
// reason: when the AVX-512 pmf kernels land they go behind an ISA guard and these
// stay outside it, so the tuned build can differential-test against them
// in-process. Kept visible rather than static so a C test can call them too.
// Deliberately NOT structured like the planned vector kernel -- they are the
// oracle, and a shared shape would let one shared mistake hide in both.
void pmf_ref_add(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params);
void pmf_ref_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params);
void pmf_ref_neg(uint64_t *out, const uint64_t *a, PMFParams params);
void pmf_ref_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params);
// T is limbs+1 words, each below 2^60, and is clobbered. Requires limbs >= 3.
void pmf_ref_reduce_wide(uint64_t *out, uint64_t *T, PMFParams params);

#endif // __ARITH_INTERNAL_H__
