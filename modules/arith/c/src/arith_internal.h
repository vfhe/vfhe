// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#ifndef __ARITH_INTERNAL_H__
#define __ARITH_INTERNAL_H__

#include <arith.h>
#include <blake3.h>

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

/* The transforms' depth-first cutover, in elements. A sub-problem at or below
   this runs all its stages to completion while it is cache resident; above it
   the transform splits and recurses.

   The right value is a property of the machine, not of the algorithm, and it
   is sized by the block's data plus the block's slice of the twiddle tables --
   so it moves when that layout does. Swept on two machines with different L2
   against the compact broadcast tables `ntt_precompute_fwd` builds, and both
   chose 2048. The curve is asymmetric: guessing high is expensive and guessing
   low is nearly free, so an unsure value should err low. */
#define NTT_LEAF_ELEMENTS 2048

void ntt_scalar_precompute(uint64_t n, Modulus mod, uint64_t root_of_unity, uint64_t ***out_ws);
void ntt_scalar_free_precompute(uint64_t **ws);
// `ws` is the only thing the plan cannot supply: forward and inverse read
// different tables. Length and modulus come from the plan.
void ntt_CT_NR_gen(uint64_t *a, uint64_t *ws, NTT_Plan plan);
void ntt_GS_RN_gen(uint64_t *a, uint64_t *ws, NTT_Plan plan);

/* The vectorized kernels come in three families, one per Shoup radix, and this
   is the single place a modulus is matched to one. Every caller that needs the
   choice -- the element-wise dispatchers in mod.c, and both the table builder
   and the transform dispatchers in ntt.c -- must take it from here.
   `ntt_precompute_*` bakes the radix into the twiddle constants, so a table
   built for one radix and consumed by a kernel expecting another is silent
   corruption rather than a crash. That is why the choice is centralized and
   why an NTT_Plan carries the radix it was built at.

   The bound on each family is Harvey's lazy-reduction requirement 4q <= B for
   its radix B, which the transforms impose: a butterfly's un-reduced operand
   is whatever the previous stage left, so values live in [0, 4q), and the
   Shoup step is only in range while that operand is below B. Hence

       radix 2^32  ->  q < 2^30
       radix 2^52  ->  q < 2^50
       radix 2^64  ->  q < 2^62

   The element-wise kernels are bounded differently -- their inputs are already
   in [0, q), and what constrains the narrow family there is that a product of
   two of them stay exact in 64 bits, i.e. q < 2^32. One choice serves both
   because the transform's is the tighter, so a modulus admitted here is
   admitted by either. Do not relax it to the element-wise bound.

   MOD_CLASS32_MAX_BITS decides whether the radix-2^32 family is reachable at
   all. Every prime it accepts is also legal input to the IFMA family, which is
   faster where IFMA exists, so the narrow kernels are what a build without
   IFMA falls back to. At 0 the arm is unreachable and the comparison folds
   away at compile time, leaving a single branch. A build may override it, and
   the cross-class test does. */
#ifndef MOD_CLASS32_MAX_BITS
#if VFHE_HAVE_AVX512IFMA
#define MOD_CLASS32_MAX_BITS 0
#else
#define MOD_CLASS32_MAX_BITS 30
#endif
#endif

// The Shoup radix a family reduces at, as its base-2 logarithm. Also the value
// NTT_Plan::shoup_shift carries, and what the dispatchers switch on.
#define MOD_SHIFT_32 32
#define MOD_SHIFT_50 52
#define MOD_SHIFT_64 64

static inline uint64_t mod_shoup_shift(uint64_t q)
{
    if (MOD_CLASS32_MAX_BITS && q < (1ULL << MOD_CLASS32_MAX_BITS))
        return MOD_SHIFT_32;
    return q < (1ULL << 50) ? MOD_SHIFT_50 : MOD_SHIFT_64;
}

/* A plan whose tables are built at `shoup_shift` instead of the radix
   `mod_shoup_shift` would pick for this modulus, so `ntt_forward` and
   `ntt_reverse` reach the matching family. Requires the modulus to be inside
   that radix's bound; the caller is responsible for that.

   A prime below several bounds is legal input to every family that accepts it,
   and an NTT is exact, so all of them owe the same words. That equality is the
   only construction that can see a bound set slightly too loose -- an ordinary
   test exercises a family at the modulus size its own dispatch chose, which is
   where its bound is least likely to be violated. Test-only; the library
   itself always goes through `ntt_new_plan`. */
NTT_Plan ntt_new_plan_at_shift(uint64_t n, Modulus mod, uint64_t shoup_shift);

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

// a^exp, with the exponent as `exp_limbs` little-endian 52-bit limbs (the form
// PMFParams holds p in). Square-and-multiply over pmf_ref_mul.
void pmf_ref_pow(uint64_t *out, const uint64_t *a, const uint64_t *exp, uint64_t exp_limbs,
                 PMFParams params);

// The scalar transform (pmf_ntt.c): the loops of pmf_vec_ntt_* over the element
// kernels above, on `n` elements laid out PMF_LANES words apart, in place. Same
// basis and order as the vector transform, and the oracle it is tested against.
void pmf_ref_ntt_forward(uint64_t *a, PMFNTTPlan plan);
void pmf_ref_ntt_inverse(uint64_t *a, PMFNTTPlan plan);

// One rejection-sampled element, drawn from `hasher`'s output stream starting at
// *stream_offset, which is advanced past whatever the draw consumed. Internal so
// that a caller sampling many elements -- the vector sampler -- keeps a single
// stream instead of deriving one per element.
void pmf_sample_stream(uint64_t *out, blake3_hasher *hasher, uint64_t *stream_offset,
                       PMFParams params);

#endif // __ARITH_INTERNAL_H__
