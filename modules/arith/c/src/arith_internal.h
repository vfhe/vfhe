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

/* None of the vectorized kernels has a tail, so a length is required to be a
   whole number of lane groups and not merely at least one: at n = 15 an
   element-wise kernel computes the first 8 and leaves the rest as it found
   them. Every caller passes a power of two -- a row is `N` words and the
   split-degree paths pass `N / split_degree`, both powers of two -- so the
   dispatchers' `< MOD_MIN_VECTOR_LEN` guard is enough in practice. Preserve
   that if you add a caller. */

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

/* The same cutover for the 32-bit-word transform. Its own sweep chose 4096
   where the 64-bit one chose 2048, which is what halving the footprint should
   do: the leaf is sized by the block's data plus its slice of the tables, and
   both halve. Its curve is also far flatter -- at n = 65536 the spread across
   leaves 128..16384 was about 4%, against 22% for the 64-bit transform -- so
   the choice matters less here as well as landing higher. */
#define NTT_LEAF_ELEMENTS_W32 4096

void ntt_w32_precompute(NTT_Plan plan);
void ntt_w32_free(NTT_Plan plan);

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

/* Coefficients held as `uint32_t`, 16 to an AVX-512 vector instead of 8.
   Applies only where the lazy range fits a 32-bit lane: values live in
   [0, 4q), so `q <= 2^30`. That is the same ceiling the radix-2^32 Shoup form
   already carries, which is why the word width costs no modulus range that was
   not already spent.

   A prime under `RNS_NARROW_MAX_BITS` bits is stored narrow. This is a
   property of the prime, not a choice a caller makes, so two elements over the
   same prime always agree on width and no conversion between widths exists.

   Inputs and outputs are canonical, in [0, q); `n` is a multiple of 16.
   Element-wise operations at the conversion boundaries (`reduce_signed`,
   `reduce_array_mp`) have no narrow form: they read 64-bit or 128-bit inputs,
   so they reduce into a wide scratch row and the caller narrows. */
#define RNS_NARROW_MAX_BITS 30

static inline bool rns_prime_is_narrow(uint64_t q) { return q < (1ULL << RNS_NARROW_MAX_BITS); }

// Whether prime index `i` of this base is stored narrow. The base is the only
// authority; never re-derive this from a modulus at a call site, or a row can
// be allocated at one width and read at the other.
static inline bool rns_row_is_narrow(RNS_Base base, size_t i)
{
    return (base->narrow_mask >> i) & 1ULL;
}

/* Reaching a row.
 *
 * The rule for `rns_polynomial.c` and the `_rns` backend files is: **branch on
 * the width once per row, then run typed code**. Never per coefficient, and
 * never by converting a row to the other width -- a narrow row exists in order
 * to be half the bytes and twice the lanes, and both are lost the moment it is
 * widened. The macros below are that branch; what they call are the kernels.
 *
 * Where a body is identical apart from the word type -- the coefficient
 * shifts, the block products, the slot moves -- it lives in `rns_row_ops.inc`,
 * which is included twice, so there is one source and two instantiations
 * rather than a runtime test inside the loop.
 */
#define rns_row64(p, i)                                                                            \
    (assert(!rns_row_is_narrow((p)->base, (i)) && (p)->rows64[(i)] != NULL), (p)->rows64[(i)])
#define rns_row32(p, i)                                                                            \
    (assert(rns_row_is_narrow((p)->base, (i)) && (p)->rows32[(i)] != NULL), (p)->rows32[(i)])

/* One row through the element-wise kernels, at the row's own width: one branch
   per row, then a vectorized kernel over 16 lanes (narrow) or 8 (wide). `fn`
   names the 64-bit kernel; the narrow one is `fn##_w32`. */
#define RNS_ROW_BINOP(fn, out, a, b, i, n, mod)                                                    \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((out)->base, (i)))                                                   \
            fn##_w32((out)->rows32[(i)], (a)->rows32[(i)], (b)->rows32[(i)], (n), (mod));          \
        else                                                                                       \
            fn((out)->rows64[(i)], (a)->rows64[(i)], (b)->rows64[(i)], (n), (mod));                \
    } while (0)

#define RNS_ROW_UNOP(fn, out, a, i, n, mod)                                                        \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((out)->base, (i)))                                                   \
            fn##_w32((out)->rows32[(i)], (a)->rows32[(i)], (n), (mod));                            \
        else                                                                                       \
            fn((out)->rows64[(i)], (a)->rows64[(i)], (n), (mod));                                  \
    } while (0)

#define RNS_ROW_SCALAROP(fn, out, a, sc, i, n, mod)                                                \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((out)->base, (i)))                                                   \
            fn##_w32((out)->rows32[(i)], (a)->rows32[(i)], (sc), (n), (mod));                      \
        else                                                                                       \
            fn((out)->rows64[(i)], (a)->rows64[(i)], (sc), (n), (mod));                            \
    } while (0)

// Whole-row zero and copy, at the row's width.
#define RNS_ROW_ZERO(p, i, n)                                                                      \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((p)->base, (i)))                                                     \
            memset((p)->rows32[(i)], 0, (n) * sizeof(uint32_t));                                   \
        else                                                                                       \
            memset((p)->rows64[(i)], 0, (n) * sizeof(uint64_t));                                   \
    } while (0)

#define RNS_ROW_COPY(out, in, i, n)                                                                \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((out)->base, (i)))                                                   \
            memcpy((out)->rows32[(i)], (in)->rows32[(i)], (n) * sizeof(uint32_t));                 \
        else                                                                                       \
            memcpy((out)->rows64[(i)], (in)->rows64[(i)], (n) * sizeof(uint64_t));                 \
    } while (0)

/* Two rows of one prime have one width -- width is a function of the prime --
   so equality is a compare at that width, never a cross-width one. */
#define RNS_ROW_EQ(a, b, i, n)                                                                     \
    (rns_row_is_narrow((a)->base, (i))                                                             \
         ? memcmp((a)->rows32[(i)], (b)->rows32[(i)], (n) * sizeof(uint32_t)) == 0                 \
         : memcmp((a)->rows64[(i)], (b)->rows64[(i)], (n) * sizeof(uint64_t)) == 0)

/* Reducing one row into another crosses a modulus boundary, so it is the one
   element-wise operation whose source and destination can differ in width. */
#define RNS_ROW_REDUCE(out, in, oi, ii, n, mod)                                                    \
    do                                                                                             \
    {                                                                                              \
        const bool rr_on_ = rns_row_is_narrow((out)->base, (oi));                                  \
        const bool rr_in_ = rns_row_is_narrow((in)->base, (ii));                                   \
        if (rr_on_ && rr_in_)                                                                      \
            mod_eltwise_reduce_w32((out)->rows32[(oi)], (in)->rows32[(ii)], (n), (mod));           \
        else if (rr_on_)                                                                           \
            mod_eltwise_reduce_narrow_from_wide((out)->rows32[(oi)], (in)->rows64[(ii)], (n),      \
                                                (mod));                                            \
        else if (rr_in_)                                                                           \
            mod_eltwise_reduce_wide_from_narrow((out)->rows64[(oi)], (in)->rows32[(ii)], (n),      \
                                                (mod));                                            \
        else                                                                                       \
            mod_eltwise_reduce((out)->rows64[(oi)], (in)->rows64[(ii)], (n), (mod));               \
    } while (0)

// A signed 64-bit array reduced into one row: the samplers and the permutation.
#define RNS_ROW_REDUCE_SIGNED(out, src, i, n, mod)                                                 \
    do                                                                                             \
    {                                                                                              \
        if (rns_row_is_narrow((out)->base, (i)))                                                   \
            mod_eltwise_reduce_signed_w32((out)->rows32[(i)], (src), (n), (mod));                  \
        else                                                                                       \
            mod_eltwise_reduce_signed((out)->rows64[(i)], (src), (n), (mod));                      \
    } while (0)

// 32-bit-word element-wise kernels. See mod_w32.c for the bounds and for why
// every engine has them.
void mod_eltwise_mul_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_mul_addto_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n,
                               Modulus mod);
void mod_eltwise_mul_subto_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n,
                               Modulus mod);
void mod_eltwise_scale_w32(uint32_t *out, uint32_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_fma_w32(uint32_t *out, uint32_t *in, uint64_t scale, uint64_t n, Modulus mod);
void mod_eltwise_add_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_sub_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod);
void mod_eltwise_negate_w32(uint32_t *out, uint32_t *in, uint64_t n, Modulus mod);
void mod_eltwise_add_scalar_w32(uint32_t *out, uint32_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod);
void mod_eltwise_sub_scalar_w32(uint32_t *out, uint32_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod);
void mod_eltwise_reduce_w32(uint32_t *out, uint32_t *in, uint64_t n, Modulus mod);
void mod_eltwise_reduce_signed_w32(uint32_t *out, int64_t *in, uint64_t n, Modulus mod);

// The three width-changing kernels; see the note above their definitions.
void mod_narrow_w32(uint32_t *out, const uint64_t *in, uint64_t n);
void mod_widen_w32(uint64_t *out, const uint32_t *in, uint64_t n);
void mod_eltwise_reduce_narrow_from_wide(uint32_t *out, uint64_t *in, uint64_t n, Modulus mod);
void mod_eltwise_reduce_wide_from_narrow(uint64_t *out, uint32_t *in, uint64_t n, Modulus mod);

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
