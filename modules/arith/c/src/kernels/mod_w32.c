// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>

#include "arith_internal.h"

/* Element-wise modular arithmetic on coefficients held as `uint32_t`, so an
   AVX-512 vector covers 16 of them instead of 8.
 *
 * Applies only where the lazy range fits a 32-bit lane: values live in
 * [0, 4q) and that needs q <= 2^30, which is the same ceiling the radix-2^32
 * Shoup form already carries (see mod_shoup_shift). `RNS_NARROW_MAX_BITS` is
 * the bound, and a row is stored narrow exactly when its prime is under it.
 *
 * These kernels need AVX512F, DQ and VL but not IFMA: 32-bit products go
 * through `vpmuludq`, which is exact here because every operand is below
 * 2^32. They sit under the same guard as the rest of `kernels/` only because
 * that is where the dispatchers live.
 *
 * All inputs and outputs are canonical, in [0, q).
 *
 * Two things every entry point here guarantees, because callers are per-row
 * loops that cannot reasonably check either:
 *
 *  - **Every length works.** The vector bodies consume 16 lanes at a time and
 *    would compute nothing at all for a shorter array, so each entry point
 *    routes `n < W32_MIN_VECTOR_LEN` to the scalar body. A row is `N` words but
 *    callers also pass `N / split_degree`, which can be under 16.
 *  - **Every engine has them.** Narrow storage is derived from the prime, so it
 *    is the same on every engine -- the engine is a build parameter and must
 *    not change how a ring is laid out. Without AVX-512 the scalar bodies are
 *    the whole implementation, exactly as `mod_portable.c` forwards to
 *    `mod_scalar.c` for the 64-bit families. */

#define W32_LANES 16
// Below this the vector bodies cannot run: one lane group is 16 coefficients.
#define W32_MIN_VECTOR_LEN 16

/* --- scalar bodies: the whole implementation without AVX-512, and the tail
   for a short array with it. Written against `modq` / `mul_modq`, so they
   agree with the 64-bit scalar kernels by construction. --- */

static void w32_mul_scalar(uint32_t *out, const uint32_t *a, const uint32_t *b, uint64_t n,
                           Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)mul_modq(a[i], b[i], mod);
}

static void w32_mul_addto_scalar(uint32_t *out, const uint32_t *a, const uint32_t *b, uint64_t n,
                                 Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)add_modq(out[i], mul_modq(a[i], b[i], mod), mod->q);
}

static void w32_mul_subto_scalar(uint32_t *out, const uint32_t *a, const uint32_t *b, uint64_t n,
                                 Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)sub_modq(out[i], mul_modq(a[i], b[i], mod), mod->q);
}

static void w32_scale_scalar(uint32_t *out, const uint32_t *a, uint64_t s, uint64_t n, Modulus mod)
{
    const uint64_t sr = modq(s, mod);
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)mul_modq(a[i], sr, mod);
}

static void w32_fma_scalar(uint32_t *out, const uint32_t *a, uint64_t s, uint64_t n, Modulus mod)
{
    const uint64_t sr = modq(s, mod);
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)add_modq(out[i], mul_modq(a[i], sr, mod), mod->q);
}

static void w32_add_scalar_body(uint32_t *out, const uint32_t *a, const uint32_t *b, uint64_t n,
                                Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)add_modq(a[i], b[i], mod->q);
}

static void w32_sub_scalar_body(uint32_t *out, const uint32_t *a, const uint32_t *b, uint64_t n,
                                Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)sub_modq(a[i], b[i], mod->q);
}

static void w32_negate_scalar(uint32_t *out, const uint32_t *a, uint64_t n, Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)negate_modq(a[i], mod->q);
}

static void w32_add_scalarv_scalar(uint32_t *out, const uint32_t *a, uint64_t s, uint64_t n,
                                   Modulus mod)
{
    const uint64_t sr = modq(s, mod);
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)add_modq(a[i], sr, mod->q);
}

static void w32_sub_scalarv_scalar(uint32_t *out, const uint32_t *a, uint64_t s, uint64_t n,
                                   Modulus mod)
{
    const uint64_t sr = modq(s, mod);
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)sub_modq(a[i], sr, mod->q);
}

/* --- width changes. Only three operations legitimately change a word width,
   and each is here as one vectorized pass rather than a widening copy plus a
   second kernel:

     - narrowing a 64-bit result into a narrow row (a sampler, or a reduction
       whose destination modulus is narrow),
     - widening a narrow row into a 64-bit buffer, which after the 32-bit-word
       transform lands is needed nowhere in the row layer,
     - reducing across a modulus boundary, where the source row and the
       destination row belong to different primes and so may differ in width.

   Everything else -- add, sub, mul, scale, fma, negate, the block products,
   the coefficient shifts -- keeps both operands at one width and never
   converts. --- */

static void w32_narrow_scalar(uint32_t *out, const uint64_t *in, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        assert(in[i] <= UINT32_MAX);
        out[i] = (uint32_t)in[i];
    }
}

static void w32_widen_scalar(uint64_t *out, const uint32_t *in, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = in[i];
}

static void w32_reduce_signed_scalar(uint32_t *out, const int64_t *in, uint64_t n, Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
    {
        const int64_t v = in[i];
        const uint64_t a = modq(v < 0 ? (uint64_t)(-v) : (uint64_t)v, mod);
        out[i] = (uint32_t)(v < 0 ? negate_modq(a, mod->q) : a);
    }
}

static void w32_reduce_from_wide_scalar(uint32_t *out, const uint64_t *in, uint64_t n, Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)modq(in[i], mod);
}

static void w32_reduce_to_wide_scalar(uint64_t *out, const uint32_t *in, uint64_t n, Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = modq(in[i], mod);
}

static void w32_reduce_scalar(uint32_t *out, const uint32_t *in, uint64_t n, Modulus mod)
{
    for (uint64_t i = 0; i < n; i++)
        out[i] = (uint32_t)modq(in[i], mod);
}

#if VFHE_HAVE_AVX512IFMA

// The high 64 bits of a 64x64 product, minus the lo*lo partial: at most 1 too
// small, which the two conditional subtracts downstream absorb. Same trade the
// 64-bit families make; duplicated rather than shared because each kernel file
// is compiled on its own and these are the only two users.
static inline __m512i w32_mulhi_approx_64(__m512i x, __m512i y)
{
    const __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
    const __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
    const __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
    const __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);
    const __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);
    const __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);
    const __m512i mid2 = _mm512_add_epi64(z_hi_lo, _mm512_and_si512(z_lo_hi, lo_mask));
    return _mm512_add_epi64(_mm512_add_epi64(z_hi_hi, _mm512_srli_epi64(z_lo_hi, 32)),
                            _mm512_srli_epi64(mid2, 32));
}

/* One Barrett reduction of eight 64-bit products, each below 2^60.
   `q_hat` stays under 2^32, so its multiply by q is exact in `vpmuludq`. */
static inline __m512i w32_reduce_prod(__m512i prod, __m512i q64, __m512i barr_lo,
                                      unsigned prod_right_shift)
{
    const __m512i c1 = _mm512_srli_epi64(prod, prod_right_shift);
    const __m512i q_hat = w32_mulhi_approx_64(c1, barr_lo);
    __m512i r = _mm512_sub_epi64(prod, _mm512_mul_epu32(q_hat, q64));
    r = _mm512_min_epu64(r, _mm512_sub_epi64(r, q64));
    return _mm512_min_epu64(r, _mm512_sub_epi64(r, q64));
}

/* 16 modular products at once. A 32x32 product does not fit its own lane, so
   the vector is split into its even and odd 32-bit positions -- `vpmuludq`
   reads the low half of each 64-bit lane -- reduced as two vectors of eight
   64-bit products, and recombined. That recombination is the price of the
   packed layout; what it buys is half the memory traffic. */
static inline __m512i w32_prodmod(__m512i a, __m512i b, __m512i q64, __m512i barr_lo,
                                  unsigned prod_right_shift)
{
    const __m512i pe = _mm512_mul_epu32(a, b);
    const __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32));
    const __m512i re = w32_reduce_prod(pe, q64, barr_lo, prod_right_shift);
    const __m512i ro = w32_reduce_prod(po, q64, barr_lo, prod_right_shift);
    // both results are below q < 2^32, so each occupies the low half of its
    // 64-bit lane exactly
    return _mm512_or_si512(re, _mm512_slli_epi64(ro, 32));
}

// Locals every kernel below needs. `q32` is the modulus broadcast over 32-bit
// lanes for the add/sub family; `q64` over 64-bit lanes for the product path.
#define W32_LOCALS                                                                                 \
    __attribute__((unused)) const __m512i q32 = _mm512_set1_epi32((int)mod->q);                    \
    __attribute__((unused)) const __m512i q64 = _mm512_set1_epi64((long long)mod->q);              \
    __attribute__((unused)) const __m512i barr_lo = _mm512_set1_epi64((long long)mod->barr_lo);    \
    __attribute__((unused)) const unsigned prs = (unsigned)mod->prod_right_shift

static inline __m512i w32_add(__m512i a, __m512i b, __m512i q32)
{
    // a, b < q <= 2^30, so the sum cannot leave a 32-bit lane
    const __m512i s = _mm512_add_epi32(a, b);
    return _mm512_min_epu32(s, _mm512_sub_epi32(s, q32));
}

static inline __m512i w32_sub(__m512i a, __m512i b, __m512i q32)
{
    /* The subtraction wraps exactly when b > a. Adding q back then also wraps,
       landing on the right residue, and the wrapped value is the smaller of
       the two -- so one unsigned min selects the correct branch. */
    const __m512i d = _mm512_sub_epi32(a, b);
    return _mm512_min_epu32(d, _mm512_add_epi32(d, q32));
}

static void w32_mul_vec(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in1 + i);
        const __m512i b = _mm512_loadu_si512((const __m512i *)in2 + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_prodmod(a, b, q64, barr_lo, prs));
    }
}

static void w32_mul_addto_vec(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in1 + i);
        const __m512i b = _mm512_loadu_si512((const __m512i *)in2 + i);
        const __m512i o = _mm512_loadu_si512((const __m512i *)out + i);
        _mm512_storeu_si512((__m512i *)out + i,
                            w32_add(o, w32_prodmod(a, b, q64, barr_lo, prs), q32));
    }
}

static void w32_mul_subto_vec(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in1 + i);
        const __m512i b = _mm512_loadu_si512((const __m512i *)in2 + i);
        const __m512i o = _mm512_loadu_si512((const __m512i *)out + i);
        _mm512_storeu_si512((__m512i *)out + i,
                            w32_sub(o, w32_prodmod(a, b, q64, barr_lo, prs), q32));
    }
}

/* `scale` and `fma` multiply by a scalar fixed across the array. A fixed
   operand admits Shoup's form -- precompute s' = floor(s * 2^32 / q) once and
   the reduction is a multiply-high, a multiply and a subtract -- so they do
   not pay the two-varying-operand Barrett above. The multiply-high is the top
   half of `vpmuludq`, which is exactly the quotient at this radix. */
static inline __m512i w32_shoupmod(__m512i a, __m512i s64, __m512i sp64, __m512i q64)
{
    const __m512i ae = _mm512_and_si512(a, _mm512_set1_epi64(0x00000000ffffffff));
    const __m512i ao = _mm512_srli_epi64(a, 32);

    const __m512i qe = _mm512_srli_epi64(_mm512_mul_epu32(sp64, ae), 32);
    __m512i re = _mm512_sub_epi64(_mm512_mul_epu32(s64, ae), _mm512_mul_epu32(qe, q64));
    re = _mm512_min_epu64(re, _mm512_sub_epi64(re, q64));

    const __m512i qo = _mm512_srli_epi64(_mm512_mul_epu32(sp64, ao), 32);
    __m512i ro = _mm512_sub_epi64(_mm512_mul_epu32(s64, ao), _mm512_mul_epu32(qo, q64));
    ro = _mm512_min_epu64(ro, _mm512_sub_epi64(ro, q64));

    return _mm512_or_si512(re, _mm512_slli_epi64(ro, 32));
}

#define W32_SHOUP_LOCALS(scalar)                                                                   \
    const uint64_t s_ = modq((scalar), mod);                                                       \
    const __m512i s64 = _mm512_set1_epi64((long long)s_);                                          \
    const __m512i sp64 = _mm512_set1_epi64((long long)(((unsigned __int128)s_ << 32) / mod->q))

static void w32_scale_vec(uint32_t *out, uint32_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    W32_SHOUP_LOCALS(scale);
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_shoupmod(a, s64, sp64, q64));
    }
}

static void w32_fma_vec(uint32_t *out, uint32_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    W32_SHOUP_LOCALS(scale);
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in + i);
        const __m512i o = _mm512_loadu_si512((const __m512i *)out + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_add(o, w32_shoupmod(a, s64, sp64, q64), q32));
    }
}

static void w32_add_vec(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in1 + i);
        const __m512i b = _mm512_loadu_si512((const __m512i *)in2 + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_add(a, b, q32));
    }
}

static void w32_sub_vec(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in1 + i);
        const __m512i b = _mm512_loadu_si512((const __m512i *)in2 + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_sub(a, b, q32));
    }
}

static void w32_negate_vec(uint32_t *out, uint32_t *in, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    const __m512i zero = _mm512_setzero_si512();
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in + i);
        // 0 negates to 0, not to q
        const __mmask16 nz = _mm512_cmpneq_epu32_mask(a, zero);
        _mm512_storeu_si512((__m512i *)out + i, _mm512_maskz_sub_epi32(nz, q32, a));
    }
}

static void w32_add_scalar_vec(uint32_t *out, uint32_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod)
{
    W32_LOCALS;
    const __m512i s = _mm512_set1_epi32((int)modq(scalar, mod));
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_add(a, s, q32));
    }
}

static void w32_sub_scalar_vec(uint32_t *out, uint32_t *in, uint64_t scalar, uint64_t n,
                               Modulus mod)
{
    W32_LOCALS;
    const __m512i s = _mm512_set1_epi32((int)modq(scalar, mod));
    for (size_t i = 0; i < n / W32_LANES; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in + i);
        _mm512_storeu_si512((__m512i *)out + i, w32_sub(a, s, q32));
    }
}

/* Barrett-reduce eight arbitrary 64-bit words. Same constants the 64-bit
   families use, so it agrees with them by construction. */
static inline __m512i w32_reduce_word8(__m512i v, __m512i q64, __m512i barr_lo, unsigned prs)
{
    const __m512i c1 = _mm512_srli_epi64(v, prs);
    const __m512i q_hat = w32_mulhi_approx_64(c1, barr_lo);
    __m512i r = _mm512_sub_epi64(v, _mm512_mullo_epi64(q_hat, q64));
    r = _mm512_min_epu64(r, _mm512_sub_epi64(r, q64));
    return _mm512_min_epu64(r, _mm512_sub_epi64(r, q64));
}

// Eight 64-bit words to eight 32-bit lanes, and back. One instruction each.
static void w32_narrow_vec(uint32_t *out, const uint64_t *in, uint64_t n)
{
    for (uint64_t i = 0; i < n / 8; i++)
        _mm256_storeu_si256((__m256i *)(out + 8 * i),
                            _mm512_cvtepi64_epi32(_mm512_loadu_si512((const __m512i *)in + i)));
}

static void w32_widen_vec(uint64_t *out, const uint32_t *in, uint64_t n)
{
    for (uint64_t i = 0; i < n / 8; i++)
        _mm512_storeu_si512((__m512i *)out + i, _mm512_cvtepu32_epi64(_mm256_loadu_si256(
                                                    (const __m256i *)(in + 8 * i))));
}

static void w32_reduce_signed_vec(uint32_t *out, const int64_t *in, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    const __m512i zero = _mm512_setzero_si512();
    for (uint64_t i = 0; i < n / 8; i++)
    {
        const __m512i a = _mm512_loadu_si512((const __m512i *)in + i);
        const __mmask8 neg = _mm512_movepi64_mask(a);
        const __m512i abs = _mm512_mask_sub_epi64(a, neg, zero, a);
        __m512i r = w32_reduce_word8(abs, q64, barr_lo, prs);
        // negate the ones that were negative, leaving zero at zero
        const __m512i nr = _mm512_sub_epi64(q64, r);
        const __mmask8 flip = neg & _mm512_cmpneq_epu64_mask(r, zero);
        r = _mm512_mask_blend_epi64(flip, r, nr);
        _mm256_storeu_si256((__m256i *)(out + 8 * i), _mm512_cvtepi64_epi32(r));
    }
}

static void w32_reduce_from_wide_vec(uint32_t *out, const uint64_t *in, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (uint64_t i = 0; i < n / 8; i++)
    {
        const __m512i r =
            w32_reduce_word8(_mm512_loadu_si512((const __m512i *)in + i), q64, barr_lo, prs);
        _mm256_storeu_si256((__m256i *)(out + 8 * i), _mm512_cvtepi64_epi32(r));
    }
}

static void w32_reduce_to_wide_vec(uint64_t *out, const uint32_t *in, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (uint64_t i = 0; i < n / 8; i++)
    {
        const __m512i v = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)(in + 8 * i)));
        _mm512_storeu_si512((__m512i *)out + i, w32_reduce_word8(v, q64, barr_lo, prs));
    }
}

static void w32_reduce_vec(uint32_t *out, const uint32_t *in, uint64_t n, Modulus mod)
{
    W32_LOCALS;
    for (uint64_t i = 0; i < n / 8; i++)
    {
        const __m512i v = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)(in + 8 * i)));
        const __m512i r = w32_reduce_word8(v, q64, barr_lo, prs);
        _mm256_storeu_si256((__m256i *)(out + 8 * i), _mm512_cvtepi64_epi32(r));
    }
}

#endif // VFHE_HAVE_AVX512IFMA

/* --- entry points: length guard first, then whatever this engine has --- */

#if VFHE_HAVE_AVX512IFMA
#define W32_RUN(vec_call, scalar_call)                                                             \
    do                                                                                             \
    {                                                                                              \
        if (n < W32_MIN_VECTOR_LEN)                                                                \
            scalar_call;                                                                           \
        else                                                                                       \
            vec_call;                                                                              \
    } while (0)
#else
#define W32_RUN(vec_call, scalar_call)                                                             \
    do                                                                                             \
    {                                                                                              \
        scalar_call;                                                                               \
    } while (0)
#endif

void mod_eltwise_mul_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_RUN(w32_mul_vec(out, in1, in2, n, mod), w32_mul_scalar(out, in1, in2, n, mod));
}

void mod_eltwise_mul_addto_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_RUN(w32_mul_addto_vec(out, in1, in2, n, mod), w32_mul_addto_scalar(out, in1, in2, n, mod));
}

void mod_eltwise_mul_subto_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_RUN(w32_mul_subto_vec(out, in1, in2, n, mod), w32_mul_subto_scalar(out, in1, in2, n, mod));
}

void mod_eltwise_scale_w32(uint32_t *out, uint32_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    W32_RUN(w32_scale_vec(out, in, scale, n, mod), w32_scale_scalar(out, in, scale, n, mod));
}

void mod_eltwise_fma_w32(uint32_t *out, uint32_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    W32_RUN(w32_fma_vec(out, in, scale, n, mod), w32_fma_scalar(out, in, scale, n, mod));
}

void mod_eltwise_add_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_RUN(w32_add_vec(out, in1, in2, n, mod), w32_add_scalar_body(out, in1, in2, n, mod));
}

void mod_eltwise_sub_w32(uint32_t *out, uint32_t *in1, uint32_t *in2, uint64_t n, Modulus mod)
{
    W32_RUN(w32_sub_vec(out, in1, in2, n, mod), w32_sub_scalar_body(out, in1, in2, n, mod));
}

void mod_eltwise_negate_w32(uint32_t *out, uint32_t *in, uint64_t n, Modulus mod)
{
    W32_RUN(w32_negate_vec(out, in, n, mod), w32_negate_scalar(out, in, n, mod));
}

void mod_eltwise_add_scalar_w32(uint32_t *out, uint32_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod)
{
    W32_RUN(w32_add_scalar_vec(out, in, scalar, n, mod),
            w32_add_scalarv_scalar(out, in, scalar, n, mod));
}

void mod_eltwise_sub_scalar_w32(uint32_t *out, uint32_t *in, uint64_t scalar, uint64_t n,
                                Modulus mod)
{
    W32_RUN(w32_sub_scalar_vec(out, in, scalar, n, mod),
            w32_sub_scalarv_scalar(out, in, scalar, n, mod));
}

/* The width-changing entry points. These take `n / 8` lanes rather than
   `n / 16`, since one side is 64-bit, so their guard is MOD_MIN_VECTOR_LEN. */
#if VFHE_HAVE_AVX512IFMA
#define W32_RUN8(vec_call, scalar_call)                                                            \
    do                                                                                             \
    {                                                                                              \
        if (n < MOD_MIN_VECTOR_LEN)                                                                \
            scalar_call;                                                                           \
        else                                                                                       \
            vec_call;                                                                              \
    } while (0)
#else
#define W32_RUN8(vec_call, scalar_call)                                                            \
    do                                                                                             \
    {                                                                                              \
        scalar_call;                                                                               \
    } while (0)
#endif

void mod_narrow_w32(uint32_t *out, const uint64_t *in, uint64_t n)
{
    W32_RUN8(w32_narrow_vec(out, in, n), w32_narrow_scalar(out, in, n));
}

void mod_widen_w32(uint64_t *out, const uint32_t *in, uint64_t n)
{
    W32_RUN8(w32_widen_vec(out, in, n), w32_widen_scalar(out, in, n));
}

void mod_eltwise_reduce_signed_w32(uint32_t *out, int64_t *in, uint64_t n, Modulus mod)
{
    W32_RUN8(w32_reduce_signed_vec(out, in, n, mod), w32_reduce_signed_scalar(out, in, n, mod));
}

void mod_eltwise_reduce_w32(uint32_t *out, uint32_t *in, uint64_t n, Modulus mod)
{
    W32_RUN8(w32_reduce_vec(out, in, n, mod), w32_reduce_scalar(out, in, n, mod));
}

void mod_eltwise_reduce_narrow_from_wide(uint32_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    W32_RUN8(w32_reduce_from_wide_vec(out, in, n, mod),
             w32_reduce_from_wide_scalar(out, in, n, mod));
}

void mod_eltwise_reduce_wide_from_narrow(uint64_t *out, uint32_t *in, uint64_t n, Modulus mod)
{
    W32_RUN8(w32_reduce_to_wide_vec(out, in, n, mod), w32_reduce_to_wide_scalar(out, in, n, mod));
}
