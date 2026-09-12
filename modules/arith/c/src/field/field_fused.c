// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The extension product as one pass, with the reductions delayed.
//
// field_vector.c's schoolbook form is 2d^2 + d - 1 whole-plane passes, each a
// *reduced* multiply-accumulate into a (2d-1)-plane accumulator that it
// streams through memory. Both halves of that are avoidable: the accumulator
// belongs in registers, and a sum of d products needs one reduction, not d.
//
// This file is the version that does neither. For a block of eight elements it
// holds the operands in registers, accumulates one output coefficient at a
// time into a redundant (hi, lo) pair, and reduces once. `madd52lo` / `madd52hi`
// accumulate into a 64-bit lane natively, which is what makes the delay free --
// the instruction already adds into its destination.
//
// It is not written per degree: `d` is a compile-time constant at every
// instantiation below, so the index arithmetic folds away and the loop unrolls
// exactly as a hand-written kernel would, from one source.
#include <arith.h>
#include "arith_internal.h"

#if VFHE_HAVE_AVX512IFMA

// The bound on the delay. A product of two values below q is below q^2, so its
// high half is below q^2 / 2^52; the accumulator holds d of them plus the carry
// out of the low half, and the reduction below needs that under 2^52. With
// q < 2^50 -- which the 2^52 Shoup family guarantees -- that admits d up to 15.
// Eight is the cap because it is also where the operands stop fitting in the
// register file: 3d + 2 live vectors, 26 of the 32 at d = 8.
#define FIELD_MAX_FUSED_D 8

// Elements per block: the lanes of one zmm at 64 bits, which is what the loads
// and stores below step by. It equals MOD_MIN_VECTOR_LEN today, but for its own
// reason -- that one is the length below which the eltwise kernels have nothing
// to do. `field_fused_applies` checks the length against this one.
#define FIELD_FUSED_LANES 8

// One 128-bit value reduced mod q: mod50.c's own _mm512_hexl_reduce_epi64,
// which is the general form of the reduction the per-pass kernels use on a
// single product. Folding v_hi in twice through 2^52 mod q is what lets the
// Barrett below see a value it can handle.
static inline __m512i field_reduce128(__m512i v_hi, __m512i v_lo, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q = _mm512_set1_epi64(-(int64_t)mod->q);
    const __m512i m_ifma = _mm512_set1_epi64(mod->ifma_barr_lo);
    const __m512i v_w1 = _mm512_set1_epi64(mod->mp_w1);

    __m512i h2 = _mm512_madd52hi_epu64(zero, v_hi, v_w1);
    __m512i l2 = _mm512_madd52lo_epu64(v_lo, v_hi, v_w1);
    __m512i h3 = _mm512_madd52hi_epu64(zero, h2, v_w1);
    __m512i l3 = _mm512_madd52lo_epu64(l2, h2, v_w1);
    h3 = _mm512_add_epi64(h3, _mm512_srli_epi64(l3, 52));
    l3 = _mm512_and_epi64(l3, mask52);

    const __m128i shift = _mm_cvtsi64_si128(mod->ifma_prod_right_shift);
    const __m128i shift_rev = _mm_cvtsi64_si128(52 - mod->ifma_prod_right_shift);
    __m512i c1 = _mm512_or_si512(_mm512_srl_epi64(l3, shift), _mm512_sll_epi64(h3, shift_rev));
    __m512i q_hat = _mm512_madd52hi_epu64(zero, c1, m_ifma);
    __m512i res = _mm512_and_epi64(_mm512_madd52lo_epu64(l3, q_hat, neg_q), mask52);
    res = _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
    return _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
}

// a * s for a broadcast s whose Shoup constant sp is already formed.
static inline __m512i field_shoup(__m512i a, __m512i s, __m512i sp, __m512i q_vec, __m512i neg_q)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    __m512i q_hat = _mm512_madd52hi_epu64(zero, sp, a);
    __m512i res = _mm512_and_epi64(
        _mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero, s, a), q_hat, neg_q), mask52);
    return _mm512_min_epu64(res, _mm512_sub_epi64(res, q_vec));
}

// Output coefficient k takes a_i against b_(k-i mod d), and the terms where
// i + j ran past d carry a factor of w -- so they take w*b_j instead, which is
// the only thing the quotient x^d = w costs. j = 0 is never one of them, which
// is why w*b_0 is never formed.
#define FIELD_FUSED_WRAPS(I, K, D) ((I) > (K))
#define FIELD_FUSED_PARTNER(I, K, D) (((K) + (D) - (I)) % (D))

// `out = a * b`, both plane arrays, over `nvec` blocks of eight elements.
// `d` is constant at every call site, so this unrolls completely.
static inline void field_fused_mul_impl(uint64_t *const *out, uint64_t *const *a,
                                        uint64_t *const *b, uint64_t nvec, const uint64_t d,
                                        uint64_t w, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i q_vec = _mm512_set1_epi64(mod->q);
    const __m512i neg_q = _mm512_set1_epi64(-(int64_t)mod->q);
    const uint64_t ws = modq(w, mod);
    const __m512i w_vec = _mm512_set1_epi64(ws);
    const __m512i w_pre = _mm512_set1_epi64((uint64_t)(((unsigned __int128)ws << 52) / mod->q));

    for (uint64_t v = 0; v < nvec; v++)
    {
        __m512i av[FIELD_MAX_FUSED_D], bv[FIELD_MAX_FUSED_D], wb[FIELD_MAX_FUSED_D];
        for (uint64_t i = 0; i < d; i++)
        {
            av[i] = ((const __m512i *)a[i])[v];
            bv[i] = ((const __m512i *)b[i])[v];
        }
        // In registers rather than as d - 1 passes over memory, which is worth
        // about a fifth of the kernel.
        for (uint64_t j = 1; j < d; j++)
            wb[j] = field_shoup(bv[j], w_vec, w_pre, q_vec, neg_q);

        for (uint64_t k = 0; k < d; k++)
        {
            __m512i lo = zero, hi = zero;
            for (uint64_t i = 0; i < d; i++)
            {
                const uint64_t j = FIELD_FUSED_PARTNER(i, k, d);
                const __m512i p = FIELD_FUSED_WRAPS(i, k, d) ? wb[j] : bv[j];
                lo = _mm512_madd52lo_epu64(lo, av[i], p);
                hi = _mm512_madd52hi_epu64(hi, av[i], p);
            }
            hi = _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52));
            lo = _mm512_and_epi64(lo, mask52);
            ((__m512i *)out[k])[v] = field_reduce128(hi, lo, mod);
        }
    }
}

// The same against one element broadcast over the whole run: the w-scaled
// partners are d - 1 scalars, formed once instead of per block.
static inline void field_fused_scale_impl(uint64_t *const *out, uint64_t *const *a,
                                          const uint64_t *s, uint64_t nvec, const uint64_t d,
                                          uint64_t w, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const uint64_t ws = modq(w, mod);
    __m512i bv[FIELD_MAX_FUSED_D], wb[FIELD_MAX_FUSED_D];
    for (uint64_t j = 0; j < d; j++)
    {
        const uint64_t sj = modq(s[j], mod);
        bv[j] = _mm512_set1_epi64(sj);
        wb[j] = _mm512_set1_epi64(mul_modq(ws, sj, mod));
    }

    for (uint64_t v = 0; v < nvec; v++)
    {
        __m512i av[FIELD_MAX_FUSED_D];
        for (uint64_t i = 0; i < d; i++)
            av[i] = ((const __m512i *)a[i])[v];

        for (uint64_t k = 0; k < d; k++)
        {
            __m512i lo = zero, hi = zero;
            for (uint64_t i = 0; i < d; i++)
            {
                const uint64_t j = FIELD_FUSED_PARTNER(i, k, d);
                const __m512i p = FIELD_FUSED_WRAPS(i, k, d) ? wb[j] : bv[j];
                lo = _mm512_madd52lo_epu64(lo, av[i], p);
                hi = _mm512_madd52hi_epu64(hi, av[i], p);
            }
            hi = _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52));
            lo = _mm512_and_epi64(lo, mask52);
            ((__m512i *)out[k])[v] = field_reduce128(hi, lo, mod);
        }
    }
}

// The instantiations. A degree in between takes the runtime-`d` one, which is
// the same source without the unrolling.
#define FIELD_FUSED_AT(DEGREE)                                                                     \
    static void field_fused_mul_##DEGREE(uint64_t *const *out, uint64_t *const *a,                 \
                                         uint64_t *const *b, uint64_t nvec, uint64_t w,            \
                                         Modulus mod)                                              \
    {                                                                                              \
        field_fused_mul_impl(out, a, b, nvec, DEGREE, w, mod);                                     \
    }                                                                                              \
    static void field_fused_scale_##DEGREE(uint64_t *const *out, uint64_t *const *a,               \
                                           const uint64_t *s, uint64_t nvec, uint64_t w,           \
                                           Modulus mod)                                            \
    {                                                                                              \
        field_fused_scale_impl(out, a, s, nvec, DEGREE, w, mod);                                   \
    }
FIELD_FUSED_AT(2)
FIELD_FUSED_AT(4)
FIELD_FUSED_AT(8)
#undef FIELD_FUSED_AT

int field_fused_applies(Modulus mod, uint64_t d, uint64_t n)
{
    // The 2^52 family is the one with a widening multiply that accumulates;
    // the other two would have to build the product from 32-bit pieces, which
    // costs more than the reductions this saves.
    return mod_shoup_shift(mod->q) == MOD_SHIFT_50 && d >= 1 && d <= FIELD_MAX_FUSED_D &&
           n >= FIELD_FUSED_LANES && (n % FIELD_FUSED_LANES) == 0;
}

void field_fused_mul(uint64_t *const *out, uint64_t *const *a, uint64_t *const *b, uint64_t n,
                     uint64_t d, uint64_t w, Modulus mod)
{
    const uint64_t nvec = n / FIELD_FUSED_LANES;
    switch (d)
    {
    case 2:
        field_fused_mul_2(out, a, b, nvec, w, mod);
        break;
    case 4:
        field_fused_mul_4(out, a, b, nvec, w, mod);
        break;
    case 8:
        field_fused_mul_8(out, a, b, nvec, w, mod);
        break;
    default:
        field_fused_mul_impl(out, a, b, nvec, d, w, mod);
        break;
    }
}

void field_fused_scale(uint64_t *const *out, uint64_t *const *a, const uint64_t *s, uint64_t n,
                       uint64_t d, uint64_t w, Modulus mod)
{
    const uint64_t nvec = n / FIELD_FUSED_LANES;
    switch (d)
    {
    case 2:
        field_fused_scale_2(out, a, s, nvec, w, mod);
        break;
    case 4:
        field_fused_scale_4(out, a, s, nvec, w, mod);
        break;
    case 8:
        field_fused_scale_8(out, a, s, nvec, w, mod);
        break;
    default:
        field_fused_scale_impl(out, a, s, nvec, d, w, mod);
        break;
    }
}

#else // !VFHE_HAVE_AVX512IFMA

// No widening accumulate on this engine, so the schoolbook passes stay.
int field_fused_applies(Modulus mod, uint64_t d, uint64_t n)
{
    (void)mod;
    (void)d;
    (void)n;
    return 0;
}

void field_fused_mul(uint64_t *const *out, uint64_t *const *a, uint64_t *const *b, uint64_t n,
                     uint64_t d, uint64_t w, Modulus mod)
{
    (void)out;
    (void)a;
    (void)b;
    (void)n;
    (void)d;
    (void)w;
    (void)mod;
}

void field_fused_scale(uint64_t *const *out, uint64_t *const *a, const uint64_t *s, uint64_t n,
                       uint64_t d, uint64_t w, Modulus mod)
{
    (void)out;
    (void)a;
    (void)s;
    (void)n;
    (void)d;
    (void)w;
    (void)mod;
}

#endif
