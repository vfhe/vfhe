// SPDX-FileCopyrightText: 2026 The vfhe authors
// SPDX-License-Identifier: Apache-2.0
//
// The IFMA pieces of the extension product with delayed reductions, shared by
// the kernels built from them (field_fused.c, field_vector_fold_ifma.c).
// Products of canonical coefficients below q < 2^50 are accumulated as
// redundant (hi, lo) pairs with madd52hi / madd52lo and reduced once.
#ifndef VFHE_FIELD_FUSED_H
#define VFHE_FIELD_FUSED_H

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

#endif // VFHE_HAVE_AVX512IFMA

#endif // VFHE_FIELD_FUSED_H
