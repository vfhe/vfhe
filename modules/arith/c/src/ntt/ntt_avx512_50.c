// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_avx512_50.c
 * @brief AVX-512 IFMA NTT backend for primes q < 2^50.
 *
 * Butterflies use the 52-bit fused multiply-add units with a 52-bit-shift
 * Barrett preconditioner (HEXL-style lazy butterflies, values in [0, 4q)).
 * The recursion/interleaving driver is shared: see ntt_avx512_driver.inc.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/config.h>
#include <arith/nt.h>

#include "ntt_backends.h"

#if VFHE_MP_SIMD

#include <immintrin.h>

static inline void bfly_fwd(__m512i *X, __m512i *Y, __m512i W, __m512i W_precon,
                            __m512i neg_modulus, __m512i twice_modulus)
{
    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, twice_modulus));
    const __m512i zero = _mm512_setzero_si512();
    __m512i Q = _mm512_madd52hi_epu64(zero, W_precon, *Y);
    __m512i W_Y = _mm512_madd52lo_epu64(zero, W, *Y);
    __m512i T = _mm512_madd52lo_epu64(W_Y, Q, neg_modulus);
    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    T = _mm512_and_epi64(T, low52b_mask);

    __m512i twice_mod_minus_T = _mm512_sub_epi64(twice_modulus, T);
    *Y = _mm512_add_epi64(*X, twice_mod_minus_T);
    *X = _mm512_add_epi64(*X, T);
}

static inline void bfly_inv(__m512i *X, __m512i *Y, __m512i W, __m512i W_precon,
                            __m512i neg_modulus, __m512i twice_modulus)
{
    __m512i Y_minus_2q = _mm512_sub_epi64(*Y, twice_modulus);
    __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

    *X = _mm512_add_epi64(*X, Y_minus_2q);
    __mmask8 sign_bits = _mm512_movepi64_mask(*X);
    *X = _mm512_mask_add_epi64(*X, sign_bits, *X, twice_modulus);

    const __m512i zero = _mm512_setzero_si512();
    __m512i Q = _mm512_madd52hi_epu64(zero, W_precon, T);
    __m512i Q_p = _mm512_madd52lo_epu64(zero, Q, neg_modulus);
    *Y = _mm512_madd52lo_epu64(Q_p, W, T);

    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    *Y = _mm512_and_epi64(*Y, low52b_mask);
}

static inline void bfly_inv_final(__m512i *X, __m512i *Y, __m512i v_inv_n, __m512i v_inv_n_prime,
                                  __m512i v_inv_n_w, __m512i v_inv_n_w_prime, __m512i minus_q,
                                  __m512i q2, __m512i q_vec)
{
    __m512i Y_minus_2q = _mm512_sub_epi64(*Y, q2);
    __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

    __m512i X_plus_Y = _mm512_add_epi64(*X, *Y);
    X_plus_Y = _mm512_min_epu64(X_plus_Y, _mm512_sub_epi64(X_plus_Y, q2));

    const __m512i zero = _mm512_setzero_si512();
    __m512i Q1 = _mm512_madd52hi_epu64(zero, v_inv_n_prime, X_plus_Y);
    __m512i Q1_p = _mm512_madd52lo_epu64(zero, Q1, minus_q);
    *X = _mm512_madd52lo_epu64(Q1_p, v_inv_n, X_plus_Y);

    __m512i Q2 = _mm512_madd52hi_epu64(zero, v_inv_n_w_prime, T);
    __m512i Q2_p = _mm512_madd52lo_epu64(zero, Q2, minus_q);
    *Y = _mm512_madd52lo_epu64(Q2_p, v_inv_n_w, T);

    const __m512i low52b_mask = _mm512_set1_epi64((1ULL << 52) - 1);
    *X = _mm512_and_epi64(*X, low52b_mask);
    *Y = _mm512_and_epi64(*Y, low52b_mask);

    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, q_vec));
    *Y = _mm512_min_epu64(*Y, _mm512_sub_epi64(*Y, q_vec));
}

#define NTT_TIER_SHIFT 52
#define NTT_BACKEND_INIT ntt_backend_avx512_50_init
#include "ntt_avx512_driver.inc"

#endif // VFHE_MP_SIMD
