// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_avx512_32.c
 * @brief AVX-512 NTT backend for primes q < 2^32.
 *
 * Butterflies use `mullo` products with a 32-bit-shift Barrett
 * preconditioner (Harvey-style lazy butterflies, values in [0, 4q)).
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
    __m512i Q = _mm512_mullo_epi64(W_precon, *Y);
    Q = _mm512_srli_epi64(Q, 32);
    __m512i W_Y = _mm512_mullo_epi64(W, *Y);
    __m512i T = _mm512_add_epi64(W_Y, _mm512_mullo_epi64(Q, neg_modulus));

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

    __m512i Q = _mm512_mullo_epi64(W_precon, T);
    Q = _mm512_srli_epi64(Q, 32);
    __m512i Q_p = _mm512_mullo_epi64(Q, neg_modulus);
    __m512i prod = _mm512_mullo_epi64(W, T);
    *Y = _mm512_add_epi64(prod, Q_p);
}

static inline void bfly_inv_final(__m512i *X, __m512i *Y, __m512i v_inv_n, __m512i v_inv_n_prime,
                                  __m512i v_inv_n_w, __m512i v_inv_n_w_prime, __m512i minus_q,
                                  __m512i q2, __m512i q_vec)
{
    __m512i Y_minus_2q = _mm512_sub_epi64(*Y, q2);
    __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

    __m512i X_plus_Y = _mm512_add_epi64(*X, *Y);
    X_plus_Y = _mm512_min_epu64(X_plus_Y, _mm512_sub_epi64(X_plus_Y, q2));

    __m512i Q1 = _mm512_mullo_epi64(v_inv_n_prime, X_plus_Y);
    Q1 = _mm512_srli_epi64(Q1, 32);
    __m512i Q1_p = _mm512_mullo_epi64(Q1, minus_q);
    __m512i prod_X = _mm512_mullo_epi64(v_inv_n, X_plus_Y);
    *X = _mm512_add_epi64(prod_X, Q1_p);

    __m512i Q2 = _mm512_mullo_epi64(v_inv_n_w_prime, T);
    Q2 = _mm512_srli_epi64(Q2, 32);
    __m512i Q2_p = _mm512_mullo_epi64(Q2, minus_q);
    __m512i prod_Y = _mm512_mullo_epi64(v_inv_n_w, T);
    *Y = _mm512_add_epi64(prod_Y, Q2_p);

    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, q_vec));
    *Y = _mm512_min_epu64(*Y, _mm512_sub_epi64(*Y, q_vec));
}

#define NTT_TIER_SHIFT 32
#define NTT_BACKEND_INIT ntt_backend_avx512_32_init
#include "ntt_avx512_driver.inc"

#endif // VFHE_MP_SIMD
