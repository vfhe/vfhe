// SPDX-License-Identifier: Apache-2.0
/**
 * @file ntt_avx512_64.c
 * @brief AVX-512 NTT backend for primes up to 2^63.
 *
 * Butterflies assemble 64x64 products from 32-bit partials (no single-
 * instruction full multiply at this width) with a 64-bit-shift Barrett
 * preconditioner. The recursion/interleaving driver is shared: see
 * ntt_avx512_driver.inc.
 */
#include <stdlib.h>
#include <string.h>

#include <arith/config.h>
#include <arith/nt.h>

#include "ntt_backends.h"

#if VFHE_MP_SIMD

#include <immintrin.h>

/** Approximate high 64 bits of a 64x64 product (error <= 1). */
static inline __m512i bfly_mulhi_approx(__m512i x, __m512i y)
{
    __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
    __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
    __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
    __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);
    __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);
    __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);

    __m512i sum_lo = _mm512_and_si512(z_lo_hi, lo_mask);
    __m512i sum_mid = _mm512_srli_epi64(z_lo_hi, 32);
    __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
    __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
    __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
    return _mm512_add_epi64(sum_hi, sum_mid2_hi);
}

/** Exact high 64 bits of a 64x64 product. */
static inline __m512i bfly_mulhi(__m512i x, __m512i y)
{
    __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
    __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
    __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
    __m512i z_lo_lo = _mm512_mul_epu32(x, y);
    __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);
    __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);
    __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);

    __m512i z_lo_lo_shift = _mm512_srli_epi64(z_lo_lo, 32);
    __m512i sum_tmp = _mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
    __m512i sum_lo = _mm512_and_si512(sum_tmp, lo_mask);
    __m512i sum_mid = _mm512_srli_epi64(sum_tmp, 32);
    __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
    __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
    __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
    return _mm512_add_epi64(sum_hi, sum_mid2_hi);
}

static inline void bfly_fwd(__m512i *X, __m512i *Y, __m512i W, __m512i W_precon,
                            __m512i neg_modulus, __m512i twice_modulus)
{
    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, twice_modulus));
    __m512i Q = bfly_mulhi_approx(W_precon, *Y);
    __m512i W_Y = _mm512_mullo_epi64(W, *Y);
    __m512i T = _mm512_add_epi64(W_Y, _mm512_mullo_epi64(Q, neg_modulus));
    T = _mm512_min_epu64(T, _mm512_sub_epi64(T, twice_modulus));

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

    __m512i Q = bfly_mulhi_approx(W_precon, T);
    __m512i Q_p = _mm512_mullo_epi64(Q, neg_modulus);
    __m512i prod = _mm512_mullo_epi64(W, T);
    *Y = _mm512_add_epi64(prod, Q_p);
    *Y = _mm512_min_epu64(*Y, _mm512_sub_epi64(*Y, twice_modulus));
}

static inline void bfly_inv_final(__m512i *X, __m512i *Y, __m512i v_inv_n, __m512i v_inv_n_prime,
                                  __m512i v_inv_n_w, __m512i v_inv_n_w_prime, __m512i minus_q,
                                  __m512i q2, __m512i q_vec)
{
    __m512i Y_minus_2q = _mm512_sub_epi64(*Y, q2);
    __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

    __m512i X_plus_Y = _mm512_add_epi64(*X, *Y);
    X_plus_Y = _mm512_min_epu64(X_plus_Y, _mm512_sub_epi64(X_plus_Y, q2));

    __m512i Q1 = bfly_mulhi(v_inv_n_prime, X_plus_Y);
    __m512i Q1_p = _mm512_mullo_epi64(Q1, minus_q);
    __m512i prod_X = _mm512_mullo_epi64(v_inv_n, X_plus_Y);
    *X = _mm512_add_epi64(prod_X, Q1_p);

    __m512i Q2 = bfly_mulhi(v_inv_n_w_prime, T);
    __m512i Q2_p = _mm512_mullo_epi64(Q2, minus_q);
    __m512i prod_Y = _mm512_mullo_epi64(v_inv_n_w, T);
    *Y = _mm512_add_epi64(prod_Y, Q2_p);

    *X = _mm512_min_epu64(*X, _mm512_sub_epi64(*X, q_vec));
    *Y = _mm512_min_epu64(*Y, _mm512_sub_epi64(*Y, q_vec));
}

#define NTT_TIER_SHIFT 64
#define NTT_BACKEND_INIT ntt_backend_avx512_64_init
#include "ntt_avx512_driver.inc"

#endif // VFHE_MP_SIMD
