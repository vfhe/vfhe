/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mp.c
 * @brief The multiprecision layer: base-2^52 scalar load/scale/subtract, the
 *        polynomial scale/negate/add operations, and the CRT bridge into an
 *        RNS polynomial.
 */
#include <stdint.h>
#include <stdlib.h>

#include "arith.h"
#include "test_arith_support.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_mp_scalar_roundtrip_and_ops(void)
{
    /* value = d0 + d1 * 2^52, small enough that products fit __int128. */
    uint64_t digits[2] = {0x000ABCDEF012345ULL, 0x00000000000FFULL};
    mp_scalar_t s = mp_scalar_load(digits, 2);
    TEST_ASSERT_EQUAL_UINT64(2, mp_scalar_digit_count(s));
    TEST_ASSERT_EQUAL_UINT64(digits[0], mp_scalar_digit(s, 0));
    TEST_ASSERT_EQUAL_UINT64(digits[1], mp_scalar_digit(s, 1));

    unsigned __int128 value = (unsigned __int128)digits[0] + ((unsigned __int128)digits[1] << 52);

    /* scale by a word m: reconstruct out = value * m from 3 digits. */
    const uint64_t m_word = 0x12345;
    mp_vector_t *m = mp_vector_load(m_word);
    uint64_t zero3[3] = {0, 0, 0};
    mp_scalar_t out = mp_scalar_load(zero3, 3);
    mp_scalar_scale(out, s, m);
    unsigned __int128 got = 0;
    for (int i = 2; i >= 0; i--)
        got = (got << 52) | mp_scalar_digit(out, (uint64_t)i);
    TEST_ASSERT_TRUE(got == value * m_word);

    /* a - b for a > b. */
    uint64_t b_digits[2] = {0x0000000000123ULL, 0x000000000000AULL};
    mp_scalar_t b = mp_scalar_load(b_digits, 2);
    mp_scalar_t diff = mp_scalar_load(zero3, 2);
    mp_scalar_sub(diff, s, b);
    unsigned __int128 b_val =
        (unsigned __int128)b_digits[0] + ((unsigned __int128)b_digits[1] << 52);
    unsigned __int128 d_val = (unsigned __int128)mp_scalar_digit(diff, 0) +
                              ((unsigned __int128)mp_scalar_digit(diff, 1) << 52);
    TEST_ASSERT_TRUE(d_val == value - b_val);

    mp_scalar_free(s);
    mp_scalar_free(b);
    mp_scalar_free(diff);
    mp_scalar_free(out);
    free(m);
}

void test_mp_polynomial_ops(void)
{
    enum
    {
        N = 8
    };
    mp_polynomial_t p = mp_polynomial_new(N, 2);
    for (uint64_t i = 0; i < N; i++)
    {
        p->coeffs[0][i] = 1000 + i;
        p->coeffs[1][i] = 7 * i;
    }

    /* Digit-local scale: with every digit product below 2^52 there are no high
     * halves, so out_j = in_j * s exactly (see arith/mp.h for the fixed-point
     * convention governing the general case). */
    const uint64_t s = 12345;
    mp_polynomial_t out = mp_polynomial_new(N, 2);
    mp_polynomial_scale_fixedpoint(out, p, s);
    for (uint64_t i = 0; i < N; i++)
    {
        TEST_ASSERT_EQUAL_UINT64((1000 + i) * s, out->coeffs[0][i]);
        TEST_ASSERT_EQUAL_UINT64(7 * i * s, out->coeffs[1][i]);
    }

    /* a + (-a) propagates to zero (radix complement). */
    mp_polynomial_t neg = mp_polynomial_new(N, 2), sum = mp_polynomial_new(N, 2);
    mp_polynomial_negate(neg, p);
    mp_polynomial_add(sum, p, neg);
    mp_polynomial_propagate_carry(sum);
    for (uint64_t i = 0; i < N; i++)
    {
        TEST_ASSERT_EQUAL_UINT64(0, sum->coeffs[0][i]);
        TEST_ASSERT_EQUAL_UINT64(0, sum->coeffs[1][i]);
    }

    mp_polynomial_free(p);
    mp_polynomial_free(out);
    mp_polynomial_free(neg);
    mp_polynomial_free(sum);
}

void test_mp_polynomial_to_poly(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    /* value = d0 + d1 * 2^52 reduced into each limb. */
    mp_polynomial_t mp = mp_polynomial_new(N, 2);
    for (uint64_t i = 0; i < N; i++)
    {
        mp->coeffs[0][i] = lcg() & ((1ULL << 52) - 1);
        mp->coeffs[1][i] = lcg() & 0xFFFF;
    }
    rns_poly_t p = poly_new(r, 0x3);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, mp_polynomial_to_poly(p, mp));
    for (int limb = 0; limb < 2; limb++)
    {
        for (uint64_t i = 0; i < N; i++)
        {
            unsigned __int128 v =
                (unsigned __int128)mp->coeffs[0][i] + ((unsigned __int128)mp->coeffs[1][i] << 52);
            TEST_ASSERT_EQUAL_UINT64((uint64_t)(v % primes[limb]),
                                     poly_limb_data(p, (uint64_t)limb)[i]);
        }
    }

    mp_polynomial_free(mp);
    poly_free(p);
    ring_free(r);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_mp_scalar_roundtrip_and_ops);
    RUN_TEST(test_mp_polynomial_ops);
    RUN_TEST(test_mp_polynomial_to_poly);
    return UNITY_END();
}
