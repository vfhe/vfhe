/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_tower.c
 * @brief RNS tower operations: rounding division, exact fast base extension,
 *        and the scaled-lift / round-divide roundtrip.
 */
#include <stdint.h>

#include "arith.h"
#include "test_arith_support.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_tower_div_round(void)
{
    const uint64_t N = 16;
    uint64_t primes[2];
    primes[0] = nt_next_ntt_prime(1ULL << 20, N, false);
    primes[1] = nt_next_ntt_prime(primes[0], N, false);
    ring_t r = ring_new(primes, 1, N, 2);

    /* p = c * q1: round division by q1 must recover c exactly. */
    const uint64_t c = 12345;
    rns_poly_t p = poly_new(r, 0x3);
    uint64_t v[16] = {0};
    v[0] = c;
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_from_int_array(p, v));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_to_coeff(p, p));
    uint64_t s[2] = {primes[1] % primes[0], 0};
    poly_scale_vec(p, p, s);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_div_round(p, 0x2));
    TEST_ASSERT_EQUAL_UINT64(0x1, poly_mask(p));
    TEST_ASSERT_EQUAL_UINT64(c, poly_limb_data(p, 0)[0]);

    poly_free(p);
    ring_free(r);
}

void test_tower_base_convert_exact_single_source(void)
{
    /* One source limb (w = 1): fast base extension is exact. */
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    uint64_t v[N];
    for (uint64_t i = 0; i < N; i++)
        v[i] = lcg() % 10000;
    rns_poly_t small = poly_new(r, 0x1), big = poly_new(r, 0x3);
    poly_from_int_array(small, v);
    poly_to_coeff(small, small);

    baseconv_plan_t plan = baseconv_plan_new(r, 0x1, 0x3);
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_base_convert(big, small, plan));
    TEST_ASSERT_EQUAL_UINT64(0x3, poly_mask(big));
    for (uint64_t i = 0; i < N; i++)
    {
        TEST_ASSERT_EQUAL_UINT64(v[i] % primes[0], poly_limb_data(big, 0)[i]);
        TEST_ASSERT_EQUAL_UINT64(v[i] % primes[1], poly_limb_data(big, 1)[i]);
    }
    baseconv_plan_free(plan);

    poly_free(small);
    poly_free(big);
    ring_free(r);
}

void test_tower_scaled_lift_div_roundtrip(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    uint64_t v[N];
    for (uint64_t i = 0; i < N; i++)
        v[i] = lcg() % primes[0];
    rns_poly_t small = poly_new(r, 0x1), big = poly_new(r, 0x3);
    poly_from_int_array(small, v);
    poly_to_coeff(small, small);

    /* v -> v * q1 on both limbs (exact), then round-divide q1 away. */
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_scaled_lift(big, small, NULL));
    TEST_ASSERT_EQUAL_INT(VFHE_OK, poly_div_round(big, 0x2));
    TEST_ASSERT_EQUAL_UINT64(0x1, poly_mask(big));
    for (uint64_t i = 0; i < N; i++)
    {
        TEST_ASSERT_EQUAL_UINT64(v[i] % primes[0], poly_limb_data(big, 0)[i]);
    }

    poly_free(small);
    poly_free(big);
    ring_free(r);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_tower_div_round);
    RUN_TEST(test_tower_base_convert_exact_single_source);
    RUN_TEST(test_tower_scaled_lift_div_roundtrip);
    return UNITY_END();
}
