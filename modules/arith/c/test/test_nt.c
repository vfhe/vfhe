/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_nt.c
 * @brief Number-theory primitives: primality, modular power/inverse, and the
 *        deterministic 2n-th root of unity used by the NTT plan.
 */
#include <stdint.h>

#include "arith.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_nt_primality_and_modular(void)
{
    TEST_ASSERT_TRUE(nt_is_prime(97));
    TEST_ASSERT_FALSE(nt_is_prime(100));
    TEST_ASSERT_EQUAL_UINT64(24, nt_power_mod(2, 10, 1000)); /* 1024 mod 1000 */
    const uint64_t q = 12289, a = 1234;
    const uint64_t inv = nt_inverse_mod(a, q);
    TEST_ASSERT_EQUAL_UINT64(1, (uint64_t)(((unsigned __int128)a * inv) % q));
}

void test_nt_root_of_unity(void)
{
    /* 2n-th root produced by the plan layer must satisfy w^n == -1. */
    const uint64_t n = 8;
    const uint64_t p = nt_next_ntt_prime(1ULL << 20, n, false);
    const uint64_t w = nt_gen_root_of_unity(p, 2 * n);
    TEST_ASSERT_EQUAL_UINT64(p - 1, nt_power_mod(w, n, p));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_nt_primality_and_modular);
    RUN_TEST(test_nt_root_of_unity);
    return UNITY_END();
}
