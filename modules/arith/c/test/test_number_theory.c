/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_number_theory.c
 * @brief Primality, special-prime search, and Nth-root-of-unity.
 */
#include <arith.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_is_prime(void)
{
    TEST_ASSERT_TRUE(is_prime(562949954142209ULL));
    TEST_ASSERT_FALSE(is_prime(100));
}

void test_next_special_prime_primitive(void)
{
    const uint64_t n = 1024, start = 1ULL << 40;
    const uint64_t sp = next_special_prime(start, n, true);
    TEST_ASSERT_TRUE(sp > start);
    TEST_ASSERT_TRUE(is_prime(sp));
    TEST_ASSERT_EQUAL_UINT64(1, sp % (2 * n)); /* q == 1 (mod 2N) */
    TEST_ASSERT_TRUE(sp % (4 * n) != 1);       /* admits a primitive 2N-th root */
}

void test_next_special_prime_nonprimitive(void)
{
    const uint64_t n = 1024, start = 1ULL << 40;
    const uint64_t sp = next_special_prime(start, n, false);
    TEST_ASSERT_TRUE(sp > start);
    TEST_ASSERT_TRUE(is_prime(sp));
    TEST_ASSERT_EQUAL_UINT64(1, sp % (2 * n));
}

void test_nth_root_of_unity(void)
{
    const uint64_t q = 769; /* 768 = 256 * 3 */
    const uint64_t root = generate_Nth_root_of_unity(q, 16);
    uint64_t res = 1;
    for (int i = 0; i < 8; i++)
        res = (res * root) % q;
    TEST_ASSERT_EQUAL_UINT64(q - 1, res); /* root^8 == -1 (mod q) */
}

void test_inverse_mod_eea(void)
{
    const uint64_t p = 1073739937; /* prime modulus */
    for (uint64_t a = 1; a < 1000; a++)
    {
        uint64_t inv = inverse_mod_eea(a, p);
        uint64_t prod = (uint64_t)(((unsigned __int128)inv * a) % p);
        TEST_ASSERT_EQUAL_UINT64(1, prod);
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_is_prime);
    RUN_TEST(test_next_special_prime_primitive);
    RUN_TEST(test_next_special_prime_nonprimitive);
    RUN_TEST(test_nth_root_of_unity);
    RUN_TEST(test_inverse_mod_eea);
    return UNITY_END();
}
