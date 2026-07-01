/* SPDX-License-Identifier: Apache-2.0 */
#include <stdint.h>

#include "crypto.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_crypto_sample_known_values(void)
{
    TEST_ASSERT_EQUAL_UINT64(1442695040888963407ULL, crypto_sample(0));
    TEST_ASSERT_EQUAL_UINT64(7806831264735756412ULL, crypto_sample(1));
}

void test_crypto_sample_twice_composes(void)
{
    TEST_ASSERT_EQUAL_UINT64(crypto_sample(crypto_sample(5)), crypto_sample_twice(5));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_crypto_sample_known_values);
    RUN_TEST(test_crypto_sample_twice_composes);
    return UNITY_END();
}
