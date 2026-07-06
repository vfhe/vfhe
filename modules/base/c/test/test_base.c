/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_base.c
 * @brief Unit tests for the base utilities: alloc, bits, modswitch.
 */
#include <stdint.h>
#include <stdlib.h>

#include "base.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---------------------------------------------------------------- cpu ---- */

void test_cpu_detect(void)
{
    cpu_features f1, f2;
    cpu_detect(&f1);
    cpu_detect(&f2); /* stateless: two queries must agree */
    TEST_ASSERT_EQUAL_INT(f1.avx2, f2.avx2);
    TEST_ASSERT_EQUAL_INT(f1.avx512f, f2.avx512f);
    TEST_ASSERT_EQUAL_INT(f1.avx512ifma, f2.avx512ifma);
    TEST_ASSERT_EQUAL_INT(f1.aes, f2.aes);

    /* Feature implications hold on every host (all-false off x86). */
    if (f1.avx512ifma)
        TEST_ASSERT_TRUE(f1.avx512f);
    if (f1.avx512f)
        TEST_ASSERT_TRUE(f1.avx2);
}

/* -------------------------------------------------------------- alloc ---- */

void test_safe_aligned_malloc(void)
{
    uint64_t *p = (uint64_t *)safe_aligned_malloc(64 * sizeof(uint64_t));
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_EQUAL_UINT64(0, ((uintptr_t)p) & 63); /* 64-byte aligned */
    free(p);

    /* Small sizes must keep the alignment guarantee too. */
    uint8_t *q = (uint8_t *)safe_aligned_malloc(1);
    TEST_ASSERT_EQUAL_UINT64(0, ((uintptr_t)q) & 63);
    free(q);
}

/* --------------------------------------------------------------- bits ---- */

void test_next_power_of_2(void)
{
    TEST_ASSERT_EQUAL_UINT64(1, next_power_of_2(0));
    TEST_ASSERT_EQUAL_UINT64(1, next_power_of_2(1));
    TEST_ASSERT_EQUAL_UINT64(2, next_power_of_2(2));
    TEST_ASSERT_EQUAL_UINT64(8, next_power_of_2(5));
    TEST_ASSERT_EQUAL_UINT64(8, next_power_of_2(8));
    TEST_ASSERT_EQUAL_UINT64(16, next_power_of_2(9));
    TEST_ASSERT_EQUAL_UINT64(1ULL << 63, next_power_of_2((1ULL << 62) + 1));
    TEST_ASSERT_EQUAL_UINT64(1ULL << 63, next_power_of_2(1ULL << 63));
}

void test_byte_and_word_reversal(void)
{
    TEST_ASSERT_EQUAL_UINT8(0x80, char_rev(0x01));
    TEST_ASSERT_EQUAL_UINT8(0x01, char_rev(0x80));
    TEST_ASSERT_EQUAL_UINT8(0xA5, char_rev(0xA5)); /* palindrome byte */

    TEST_ASSERT_EQUAL_UINT32(0x80000000u, int_rev(1));
    TEST_ASSERT_EQUAL_UINT32(1, int_rev(0x80000000u));
    /* Involution on a sweep of values. */
    for (uint32_t i = 0; i < 64; i++)
    {
        uint32_t v = i * 2654435761u;
        TEST_ASSERT_EQUAL_UINT32(v, int_rev(int_rev(v)));
    }
}

void test_bit_rev_permutation(void)
{
    /* Involution for the square case (log_n == log2(n)). */
    uint64_t in[16], out[16], back[16];
    for (uint64_t i = 0; i < 16; i++)
        in[i] = i * 7 + 1;
    bit_rev(out, in, 16, 4);
    bit_rev(back, out, 16, 4);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(in, back, 16);

    /* Exact index math for n = 8: out[i] = in[bitrev_3(i)]. */
    static const uint64_t rev3[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    bit_rev(out, in, 8, 3);
    for (uint64_t i = 0; i < 8; i++)
    {
        TEST_ASSERT_EQUAL_UINT64(in[rev3[i]], out[i]);
    }

    /* Oversampled case (log_n = log2(n) + 1), as used by the NTT twist
     * tables: indices reverse in 4 bits, sampling the double-length input. */
    uint64_t in16[16];
    for (uint64_t i = 0; i < 16; i++)
        in16[i] = 100 + i;
    bit_rev(out, in16, 8, 4);
    for (uint64_t i = 0; i < 8; i++)
    {
        uint64_t r = ((i & 1) << 3) | ((i & 2) << 1) | ((i & 4) >> 1) | ((i & 8) >> 3);
        TEST_ASSERT_EQUAL_UINT64(in16[r], out[i]);
    }
}

/* ---------------------------------------------------------- modswitch ---- */

void test_mod_switch(void)
{
    /* Identity when p == q. */
    TEST_ASSERT_EQUAL_UINT64(1234, mod_switch(1234, 4096, 4096));
    /* Exact doubling / halving between powers of two. */
    TEST_ASSERT_EQUAL_UINT64(2468, mod_switch(1234, 4096, 8192));
    TEST_ASSERT_EQUAL_UINT64(617, mod_switch(1234, 4096, 2048));
    /* Rounding: 3 * 5 / 8 = 1.875 -> 2. */
    TEST_ASSERT_EQUAL_UINT64(2, mod_switch(3, 8, 5));
    /* Wrap: v = p - 1 maps to q - 1 + rounding, reduced into [0, q). */
    uint64_t r = mod_switch(4095, 4096, 17);
    TEST_ASSERT_TRUE(r < 17);
}

void test_array_mod_switch_consistency(void)
{
    uint64_t in[8], out[8];
    for (uint64_t i = 0; i < 8; i++)
        in[i] = i * 511;
    array_mod_switch(out, in, 4096, 97, 8);
    for (uint64_t i = 0; i < 8; i++)
    {
        TEST_ASSERT_EQUAL_UINT64(mod_switch(in[i], 4096, 97), out[i]);
    }
}

void test_array_reduce_mod_N(void)
{
    uint64_t in[4] = {0xFFFF, 0x10001, 0x12345, 7};
    uint64_t out[4];
    array_reduce_mod_N(out, in, 4, 60000); /* window = 65536 */
    TEST_ASSERT_EQUAL_UINT64(0xFFFF, out[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out[1]);
    TEST_ASSERT_EQUAL_UINT64(0x2345, out[2]);
    TEST_ASSERT_EQUAL_UINT64(7, out[3]);
}

void test_array_mod_switch_from_2k_alias_and_copy(void)
{
    /* The aliased and the two-buffer form must agree (regression: the
     * two-buffer form used to read the unmasked input). */
    const uint64_t q = 12289, n = 16;
    uint64_t in[16], aliased[16], copied[16];
    for (uint64_t i = 0; i < n; i++)
    {
        in[i] = i * 0x9E3779B97F4A7C15ULL; /* full-width values */
        aliased[i] = in[i];
    }
    array_mod_switch_from_2k(aliased, aliased, q, q, n);
    array_mod_switch_from_2k(copied, in, q, q, n);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(aliased, copied, n);
    for (uint64_t i = 0; i < n; i++)
    {
        TEST_ASSERT_TRUE(copied[i] < q);
    }
}

void test_array_additive_inverse_mod_switch(void)
{
    /* Values in the upper half of [0, p) are re-centered as negatives. */
    const uint64_t p = 16, q = 97;
    uint64_t in[4] = {0, 3, 15, 9}; /* 15 = -1, 9 = -7 */
    uint64_t out[4];
    array_additive_inverse_mod_switch(out, in, p, q, 4);
    TEST_ASSERT_EQUAL_UINT64(0, out[0]);
    TEST_ASSERT_EQUAL_UINT64(3, out[1]);
    TEST_ASSERT_EQUAL_UINT64(q - 1, out[2]);
    TEST_ASSERT_EQUAL_UINT64(q - 7, out[3]);
}

void test_mod_dist(void)
{
    TEST_ASSERT_EQUAL_UINT64(0, mod_dist(5, 5, 17));
    TEST_ASSERT_EQUAL_UINT64(2, mod_dist(3, 1, 17));
    TEST_ASSERT_EQUAL_UINT64(2, mod_dist(1, 3, 17));  /* symmetric */
    TEST_ASSERT_EQUAL_UINT64(1, mod_dist(0, 16, 17)); /* wraps around */
    TEST_ASSERT_EQUAL_UINT64(8, mod_dist(0, 8, 17));  /* farthest point */
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_cpu_detect);
    RUN_TEST(test_safe_aligned_malloc);
    RUN_TEST(test_next_power_of_2);
    RUN_TEST(test_byte_and_word_reversal);
    RUN_TEST(test_bit_rev_permutation);
    RUN_TEST(test_mod_switch);
    RUN_TEST(test_array_mod_switch_consistency);
    RUN_TEST(test_array_reduce_mod_N);
    RUN_TEST(test_array_mod_switch_from_2k_alias_and_copy);
    RUN_TEST(test_array_additive_inverse_mod_switch);
    RUN_TEST(test_mod_dist);
    return UNITY_END();
}
