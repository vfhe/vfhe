/* SPDX-License-Identifier: Apache-2.0 */
#include <stdint.h>

#include "arith.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_ntt_forward_32_reduces_mod_q(void)
{
    uint64_t in[3] = {10, 20, 100};
    uint64_t out[3] = {0};
    NTT_proc proc = {.n = 3, .modulus = 7};
    ntt_forward_32(out, in, proc);
    TEST_ASSERT_EQUAL_UINT64(3, out[0]);
    TEST_ASSERT_EQUAL_UINT64(6, out[1]);
    TEST_ASSERT_EQUAL_UINT64(2, out[2]);
}

void test_asm_mul64(void)
{
    TEST_ASSERT_EQUAL_UINT64(42, asm_mul64(6, 7));
    TEST_ASSERT_EQUAL_UINT64(0, asm_mul64((uint64_t)1 << 63, 4));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_ntt_forward_32_reduces_mod_q);
    RUN_TEST(test_asm_mul64);
    return UNITY_END();
}
