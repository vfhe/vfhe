/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_ntt.c
 * @brief The NTT plan: forward/inverse roundtrips across a size x prime matrix
 *        and the convolution theorem against a negacyclic schoolbook oracle.
 */
#include <stdint.h>
#include <stdlib.h>

#include "arith.h"
#include "base.h"
#include "test_arith_support.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_ntt_roundtrip(void)
{
    const uint64_t n = 8;
    const uint64_t q = nt_next_ntt_prime(1ULL << 20, n, false);
    zq_ctx zq;
    zq_ctx_init(&zq, q);
    ntt_plan plan;
    TEST_ASSERT_EQUAL_INT(VFHE_OK, ntt_plan_init(&plan, n, &zq));

    uint64_t in[8], fwd[8], back[8];
    for (uint64_t i = 0; i < n; i++)
        in[i] = (i * 123 + 7) % q;
    ntt_forward(&plan, fwd, in);
    ntt_inverse(&plan, back, fwd);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(in, back, n);
    ntt_plan_clear(&plan);
}

void test_ntt_roundtrip_matrix(void)
{
    const uint64_t sizes[3] = {8, 64, 256};
    const uint64_t qbits[3] = {20, 40, 59};
    for (int si = 0; si < 3; si++)
    {
        for (int qi = 0; qi < 3; qi++)
        {
            const uint64_t n = sizes[si];
            const uint64_t q = nt_next_ntt_prime(1ULL << qbits[qi], n, false);
            zq_ctx zq;
            zq_ctx_init(&zq, q);
            ntt_plan plan;
            TEST_ASSERT_EQUAL_INT(VFHE_OK, ntt_plan_init(&plan, n, &zq));

            uint64_t *in = (uint64_t *)safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *tmp = (uint64_t *)safe_aligned_malloc(n * sizeof(uint64_t));
            for (uint64_t i = 0; i < n; i++)
                in[i] = lcg() % q;
            ntt_forward(&plan, tmp, in);
            ntt_inverse(&plan, tmp, tmp); /* in-place inverse */
            TEST_ASSERT_EQUAL_UINT64_ARRAY(in, tmp, n);

            free(in);
            free(tmp);
            ntt_plan_clear(&plan);
        }
    }
}

void test_ntt_convolution_theorem(void)
{
    /* fwd(a) . fwd(b) followed by inverse must equal the negacyclic product
     * -- exercises plan + zq below the ring layer. */
    const uint64_t n = 32;
    const uint64_t q = nt_next_ntt_prime(1ULL << 40, n, false);
    zq_ctx zq;
    zq_ctx_init(&zq, q);
    ntt_plan plan;
    TEST_ASSERT_EQUAL_INT(VFHE_OK, ntt_plan_init(&plan, n, &zq));

    uint64_t a[32], b[32], fa[32], fb[32], prod[32], expect[32];
    for (uint64_t i = 0; i < n; i++)
    {
        a[i] = lcg() % q;
        b[i] = lcg() % q;
    }
    ntt_forward(&plan, fa, a);
    ntt_forward(&plan, fb, b);
    zq_arr_mul(&zq, prod, fa, fb, n);
    ntt_inverse(&plan, prod, prod);

    negacyclic_schoolbook(expect, a, b, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(expect, prod, n);
    ntt_plan_clear(&plan);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_ntt_roundtrip);
    RUN_TEST(test_ntt_roundtrip_matrix);
    RUN_TEST(test_ntt_convolution_theorem);
    return UNITY_END();
}
