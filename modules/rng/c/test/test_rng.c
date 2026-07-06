/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_rng.c
 * @brief Unit tests for the rng module: byte stream, Gaussian sampling, and
 *        sparse ternary vectors. Statistical bounds are deliberately loose
 *        (many standard deviations) so the suite never flakes.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "rng.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* -------------------------------------------------------------- stream ---- */

void test_random_bytes_not_all_zero(void)
{
    uint8_t buf[64];
    memset(buf, 0, sizeof(buf));
    rng_random_bytes(sizeof(buf), buf);
    int any = 0;
    for (size_t i = 0; i < sizeof(buf); i++)
        any |= buf[i];
    TEST_ASSERT_TRUE(any); /* all-zero would be astronomically unlikely */
}

void test_random_bytes_draws_differ(void)
{
    /* Consecutive draws must differ, on both the buffered path (< 512) and
     * the direct path (>= 512). */
    uint8_t a[64], b[64];
    rng_random_bytes(sizeof(a), a);
    rng_random_bytes(sizeof(b), b);
    TEST_ASSERT_TRUE(memcmp(a, b, sizeof(a)) != 0);

    static uint8_t big_a[2048], big_b[2048];
    rng_random_bytes(sizeof(big_a), big_a);
    rng_random_bytes(sizeof(big_b), big_b);
    TEST_ASSERT_TRUE(memcmp(big_a, big_b, sizeof(big_a)) != 0);
}

void test_random_bytes_rough_uniformity(void)
{
    /* Mean of 8192 uniform bytes: expectation 127.5, sd of the mean ~ 0.8.
     * A +-12 window is ~15 sigma -- catches a broken generator, never a
     * healthy one. */
    static uint8_t buf[8192];
    rng_random_bytes(sizeof(buf), buf);
    double sum = 0;
    for (size_t i = 0; i < sizeof(buf); i++)
        sum += buf[i];
    double mean = sum / (double)sizeof(buf);
    TEST_ASSERT_TRUE(mean > 115.0 && mean < 140.0);
}

void test_random_bytes_odd_sizes(void)
{
    /* Sizes around the buffer-refill boundary must fill every byte; check
     * that a canary past the requested length survives. */
    for (uint64_t len = 1; len <= 33; len += 8)
    {
        uint8_t buf[64];
        memset(buf, 0xAB, sizeof(buf));
        rng_random_bytes(len, buf);
        TEST_ASSERT_EQUAL_UINT8(0xAB, buf[len]);
    }
}

/* ------------------------------------------------------------ gaussian ---- */

void test_gaussian_moments(void)
{
    const int n = 4000;
    const double sigma = 3.2;
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < n; i++)
    {
        double x = rng_gaussian(sigma);
        TEST_ASSERT_TRUE(isfinite(x));
        sum += x;
        sum_sq += x * x;
    }
    double mean = sum / n;
    double var = sum_sq / n - mean * mean;
    /* sd of the sample mean ~ 0.05: |mean| < 0.5 is ~10 sigma. */
    TEST_ASSERT_TRUE(fabs(mean) < 0.5);
    /* Variance should be near sigma^2 = 10.24. */
    TEST_ASSERT_TRUE(var > 6.0 && var < 16.0);
}

/* ------------------------------------------------------- sparse ternary ---- */

void test_sparse_ternary_weight_and_balance(void)
{
    const uint64_t n = 64, h = 8, q = 12289;
    uint64_t out[64];
    rng_sparse_ternary(out, n, h, q);
    uint64_t plus = 0, minus = 0, other = 0, sum = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        sum = (sum + out[i]) % q;
        if (out[i] == 1)
            plus++;
        else if (out[i] == q - 1)
            minus++;
        else if (out[i] != 0)
            other++;
    }
    TEST_ASSERT_EQUAL_UINT64(h / 2, plus);  /* exactly h/2 of +1 */
    TEST_ASSERT_EQUAL_UINT64(h / 2, minus); /* exactly h/2 of -1 */
    TEST_ASSERT_EQUAL_UINT64(0, other);     /* nothing but 0 / +-1 */
    TEST_ASSERT_EQUAL_UINT64(0, sum);       /* balanced: sums to 0 mod q */
}

void test_sparse_ternary_dense_and_varied(void)
{
    /* Half-dense vector: the rejection loop must still terminate and hold
     * the invariants. */
    const uint64_t n = 32, h = 16, q = 65537;
    uint64_t a[32], b[32];
    rng_sparse_ternary(a, n, h, q);
    rng_sparse_ternary(b, n, h, q);
    uint64_t nz = 0;
    for (uint64_t i = 0; i < n; i++)
        nz += (a[i] != 0);
    TEST_ASSERT_EQUAL_UINT64(h, nz);
    /* Two draws with 16 random support positions colliding entirely is
     * ~2^-30: treat equality as failure. */
    TEST_ASSERT_TRUE(memcmp(a, b, sizeof(a)) != 0);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_random_bytes_not_all_zero);
    RUN_TEST(test_random_bytes_draws_differ);
    RUN_TEST(test_random_bytes_rough_uniformity);
    RUN_TEST(test_random_bytes_odd_sizes);
    RUN_TEST(test_gaussian_moments);
    RUN_TEST(test_sparse_ternary_weight_and_balance);
    RUN_TEST(test_sparse_ternary_dense_and_varied);
    return UNITY_END();
}
