// SPDX-FileCopyrightText: 2026 The vFHE Authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_normal.c
 * @brief The two Gaussian samplers: finite, of the right moments, and the
 * seeded one a function of its seed alone.
 */
#include <math.h>
#include <normal.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

#define SIGMA 3.2
#define DRAWS 200000

/* Mean and variance of `n` samples against sigma: the mean within a few
 * standard errors of 0, the variance within 5% (rounding adds 1/12). */
static void assert_moments(const double *x, uint64_t n, double variance)
{
    double sum = 0, sq = 0;
    for (uint64_t i = 0; i < n; i++)
        sum += x[i];
    const double mean = sum / (double)n;
    for (uint64_t i = 0; i < n; i++)
        sq += (x[i] - mean) * (x[i] - mean);
    TEST_ASSERT_TRUE(fabs(mean) < 4 * sqrt(variance / (double)n));
    TEST_ASSERT_TRUE(fabs(sq / (double)n - variance) < 0.05 * variance);
}

void test_entropy_samples_are_finite_and_gaussian(void)
{
    double *x = (double *)malloc(DRAWS * sizeof(double));
    for (uint64_t i = 0; i < DRAWS; i++)
    {
        x[i] = normal_random(NULL, SIGMA);
        TEST_ASSERT_TRUE(isfinite(x[i]));
    }
    assert_moments(x, DRAWS, SIGMA * SIGMA);
    free(x);
}

void test_seeded_samples_are_gaussian_integers(void)
{
    const uint8_t seed[] = "a secret seed";
    int64_t *out = (int64_t *)malloc(DRAWS * sizeof(int64_t));
    double *x = (double *)malloc(DRAWS * sizeof(double));
    normal_from_seed(out, DRAWS, SIGMA, seed, sizeof seed);
    for (uint64_t i = 0; i < DRAWS; i++)
        x[i] = (double)out[i];
    assert_moments(x, DRAWS, SIGMA * SIGMA + 1.0 / 12.0);
    free(x);
    free(out);
}

/* A function of the seed: the same seed gives the same samples however many
 * are asked for, across the 256-sample chunk, and another seed differs. */
void test_seeded_samples_are_a_function_of_the_seed(void)
{
    const uint8_t seed[] = "a secret seed", other[] = "a secret seeD";
    int64_t a[1000], b[1000];
    normal_from_seed(a, 1000, SIGMA, seed, sizeof seed);
    normal_from_seed(b, 300, SIGMA, seed, sizeof seed);
    TEST_ASSERT_EQUAL_INT64_ARRAY(a, b, 300);
    normal_from_seed(b, 1000, SIGMA, seed, sizeof seed);
    TEST_ASSERT_EQUAL_INT64_ARRAY(a, b, 1000);
    normal_from_seed(b, 1000, SIGMA, other, sizeof other);
    TEST_ASSERT_TRUE(memcmp(a, b, sizeof a) != 0);
    normal_from_seed(b, 0, SIGMA, seed, sizeof seed); /* writes nothing */
}

/* Sigma scales the samples. */
void test_sigma_scales(void)
{
    const uint8_t seed[] = "scale";
    double *x = (double *)malloc(DRAWS * sizeof(double));
    int64_t *out = (int64_t *)malloc(DRAWS * sizeof(int64_t));
    normal_from_seed(out, DRAWS, 100.0, seed, sizeof seed);
    for (uint64_t i = 0; i < DRAWS; i++)
        x[i] = (double)out[i];
    assert_moments(x, DRAWS, 100.0 * 100.0 + 1.0 / 12.0);
    free(out);
    free(x);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_entropy_samples_are_finite_and_gaussian);
    RUN_TEST(test_seeded_samples_are_gaussian_integers);
    RUN_TEST(test_seeded_samples_are_a_function_of_the_seed);
    RUN_TEST(test_sigma_scales);
    return UNITY_END();
}
