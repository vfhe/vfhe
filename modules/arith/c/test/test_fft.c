// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_fft.c
 * @brief The negacyclic FFT against a direct evaluation of its defining sum,
 *        roundtrips up to the largest length, exact integer convolution
 *        against a schoolbook oracle, and the vector transform against the
 *        scalar one.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <arith.h>
#include <util.h> /* safe_aligned_malloc: the vector transform needs 64-byte-aligned buffers */

#include "arith_internal.h" /* fft_forward_scalar, fft_inverse_scalar */

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint64_t next_random(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}
// Uniform in [-1, 1).
static double random_unit(void) { return (double)(next_random() >> 11) * 0x1p-52 - 1.0; }

static unsigned log2_exact(uint64_t n)
{
    unsigned b = 0;
    while ((1ULL << b) < n)
        b++;
    return b;
}

static uint64_t reverse_bits(uint64_t x, unsigned bits)
{
    uint64_t r = 0;
    for (unsigned b = 0; b < bits; b++)
        r |= ((x >> b) & 1) << (bits - 1 - b);
    return r;
}

static double *random_buffer(uint64_t n)
{
    double *x = safe_aligned_malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++)
        x[i] = random_unit();
    return x;
}

void test_fft_matches_the_defining_sum(void)
{
    const long double pi = 3.141592653589793238462643383279502884L;
    for (uint64_t n = 2; n <= 4096; n *= 2)
    {
        const uint64_t m = n / 2;
        const unsigned bits = log2_exact(m);
        FFT_Plan plan = fft_new_plan(n);
        double *a = random_buffer(n), *x = safe_aligned_malloc(n * sizeof(double));
        memcpy(x, a, n * sizeof(double));
        fft_forward(x, plan);
        // zeta^t = exp(i pi t / n), t < 2n
        long double *cs = malloc(2 * n * sizeof(long double));
        long double *sn = malloc(2 * n * sizeof(long double));
        for (uint64_t t = 0; t < 2 * n; t++)
        {
            cs[t] = cosl(pi * (long double)t / (long double)n);
            sn[t] = sinl(pi * (long double)t / (long double)n);
        }
        long double err = 0, norm = 0;
        for (uint64_t k = 0; k < m; k++)
        {
            // f(zeta^(4k+1))
            long double re = 0, im = 0;
            for (uint64_t j = 0; j < n; j++)
            {
                const uint64_t t = ((4 * k + 1) * j) % (2 * n);
                re += a[j] * cs[t];
                im += a[j] * sn[t];
            }
            const uint64_t at = reverse_bits(k, bits);
            err += (x[at] - re) * (x[at] - re) + (x[m + at] - im) * (x[m + at] - im);
            norm += re * re + im * im;
        }
        TEST_ASSERT_TRUE_MESSAGE(sqrtl(err / norm) < 1e-15L, "relative error above 1e-15");
        free(cs);
        free(sn);
        free(a);
        free(x);
        fft_free_plan(plan);
    }
}

void test_fft_roundtrip_every_length(void)
{
    for (uint64_t n = 2; n <= (1ULL << 20); n *= 2)
    {
        FFT_Plan plan = fft_new_plan(n);
        TEST_ASSERT_NOT_NULL(plan);
        double *a = random_buffer(n), *x = safe_aligned_malloc(n * sizeof(double));
        memcpy(x, a, n * sizeof(double));
        fft_forward(x, plan);
        fft_inverse(x, plan);
        double worst = 0;
        for (uint64_t i = 0; i < n; i++)
            worst = fmax(worst, fabs(x[i] - a[i]));
        TEST_ASSERT_TRUE_MESSAGE(worst < 1e-13, "roundtrip error above 1e-13");
        free(a);
        free(x);
        fft_free_plan(plan);
    }
}

void test_fft_negacyclic_convolution_is_exact(void)
{
    // |a_i|, |b_i| < 2^12: products below 2^24 n, far inside double precision,
    // so the rounded result must equal the integer convolution exactly.
    for (uint64_t n = 2; n <= 2048; n *= 2)
    {
        FFT_Plan plan = fft_new_plan(n);
        double *a = safe_aligned_malloc(n * sizeof(double));
        double *b = safe_aligned_malloc(n * sizeof(double));
        int64_t *ai = malloc(n * sizeof(int64_t)), *bi = malloc(n * sizeof(int64_t));
        int64_t *expected = calloc(n, sizeof(int64_t));
        for (uint64_t i = 0; i < n; i++)
        {
            ai[i] = (int64_t)(next_random() % 8191) - 4095;
            bi[i] = (int64_t)(next_random() % 8191) - 4095;
            a[i] = (double)ai[i];
            b[i] = (double)bi[i];
        }
        for (uint64_t i = 0; i < n; i++)
            for (uint64_t j = 0; j < n; j++)
            {
                const int64_t p = ai[i] * bi[j];
                if (i + j < n)
                    expected[i + j] += p;
                else
                    expected[i + j - n] -= p; // X^n = -1
            }
        fft_forward(a, plan);
        fft_forward(b, plan);
        fft_pointwise_mul(a, a, b, plan);
        fft_inverse(a, plan);
        for (uint64_t i = 0; i < n; i++)
            TEST_ASSERT_EQUAL_INT64(expected[i], llrint(a[i]));
        free(a);
        free(b);
        free(ai);
        free(bi);
        free(expected);
        fft_free_plan(plan);
    }
}

void test_fft_vector_transform_matches_the_scalar_one(void)
{
    // The scalar transform is the whole implementation on the portable
    // engine; on an engine with the vector one, the two must agree to within
    // rounding at every length the vector code handles.
    for (uint64_t n = 32; n <= 65536; n *= 2)
    {
        FFT_Plan plan = fft_new_plan(n);
        double *a = random_buffer(n);
        double *x = safe_aligned_malloc(n * sizeof(double));
        double *y = safe_aligned_malloc(n * sizeof(double));
        memcpy(x, a, n * sizeof(double));
        memcpy(y, a, n * sizeof(double));
        fft_forward(x, plan);
        fft_forward_scalar(y, plan);
        double worst = 0;
        for (uint64_t i = 0; i < n; i++)
            worst = fmax(worst, fabs(x[i] - y[i]));
        TEST_ASSERT_TRUE_MESSAGE(worst < 1e-12 * sqrt((double)n), "forward transforms disagree");
        fft_inverse(x, plan);
        fft_inverse_scalar(y, plan);
        worst = 0;
        for (uint64_t i = 0; i < n; i++)
            worst = fmax(worst, fabs(x[i] - y[i]));
        TEST_ASSERT_TRUE_MESSAGE(worst < 1e-13, "inverse transforms disagree");
        free(a);
        free(x);
        free(y);
        fft_free_plan(plan);
    }
}

void test_fft_plan_refuses_a_bad_length(void)
{
    const uint64_t bad[] = {0, 1, 3, 48, 1ULL << 21};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        TEST_ASSERT_NULL(fft_new_plan(bad[i]));
    fft_free_plan(NULL);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_fft_matches_the_defining_sum);
    RUN_TEST(test_fft_roundtrip_every_length);
    RUN_TEST(test_fft_negacyclic_convolution_is_exact);
    RUN_TEST(test_fft_vector_transform_matches_the_scalar_one);
    RUN_TEST(test_fft_plan_refuses_a_bad_length);
    return UNITY_END();
}
