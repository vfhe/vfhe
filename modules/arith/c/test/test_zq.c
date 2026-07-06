/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_zq.c
 * @brief The Z_q elementwise kernels: scalar helpers, the smoke array ops, and
 *        a randomized __int128 oracle across every reduction tier.
 */
#include <stdint.h>
#include <string.h>

#include "arith.h"
#include "test_arith_support.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_zq_scalar(void)
{
    const uint64_t q = 97;
    TEST_ASSERT_EQUAL_UINT64(3, zq_scalar_add(50, 50, q)); /* 100 mod 97 */
    TEST_ASSERT_EQUAL_UINT64(94, zq_scalar_sub(0, 3, q));  /* -3 mod 97 */
    TEST_ASSERT_EQUAL_UINT64(0, zq_scalar_negate(0, q));
    TEST_ASSERT_EQUAL_UINT64(q - 5, zq_scalar_negate(5, q));

    zq_ctx zq;
    zq_ctx_init(&zq, 1152921504606846883ULL); /* a 60-bit prime */
    const uint64_t a = 987654321987654321ULL, b = 123456789123456789ULL;
    const uint64_t expect = (uint64_t)(((unsigned __int128)a * b) % zq.q);
    TEST_ASSERT_EQUAL_UINT64(expect, zq_scalar_mul(a, b, &zq));
}

void test_zq_arrays(void)
{
    const uint64_t n = 16;
    const uint64_t q = nt_next_ntt_prime(1ULL << 40, n, false);
    zq_ctx zq;
    zq_ctx_init(&zq, q);

    uint64_t a[16], b[16], out[16];
    for (uint64_t i = 0; i < n; i++)
    {
        a[i] = (i * 1234567891ULL) % q;
        b[i] = (i * 987654321ULL + 17) % q;
    }

    zq_arr_mul(&zq, out, a, b, n);
    for (uint64_t i = 0; i < n; i++)
        TEST_ASSERT_EQUAL_UINT64((uint64_t)(((unsigned __int128)a[i] * b[i]) % q), out[i]);

    zq_arr_add(&zq, out, a, b, n);
    for (uint64_t i = 0; i < n; i++)
        TEST_ASSERT_EQUAL_UINT64((a[i] + b[i]) % q, out[i]);

    /* Fused accumulate: out += a * b must equal the two-step reference. */
    uint64_t acc[16];
    memcpy(acc, b, sizeof(acc));
    zq_arr_mul_addto(&zq, acc, a, b, n);
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t prod = (uint64_t)(((unsigned __int128)a[i] * b[i]) % q);
        TEST_ASSERT_EQUAL_UINT64((b[i] + prod) % q, acc[i]);
    }
}

static void zq_ops_oracle_for_prime(uint64_t q)
{
    enum
    {
        N = 32
    };
    zq_ctx zq;
    zq_ctx_init(&zq, q);

    uint64_t a[N], b[N], out[N], ref[N];
    int64_t s_in[N];
    uint64_t hi[N], lo[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = lcg() % q;
        b[i] = lcg() % q;
        s_in[i] = (int64_t)(lcg() >> 1) * ((i & 1) ? -1 : 1);
        hi[i] = lcg();
        lo[i] = lcg();
    }
    const uint64_t s = lcg();

#define PROD(x, y) ((uint64_t)(((unsigned __int128)(x) * (y)) % q))

    zq_arr_mul(&zq, out, a, b, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64(PROD(a[i], b[i]), out[i]);

    memcpy(out, b, sizeof(out));
    zq_arr_mul_addto(&zq, out, a, b, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((b[i] + PROD(a[i], b[i])) % q, out[i]);

    memcpy(out, b, sizeof(out));
    zq_arr_mul_subto(&zq, out, a, b, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((b[i] + q - PROD(a[i], b[i])) % q, out[i]);

    zq_arr_scale(&zq, out, a, s, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64(PROD(a[i], s % q), out[i]);

    memcpy(out, b, sizeof(out));
    zq_arr_scale_addto(&zq, out, a, s, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((b[i] + PROD(a[i], s % q)) % q, out[i]);

    zq_arr_add(&zq, out, a, b, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((a[i] + b[i]) % q, out[i]);

    zq_arr_sub(&zq, out, a, b, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((a[i] + q - b[i]) % q, out[i]);

    zq_arr_negate(&zq, out, a, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((q - a[i]) % q, out[i]);

    zq_arr_add_scalar(&zq, out, a, s, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((a[i] + s % q) % q, out[i]);

    zq_arr_sub_scalar(&zq, out, a, s, N);
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64((a[i] + q - s % q) % q, out[i]);

    zq_arr_reduce(&zq, out, hi, N); /* arbitrary 64-bit inputs */
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_UINT64(hi[i] % q, out[i]);

    zq_arr_reduce_signed(&zq, out, s_in, N);
    for (int i = 0; i < N; i++)
    {
        uint64_t abs_v = (s_in[i] < 0) ? (uint64_t)(-s_in[i]) : (uint64_t)s_in[i];
        uint64_t r = abs_v % q;
        ref[i] = (s_in[i] < 0 && r != 0) ? q - r : r;
        TEST_ASSERT_EQUAL_UINT64(ref[i], out[i]);
    }

    zq_arr_reduce_wide(&zq, out, hi, lo, N);
    for (int i = 0; i < N; i++)
    {
        unsigned __int128 v = (((unsigned __int128)hi[i]) << 64) | lo[i];
        TEST_ASSERT_EQUAL_UINT64((uint64_t)(v % q), out[i]);
    }

    /* Scalar wide reduction, randomized. */
    for (int i = 0; i < 200; i++)
    {
        unsigned __int128 v = (((unsigned __int128)lcg()) << 64) | lcg();
        TEST_ASSERT_EQUAL_UINT64((uint64_t)(v % q), zq_reduce_u128(v, &zq));
    }
#undef PROD
}

void test_zq_oracle_all_tiers(void)
{
    /* One prime per reduction tier (32 / 50 / 64-bit kernels on SIMD builds;
     * same portable kernels but different constants otherwise). */
    zq_ops_oracle_for_prime(nt_next_ntt_prime(1ULL << 20, 16, false));
    zq_ops_oracle_for_prime(nt_next_ntt_prime(1ULL << 45, 16, false));
    zq_ops_oracle_for_prime(nt_next_ntt_prime(1ULL << 59, 16, false));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_zq_scalar);
    RUN_TEST(test_zq_arrays);
    RUN_TEST(test_zq_oracle_all_tiers);
    return UNITY_END();
}
