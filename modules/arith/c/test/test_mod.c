/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mod.c
 * @brief Elementwise modular kernels checked against scalar __int128 oracles
 *        across a sweep of prime bit-sizes.
 */
#include <stdint.h>
#include <stdlib.h>

#include <arith.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- scalar reference oracles ---- */

static uint64_t ref_modq(unsigned __int128 x, uint64_t q) { return (uint64_t)(x % q); }

static void ref_mul(uint64_t *o, uint64_t *a, uint64_t *b, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        o[i] = (uint64_t)(((unsigned __int128)a[i] * b[i]) % q);
}
static void ref_add(uint64_t *o, uint64_t *a, uint64_t *b, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        o[i] = (a[i] + b[i]) % q;
}
static void ref_sub(uint64_t *o, uint64_t *a, uint64_t *b, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        o[i] = (a[i] + q - b[i]) % q;
}
static void ref_scale(uint64_t *o, uint64_t *a, uint64_t s, uint64_t n, uint64_t q)
{
    s %= q;
    for (uint64_t i = 0; i < n; i++)
        o[i] = (uint64_t)(((unsigned __int128)a[i] * s) % q);
}
static void ref_fma(uint64_t *o, uint64_t *a, uint64_t s, uint64_t n, uint64_t q)
{
    s %= q;
    for (uint64_t i = 0; i < n; i++)
        o[i] = (o[i] + (uint64_t)(((unsigned __int128)a[i] * s) % q)) % q;
}
static void ref_add_scalar(uint64_t *o, uint64_t *a, uint64_t s, uint64_t n, uint64_t q)
{
    s %= q;
    for (uint64_t i = 0; i < n; i++)
        o[i] = (a[i] + s) % q;
}
static void ref_sub_scalar(uint64_t *o, uint64_t *a, uint64_t s, uint64_t n, uint64_t q)
{
    s %= q;
    for (uint64_t i = 0; i < n; i++)
        o[i] = (a[i] + q - s) % q;
}
static void ref_negate(uint64_t *o, uint64_t *a, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        o[i] = (q - (a[i] % q)) % q;
}
static void ref_reduce(uint64_t *o, uint64_t *a, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        o[i] = a[i] % q;
}
static void ref_reduce_signed(uint64_t *o, int64_t *a, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
    {
        int64_t v = a[i];
        uint64_t r = ((v < 0) ? -(uint64_t)v : (uint64_t)v) % q;
        o[i] = (v < 0) ? (r == 0 ? 0 : q - r) : r;
    }
}
static void ref_reduce_mp(uint64_t *o, uint64_t *hi, uint64_t *lo, uint64_t n, uint64_t q)
{
    for (uint64_t i = 0; i < n; i++)
        o[i] = (uint64_t)((((unsigned __int128)hi[i] << 64) | lo[i]) % q);
}

/* ---- kernels vs oracles for one prime ---- */

static void check_ops(uint64_t q_bits)
{
    const uint64_t n = 1024;
    const uint64_t q = next_special_prime(1ULL << q_bits, n, true);
    NTT_proc proc = ntt_new_proc(n, q);

    uint64_t *in1 = malloc(n * sizeof(uint64_t));
    uint64_t *in2 = malloc(n * sizeof(uint64_t));
    uint64_t *out = malloc(n * sizeof(uint64_t));
    uint64_t *ref = malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
    {
        in1[i] = (0x123456789ABCDEFULL ^ (i * 0x1337BEEFULL)) % q;
        in2[i] = (0xFEDCBA987654321ULL ^ (i * 0xDEADBEEFULL)) % q;
    }

    const uint64_t scale = 0x123456789ABCDEFULL, scalar = 0x9876543210ABCDEFULL;

    mod_eltwise_add(out, in1, in2, n, proc);
    ref_add(ref, in1, in2, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_sub(out, in1, in2, n, proc);
    ref_sub(ref, in1, in2, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_mul(out, in1, in2, n, proc);
    ref_mul(ref, in1, in2, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_scale(out, in1, scale, n, proc);
    ref_scale(ref, in1, scale, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    for (uint64_t i = 0; i < n; i++)
        out[i] = ref[i] = i % q;
    mod_eltwise_fma(out, in1, scale, n, proc);
    ref_fma(ref, in1, scale, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_add_scalar(out, in1, scalar, n, proc);
    ref_add_scalar(ref, in1, scalar, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_sub_scalar(out, in1, scalar, n, proc);
    ref_sub_scalar(ref, in1, scalar, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_negate(out, in1, n, proc);
    ref_negate(ref, in1, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    uint64_t *large = malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
        large[i] = 0xFFFFFFFFFFFFFFFFULL ^ (i * 0x12345678ULL);
    mod_eltwise_reduce(out, large, n, proc);
    ref_reduce(ref, large, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);
    free(large);

    int64_t *sig = malloc(n * sizeof(int64_t));
    for (uint64_t i = 0; i < n; i++)
    {
        if (i == 0)
            sig[i] = 0;
        else if (i == 1)
            sig[i] = INT64_MIN;
        else if (i == 2)
            sig[i] = -1;
        else if (i == 3)
            sig[i] = 1;
        else
            sig[i] = (i % 2 ? 1 : -1) * (int64_t)(i * 0x12345678ULL);
    }
    mod_eltwise_reduce_signed(out, sig, n, proc);
    ref_reduce_signed(ref, sig, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);
    free(sig);

    uint64_t *hi = malloc(n * sizeof(uint64_t)), *lo = malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
    {
        hi[i] = 0xAAAAAAAAAAAAAAAAULL ^ (i * 0x11111111ULL);
        lo[i] = 0x5555555555555555ULL ^ (i * 0x22222222ULL);
    }
    mod_reduce_array_mp(out, hi, lo, n, proc);
    ref_reduce_mp(ref, hi, lo, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);
    free(hi);
    free(lo);

    free(in1);
    free(in2);
    free(out);
    free(ref);
    ntt_free_proc(proc);
}

void test_modq_scalar(void)
{
    const uint64_t q = next_special_prime(1ULL << 50, 1024, true);
    NTT_proc proc = ntt_new_proc(1024, q);
    for (int i = 0; i < 200; i++)
    {
        unsigned __int128 v = ((unsigned __int128)rand() << 64) | (unsigned)rand();
        TEST_ASSERT_EQUAL_UINT64(ref_modq(v, q), modq(v, proc));
    }
    ntt_free_proc(proc);
}

void test_mod_eltwise_sweep(void)
{
    const uint64_t bits[] = {10, 20, 30, 40, 50, 60, 62};
    for (unsigned i = 0; i < sizeof(bits) / sizeof(*bits); i++)
        check_ops(bits[i]);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_modq_scalar);
    RUN_TEST(test_mod_eltwise_sweep);
    return UNITY_END();
}
