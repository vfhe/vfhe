// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mod.c
 * @brief Elementwise modular kernels checked against scalar __int128 oracles
 *        across a sweep of prime bit-sizes.
 */
#include <stdint.h>
#include <stdlib.h>

#include <arith.h>
#include <util.h> /* safe_aligned_malloc: the SIMD kernels need 64-byte-aligned buffers */

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

/* `n` selects which implementation the dispatchers in mod.c reach: at or above
   MOD_MIN_VECTOR_LEN (8) the vectorized kernels, below it the size-generic
   scalar ones in mod_scalar.c. On the portable engine both settings land on the
   scalar kernels. The root-of-unity order the prime is chosen for is fixed
   separately, so a short `n` still gets a usable prime. */
static void check_ops(uint64_t q_bits, uint64_t n)
{
    const uint64_t q = next_special_prime(1ULL << q_bits, 1024, true);
    Modulus mod = mod_new(q);

    uint64_t *in1 = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *in2 = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *out = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *ref = safe_aligned_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
    {
        in1[i] = (0x123456789ABCDEFULL ^ (i * 0x1337BEEFULL)) % q;
        in2[i] = (0xFEDCBA987654321ULL ^ (i * 0xDEADBEEFULL)) % q;
    }

    const uint64_t scale = 0x123456789ABCDEFULL, scalar = 0x9876543210ABCDEFULL;

    mod_eltwise_add(out, in1, in2, n, mod);
    ref_add(ref, in1, in2, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_sub(out, in1, in2, n, mod);
    ref_sub(ref, in1, in2, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_mul(out, in1, in2, n, mod);
    ref_mul(ref, in1, in2, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_scale(out, in1, scale, n, mod);
    ref_scale(ref, in1, scale, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    for (uint64_t i = 0; i < n; i++)
        out[i] = ref[i] = i % q;
    mod_eltwise_fma(out, in1, scale, n, mod);
    ref_fma(ref, in1, scale, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_add_scalar(out, in1, scalar, n, mod);
    ref_add_scalar(ref, in1, scalar, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_sub_scalar(out, in1, scalar, n, mod);
    ref_sub_scalar(ref, in1, scalar, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    mod_eltwise_negate(out, in1, n, mod);
    ref_negate(ref, in1, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);

    uint64_t *large = safe_aligned_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
        large[i] = 0xFFFFFFFFFFFFFFFFULL ^ (i * 0x12345678ULL);
    mod_eltwise_reduce(out, large, n, mod);
    ref_reduce(ref, large, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);
    free(large);

    int64_t *sig = safe_aligned_malloc(n * sizeof(int64_t));
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
    mod_eltwise_reduce_signed(out, sig, n, mod);
    ref_reduce_signed(ref, sig, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);
    free(sig);

    uint64_t *hi = safe_aligned_malloc(n * sizeof(uint64_t)),
             *lo = safe_aligned_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
    {
        hi[i] = 0xAAAAAAAAAAAAAAAAULL ^ (i * 0x11111111ULL);
        lo[i] = 0x5555555555555555ULL ^ (i * 0x22222222ULL);
    }
    mod_reduce_array_mp(out, hi, lo, n, mod);
    ref_reduce_mp(ref, hi, lo, n, q);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, out, n);
    free(hi);
    free(lo);

    free(in1);
    free(in2);
    free(out);
    free(ref);
    mod_free(mod);
}

void test_modq_one_and_two_words(void)
{
    /* The two-word path folds on the precomputed residues of 2^52 and 2^104 and
       shifts by k - 64, all of which move with the size of q -- so sweep it. */
    const uint64_t bits[] = {10, 20, 30, 40, 50, 60, 62};
    for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
    {
        const uint64_t q = next_special_prime(1ULL << bits[b], 1024, true);
        Modulus mod = mod_new(q);
        for (int i = 0; i < 200; i++)
        {
            const uint64_t hi = ((uint64_t)rand() << 32) | (unsigned)rand();
            const uint64_t lo = ((uint64_t)rand() << 32) | (unsigned)rand();
            const unsigned __int128 v = ((unsigned __int128)hi << 64) | lo;
            TEST_ASSERT_EQUAL_UINT64(ref_modq(v, q), modq_wide(hi, lo, mod));
            /* the one-word entry point and the two-word one with hi = 0 must
               agree with each other and with the oracle */
            TEST_ASSERT_EQUAL_UINT64(ref_modq(lo, q), modq(lo, mod));
            TEST_ASSERT_EQUAL_UINT64(ref_modq(lo, q), modq_wide(0, lo, mod));
            /* and mul_modq must match a 128-bit product reduced directly */
            const uint64_t a = lo % q, c = hi % q;
            TEST_ASSERT_EQUAL_UINT64(ref_modq((unsigned __int128)a * c, q), mul_modq(a, c, mod));
        }
        /* boundaries of the internal paths: the 2^52 fast path and the limbs */
        const uint64_t edges[] = {0, 1, q - 1, (1ULL << 52) - 1, 1ULL << 52, UINT64_MAX};
        for (unsigned e = 0; e < sizeof(edges) / sizeof(*edges); e++)
        {
            TEST_ASSERT_EQUAL_UINT64(ref_modq(edges[e], q), modq(edges[e], mod));
            TEST_ASSERT_EQUAL_UINT64(ref_modq(edges[e], q), modq_wide(0, edges[e], mod));
            TEST_ASSERT_EQUAL_UINT64(ref_modq(((unsigned __int128)edges[e] << 64) | edges[e], q),
                                     modq_wide(edges[e], edges[e], mod));
        }
        mod_free(mod);
    }
}

void test_mod_eltwise_sweep(void)
{
    const uint64_t bits[] = {10, 20, 30, 40, 50, 60, 62};
    for (unsigned i = 0; i < sizeof(bits) / sizeof(*bits); i++)
        check_ops(bits[i], 1024);
}

/* The same oracles at lengths below MOD_MIN_VECTOR_LEN, which is what the
   dispatchers route to the scalar kernels. The sweep above only reaches those
   on the portable engine, where they are the whole implementation. */
void test_mod_eltwise_sweep_scalar_path(void)
{
    const uint64_t bits[] = {10, 20, 30, 40, 50, 60, 62};
    const uint64_t lengths[] = {1, 2, 4, 7};
    for (unsigned i = 0; i < sizeof(bits) / sizeof(*bits); i++)
        for (unsigned j = 0; j < sizeof(lengths) / sizeof(*lengths); j++)
            check_ops(bits[i], lengths[j]);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_modq_one_and_two_words);
    RUN_TEST(test_mod_eltwise_sweep);
    RUN_TEST(test_mod_eltwise_sweep_scalar_path);
    return UNITY_END();
}
