// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_mod.c
 * @brief Elementwise modular kernels checked against scalar __int128 oracles
 *        across a sweep of prime bit-sizes.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arith.h>
#include <util.h> /* safe_aligned_malloc: the SIMD kernels need 64-byte-aligned buffers */

#include "arith_internal.h" /* MOD_SHIFT_*, mod_shoup_shift, the per-family kernels */

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
    const uint64_t bits[] = {10, 20, 29, 30, 31, 49, 50, 60, 61};
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
    /* next_special_prime returns a prime *above* 2^b, so 29/30 straddle the
       radix-2^32 family's bound and 49/50 straddle the IFMA family's. */
    const uint64_t bits[] = {10, 20, 29, 30, 31, 49, 50, 60, 61};
    for (unsigned i = 0; i < sizeof(bits) / sizeof(*bits); i++)
        check_ops(bits[i], 1024);
}

/* The same oracles at lengths below MOD_MIN_VECTOR_LEN, which is what the
   dispatchers route to the scalar kernels. The sweep above only reaches those
   on the portable engine, where they are the whole implementation. */
void test_mod_eltwise_sweep_scalar_path(void)
{
    const uint64_t bits[] = {10, 20, 29, 30, 31, 49, 50, 60, 61};
    const uint64_t lengths[] = {1, 2, 4, 7};
    for (unsigned i = 0; i < sizeof(bits) / sizeof(*bits); i++)
        for (unsigned j = 0; j < sizeof(lengths) / sizeof(*lengths); j++)
            check_ops(bits[i], lengths[j]);
}

/* ---- the family the element-wise dispatchers pick ---- */

#if VFHE_HAVE_AVX512IFMA
/* Same construction as the transform's cross-family check, for the kernels
   that carry no tables. A modulus every family accepts must give one answer
   from all of them, and the family the dispatcher picks must be one that
   accepts it -- stated from Harvey's 4q <= B rather than read back off the
   dispatcher. */
static bool family_admits(uint64_t q, uint64_t shift) { return q < (1ULL << (shift - 2)); }

typedef void (*eltwise_mul_fn)(uint64_t *, uint64_t *, uint64_t *, uint64_t, Modulus);
typedef void (*eltwise_scale_fn)(uint64_t *, uint64_t *, uint64_t, uint64_t, Modulus);

void test_mod_eltwise_families_agree_where_they_overlap(void)
{
    const uint64_t shifts[] = {MOD_SHIFT_32, MOD_SHIFT_50, MOD_SHIFT_64};
    const eltwise_mul_fn muls[] = {mod_eltwise_mul_32, mod_eltwise_mul_50, mod_eltwise_mul_64};
    const eltwise_scale_fn scales[] = {mod_eltwise_scale_32, mod_eltwise_scale_50,
                                       mod_eltwise_scale_64};
    const eltwise_scale_fn fmas[] = {mod_eltwise_fma_32, mod_eltwise_fma_50, mod_eltwise_fma_64};
    const uint64_t bits[] = {10, 20, 29, 30, 31, 49, 50, 60, 61};
    const uint64_t n = 64;

    for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
    {
        const uint64_t q = next_special_prime(1ULL << bits[b], 1024, true);
        Modulus mod = mod_new(q);

        TEST_ASSERT_TRUE_MESSAGE(family_admits(q, mod_shoup_shift(q)),
                                 "element-wise dispatch chose a family that cannot hold this "
                                 "modulus");

        uint64_t *in1 = safe_aligned_malloc(n * sizeof(uint64_t));
        uint64_t *in2 = safe_aligned_malloc(n * sizeof(uint64_t));
        uint64_t *out = safe_aligned_malloc(n * sizeof(uint64_t));
        uint64_t *first = safe_aligned_malloc(n * sizeof(uint64_t));
        for (uint64_t i = 0; i < n; i++)
        {
            in1[i] = (0x9E3779B97F4A7C15ULL * (i + 1)) % q;
            in2[i] = (0xC2B2AE3D27D4EB4FULL * (i + 3)) % q;
        }
        const uint64_t scalar = in2[1];

        for (unsigned op = 0; op < 3; op++)
        {
            int have_first = 0;
            for (unsigned k = 0; k < sizeof(shifts) / sizeof(*shifts); k++)
            {
                if (!family_admits(q, shifts[k]))
                    continue;
                if (op == 0)
                    muls[k](out, in1, in2, n, mod);
                else
                {
                    /* scale and fma read `out`, so seed it the same way each time */
                    for (uint64_t i = 0; i < n; i++)
                        out[i] = in2[i];
                    (op == 1 ? scales[k] : fmas[k])(out, in1, scalar, n, mod);
                }
                if (have_first)
                    TEST_ASSERT_EQUAL_UINT64_ARRAY(first, out, n);
                else
                {
                    memcpy(first, out, n * sizeof(uint64_t));
                    have_first = 1;
                }
            }
            TEST_ASSERT_TRUE(have_first);
        }

        free(in1);
        free(in2);
        free(out);
        free(first);
        mod_free(mod);
    }
}
#endif

/* The widest modulus any family can hold is set by the lazy range: values live
   in [0, 4q) between butterfly stages, so 4q must fit the reduction radix, and
   the widest radix is 2^64. Above that there is no family to dispatch to, and
   the failure would be a wrong value rather than a refusal. */
void test_mod_new_refuses_a_modulus_no_family_can_hold(void)
{
    TEST_ASSERT_NULL(mod_new(1ULL << 62));
    TEST_ASSERT_NULL(mod_new((1ULL << 63) - 1));
    /* next_special_prime(1ULL << 62, ...) lands above the bound, which is how
       a 63-bit prime used to reach the kernels. */
    TEST_ASSERT_NULL(mod_new(next_special_prime(1ULL << 62, 1024, true)));

    /* and the largest in-contract modulus is still accepted */
    Modulus mod = mod_new(next_special_prime(1ULL << 61, 1024, true));
    TEST_ASSERT_NOT_NULL(mod);
    TEST_ASSERT_TRUE(mod->q < (1ULL << 62));
    mod_free(mod);
}

/* The 32-bit-word kernels against the 64-bit ones, at a prime both accept.
   Storing a coefficient in 32 bits does not change its value, so the two must
   agree word for word -- which makes the 64-bit family, already checked
   against an __int128 oracle above, the oracle for these. */
void test_mod_eltwise_w32_matches_the_wide_kernels(void)
{
    const uint64_t bits[] = {10, 20, 29};
    /* Powers of two, short ones included. Two reasons for that shape: a row
       operation can be handed `N / split_degree`, which is a power of two and
       may be under a lane group -- below 16 the 32-bit-word bodies would
       compute nothing at all, so they route to scalar, and unguarded that is
       ARITH-7 again. And a power of two is all any caller passes, which is
       what makes it sound that none of these kernels has a tail: their real
       precondition is a whole number of lane groups, not merely one. */
    const uint64_t lengths[] = {1, 2, 4, 8, 16, 32, 64, 1024};

    for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
    {
        const uint64_t q = next_special_prime(1ULL << bits[b], 1024, true);
        TEST_ASSERT_TRUE(rns_prime_is_narrow(q));
        Modulus mod = mod_new(q);

        for (unsigned li = 0; li < sizeof(lengths) / sizeof(*lengths); li++)
        {
            const uint64_t n = lengths[li];
            uint64_t *w1 = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *w2 = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *wo = safe_aligned_malloc(n * sizeof(uint64_t));
            uint32_t *n1 = safe_aligned_malloc(n * sizeof(uint32_t));
            uint32_t *n2 = safe_aligned_malloc(n * sizeof(uint32_t));
            uint32_t *no = safe_aligned_malloc(n * sizeof(uint32_t));

            for (uint64_t i = 0; i < n; i++)
            {
                w1[i] = (0x9E3779B97F4A7C15ULL * (i + 1)) % q;
                w2[i] = (0xC2B2AE3D27D4EB4FULL * (i + 3)) % q;
                n1[i] = (uint32_t)w1[i];
                n2[i] = (uint32_t)w2[i];
            }
            const uint64_t scalar = w2[1];

/* Run the wide kernel and the narrow one from the same inputs, then require
   every coefficient to match. `seed_out` says whether the operation reads its
   output (the accumulating ones do). */
#define W32_CHECK(seed_out, wide_call, narrow_call)                                                \
    do                                                                                             \
    {                                                                                              \
        for (uint64_t i = 0; i < n; i++)                                                           \
        {                                                                                          \
            wo[i] = (seed_out) ? w1[i] : 0;                                                        \
            no[i] = (uint32_t)wo[i];                                                               \
        }                                                                                          \
        wide_call;                                                                                 \
        narrow_call;                                                                               \
        for (uint64_t i = 0; i < n; i++)                                                           \
        {                                                                                          \
            TEST_ASSERT_TRUE(wo[i] < q);                                                           \
            TEST_ASSERT_EQUAL_UINT64(wo[i], (uint64_t)no[i]);                                      \
        }                                                                                          \
    } while (0)

            W32_CHECK(0, mod_eltwise_mul(wo, w1, w2, n, mod),
                      mod_eltwise_mul_w32(no, n1, n2, n, mod));
            W32_CHECK(1, mod_eltwise_mul_addto(wo, w1, w2, n, mod),
                      mod_eltwise_mul_addto_w32(no, n1, n2, n, mod));
            W32_CHECK(1, mod_eltwise_mul_subto(wo, w1, w2, n, mod),
                      mod_eltwise_mul_subto_w32(no, n1, n2, n, mod));
            W32_CHECK(0, mod_eltwise_scale(wo, w1, scalar, n, mod),
                      mod_eltwise_scale_w32(no, n1, scalar, n, mod));
            W32_CHECK(1, mod_eltwise_fma(wo, w1, scalar, n, mod),
                      mod_eltwise_fma_w32(no, n1, scalar, n, mod));
            W32_CHECK(0, mod_eltwise_add(wo, w1, w2, n, mod),
                      mod_eltwise_add_w32(no, n1, n2, n, mod));
            W32_CHECK(0, mod_eltwise_sub(wo, w1, w2, n, mod),
                      mod_eltwise_sub_w32(no, n1, n2, n, mod));
            W32_CHECK(0, mod_eltwise_negate(wo, w1, n, mod),
                      mod_eltwise_negate_w32(no, n1, n, mod));
            W32_CHECK(0, mod_eltwise_add_scalar(wo, w1, scalar, n, mod),
                      mod_eltwise_add_scalar_w32(no, n1, scalar, n, mod));
            W32_CHECK(0, mod_eltwise_sub_scalar(wo, w1, scalar, n, mod),
                      mod_eltwise_sub_scalar_w32(no, n1, scalar, n, mod));
#undef W32_CHECK

            free(w1);
            free(w2);
            free(wo);
            free(n1);
            free(n2);
            free(no);
        }
        mod_free(mod);
    }
}

/* The extremes, which the structured inputs above never produce: 0 and q-1 in
   every position of a vector, where a conditional subtract is either taken or
   skipped for the whole lane group. */
void test_mod_eltwise_w32_at_the_range_edges(void)
{
    const uint64_t q = next_special_prime(1ULL << 29, 1024, true);
    Modulus mod = mod_new(q);
    const uint64_t n = 64;
    const uint64_t edges[] = {0, 1, 2, q - 2, q - 1};
    const unsigned ne = sizeof(edges) / sizeof(*edges);

    uint64_t *w1 = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *w2 = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *wo = safe_aligned_malloc(n * sizeof(uint64_t));
    uint32_t *n1 = safe_aligned_malloc(n * sizeof(uint32_t));
    uint32_t *n2 = safe_aligned_malloc(n * sizeof(uint32_t));
    uint32_t *no = safe_aligned_malloc(n * sizeof(uint32_t));

    for (unsigned e1 = 0; e1 < ne; e1++)
    {
        for (unsigned e2 = 0; e2 < ne; e2++)
        {
            for (uint64_t i = 0; i < n; i++)
            {
                w1[i] = edges[e1];
                w2[i] = edges[e2];
                n1[i] = (uint32_t)w1[i];
                n2[i] = (uint32_t)w2[i];
                wo[i] = edges[(e1 + e2) % ne];
                no[i] = (uint32_t)wo[i];
            }
            mod_eltwise_mul_addto(wo, w1, w2, n, mod);
            mod_eltwise_mul_addto_w32(no, n1, n2, n, mod);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(wo[i], (uint64_t)no[i]);

            for (uint64_t i = 0; i < n; i++)
            {
                wo[i] = 0;
                no[i] = 0;
            }
            mod_eltwise_sub(wo, w1, w2, n, mod);
            mod_eltwise_sub_w32(no, n1, n2, n, mod);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(wo[i], (uint64_t)no[i]);

            mod_eltwise_scale(wo, w1, edges[e2], n, mod);
            mod_eltwise_scale_w32(no, n1, edges[e2], n, mod);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(wo[i], (uint64_t)no[i]);

            mod_eltwise_negate(wo, w1, n, mod);
            mod_eltwise_negate_w32(no, n1, n, mod);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(wo[i], (uint64_t)no[i]);
        }
    }
    free(w1);
    free(w2);
    free(wo);
    free(n1);
    free(n2);
    free(no);
    mod_free(mod);
}

/* The width-changing kernels against the wide path. These are the only
   operations allowed to convert a word width, so each is checked against
   "do it in 64 bits, then move the width" done the long way. */
void test_mod_w32_width_changes_match_the_wide_path(void)
{
    const uint64_t narrow_bits[] = {10, 20, 29};
    const uint64_t wide_bits[] = {31, 40, 50, 60};
    const uint64_t lengths[] = {8, 16, 64, 1024};

    for (unsigned li = 0; li < sizeof(lengths) / sizeof(*lengths); li++)
    {
        const uint64_t n = lengths[li];
        uint64_t *w = safe_aligned_malloc(n * sizeof(uint64_t));
        uint64_t *wref = safe_aligned_malloc(n * sizeof(uint64_t));
        uint32_t *nar = safe_aligned_malloc(n * sizeof(uint32_t));
        uint32_t *nref = safe_aligned_malloc(n * sizeof(uint32_t));
        int64_t *sgn = safe_aligned_malloc(n * sizeof(int64_t));

        for (unsigned b = 0; b < sizeof(narrow_bits) / sizeof(*narrow_bits); b++)
        {
            const uint64_t qn = next_special_prime(1ULL << narrow_bits[b], 1024, true);
            TEST_ASSERT_TRUE(rns_prime_is_narrow(qn));
            Modulus mn = mod_new(qn);

            for (uint64_t i = 0; i < n; i++)
            {
                nar[i] = (uint32_t)((0x9E3779B97F4A7C15ULL * (i + 1)) % qn);
                w[i] = (0xC2B2AE3D27D4EB4FULL * (i + 3));
                sgn[i] = (i & 1) ? -(int64_t)((i * 7919) % qn) : (int64_t)((i * 104729) % qn);
            }

            // widen then narrow is the identity on a canonical narrow row
            mod_widen_w32(wref, nar, n);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64((uint64_t)nar[i], wref[i]);
            mod_narrow_w32(nref, wref, n);
            TEST_ASSERT_EQUAL_UINT32_ARRAY(nar, nref, n);

            // a signed array reduced straight into a narrow row
            mod_eltwise_reduce_signed_w32(nref, sgn, n, mn);
            mod_eltwise_reduce_signed(wref, sgn, n, mn);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(wref[i], (uint64_t)nref[i]);

            // an arbitrary 64-bit array reduced into a narrow row
            mod_eltwise_reduce_narrow_from_wide(nref, w, n, mn);
            mod_eltwise_reduce(wref, w, n, mn);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(wref[i], (uint64_t)nref[i]);

            // a narrow row reduced mod its own prime is the identity
            mod_eltwise_reduce_w32(nref, nar, n, mn);
            TEST_ASSERT_EQUAL_UINT32_ARRAY(nar, nref, n);

            for (unsigned c = 0; c < sizeof(wide_bits) / sizeof(*wide_bits); c++)
            {
                const uint64_t qw = next_special_prime(1ULL << wide_bits[c], 1024, true);
                TEST_ASSERT_FALSE(rns_prime_is_narrow(qw));
                Modulus mw = mod_new(qw);

                // narrow row -> wide row, the cross-modulus direction
                mod_eltwise_reduce_wide_from_narrow(w, nar, n, mw);
                mod_widen_w32(wref, nar, n);
                mod_eltwise_reduce(wref, wref, n, mw);
                TEST_ASSERT_EQUAL_UINT64_ARRAY(wref, w, n);
                mod_free(mw);
            }
            mod_free(mn);
        }
        free(w);
        free(wref);
        free(nar);
        free(nref);
        free(sgn);
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_modq_one_and_two_words);
    RUN_TEST(test_mod_eltwise_sweep);
    RUN_TEST(test_mod_eltwise_sweep_scalar_path);
    RUN_TEST(test_mod_new_refuses_a_modulus_no_family_can_hold);
#if VFHE_HAVE_AVX512IFMA
    RUN_TEST(test_mod_eltwise_families_agree_where_they_overlap);
#endif
    RUN_TEST(test_mod_eltwise_w32_matches_the_wide_kernels);
    RUN_TEST(test_mod_eltwise_w32_at_the_range_edges);
    RUN_TEST(test_mod_w32_width_changes_match_the_wide_path);
    return UNITY_END();
}
