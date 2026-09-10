// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_ntt.c
 * @brief NTT forward/inverse roundtrips across a size x prime matrix, a fixed
 *        30-bit prime, and the negacyclic convolution theorem against a
 *        schoolbook oracle.
 */
#include <stdlib.h>
#include <string.h>

#include <arith.h>
#include <util.h> /* safe_aligned_malloc: the SIMD kernels need 64-byte-aligned buffers */

#include "arith_internal.h" /* MOD_SHIFT_*, ntt_new_plan_at_shift */

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void roundtrip(uint64_t n, uint64_t q)
{
    Modulus mod = mod_new(q);
    NTT_Plan plan = ntt_new_plan(n, mod);
    uint64_t *in = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *fwd = safe_aligned_malloc(n * sizeof(uint64_t));
    uint64_t *back = safe_aligned_malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
        in[i] = (0x123456789ABCDEFULL + i) % q;
    ntt_forward(fwd, in, plan);
    ntt_reverse(back, fwd, plan);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(in, back, n);
    free(in);
    free(fwd);
    free(back);
    ntt_free_plan(plan);
    mod_free(mod);
}

void test_ntt_roundtrip_matrix(void)
{
    const uint64_t sizes[] = {64, 256, 1024, 4096};
    /* next_special_prime returns a prime *above* 2^b, so a size that names a
       boundary lands just past it: 29 and 30 straddle the radix-2^32 family's
       q < 2^30, and 49 and 50 straddle the IFMA family's q < 2^50. Every
       derived constant is a function of bits(q), so a family tested at one
       modulus size is tested at one point. */
    const uint64_t bits[] = {20, 29, 30, 31, 49, 50, 60, 61};
    for (unsigned s = 0; s < sizeof(sizes) / sizeof(*sizes); s++)
        for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
            roundtrip(sizes[s], next_special_prime(1ULL << bits[b], sizes[s], true));
}

void test_ntt_roundtrip_30bit(void)
{
    roundtrip(1024, 1073643521ULL); /* fixed 30-bit prime, q == 1 (mod 2N) */
}

void test_ntt_negacyclic_convolution(void)
{
    const uint64_t sizes[] = {64, 256, 1024};
    const uint64_t bits[] = {20, 40, 60, 61};
    for (unsigned si = 0; si < sizeof(sizes) / sizeof(*sizes); si++)
    {
        for (unsigned bi = 0; bi < sizeof(bits) / sizeof(*bits); bi++)
        {
            const uint64_t n = sizes[si];
            const uint64_t q = next_special_prime(1ULL << bits[bi], n, true);
            Modulus mod = mod_new(q);
            NTT_Plan plan = ntt_new_plan(n, mod);

            uint64_t *a = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *b = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *ref = safe_aligned_malloc(n * sizeof(uint64_t));
            memset(ref, 0, n * sizeof(uint64_t));
            uint64_t *na = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *nb = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *nr = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *res = safe_aligned_malloc(n * sizeof(uint64_t));

            for (uint64_t i = 0; i < n; i++)
            {
                a[i] = (i + 123) % q;
                b[i] = (2 * i + 456) % q;
            }
            /* negacyclic schoolbook: X^n == -1 */
            for (uint64_t i = 0; i < n; i++)
                for (uint64_t j = 0; j < n; j++)
                {
                    uint64_t p = (uint64_t)(((unsigned __int128)a[i] * b[j]) % q);
                    if (i + j < n)
                        ref[i + j] = (ref[i + j] + p) % q;
                    else
                        ref[i + j - n] = (ref[i + j - n] + q - p) % q;
                }

            ntt_forward(na, a, plan);
            ntt_forward(nb, b, plan);
            mod_eltwise_mul(nr, na, nb, n, mod);
            ntt_reverse(res, nr, plan);
            TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, res, n);

            free(a);
            free(b);
            free(ref);
            free(na);
            free(nb);
            free(nr);
            free(res);
            ntt_free_plan(plan);
            mod_free(mod);
        }
    }
}

void test_ntt_plan_refuses_a_modulus_without_the_root(void)
{
    /* 2^61 - 1: q - 1 has 2-adicity 1, so no length above 1 has a 2n-th root. */
    Modulus mod = mod_new((1ULL << 61) - 1);
    TEST_ASSERT_NULL(ntt_new_plan(64, mod));
    TEST_ASSERT_NULL(ntt_new_plan(2, mod));
    NTT_Plan plan = ntt_new_plan(1, mod);
    TEST_ASSERT_NOT_NULL(plan);
    ntt_free_plan(plan);
    mod_free(mod);
}

/* ---- the family a modulus is dispatched to, and agreement between families ---- */

/* Harvey's lazy-reduction bound for Shoup radix 2^shift, stated here
   independently of the dispatcher so that this is a check and not an echo:
   a butterfly's un-reduced operand can be as large as 4q, and the reduction is
   only in range while that is below the radix. HEXL writes it 1 << (shift - 2).

   Shift 0 is the scalar tables, which carry no Shoup constants and so no radix
   bound: every modulus a plan can hold is admissible. An engine without
   vectorized transforms reports it for every plan, and so does any plan below
   NTT_MIN_VECTOR_LEN. */
static bool family_admits(uint64_t q, uint64_t shift)
{
    return shift == 0 || q < (1ULL << (shift - 2));
}

/* The dispatcher must never hand a modulus to a family whose bound it breaks.
   A radix-2^32 plan built for a prime at or above 2^30 is not off by a
   multiple of q, it is garbage, and it is garbage *rarely* -- a prime just
   past the bound corrupts a small enough fraction of inputs that a roundtrip
   matrix will not see it. So assert the bound directly rather than hoping a
   value check trips. */
void test_ntt_plan_stays_inside_its_family_bound(void)
{
    const uint64_t bits[] = {20, 29, 30, 31, 49, 50, 60, 61};
    for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
    {
        const uint64_t q = next_special_prime(1ULL << bits[b], 1024, true);
        Modulus mod = mod_new(q);
        NTT_Plan plan = ntt_new_plan(1024, mod);
        TEST_ASSERT_NOT_NULL(plan);
        TEST_ASSERT_TRUE_MESSAGE(family_admits(q, plan->shoup_shift),
                                 "plan dispatched to a family that cannot hold this modulus");
        ntt_free_plan(plan);
        mod_free(mod);
    }
}

#if VFHE_HAVE_AVX512IFMA
/* A modulus small enough for several families is legal input to every one of
   them, and an NTT is exact, so they owe the same words. Nothing else tests a
   family at a modulus size its own dispatch would not choose, which is exactly
   where a bound too loose by a couple of bits hides. */
void test_ntt_families_agree_where_they_overlap(void)
{
    const uint64_t shifts[] = {MOD_SHIFT_32, MOD_SHIFT_50, MOD_SHIFT_64};
    const uint64_t sizes[] = {16, 64, 256, 1024, 4096};
    const uint64_t bits[] = {20, 29, 30, 31, 49, 50, 60, 61};

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(*sizes); s++)
    {
        const uint64_t n = sizes[s];
        for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
        {
            const uint64_t q = next_special_prime(1ULL << bits[b], n, true);
            Modulus mod = mod_new(q);

            uint64_t *in = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *first = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *out = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *back = safe_aligned_malloc(n * sizeof(uint64_t));
            for (uint64_t i = 0; i < n; i++)
                in[i] = (0x9E3779B97F4A7C15ULL * (i + 1)) % q;

            int have_first = 0;
            for (unsigned k = 0; k < sizeof(shifts) / sizeof(*shifts); k++)
            {
                if (!family_admits(q, shifts[k]))
                    continue;
                NTT_Plan plan = ntt_new_plan_at_shift(n, mod, shifts[k]);
                TEST_ASSERT_NOT_NULL(plan);
                ntt_forward(out, in, plan);
                ntt_reverse(back, out, plan);
                /* each family inverts its own forward */
                TEST_ASSERT_EQUAL_UINT64_ARRAY(in, back, n);
                if (have_first)
                    TEST_ASSERT_EQUAL_UINT64_ARRAY(first, out, n);
                else
                {
                    memcpy(first, out, n * sizeof(uint64_t));
                    have_first = 1;
                }
                ntt_free_plan(plan);
            }
            TEST_ASSERT_TRUE(have_first);

            free(in);
            free(first);
            free(out);
            free(back);
            mod_free(mod);
        }
    }
}
#endif

#if VFHE_HAVE_AVX512IFMA
/* The 32-bit-word transform against the 64-bit one over the same plan.
   Storing a coefficient in 32 bits does not change its value and an NTT is
   exact, so the two owe identical words -- which makes the 64-bit transform,
   already checked against the direct-evaluation oracle above, the oracle here.
   Sizes span the guard at 32, the depth-first cutover at 4096, and one length
   below the guard where the entry point falls back to widening. */
void test_ntt_w32_matches_the_wide_transform(void)
{
    const uint64_t bits[] = {20, 29};
    const uint64_t sizes[] = {16, 32, 64, 256, 1024, 4096, 8192};

    for (unsigned b = 0; b < sizeof(bits) / sizeof(*bits); b++)
    {
        for (unsigned s = 0; s < sizeof(sizes) / sizeof(*sizes); s++)
        {
            const uint64_t n = sizes[s];
            const uint64_t q = next_special_prime(1ULL << bits[b], n, true);
            TEST_ASSERT_TRUE(rns_prime_is_narrow(q));
            Modulus mod = mod_new(q);
            NTT_Plan plan = ntt_new_plan(n, mod);
            TEST_ASSERT_NOT_NULL(plan);
            // the tables exist exactly when the length allows them
            TEST_ASSERT_EQUAL_INT(ntt_w32_applies(n, q), plan->ws_fwd32 != NULL);

            uint64_t *in64 = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *fwd64 = safe_aligned_malloc(n * sizeof(uint64_t));
            uint64_t *back64 = safe_aligned_malloc(n * sizeof(uint64_t));
            uint32_t *in32 = safe_aligned_malloc(n * sizeof(uint32_t));
            uint32_t *fwd32 = safe_aligned_malloc(n * sizeof(uint32_t));
            uint32_t *back32 = safe_aligned_malloc(n * sizeof(uint32_t));

            for (uint64_t i = 0; i < n; i++)
            {
                in64[i] = (0x9E3779B97F4A7C15ULL * (i + 1)) % q;
                in32[i] = (uint32_t)in64[i];
            }

            ntt_forward(fwd64, in64, plan);
            ntt_forward_w32(fwd32, in32, plan);
            for (uint64_t i = 0; i < n; i++)
            {
                TEST_ASSERT_TRUE(fwd64[i] < q);
                TEST_ASSERT_EQUAL_UINT64(fwd64[i], (uint64_t)fwd32[i]);
            }

            ntt_reverse(back64, fwd64, plan);
            ntt_reverse_w32(back32, fwd32, plan);
            for (uint64_t i = 0; i < n; i++)
            {
                TEST_ASSERT_EQUAL_UINT64(in64[i], back64[i]);
                TEST_ASSERT_EQUAL_UINT64(in64[i], (uint64_t)back32[i]);
            }

            // and in place, which is how rns_polynomial.c calls them
            memcpy(fwd32, in32, n * sizeof(uint32_t));
            ntt_forward_w32(fwd32, fwd32, plan);
            ntt_reverse_w32(fwd32, fwd32, plan);
            for (uint64_t i = 0; i < n; i++)
                TEST_ASSERT_EQUAL_UINT64(in64[i], (uint64_t)fwd32[i]);

            free(in64);
            free(fwd64);
            free(back64);
            free(in32);
            free(fwd32);
            free(back32);
            ntt_free_plan(plan);
            mod_free(mod);
        }
    }
}
#endif

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_ntt_roundtrip_matrix);
    RUN_TEST(test_ntt_roundtrip_30bit);
    RUN_TEST(test_ntt_negacyclic_convolution);
    RUN_TEST(test_ntt_plan_refuses_a_modulus_without_the_root);
    RUN_TEST(test_ntt_plan_stays_inside_its_family_bound);
#if VFHE_HAVE_AVX512IFMA
    RUN_TEST(test_ntt_families_agree_where_they_overlap);
    RUN_TEST(test_ntt_w32_matches_the_wide_transform);
#endif
    return UNITY_END();
}
