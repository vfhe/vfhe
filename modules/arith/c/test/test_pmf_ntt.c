// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_pmf_ntt.c
 * @brief The pseudo-Mersenne vector NTT against the scalar oracle, on every
 *        engine: forward and inverse agree element for element, the pair
 *        round-trips, the negacyclic convolution theorem holds against a
 *        schoolbook product, and a plan refuses a bad root or length.
 */
#include <stdlib.h>
#include <string.h>

#include <arith.h>
#include <util.h> /* safe_aligned_malloc: the planes must be 64-byte aligned */

#include "arith_internal.h"
#include "unity.h"

/* p = 2^260 - 22527: 5 limbs, shift 0, 2-adicity 11 -- what
   PseudoMersenneField.generate(260, two_adicity=8) picks. */
#define BITS 260
#define C 22527
#define MAX_LOG_N 10

static PMFParams params;
static uint64_t L;

void setUp(void) {}
void tearDown(void) {}

/* p >> shift as limbs; with p - 1 divisible by 2^shift this is (p - 1) / 2^shift. */
static void p_shifted(uint64_t *out, uint64_t shift)
{
    for (uint64_t k = 0; k < L; k++)
    {
        uint64_t word = params->p[k] >> shift;
        if (k + 1 < L)
            word |= (params->p[k + 1] << (PMF_LIMB_BITS - shift)) & PMF_LIMB_MASK;
        out[k] = word;
    }
}

/* The primitive 2n-th root the Python side derives: the least quadratic
   non-residue raised to (p - 1) / 2n. */
static void root_for(uint64_t *out, uint64_t logn)
{
    uint64_t g[PMF_LANES] = {0}, exp[PMF_MAX_LIMBS], check[PMF_LANES], minus_one[PMF_LANES] = {1};
    pmf_ref_neg(minus_one, minus_one, params);
    p_shifted(exp, 1);
    for (g[0] = 2;; g[0]++)
    {
        pmf_ref_pow(check, g, exp, L, params);
        if (pmf_is_equal(check, minus_one, params))
            break;
    }
    p_shifted(exp, logn + 1);
    pmf_ref_pow(out, g, exp, L, params);
}

static uint64_t lcg = 0x9E3779B97F4A7C15ULL;
static uint64_t next_word(void)
{
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg >> 12; /* 52 bits */
}

/* n elements, PMF_LANES words apart, canonical. */
static uint64_t *random_elements(uint64_t n)
{
    uint64_t *a = (uint64_t *)calloc(n * PMF_LANES, sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t raw[PMF_LANES] = {0};
        for (uint64_t k = 0; k < L; k++)
            raw[k] = next_word();
        pmf_canonicalize(a + i * PMF_LANES, raw, params);
    }
    return a;
}

static PMFVector vector_from(const uint64_t *elements, uint64_t n)
{
    PMFVector v = (PMFVector)calloc(1, sizeof(*v));
    v->n = n;
    v->allocated_n = pmf_vec_padded_length(n);
    v->params = params;
    v->limbs = (uint64_t **)malloc(L * sizeof(uint64_t *));
    for (uint64_t k = 0; k < L; k++)
    {
        v->limbs[k] = (uint64_t *)safe_aligned_malloc(v->allocated_n * sizeof(uint64_t));
        memset(v->limbs[k], 0, v->allocated_n * sizeof(uint64_t));
    }
    pmf_vec_set_range(v, 0, elements, n);
    return v;
}

static void vector_free(PMFVector v)
{
    for (uint64_t k = 0; k < L; k++)
        free(v->limbs[k]);
    free(v->limbs);
    free(v);
}

static void assert_vector_equals_elements(const PMFVector v, const uint64_t *elements)
{
    uint64_t got[PMF_LANES];
    for (uint64_t i = 0; i < v->n; i++)
    {
        pmf_vec_get_element(got, v, i);
        TEST_ASSERT_TRUE_MESSAGE(pmf_is_equal(got, elements + i * PMF_LANES, params),
                                 "vector and scalar transforms disagree");
        for (uint64_t k = L; k < PMF_LANES; k++)
            TEST_ASSERT_EQUAL_UINT64(0, got[k]);
    }
}

void test_vector_matches_scalar_oracle_and_round_trips(void)
{
    for (uint64_t logn = 0; logn <= MAX_LOG_N; logn++)
    {
        const uint64_t n = 1ULL << logn;
        uint64_t root[PMF_LANES];
        root_for(root, logn);
        PMFNTTPlan plan = pmf_ntt_new_plan(n, root, params);
        TEST_ASSERT_NOT_NULL(plan);

        uint64_t *original = random_elements(n);
        uint64_t *ref = (uint64_t *)malloc(n * PMF_LANES * sizeof(uint64_t));
        memcpy(ref, original, n * PMF_LANES * sizeof(uint64_t));
        PMFVector vec = vector_from(original, n);

        pmf_ref_ntt_forward(ref, plan);
        pmf_vec_ntt_forward(vec, plan);
        assert_vector_equals_elements(vec, ref);

        pmf_ref_ntt_inverse(ref, plan);
        pmf_vec_ntt_inverse(vec, plan);
        assert_vector_equals_elements(vec, ref);
        assert_vector_equals_elements(vec, original);

        vector_free(vec);
        free(ref);
        free(original);
        pmf_ntt_free_plan(plan);
    }
}

void test_negacyclic_convolution(void)
{
    const uint64_t logn = 5, n = 1ULL << logn;
    uint64_t root[PMF_LANES];
    root_for(root, logn);
    PMFNTTPlan plan = pmf_ntt_new_plan(n, root, params);
    TEST_ASSERT_NOT_NULL(plan);

    uint64_t *a = random_elements(n), *b = random_elements(n);
    uint64_t *expected = (uint64_t *)calloc(n * PMF_LANES, sizeof(uint64_t));
    uint64_t prod[PMF_LANES];
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++)
        {
            pmf_ref_mul(prod, a + i * PMF_LANES, b + j * PMF_LANES, params);
            uint64_t *slot = expected + ((i + j) % n) * PMF_LANES;
            if (i + j < n)
                pmf_ref_add(slot, slot, prod, params);
            else
                pmf_ref_sub(slot, slot, prod, params); /* X^n == -1 */
        }

    PMFVector va = vector_from(a, n), vb = vector_from(b, n);
    pmf_vec_ntt_forward(va, plan);
    pmf_vec_ntt_forward(vb, plan);
    pmf_vec_mul(va, va, vb);
    pmf_vec_ntt_inverse(va, plan);
    assert_vector_equals_elements(va, expected);

    vector_free(va);
    vector_free(vb);
    free(expected);
    free(a);
    free(b);
    pmf_ntt_free_plan(plan);
}

void test_plan_rejects_a_bad_root_or_length(void)
{
    uint64_t root[PMF_LANES], squared[PMF_LANES];
    root_for(root, 4);
    /* psi^2 has order n, not 2n. */
    pmf_ref_mul(squared, root, root, params);
    TEST_ASSERT_NULL(pmf_ntt_new_plan(16, squared, params));
    TEST_ASSERT_NULL(pmf_ntt_new_plan(12, root, params));
    TEST_ASSERT_NULL(pmf_ntt_new_plan(0, root, params));
    PMFNTTPlan plan = pmf_ntt_new_plan(16, root, params);
    TEST_ASSERT_NOT_NULL(plan);
    TEST_ASSERT_TRUE(pmf_is_equal(plan->root_of_unity, root, params));
    pmf_ntt_free_plan(plan);
}

int main(void)
{
    params = pmf_new_params(BITS, C);
    L = pmf_limbs(params);
    UNITY_BEGIN();
    RUN_TEST(test_vector_matches_scalar_oracle_and_round_trips);
    RUN_TEST(test_negacyclic_convolution);
    RUN_TEST(test_plan_rejects_a_bad_root_or_length);
    pmf_free_params(params);
    return UNITY_END();
}
