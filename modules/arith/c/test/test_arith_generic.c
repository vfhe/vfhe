// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_arith_generic.c
 * @brief The generic arithmetic interface against the RNS kernels it wraps.
 *
 * Every operation is computed twice, once through arith_* and once by calling
 * the polynomial_* kernel directly, and the two results are compared word for
 * word: the interface must be a routing layer and nothing else. The domain
 * rules are checked separately, since they are the part with no direct
 * equivalent to compare against.
 */
#include <stdlib.h>
#include <string.h>

#include <arith.h>
#include <arith_generic.h>
#include <misc.h>

#include "unity.h"

#define TEST_N 64
#define TEST_L 3
#define TEST_MASK ((1ULL << TEST_L) - 1)

/* 49-bit NTT-friendly primes for N=64 with split_degree 1, as gen_primes
 * produces them: each is 1 mod 2N, so the transform exists. */
static uint64_t PRIMES[TEST_L] = {0x1FFFFFFFFE281ULL, 0x1FFFFFFFFDB81ULL, 0x1FFFFFFFFD581ULL};

static RNS_Base base = NULL;
static ArithRing ring = NULL;

void setUp(void)
{
    base = new_rns_base(PRIMES, 1, TEST_N, TEST_L);
    ring = arith_rns_ring_new(TEST_N, TEST_MASK, base);
}

void tearDown(void)
{
    arith_ring_free(ring);
    ring = NULL;
}

/* Deterministic contents, distinct per element, so a routing mistake that
 * swapped operands would show up as a mismatch rather than a coincidence. */
static void fill(RNS_Polynomial p, uint64_t seed)
{
    for (uint64_t i = 0; i < TEST_L; i++)
    {
        for (uint64_t j = 0; j < TEST_N; j++)
        {
            p->coeffs[i][j] =
                (seed * 6364136223846793005ULL + j * 1442695040888963407ULL + i * 1013904223ULL) %
                PRIMES[i];
        }
    }
}

static void assert_same(RNS_Polynomial expected, RNS_Polynomial actual)
{
    for (uint64_t i = 0; i < TEST_L; i++)
    {
        TEST_ASSERT_EQUAL_UINT64_ARRAY(expected->coeffs[i], actual->coeffs[i], TEST_N);
    }
}

void test_capabilities_and_names(void)
{
    TEST_ASSERT_EQUAL_STRING("rns", arith_implementation(ring));
    TEST_ASSERT_EQUAL_STRING("ntt", arith_backend(ring));
    TEST_ASSERT_TRUE(arith_supports(ring, ARITH_CAP_CORE));
    TEST_ASSERT_TRUE(arith_supports(ring, ARITH_CAP_QUOTIENT_POLY_RING));
    TEST_ASSERT_TRUE(arith_supports(ring, ARITH_CAP_TOWER));
    /* RNS keeps two distinct representations, so the flag must be absent and
     * the mul domain must be the transformed one. */
    TEST_ASSERT_FALSE(arith_supports(ring, ARITH_CAP_DOMAINS_COINCIDE));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_MUL, arith_mul_domain(ring));
}

void test_new_starts_empty_and_zero_is_canonical(void)
{
    ArithElement e;
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_new(ring, &e));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_EMPTY, e.domain);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_zero(ring, &e));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_CANONICAL, e.domain);
    arith_free(ring, &e);
    TEST_ASSERT_NULL(e.handle);
}

void test_new_like_matches_the_model(void)
{
    ArithElement model, copy;
    arith_new(ring, &model);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_new_like(ring, &model, &copy));
    RNS_Polynomial m = arith_rns_polynomial(&model), c = arith_rns_polynomial(&copy);
    TEST_ASSERT_EQUAL_UINT64(m->rns_mask, c->rns_mask);
    TEST_ASSERT_EQUAL_PTR(m->base, c->base);
    arith_free(ring, &model);
    arith_free(ring, &copy);
}

void test_add_matches_the_kernel_in_both_domains(void)
{
    ArithElement a, b, out;
    arith_new(ring, &a);
    arith_new(ring, &b);
    arith_new(ring, &out);
    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);

    /* canonical domain */
    fill(arith_rns_polynomial(&a), 11);
    fill(arith_rns_polynomial(&b), 22);
    a.domain = b.domain = ARITH_DOMAIN_CANONICAL;
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_add(ring, &out, &a, &b));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_CANONICAL, out.domain);
    polynomial_add_RNSc_polynomial((RNSc_Polynomial)expected,
                                   (RNSc_Polynomial)arith_rns_polynomial(&a),
                                   (RNSc_Polynomial)arith_rns_polynomial(&b));
    assert_same(expected, arith_rns_polynomial(&out));

    /* mul domain: the same kernel, since addition commutes with the
     * transform, but the flag must be carried through to the result */
    a.domain = b.domain = ARITH_DOMAIN_MUL;
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_add(ring, &out, &a, &b));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_MUL, out.domain);
    polynomial_add_RNS_polynomial(expected, arith_rns_polynomial(&a), arith_rns_polynomial(&b));
    assert_same(expected, arith_rns_polynomial(&out));

    free_RNS_polynomial(expected);
    arith_free(ring, &a);
    arith_free(ring, &b);
    arith_free(ring, &out);
}

void test_mul_matches_the_kernel(void)
{
    ArithElement a, b, out;
    arith_new(ring, &a);
    arith_new(ring, &b);
    arith_new(ring, &out);
    fill(arith_rns_polynomial(&a), 33);
    fill(arith_rns_polynomial(&b), 44);
    a.domain = b.domain = ARITH_DOMAIN_MUL;

    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_mul_RNS_polynomial(expected, arith_rns_polynomial(&a), arith_rns_polynomial(&b));
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_mul(ring, &out, &a, &b));
    assert_same(expected, arith_rns_polynomial(&out));

    free_RNS_polynomial(expected);
    arith_free(ring, &a);
    arith_free(ring, &b);
    arith_free(ring, &out);
}

void test_scale_matches_the_kernel(void)
{
    ArithElement a, out;
    arith_new(ring, &a);
    arith_new(ring, &out);
    fill(arith_rns_polynomial(&a), 55);
    a.domain = ARITH_DOMAIN_CANONICAL;

    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_scale_RNSc_polynomial((RNSc_Polynomial)expected,
                                     (RNSc_Polynomial)arith_rns_polynomial(&a), 12345);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_scale_int(ring, &out, &a, 12345));
    assert_same(expected, arith_rns_polynomial(&out));

    free_RNS_polynomial(expected);
    arith_free(ring, &a);
    arith_free(ring, &out);
}

void test_domain_roundtrip_restores_the_value(void)
{
    ArithElement e;
    arith_new(ring, &e);
    fill(arith_rns_polynomial(&e), 66);
    e.domain = ARITH_DOMAIN_CANONICAL;

    RNS_Polynomial original = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_copy_RNS_polynomial(original, arith_rns_polynomial(&e));

    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_to_mul(ring, &e));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_MUL, e.domain);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_to_canonical(ring, &e));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_CANONICAL, e.domain);
    assert_same(original, arith_rns_polynomial(&e));

    free_RNS_polynomial(original);
    arith_free(ring, &e);
}

void test_converting_to_the_domain_already_held_is_a_no_op(void)
{
    ArithElement e;
    arith_new(ring, &e);
    fill(arith_rns_polynomial(&e), 77);
    e.domain = ARITH_DOMAIN_CANONICAL;

    RNS_Polynomial original = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_copy_RNS_polynomial(original, arith_rns_polynomial(&e));
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_to_canonical(ring, &e));
    assert_same(original, arith_rns_polynomial(&e));

    free_RNS_polynomial(original);
    arith_free(ring, &e);
}

void test_mismatched_domains_are_refused(void)
{
    ArithElement a, b, out;
    arith_new(ring, &a);
    arith_new(ring, &b);
    arith_new(ring, &out);
    a.domain = ARITH_DOMAIN_CANONICAL;
    b.domain = ARITH_DOMAIN_MUL;
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_add(ring, &out, &a, &b));
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_sub(ring, &out, &a, &b));
    /* an untouched element has no value to combine */
    a.domain = b.domain = ARITH_DOMAIN_EMPTY;
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_add(ring, &out, &a, &b));
    arith_free(ring, &a);
    arith_free(ring, &b);
    arith_free(ring, &out);
}

void test_multiplication_refuses_the_canonical_domain(void)
{
    ArithElement a, b, out;
    arith_new(ring, &a);
    arith_new(ring, &b);
    arith_new(ring, &out);
    a.domain = b.domain = ARITH_DOMAIN_CANONICAL;
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_mul(ring, &out, &a, &b));
    arith_free(ring, &a);
    arith_free(ring, &b);
    arith_free(ring, &out);
}

/* --- the slots the consumer modules need ------------------------------- */

void test_mul_subto_and_scale_addto_match_the_kernels(void)
{
    ArithElement a, b, out;
    arith_new(ring, &a);
    arith_new(ring, &b);
    arith_new(ring, &out);
    fill(arith_rns_polynomial(&a), 91);
    fill(arith_rns_polynomial(&b), 92);
    fill(arith_rns_polynomial(&out), 93);
    a.domain = b.domain = out.domain = ARITH_DOMAIN_MUL;

    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_copy_RNS_polynomial(expected, arith_rns_polynomial(&out));
    polynomial_mul_subto_RNS_polynomial(expected, arith_rns_polynomial(&a),
                                        arith_rns_polynomial(&b));
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_mul_subto(ring, &out, &a, &b));
    assert_same(expected, arith_rns_polynomial(&out));

    polynomial_scale_addto_RNS_polynomial(expected, arith_rns_polynomial(&a), 7);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_scale_addto(ring, &out, &a, 7));
    assert_same(expected, arith_rns_polynomial(&out));

    free_RNS_polynomial(expected);
    arith_free(ring, &a);
    arith_free(ring, &b);
    arith_free(ring, &out);
}

void test_scale_by_a_per_prime_scalar(void)
{
    ArithElement a, out;
    arith_new(ring, &a);
    arith_new(ring, &out);
    fill(arith_rns_polynomial(&a), 94);
    a.domain = ARITH_DOMAIN_MUL;

    uint64_t residues[TEST_L] = {3, 5, 7};
    ArithScalar scalar;
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_scalar_new(ring, residues, &scalar));

    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_scale_RNS_polynomial_RNS(expected, arith_rns_polynomial(&a), residues);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_scale_by(ring, &out, &a, scalar));
    assert_same(expected, arith_rns_polynomial(&out));

    arith_scalar_free(ring, &scalar);
    TEST_ASSERT_NULL(scalar.handle);
    free_RNS_polynomial(expected);
    arith_free(ring, &a);
    arith_free(ring, &out);
}

void test_permute_and_monomial_match_the_kernels(void)
{
    ArithElement a, out;
    arith_new(ring, &a);
    arith_new(ring, &out);
    fill(arith_rns_polynomial(&a), 95);
    a.domain = ARITH_DOMAIN_CANONICAL;

    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_RNSc_permute((RNSc_Polynomial)expected, (RNSc_Polynomial)arith_rns_polynomial(&a),
                            3);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_permute(ring, &out, &a, 3));
    assert_same(expected, arith_rns_polynomial(&out));

    polynomial_RNSc_mul_by_xai((RNSc_Polynomial)expected, (RNSc_Polynomial)arith_rns_polynomial(&a),
                               5);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_mul_by_monomial(ring, &out, &a, 5, 0));
    assert_same(expected, arith_rns_polynomial(&out));

    /* the fused form: out = a*X^k - a */
    polynomial_RNSc_mul_by_xai_minus1((RNSc_Polynomial)expected,
                                      (RNSc_Polynomial)arith_rns_polynomial(&a), 5);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_mul_by_monomial(ring, &out, &a, 5, 1));
    assert_same(expected, arith_rns_polynomial(&out));

    free_RNS_polynomial(expected);
    arith_free(ring, &a);
    arith_free(ring, &out);
}

void test_quotient_ring_ops_refuse_the_mul_domain(void)
{
    ArithElement a, out;
    arith_new(ring, &a);
    arith_new(ring, &out);
    a.domain = ARITH_DOMAIN_MUL;
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_permute(ring, &out, &a, 3));
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_mul_by_monomial(ring, &out, &a, 5, 0));
    TEST_ASSERT_EQUAL_INT(ARITH_BAD_DOMAIN, arith_round_division(ring, &a, ring));
    arith_free(ring, &a);
    arith_free(ring, &out);
}

void test_sampling_lands_in_the_canonical_domain(void)
{
    ArithElement e;
    arith_new(ring, &e);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_sample_uniform(ring, &e));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_CANONICAL, e.domain);
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_sample_gaussian(ring, &e, 3.2));
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_CANONICAL, e.domain);
    arith_free(ring, &e);
}

void test_from_int_array_reduces_per_prime(void)
{
    ArithElement e;
    arith_new(ring, &e);
    uint64_t values[TEST_N];
    for (uint64_t i = 0; i < TEST_N; i++)
    {
        values[i] = i * 1000 + 1;
    }
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_from_int_array(ring, &e, values, TEST_N));
    /* RNS fuses the forward transform into the load, and says so */
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_MUL, e.domain);

    /* the residues are visible once it is back in the canonical form */
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_to_canonical(ring, &e));
    RNS_Polynomial got = arith_rns_polynomial(&e);
    for (uint64_t i = 0; i < TEST_L; i++)
    {
        for (uint64_t j = 0; j < TEST_N; j++)
        {
            TEST_ASSERT_EQUAL_UINT64(values[j] % PRIMES[i], got->coeffs[i][j]);
        }
    }
    arith_free(ring, &e);
}

/* The tower slots take the destination ring, and the implementation derives
 * which primes leave. */
void test_round_division_to_a_smaller_ring(void)
{
    const uint64_t small_mask = 0x3; /* drop the top prime */
    ArithRing small = arith_rns_ring_get(TEST_N, small_mask, base);

    ArithElement e;
    arith_new(ring, &e);
    fill(arith_rns_polynomial(&e), 96);
    e.domain = ARITH_DOMAIN_CANONICAL;

    RNS_Polynomial expected = polynomial_new_RNS_polynomial(TEST_N, TEST_MASK, base);
    polynomial_copy_RNS_polynomial(expected, arith_rns_polynomial(&e));
    polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)expected, TEST_MASK & ~small_mask);

    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_round_division(ring, &e, small));
    RNS_Polynomial got = arith_rns_polynomial(&e);
    TEST_ASSERT_EQUAL_UINT64(expected->rns_mask, got->rns_mask);
    for (uint64_t i = 0; i < TEST_L; i++)
    {
        if (expected->rns_mask & (1ULL << i))
        {
            TEST_ASSERT_EQUAL_UINT64_ARRAY(expected->coeffs[i], got->coeffs[i], TEST_N);
        }
    }
    free_RNS_polynomial(expected);
    arith_free(ring, &e);
}

void test_the_shared_ring_is_shared(void)
{
    ArithRing again = arith_rns_ring_get(TEST_N, TEST_MASK, base);
    TEST_ASSERT_EQUAL_PTR(again, arith_rns_ring_get(TEST_N, TEST_MASK, base));
    /* a different prime set is a different ring */
    TEST_ASSERT_TRUE(again != arith_rns_ring_get(TEST_N, 0x3, base));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_capabilities_and_names);
    RUN_TEST(test_new_starts_empty_and_zero_is_canonical);
    RUN_TEST(test_new_like_matches_the_model);
    RUN_TEST(test_add_matches_the_kernel_in_both_domains);
    RUN_TEST(test_mul_matches_the_kernel);
    RUN_TEST(test_scale_matches_the_kernel);
    RUN_TEST(test_domain_roundtrip_restores_the_value);
    RUN_TEST(test_converting_to_the_domain_already_held_is_a_no_op);
    RUN_TEST(test_mismatched_domains_are_refused);
    RUN_TEST(test_multiplication_refuses_the_canonical_domain);
    RUN_TEST(test_mul_subto_and_scale_addto_match_the_kernels);
    RUN_TEST(test_scale_by_a_per_prime_scalar);
    RUN_TEST(test_permute_and_monomial_match_the_kernels);
    RUN_TEST(test_quotient_ring_ops_refuse_the_mul_domain);
    RUN_TEST(test_sampling_lands_in_the_canonical_domain);
    RUN_TEST(test_from_int_array_reduces_per_prime);
    RUN_TEST(test_round_division_to_a_smaller_ring);
    RUN_TEST(test_the_shared_ring_is_shared);
    return UNITY_END();
}
