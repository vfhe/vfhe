// SPDX-FileCopyrightText: 2026 The vFHE Authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_arith_trace.c
 * @brief The execution tape over the generic interface: which events an
 *        operation leaves, how operands are versioned, and what a reader gets.
 */
#include <stdlib.h>
#include <string.h>

#include "arith_internal.h" /* rns_row_is_narrow: row width is per prime */

#include <arith.h>
#include <arith_generic.h>
#include <arith_trace.h>

#include "unity.h"

#define TEST_N 64
#define TEST_L 4
#define TEST_MASK ((1ULL << TEST_L) - 1)

static uint64_t PRIMES[TEST_L] = {0x1FFFFFFFFE281ULL, 0x1FFFFFFFFDB81ULL, 0x1FFFFFFFFD581ULL,
                                  0x3FFFFE81ULL};

static RNS_Base base = NULL;
static ArithRing ring = NULL;

void setUp(void)
{
    base = new_rns_base(PRIMES, 1, TEST_N, TEST_L);
    ring = arith_rns_ring_new(TEST_N, TEST_MASK, base);
}

void tearDown(void)
{
    arith_trace_clear();
    arith_ring_free(ring);
    ring = NULL;
    rns_base_free(base);
    base = NULL;
}

static uint64_t coeff(RNS_Polynomial p, uint64_t i, uint64_t j)
{
    return rns_row_is_narrow(p->base, i) ? (uint64_t)p->rows32[i][j] : p->rows64[i][j];
}

static void set_coeff(RNS_Polynomial p, uint64_t i, uint64_t j, uint64_t v)
{
    if (rns_row_is_narrow(p->base, i))
        p->rows32[i][j] = (uint32_t)v;
    else
        p->rows64[i][j] = v;
}

/* A fresh canonical element with deterministic contents, distinct per seed. */
static void filled(ArithElement *e, uint64_t seed)
{
    arith_new(ring, e);
    RNS_Polynomial p = arith_rns_polynomial(e);
    for (uint64_t i = 0; i < TEST_L; i++)
        for (uint64_t j = 0; j < TEST_N; j++)
            set_coeff(
                p, i, j,
                (seed * 6364136223846793005ULL + j * 1442695040888963407ULL + i * 1013904223ULL) %
                    PRIMES[i]);
    e->domain = ARITH_DOMAIN_CANONICAL;
}

static void assert_same(RNS_Polynomial expected, RNS_Polynomial actual)
{
    TEST_ASSERT_EQUAL_UINT64(expected->rns_mask, actual->rns_mask);
    for (uint64_t i = 0; i < TEST_L; i++)
    {
        if (!(expected->rns_mask & (1ULL << i)))
            continue;
        for (uint64_t j = 0; j < TEST_N; j++)
            TEST_ASSERT_EQUAL_UINT64(coeff(expected, i, j), coeff(actual, i, j));
    }
}

static ArithTraceEvent event(uint64_t index)
{
    ArithTraceEvent e;
    TEST_ASSERT_EQUAL_INT(0, arith_trace_read(index, &e));
    return e;
}

void test_nothing_is_recorded_while_disabled(void)
{
    ArithElement x, y, z;
    filled(&x, 1);
    filled(&y, 2);
    arith_new(ring, &z);
    TEST_ASSERT_FALSE(arith_trace_is_enabled());
    arith_add(ring, &z, &x, &y);
    TEST_ASSERT_EQUAL_UINT64(0, arith_trace_length());
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &z);
}

void test_add_records_its_operands_then_the_sum(void)
{
    ArithElement x, y, z;
    filled(&x, 1);
    filled(&y, 2);
    arith_new(ring, &z);
    arith_trace_enable();
    TEST_ASSERT_TRUE(arith_trace_is_enabled());
    arith_add(ring, &z, &x, &y);
    TEST_ASSERT_EQUAL_UINT64(3, arith_trace_length());

    ArithTraceEvent e = event(0);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_INPUT, e.kind);
    TEST_ASSERT_EQUAL_UINT64(1, e.sequence);
    TEST_ASSERT_EQUAL_UINT32(0, e.n_inputs);
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_CANONICAL, e.domain);
    TEST_ASSERT_NOT_NULL(e.value);
    assert_same(arith_rns_polynomial(&x), (RNS_Polynomial)e.value);

    e = event(1);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_INPUT, e.kind);
    assert_same(arith_rns_polynomial(&y), (RNS_Polynomial)e.value);

    e = event(2);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_ADD, e.kind);
    TEST_ASSERT_EQUAL_UINT64(3, e.sequence);
    TEST_ASSERT_EQUAL_UINT32(2, e.n_inputs);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(2, e.inputs[1]);
    TEST_ASSERT_EQUAL_UINT64(TEST_N, e.N);
    TEST_ASSERT_EQUAL_UINT64(TEST_MASK, e.mask);
    TEST_ASSERT_EQUAL_PTR(base, e.base);
    TEST_ASSERT_FALSE(e.secret);

    // The reader takes the value and releases the events.
    void *taken = arith_trace_take_value(2);
    TEST_ASSERT_NOT_NULL(taken);
    assert_same(arith_rns_polynomial(&z), (RNS_Polynomial)taken);
    TEST_ASSERT_NULL(event(2).value);
    free_RNS_polynomial(taken);
    arith_trace_release(3);
    TEST_ASSERT_EQUAL_UINT64(0, arith_trace_length());
    TEST_ASSERT_EQUAL_UINT64(4, arith_trace_next_sequence());
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &z);
}

void test_a_value_an_event_wrote_is_not_recorded_again(void)
{
    ArithElement x, y, z, w;
    filled(&x, 1);
    filled(&y, 2);
    arith_new(ring, &z);
    arith_new(ring, &w);
    arith_trace_enable();
    arith_add(ring, &z, &x, &y);
    arith_sub(ring, &w, &z, &x);
    TEST_ASSERT_EQUAL_UINT64(4, arith_trace_length());
    ArithTraceEvent e = event(3);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_SUB, e.kind);
    TEST_ASSERT_EQUAL_UINT64(3, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[1]);
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &z);
    arith_free(ring, &w);
}

void test_an_accumulation_names_its_previous_version_first(void)
{
    ArithElement x, y, acc;
    filled(&x, 1);
    filled(&y, 2);
    arith_to_mul(ring, &x);
    arith_to_mul(ring, &y);
    arith_new(ring, &acc);
    arith_trace_enable();
    arith_zero_in(ring, &acc, ARITH_DOMAIN_MUL);
    arith_mul_addto(ring, &acc, &x, &y);
    TEST_ASSERT_EQUAL_UINT64(4, arith_trace_length());
    ArithTraceEvent e = event(0);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_ZERO, e.kind);
    TEST_ASSERT_NULL(e.value);
    e = event(3);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_MUL_ADDTO, e.kind);
    TEST_ASSERT_EQUAL_UINT32(3, e.n_inputs);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(2, e.inputs[1]);
    TEST_ASSERT_EQUAL_UINT64(3, e.inputs[2]);
    TEST_ASSERT_EQUAL_INT(ARITH_DOMAIN_MUL, e.domain);
    assert_same(arith_rns_polynomial(&acc), (RNS_Polynomial)e.value);
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &acc);
}

void test_copy_carries_no_payload_and_names_the_source(void)
{
    ArithElement x, c;
    filled(&x, 1);
    arith_new(ring, &c);
    arith_trace_enable();
    arith_copy(ring, &c, &x);
    TEST_ASSERT_EQUAL_UINT64(2, arith_trace_length());
    ArithTraceEvent e = event(1);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_COPY, e.kind);
    TEST_ASSERT_EQUAL_UINT32(1, e.n_inputs);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[0]);
    TEST_ASSERT_NULL(e.value);
    TEST_ASSERT_NULL(arith_trace_take_value(1));
    arith_free(ring, &x);
    arith_free(ring, &c);
}

void test_a_freed_element_is_forgotten(void)
{
    ArithElement x, y, z, w;
    filled(&x, 1);
    filled(&y, 2);
    arith_new(ring, &z);
    arith_new(ring, &w);
    arith_trace_enable();
    arith_add(ring, &z, &x, &y);
    // Whatever address the new element lands on, it is a value no event
    // wrote, so it is recorded as an input again.
    arith_free(ring, &x);
    filled(&x, 3);
    arith_add(ring, &w, &x, &y);
    TEST_ASSERT_EQUAL_UINT64(5, arith_trace_length());
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_INPUT, event(3).kind);
    ArithTraceEvent e = event(4);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_ADD, e.kind);
    TEST_ASSERT_EQUAL_UINT64(4, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(2, e.inputs[1]);
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &z);
    arith_free(ring, &w);
}

void test_round_division_is_recorded_in_place(void)
{
    ArithRing to = arith_rns_ring_new(TEST_N, TEST_MASK >> 1, base);
    ArithElement x;
    filled(&x, 1);
    arith_trace_enable();
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_round_division(ring, &x, to));
    TEST_ASSERT_EQUAL_UINT64(2, arith_trace_length());
    ArithTraceEvent e = event(0);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_INPUT, e.kind);
    TEST_ASSERT_EQUAL_UINT64(TEST_MASK, e.mask);
    e = event(1);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_ROUND_DIVISION, e.kind);
    TEST_ASSERT_EQUAL_UINT32(1, e.n_inputs);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(TEST_MASK >> 1, e.mask);
    assert_same(arith_rns_polynomial(&x), (RNS_Polynomial)e.value);
    arith_free(ring, &x);
    arith_ring_free(to);
}

void test_scale_by_records_the_scalar_and_samples_are_secret(void)
{
    ArithElement x, y, s;
    filled(&x, 1);
    arith_new(ring, &y);
    arith_new(ring, &s);
    uint64_t factors[TEST_L] = {3, 5, 7, 11};
    ArithScalar scalar;
    arith_scalar_new(ring, factors, &scalar);
    arith_trace_enable();
    arith_scale_by(ring, &y, &x, scalar);
    arith_sample_gaussian(ring, &s, 3.2);
    TEST_ASSERT_EQUAL_UINT64(3, arith_trace_length());
    ArithTraceEvent e = event(1);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_SCALE_BY, e.kind);
    TEST_ASSERT_EQUAL_UINT64(TEST_L, e.scalar_len);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(factors, e.scalar, TEST_L);
    e = event(2);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_SAMPLE_GAUSSIAN, e.kind);
    TEST_ASSERT_TRUE(e.secret);
    TEST_ASSERT_TRUE(e.sigma == 3.2);
    arith_scalar_free(ring, &scalar);
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &s);
}

void test_marks_and_disabling(void)
{
    ArithElement x, y, z;
    filled(&x, 1);
    filled(&y, 2);
    arith_new(ring, &z);
    arith_trace_enable();
    arith_trace_mark("region", 1);
    arith_trace_disable();
    TEST_ASSERT_FALSE(arith_trace_is_enabled());
    arith_add(ring, &z, &x, &y);
    TEST_ASSERT_EQUAL_UINT64(1, arith_trace_length());
    ArithTraceEvent e = event(0);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_MARK, e.kind);
    TEST_ASSERT_EQUAL_STRING("region", e.label);
    TEST_ASSERT_EQUAL_INT(1, e.enter);
    // Re-enabled, the tape goes on from where it was.
    arith_trace_enable();
    arith_add(ring, &z, &x, &y);
    TEST_ASSERT_EQUAL_UINT64(4, arith_trace_length());
    TEST_ASSERT_EQUAL_UINT64(4, event(3).sequence);
    arith_free(ring, &x);
    arith_free(ring, &y);
    arith_free(ring, &z);
}

void test_digits_and_approximate_divisions_are_recorded(void)
{
    ArithRing to = arith_rns_ring_new(TEST_N, TEST_MASK >> 2, base);
    ArithElement x, d;
    filled(&x, 1);
    arith_new(ring, &d);
    arith_trace_enable();
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_decompose_digit(ring, &d, &x, 0x2, 8, 1, 0, 1));
    TEST_ASSERT_EQUAL_INT(ARITH_OK, arith_round_division_approximate(ring, &x, to));
    TEST_ASSERT_EQUAL_UINT64(3, arith_trace_length());
    ArithTraceEvent e = event(1);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_DECOMPOSE_DIGIT, e.kind);
    TEST_ASSERT_EQUAL_UINT32(1, e.n_inputs);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(0x2, e.primes);
    TEST_ASSERT_EQUAL_UINT64(8, e.log_base);
    TEST_ASSERT_EQUAL_UINT64(1, e.level);
    TEST_ASSERT_EQUAL_UINT64(0, e.dropped);
    TEST_ASSERT_EQUAL_INT(1, e.centered);
    assert_same(arith_rns_polynomial(&d), (RNS_Polynomial)e.value);
    e = event(2);
    TEST_ASSERT_EQUAL_INT(ARITH_TRACE_ROUND_DIVISION_APPROXIMATE, e.kind);
    TEST_ASSERT_EQUAL_UINT32(1, e.n_inputs);
    TEST_ASSERT_EQUAL_UINT64(1, e.inputs[0]);
    TEST_ASSERT_EQUAL_UINT64(TEST_MASK >> 2, e.mask);
    assert_same(arith_rns_polynomial(&x), (RNS_Polynomial)e.value);
    arith_free(ring, &x);
    arith_free(ring, &d);
    arith_ring_free(to);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_nothing_is_recorded_while_disabled);
    RUN_TEST(test_add_records_its_operands_then_the_sum);
    RUN_TEST(test_a_value_an_event_wrote_is_not_recorded_again);
    RUN_TEST(test_an_accumulation_names_its_previous_version_first);
    RUN_TEST(test_copy_carries_no_payload_and_names_the_source);
    RUN_TEST(test_a_freed_element_is_forgotten);
    RUN_TEST(test_round_division_is_recorded_in_place);
    RUN_TEST(test_scale_by_records_the_scalar_and_samples_are_secret);
    RUN_TEST(test_marks_and_disabling);
    RUN_TEST(test_digits_and_approximate_divisions_are_recorded);
    return UNITY_END();
}
