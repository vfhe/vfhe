/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_vec.c
 * @brief The vector helpers: the RNS-tagged zqvec elementwise ops and the
 *        plain integer polynomial (permutation + gadget digit extraction).
 */
#include <stdint.h>

#include "arith.h"
#include "test_arith_support.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_zqvec_ops(void)
{
    enum
    {
        N = 16
    };
    uint64_t primes[2];
    ring_t r = make_ring(N, 1, primes);

    zqvec_t a = zqvec_new(r, N, 2), b = zqvec_new(r, N, 2), out = zqvec_new(r, N, 2);
    for (int l = 0; l < 2; l++)
    {
        for (uint64_t i = 0; i < N; i++)
        {
            a->rows[l][i] = lcg() % primes[l];
            b->rows[l][i] = lcg() % primes[l];
        }
    }

    zqvec_add(out, a, b);
    for (int l = 0; l < 2; l++)
        for (uint64_t i = 0; i < N; i++)
            TEST_ASSERT_EQUAL_UINT64((a->rows[l][i] + b->rows[l][i]) % primes[l], out->rows[l][i]);

    zqvec_sub(out, a, b);
    for (int l = 0; l < 2; l++)
        for (uint64_t i = 0; i < N; i++)
            TEST_ASSERT_EQUAL_UINT64((a->rows[l][i] + primes[l] - b->rows[l][i]) % primes[l],
                                     out->rows[l][i]);

    zqvec_scale(out, a, 12345);
    for (int l = 0; l < 2; l++)
        for (uint64_t i = 0; i < N; i++)
            TEST_ASSERT_EQUAL_UINT64(
                (uint64_t)(((unsigned __int128)a->rows[l][i] * (12345 % primes[l])) % primes[l]),
                out->rows[l][i]);

    zqvec_free(a);
    zqvec_free(b);
    zqvec_free(out);
    ring_free(r);
}

void test_int_poly(void)
{
    int_poly_t p = int_poly_new(8), out = int_poly_new(8);
    for (uint64_t i = 0; i < 8; i++)
        p->coeffs[i] = i + 1;

    /* Cyclic permutation: out[i * 3 mod 8] = in[i]. */
    int_poly_permute(out, p, 3);
    for (uint64_t i = 0; i < 8; i++)
        TEST_ASSERT_EQUAL_UINT64(p->coeffs[i], out->coeffs[(i * 3) % 8]);

    /* Digit extraction matches its definition. */
    const uint64_t Bg = 8, l = 3, bit_size = 32;
    p->coeffs[0] = 0xAABBCCDD;
    const uint64_t offset = 1ULL << (bit_size - l * Bg - 1);
    for (uint64_t i = 0; i < l; i++)
    {
        int_poly_decompose_digit(out, p, Bg, l, bit_size, i);
        const uint64_t h_bit = bit_size - (i + 1) * Bg;
        TEST_ASSERT_EQUAL_UINT64(((p->coeffs[0] + offset) >> h_bit) & 0xFF, out->coeffs[0]);
    }

    int_poly_free(p);
    int_poly_free(out);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_zqvec_ops);
    RUN_TEST(test_int_poly);
    return UNITY_END();
}
