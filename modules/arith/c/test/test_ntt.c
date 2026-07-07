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

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void roundtrip(uint64_t n, uint64_t q)
{
    NTT_proc proc = ntt_new_proc(n, q);
    uint64_t *in = malloc(n * sizeof(uint64_t));
    uint64_t *fwd = malloc(n * sizeof(uint64_t));
    uint64_t *back = malloc(n * sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
        in[i] = (0x123456789ABCDEFULL + i) % q;
    ntt_forward(fwd, in, proc);
    ntt_reverse(back, fwd, proc);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(in, back, n);
    free(in);
    free(fwd);
    free(back);
    ntt_free_proc(proc);
}

void test_ntt_roundtrip_matrix(void)
{
    const uint64_t sizes[] = {64, 256, 1024, 4096};
    const uint64_t bits[] = {20, 30, 40, 50, 60, 62};
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
    const uint64_t bits[] = {20, 40, 60, 62};
    for (unsigned si = 0; si < sizeof(sizes) / sizeof(*sizes); si++)
    {
        for (unsigned bi = 0; bi < sizeof(bits) / sizeof(*bits); bi++)
        {
            const uint64_t n = sizes[si];
            const uint64_t q = next_special_prime(1ULL << bits[bi], n, true);
            NTT_proc proc = ntt_new_proc(n, q);

            uint64_t *a = malloc(n * sizeof(uint64_t));
            uint64_t *b = malloc(n * sizeof(uint64_t));
            uint64_t *ref = calloc(n, sizeof(uint64_t));
            uint64_t *na = malloc(n * sizeof(uint64_t));
            uint64_t *nb = malloc(n * sizeof(uint64_t));
            uint64_t *nr = malloc(n * sizeof(uint64_t));
            uint64_t *res = malloc(n * sizeof(uint64_t));

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

            ntt_forward(na, a, proc);
            ntt_forward(nb, b, proc);
            mod_eltwise_mul(nr, na, nb, n, proc);
            ntt_reverse(res, nr, proc);
            TEST_ASSERT_EQUAL_UINT64_ARRAY(ref, res, n);

            free(a);
            free(b);
            free(ref);
            free(na);
            free(nb);
            free(nr);
            free(res);
            ntt_free_proc(proc);
        }
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_ntt_roundtrip_matrix);
    RUN_TEST(test_ntt_roundtrip_30bit);
    RUN_TEST(test_ntt_negacyclic_convolution);
    return UNITY_END();
}
