// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_interleave.c
 * @brief The k-way deinterleave, interleave and spread kernels checked word for word
 *        against plain loops, over every shape class the vector paths
 *        distinguish: k below, at and above a vector's lanes, k that is not
 *        a power of two, and lengths with a tail.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arith.h>

#include "arith_internal.h"

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

// Words past the end of every output, which no kernel may touch.
#define GUARD 17
#define SENTINEL 0xA5A5A5A5A5A5A5A5ULL

static const uint64_t KS[] = {1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 64};
static const uint64_t NS[] = {1, 7, 8, 15, 16, 24, 32, 40, 64, 256};

static uint64_t next_word(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state ^ (*state >> 29);
}

#define CHECK_KERNELS(W)                                                                           \
    static void check_u##W(uint64_t k, uint64_t n, uint64_t seed)                                  \
    {                                                                                              \
        /* One word of offset: the kernels may not assume aligned rows. */                         \
        uint##W##_t *in_buf = (uint##W##_t *)malloc((k * n + 1) * sizeof(uint##W##_t));            \
        uint##W##_t *in = in_buf + 1;                                                              \
        for (uint64_t i = 0; i < k * n; i++)                                                       \
            in[i] = (uint##W##_t)next_word(&seed);                                                 \
                                                                                                   \
        uint##W##_t **out = (uint##W##_t **)malloc(k * sizeof(uint##W##_t *));                     \
        for (uint64_t j = 0; j < k; j++)                                                           \
        {                                                                                          \
            out[j] = (uint##W##_t *)malloc((n + GUARD) * sizeof(uint##W##_t));                     \
            for (uint64_t m = 0; m < n + GUARD; m++)                                               \
                out[j][m] = (uint##W##_t)SENTINEL;                                                 \
        }                                                                                          \
        vec_deinterleave_u##W(out, in, k, n);                                                      \
        for (uint64_t j = 0; j < k; j++)                                                           \
        {                                                                                          \
            for (uint64_t m = 0; m < n; m++)                                                       \
                TEST_ASSERT_EQUAL_UINT64(in[j + k * m], out[j][m]);                                \
            for (uint64_t m = n; m < n + GUARD; m++)                                               \
                TEST_ASSERT_EQUAL_UINT64((uint##W##_t)SENTINEL, out[j][m]);                        \
            free(out[j]);                                                                          \
        }                                                                                          \
        free(out);                                                                                 \
                                                                                                   \
        /* The interleave, with every third row missing (read as zeros). */                        \
        const uint##W##_t **rows = (const uint##W##_t **)malloc(k * sizeof(uint##W##_t *));        \
        for (uint64_t j = 0; j < k; j++)                                                           \
            rows[j] = j % 3 == 2 ? NULL : in + j * n;                                              \
        uint##W##_t *joined_buf =                                                                  \
            (uint##W##_t *)malloc((k * n + GUARD + 1) * sizeof(uint##W##_t));                      \
        uint##W##_t *joined = joined_buf + 1;                                                      \
        for (uint64_t i = 0; i < k * n + GUARD; i++)                                               \
            joined[i] = (uint##W##_t)SENTINEL;                                                     \
        vec_interleave_k_u##W(joined, rows, k, n);                                                 \
        for (uint64_t i = 0; i < k * n; i++)                                                       \
            TEST_ASSERT_EQUAL_UINT64(rows[i % k] != NULL ? rows[i % k][i / k] : 0, joined[i]);     \
        for (uint64_t i = k * n; i < k * n + GUARD; i++)                                           \
            TEST_ASSERT_EQUAL_UINT64((uint##W##_t)SENTINEL, joined[i]);                            \
        free(joined_buf);                                                                          \
        free(rows);                                                                                \
                                                                                                   \
        uint##W##_t *spread_buf =                                                                  \
            (uint##W##_t *)malloc((k * n + GUARD + 1) * sizeof(uint##W##_t));                      \
        uint##W##_t *spread = spread_buf + 1;                                                      \
        for (uint64_t i = 0; i < k * n + GUARD; i++)                                               \
            spread[i] = (uint##W##_t)SENTINEL;                                                     \
        vec_spread_u##W(spread, in, k, n);                                                         \
        for (uint64_t i = 0; i < k * n; i++)                                                       \
            TEST_ASSERT_EQUAL_UINT64(i % k == 0 ? in[i / k] : 0, spread[i]);                       \
        for (uint64_t i = k * n; i < k * n + GUARD; i++)                                           \
            TEST_ASSERT_EQUAL_UINT64((uint##W##_t)SENTINEL, spread[i]);                            \
        free(spread_buf);                                                                          \
        free(in_buf);                                                                              \
    }

CHECK_KERNELS(64)
CHECK_KERNELS(32)

static void test_k_way_kernels_u64(void)
{
    for (size_t a = 0; a < sizeof(KS) / sizeof(KS[0]); a++)
        for (size_t b = 0; b < sizeof(NS) / sizeof(NS[0]); b++)
            check_u64(KS[a], NS[b], 1000 * a + b);
}

static void test_k_way_kernels_u32(void)
{
    for (size_t a = 0; a < sizeof(KS) / sizeof(KS[0]); a++)
        for (size_t b = 0; b < sizeof(NS) / sizeof(NS[0]); b++)
            check_u32(KS[a], NS[b], 1000 * a + b);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_k_way_kernels_u64);
    RUN_TEST(test_k_way_kernels_u32);
    return UNITY_END();
}
