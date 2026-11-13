// SPDX-FileCopyrightText: 2026 The vFHE Authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_seeded.c
 * @brief The seeded sampler against its definition, read straight off the
 * attempt streams: value i is word i of stream 0 masked, else of stream 1, ...
 */
#include <blake3.h>
#include <seeded.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static const char CONTEXT[] = "vfhe test_seeded 2026-11-04";
static const uint8_t SEED[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

/* Value i of the definition: attempt stream t is BLAKE3 in derive-key mode
 * with the context over seed || LE64(t); the first masked word below the
 * bound wins. */
static uint64_t reference(const char *context, const uint8_t *seed, size_t seed_len, uint64_t i,
                          uint64_t bound)
{
    uint64_t mask = bound - 1;
    for (int s = 1; s < 64; s <<= 1)
        mask |= mask >> s;
    for (uint64_t t = 0;; t++)
    {
        blake3_hasher hasher;
        uint8_t tag[8], word[8];
        for (int k = 0; k < 8; k++)
            tag[k] = (uint8_t)(t >> (8 * k));
        blake3_hasher_init_derive_key(&hasher, context);
        blake3_hasher_update(&hasher, seed, seed_len);
        blake3_hasher_update(&hasher, tag, sizeof tag);
        blake3_hasher_finalize_seek(&hasher, i * 8, word, sizeof word);
        uint64_t v = 0;
        for (int k = 0; k < 8; k++)
            v |= (uint64_t)word[k] << (8 * k);
        if ((v & mask) < bound)
            return v & mask;
    }
}

static void assert_matches_definition(uint64_t bound, uint64_t start, uint64_t count)
{
    uint64_t *out = (uint64_t *)malloc(count * sizeof(uint64_t));
    seeded_below_from(out, count, start, bound, CONTEXT, SEED, sizeof SEED);
    for (uint64_t i = 0; i < count; i++)
    {
        TEST_ASSERT_TRUE(out[i] < bound);
        TEST_ASSERT_EQUAL_UINT64(reference(CONTEXT, SEED, sizeof SEED, start + i, bound), out[i]);
    }
    free(out);
}

/* Bounds that never reject, that reject half, powers of two (whose mask is
 * their own bits, no wider), the edges of 64 bits; windows crossing the
 * stack run (64), the bulk run (4096) and starting unaligned. */
void test_matches_definition(void)
{
    const uint64_t bounds[] = {1,
                               2,
                               3,
                               256,
                               257,
                               (1ULL << 30) - 35,
                               1ULL << 32,
                               (1ULL << 32) + 1,
                               (1ULL << 61) - 1,
                               UINT64_MAX};
    const uint64_t windows[][2] = {{0, 1}, {0, 37}, {3, 64}, {5, 65}, {7, 1029}, {4090, 4200}};
    for (size_t b = 0; b < sizeof bounds / sizeof bounds[0]; b++)
        for (size_t w = 0; w < sizeof windows / sizeof windows[0]; w++)
            assert_matches_definition(bounds[b], windows[w][0], windows[w][1]);
}

/* The plain entry point is the window from 0, and a window is the slice of
 * the longer draw it sits in. */
void test_windows_are_slices_of_the_sequence(void)
{
    const uint64_t bound = 257; /* rejects about half, so positions fall through */
    uint64_t whole[300], part[300];
    seeded_below(whole, 300, bound, CONTEXT, SEED, sizeof SEED);
    seeded_below_from(part, 300, 0, bound, CONTEXT, SEED, sizeof SEED);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(whole, part, 300);
    for (uint64_t start = 0; start < 300; start += 37)
    {
        const uint64_t count = 300 - start < 50 ? 300 - start : 50;
        seeded_below_from(part, count, start, bound, CONTEXT, SEED, sizeof SEED);
        TEST_ASSERT_EQUAL_UINT64_ARRAY(&whole[start], part, count);
    }
}

/* Another context or another seed is an unrelated stream. */
void test_context_and_seed_separate_streams(void)
{
    uint64_t a[64], b[64];
    seeded_below(a, 64, 1ULL << 40, CONTEXT, SEED, sizeof SEED);
    seeded_below(b, 64, 1ULL << 40, "vfhe test_seeded other", SEED, sizeof SEED);
    TEST_ASSERT_TRUE(memcmp(a, b, sizeof a) != 0);
    const uint8_t other[] = {9, 8, 7, 6, 5, 4, 3, 2, 2};
    seeded_below(b, 64, 1ULL << 40, CONTEXT, other, sizeof other);
    TEST_ASSERT_TRUE(memcmp(a, b, sizeof a) != 0);
    seeded_below(b, 64, 1ULL << 40, CONTEXT, SEED, sizeof SEED);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(a, b, 64);
}

/* A zero-length seed is a valid seed, and a count of zero writes nothing. */
void test_edge_arguments(void)
{
    uint64_t out[4] = {1, 2, 3, 4}, again[4];
    seeded_below(out, 4, 1000, CONTEXT, SEED, 0);
    seeded_below(again, 4, 1000, CONTEXT, SEED, 0);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(out, again, 4);
    const uint64_t kept[4] = {1, 2, 3, 4};
    memcpy(out, kept, sizeof out);
    seeded_below(out, 0, 1000, CONTEXT, SEED, sizeof SEED);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(kept, out, 4);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_matches_definition);
    RUN_TEST(test_windows_are_slices_of_the_sequence);
    RUN_TEST(test_context_and_seed_separate_streams);
    RUN_TEST(test_edge_arguments);
    return UNITY_END();
}
