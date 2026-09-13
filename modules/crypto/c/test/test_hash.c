// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_hash.c
 * @brief hash_batch against the one-at-a-time digest it has to reproduce.
 */
#include <blake3.h>
#include <crypto.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The digest hash_batch has to agree with: one input, one hasher. */
static void digest_alone(uint8_t *out, const uint8_t *in, uint64_t len)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, in, len);
    blake3_hasher_finalize(&hasher, out, 32);
}

static void assert_batch_matches(uint64_t count, uint64_t len)
{
    uint8_t *in = (uint8_t *)malloc(count * len);
    uint8_t *batched = (uint8_t *)malloc(count * 32);
    uint8_t alone[32];
    for (uint64_t i = 0; i < count * len; i++)
        in[i] = (uint8_t)(i * 131u + len);

    hash_batch(batched, in, count, len);
    for (uint64_t i = 0; i < count; i++)
    {
        digest_alone(alone, &in[i * len], len);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(alone, &batched[i * 32], 32);
    }
    free(in);
    free(batched);
}

/* Counts around the lane widths, which are where a batched core goes wrong:
 * 16 and 8 are the AVX-512 and AVX2 widths, and the odd counts are the runs
 * that have to fall through them to singles. */
void test_batch_matches_one_at_a_time(void)
{
    const uint64_t counts[] = {1, 2, 7, 8, 9, 15, 16, 17, 33, 257};
    for (uint64_t c = 0; c < 10; c++)
    {
        assert_batch_matches(counts[c], 64);   /* one block: a Merkle node */
        assert_batch_matches(counts[c], 128);  /* two */
        assert_batch_matches(counts[c], 1024); /* a whole chunk, the limit */
    }
}

/* A length the lanes cannot take is still hashed, just one at a time -- the
 * caller is not asked to branch, so this must be the same digest too. */
void test_lengths_outside_the_batch_still_hash(void)
{
    TEST_ASSERT_FALSE(hash_batch_fits(0));
    TEST_ASSERT_FALSE(hash_batch_fits(8));    /* not a whole block */
    TEST_ASSERT_FALSE(hash_batch_fits(96));   /* a block and a half */
    TEST_ASSERT_FALSE(hash_batch_fits(1088)); /* past one chunk */
    TEST_ASSERT_TRUE(hash_batch_fits(64));
    TEST_ASSERT_TRUE(hash_batch_fits(1024));

    assert_batch_matches(9, 8);
    assert_batch_matches(9, 96);
    assert_batch_matches(9, 1088);
}

/* Nothing is written when there is nothing to hash, on either path. */
void test_an_empty_batch_writes_nothing(void)
{
    uint8_t out[32], in[64];
    memset(out, 0xAB, sizeof out);
    memset(in, 0, sizeof in);
    hash_batch(out, in, 0, 64);
    hash_batch(out, in, 0, 8);
    for (uint64_t i = 0; i < 32; i++)
        TEST_ASSERT_EQUAL_UINT8(0xAB, out[i]);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_batch_matches_one_at_a_time);
    RUN_TEST(test_lengths_outside_the_batch_still_hash);
    RUN_TEST(test_an_empty_batch_writes_nothing);
    return UNITY_END();
}
