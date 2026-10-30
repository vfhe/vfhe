// SPDX-FileCopyrightText: 2026 The vFHE Authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_hash.c
 * @brief hash_batch and hash_chain against the one-at-a-time digests they
 *        have to reproduce.
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

/* Lengths that end in a partial block: the last block is compressed apart
 * from the whole ones, and a length under one block is that block alone.
 * 32, 40, 52 and 66 are pair leaves of F_(p^2) and of 156-, 208- and 260-bit
 * pseudo-Mersenne fields. */
void test_partial_last_blocks_match_one_at_a_time(void)
{
    const uint64_t counts[] = {1, 7, 16, 17, 300};
    const uint64_t lengths[] = {1, 8, 32, 40, 52, 63, 65, 66, 96, 127, 1000, 1023};
    for (uint64_t c = 0; c < 5; c++)
        for (uint64_t l = 0; l < 12; l++)
            assert_batch_matches(counts[c], lengths[l]);
}

/* A length past one chunk is still hashed, just one at a time -- the caller
 * is not asked to branch, so this must be the same digest too. */
void test_lengths_outside_the_batch_still_hash(void)
{
    TEST_ASSERT_FALSE(hash_batch_fits(0));
    TEST_ASSERT_FALSE(hash_batch_fits(1025)); /* past one chunk */
    TEST_ASSERT_TRUE(hash_batch_fits(1));
    TEST_ASSERT_TRUE(hash_batch_fits(96));
    TEST_ASSERT_TRUE(hash_batch_fits(1024));

    assert_batch_matches(9, 1025);
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

/* hash_chain against its definition, one link at a time: state = H(state |
 * part). Lengths up to 32 take the single-block path, the others a hasher. */
void test_chain_matches_link_by_link(void)
{
    const uint64_t lengths[] = {0, 1, 31, 32, 33, 64, 100};
    for (uint64_t l = 0; l < 7; l++)
    {
        uint64_t len = lengths[l], count = 17;
        uint8_t in[17 * 100], state[32], expected[32], link[32 + 100];
        for (uint64_t i = 0; i < count * len; i++)
            in[i] = (uint8_t)(i * 131u + len);
        for (uint64_t i = 0; i < 32; i++)
            state[i] = expected[i] = (uint8_t)(i + 7);

        hash_chain(state, in, count, len);
        for (uint64_t i = 0; i < count; i++)
        {
            memcpy(link, expected, 32);
            memcpy(&link[32], &in[i * len], len);
            digest_alone(expected, link, 32 + len);
        }
        TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, state, 32);
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_batch_matches_one_at_a_time);
    RUN_TEST(test_partial_last_blocks_match_one_at_a_time);
    RUN_TEST(test_lengths_outside_the_batch_still_hash);
    RUN_TEST(test_an_empty_batch_writes_nothing);
    RUN_TEST(test_chain_matches_link_by_link);
    return UNITY_END();
}
