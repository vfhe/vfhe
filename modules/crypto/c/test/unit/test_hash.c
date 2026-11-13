// SPDX-FileCopyrightText: 2026 The vfhe authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_hash.c
 * @brief hash_blake3_batch and hash_blake3_chain against the one-at-a-time digests they
 *        have to reproduce.
 */
#include <blake3.h>
#include <hash.h>
#include <merkle.h>
#include <parallel.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The digest hash_blake3_batch has to agree with: one input, one hasher. */
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

    hash_blake3_batch(batched, in, count, len);
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
    TEST_ASSERT_FALSE(hash_blake3_batch_fits(0));
    TEST_ASSERT_FALSE(hash_blake3_batch_fits(1025)); /* past one chunk */
    TEST_ASSERT_TRUE(hash_blake3_batch_fits(1));
    TEST_ASSERT_TRUE(hash_blake3_batch_fits(96)); /* a block and a half */
    TEST_ASSERT_TRUE(hash_blake3_batch_fits(1024));

    assert_batch_matches(9, 1025);
    assert_batch_matches(9, 1088);
}

/* Nothing is written when there is nothing to hash, on either path. */
void test_an_empty_batch_writes_nothing(void)
{
    uint8_t out[32], in[64];
    memset(out, 0xAB, sizeof out);
    memset(in, 0, sizeof in);
    hash_blake3_batch(out, in, 0, 64);
    hash_blake3_batch(out, in, 0, 8);
    for (uint64_t i = 0; i < 32; i++)
        TEST_ASSERT_EQUAL_UINT8(0xAB, out[i]);
}

/* hash_blake3_chain against its definition, one link at a time: state = H(state |
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

        hash_blake3_chain(state, in, count, len);
        for (uint64_t i = 0; i < count; i++)
        {
            memcpy(link, expected, 32);
            memcpy(&link[32], &in[i * len], len);
            digest_alone(expected, link, 32 + len);
        }
        TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, state, 32);
    }
}

/* BLAKE3's published digests of the empty string and of "abc". */
void test_known_vectors(void)
{
    const uint8_t empty[32] = {0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6, 0xa0, 0x40, 0x4d,
                               0xea, 0x36, 0xdc, 0xc9, 0x49, 0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1,
                               0x12, 0xb7, 0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62};
    const uint8_t abc[32] = {0x64, 0x37, 0xb3, 0xac, 0x38, 0x46, 0x51, 0x33, 0xff, 0xb6, 0x3b,
                             0x75, 0x27, 0x3a, 0x8d, 0xb5, 0x48, 0xc5, 0x58, 0x46, 0x5d, 0x79,
                             0xdb, 0x03, 0xfd, 0x35, 0x9c, 0x6c, 0xd5, 0xbd, 0x9d, 0x85};
    uint8_t out[32];
    hash_blake3(out, (const uint8_t *)"", 0);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(empty, out, 32);
    hash_blake3(out, (const uint8_t *)"abc", 3);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(abc, out, 32);
}

/* The stream fed in pieces is the one-shot digest, its digest can be read
 * mid-way and the stream carries on, and its state fits what it says. */
void test_stream_matches_the_one_shot_digest(void)
{
    const uint64_t lengths[] = {0, 1, 63, 64, 65, 1024, 1025, 5000};
    uint8_t in[5000], alone[32], streamed[32];
    for (uint64_t i = 0; i < sizeof in; i++)
        in[i] = (uint8_t)(i * 7 + 1);
    TEST_ASSERT_TRUE(hash_blake3_stream_size() >= sizeof(blake3_hasher));
    uint8_t *state = (uint8_t *)malloc(hash_blake3_stream_size());
    for (size_t l = 0; l < sizeof lengths / sizeof lengths[0]; l++)
    {
        const uint64_t len = lengths[l];
        digest_alone(alone, in, len);
        hash_blake3_stream_init(state);
        for (uint64_t at = 0; at < len;)
        {
            const uint64_t piece = len - at < 37 ? len - at : 37;
            hash_blake3_stream_update(state, &in[at], piece);
            at += piece;
            if (at == 37) /* mid-way: the digest so far, then carry on */
            {
                hash_blake3_stream_digest(state, streamed);
                digest_alone(alone, in, 37);
                TEST_ASSERT_EQUAL_UINT8_ARRAY(alone, streamed, 32);
                digest_alone(alone, in, len);
            }
        }
        hash_blake3_stream_digest(state, streamed);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(alone, streamed, 32);
        hash_blake3(streamed, in, len);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(alone, streamed, 32);
    }
    free(state);
}

/* A batch longer than one run of pointers into the core. */
void test_batch_past_one_pointer_run(void) { assert_batch_matches(1000, 64); }

/* The chain over no parts leaves the state alone; a run of parts equals the
 * parts absorbed one at a time. */
void test_chain_of_nothing_and_of_runs(void)
{
    uint8_t state[32], same[32], in[5 * 20];
    for (uint64_t i = 0; i < sizeof in; i++)
        in[i] = (uint8_t)(3 * i);
    for (int i = 0; i < 32; i++)
        state[i] = same[i] = (uint8_t)i;
    hash_blake3_chain(state, in, 0, 20);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(same, state, 32);
    hash_blake3_chain(state, in, 5, 20);
    for (int i = 0; i < 5; i++)
        hash_blake3_chain(same, &in[i * 20], 1, 20);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(same, state, 32);
}

/* The thread limits a batch is checked at: the default, and one that splits
 * a batch above the chunk grain into chunks that do not divide it. */
static const uint64_t LIMITS[] = {1, 3, 4};

/* hash_blake3_batch over batches of 0.5 to 1.1 MiB, enough to run in chunks
 * once the limit is raised, on each of its paths: whole blocks, a partial last
 * block, and inputs past one chunk. Every digest is the one-at-a-time digest
 * at every limit. */
void test_large_batches_match_at_any_thread_limit(void)
{
    const uint64_t counts[] = {12365, 9001, 777};
    const uint64_t lengths[] = {64, 66, 1500};
    for (uint64_t l = 0; l < 3; l++)
        for (uint64_t t = 0; t < 3; t++)
        {
            vfhe_set_num_threads(LIMITS[t]);
            assert_batch_matches(counts[l], lengths[l]);
        }
    vfhe_set_num_threads(0);
}

/* merkle_join_digests against its definition, digest by digest, over enough
 * digests to run in chunks once the limit is raised. */
void test_large_joins_match_at_any_thread_limit(void)
{
    const uint64_t count = 7001, part_count = 3;
    uint8_t *parts_buffer = (uint8_t *)malloc(part_count * count * 32);
    uint8_t *joined = (uint8_t *)malloc(count * 32);
    const uint8_t *parts[3];
    for (uint64_t i = 0; i < part_count * count * 32; i++)
        parts_buffer[i] = (uint8_t)(i * 167u + 5);
    for (uint64_t p = 0; p < part_count; p++)
        parts[p] = &parts_buffer[p * count * 32];

    for (uint64_t t = 0; t < 3; t++)
    {
        vfhe_set_num_threads(LIMITS[t]);
        memset(joined, 0, count * 32);
        merkle_join_digests(joined, parts, part_count, count);
        for (uint64_t i = 0; i < count; i++)
        {
            uint8_t concatenated[3 * 32], expected[32];
            for (uint64_t p = 0; p < part_count; p++)
                memcpy(&concatenated[p * 32], &parts[p][i * 32], 32);
            digest_alone(expected, concatenated, sizeof concatenated);
            TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, &joined[i * 32], 32);
        }
    }
    vfhe_set_num_threads(0);
    free(joined);
    free(parts_buffer);
}

/* A tree whose lower levels are hashed in chunks once the limit is raised:
 * every level, not only the root, is the same at every limit. */
void test_a_large_tree_is_the_same_at_any_thread_limit(void)
{
    const uint64_t size = (1u << 15) + 3;
    uint8_t *leaves = (uint8_t *)malloc(size * BLAKE3_OUT_LEN);
    for (uint64_t i = 0; i < size * BLAKE3_OUT_LEN; i++)
        leaves[i] = (uint8_t)(i * 97u + 1);

    Merkle reference = merkle_new(size);
    vfhe_set_num_threads(1);
    merkle_commit(reference, leaves);
    for (uint64_t t = 1; t < 3; t++)
    {
        Merkle tree = merkle_new(size);
        vfhe_set_num_threads(LIMITS[t]);
        merkle_commit(tree, leaves);
        for (uint64_t i = 0; i <= tree->log_size; i++)
            TEST_ASSERT_EQUAL_UINT8_ARRAY(reference->levels[i], tree->levels[i],
                                          (1ULL << i) * BLAKE3_OUT_LEN);
        merkle_free(tree);
    }
    vfhe_set_num_threads(0);

    /* The root is the tree over the leaves, built pair by pair. */
    uint8_t root[BLAKE3_OUT_LEN], path[16 * BLAKE3_OUT_LEN];
    merkle_get_root(root, reference);
    merkle_open(path, reference, size - 1);
    TEST_ASSERT_TRUE(merkle_verify(root, size - 1, path, reference->log_size,
                                   &leaves[(size - 1) * BLAKE3_OUT_LEN]));
    merkle_free(reference);
    free(leaves);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_known_vectors);
    RUN_TEST(test_stream_matches_the_one_shot_digest);
    RUN_TEST(test_batch_past_one_pointer_run);
    RUN_TEST(test_chain_of_nothing_and_of_runs);
    RUN_TEST(test_batch_matches_one_at_a_time);
    RUN_TEST(test_partial_last_blocks_match_one_at_a_time);
    RUN_TEST(test_lengths_outside_the_batch_still_hash);
    RUN_TEST(test_an_empty_batch_writes_nothing);
    RUN_TEST(test_chain_matches_link_by_link);
    RUN_TEST(test_large_batches_match_at_any_thread_limit);
    RUN_TEST(test_large_joins_match_at_any_thread_limit);
    RUN_TEST(test_a_large_tree_is_the_same_at_any_thread_limit);
    return UNITY_END();
}
