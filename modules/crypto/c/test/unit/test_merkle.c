// SPDX-FileCopyrightText: 2026 The vFHE Authors
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_merkle.c
 * @brief The tree against roots computed by hand, openings of every leaf, and
 * the rejections a verifier relies on.
 */
#include <hash.h>
#include <merkle.h>
#include <stdlib.h>
#include <string.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void leaf(uint8_t out[32], uint64_t i)
{
    uint8_t in[8];
    for (int k = 0; k < 8; k++)
        in[k] = (uint8_t)(i >> (8 * k));
    hash_blake3(out, in, sizeof in);
}

static void pair(uint8_t out[32], const uint8_t *left, const uint8_t *right)
{
    uint8_t both[64];
    memcpy(both, left, 32);
    memcpy(&both[32], right, 32);
    hash_blake3(out, both, sizeof both);
}

/* The root of `size` leaves, by the definition in merkle.h: zero digests pad
 * to a power of two, an internal node is the digest of its children. */
static void reference_root(uint8_t out[32], const uint8_t *leaves, uint64_t size)
{
    uint64_t padded = 1;
    while (padded < size)
        padded <<= 1;
    uint8_t *level = (uint8_t *)calloc(padded, 32);
    memcpy(level, leaves, size * 32);
    for (uint64_t n = padded; n > 1; n /= 2)
        for (uint64_t j = 0; j < n / 2; j++)
            pair(&level[j * 32], &level[2 * j * 32], &level[(2 * j + 1) * 32]);
    memcpy(out, level, 32);
    free(level);
}

void test_one_leaf_is_its_own_root(void)
{
    uint8_t digest[32], root[32];
    leaf(digest, 42);
    Merkle mk = merkle_new(1);
    merkle_commit(mk, digest);
    merkle_get_root(root, mk);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(digest, root, 32);
    uint8_t path[32];
    merkle_open(path, mk, 0); /* a height-0 path: nothing written */
    TEST_ASSERT_TRUE(merkle_verify(root, 0, path, 0, digest));
    merkle_free(mk);
}

/* Every size up to past a power of two, every leaf opened and verified, and
 * the root matched to the hand-built one. */
void test_roots_and_openings_for_every_small_size(void)
{
    for (uint64_t size = 1; size <= 18; size++)
    {
        uint8_t *leaves = (uint8_t *)malloc(size * 32);
        for (uint64_t i = 0; i < size; i++)
            leaf(&leaves[i * 32], i);
        Merkle mk = merkle_new(size);
        merkle_commit(mk, leaves);
        uint8_t root[32], expected[32];
        merkle_get_root(root, mk);
        reference_root(expected, leaves, size);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, root, 32);
        for (uint64_t i = 0; i < size; i++)
        {
            uint8_t path[64 * 32];
            merkle_open(path, mk, i);
            TEST_ASSERT_TRUE(merkle_verify(root, i, path, mk->log_size, &leaves[i * 32]));
            if (size > 1)
                TEST_ASSERT_FALSE(
                    merkle_verify(root, (i + 1) % size, path, mk->log_size, &leaves[i * 32]));
        }
        merkle_free(mk);
        free(leaves);
    }
}

void test_tampering_is_rejected(void)
{
    uint8_t leaves[16 * 32], root[32], path[4 * 32];
    for (uint64_t i = 0; i < 16; i++)
        leaf(&leaves[i * 32], i);
    Merkle mk = merkle_new(16);
    merkle_commit(mk, leaves);
    merkle_get_root(root, mk);
    merkle_open(path, mk, 5);
    TEST_ASSERT_TRUE(merkle_verify(root, 5, path, 4, &leaves[5 * 32]));
    TEST_ASSERT_FALSE(merkle_verify(root, 5, path, 4, &leaves[6 * 32])); /* leaf */
    TEST_ASSERT_FALSE(merkle_verify(root, 6, path, 4, &leaves[5 * 32])); /* index */
    TEST_ASSERT_FALSE(merkle_verify(root, 5, path, 3, &leaves[5 * 32])); /* short */
    path[0] ^= 1;
    TEST_ASSERT_FALSE(merkle_verify(root, 5, path, 4, &leaves[5 * 32])); /* sibling */
    path[0] ^= 1;
    root[31] ^= 1;
    TEST_ASSERT_FALSE(merkle_verify(root, 5, path, 4, &leaves[5 * 32])); /* root */
    merkle_free(mk);
}

/* The tree is reused: a second commit over other leaves gives their root. */
void test_recommit_reuses_the_tree(void)
{
    uint8_t a[8 * 32], b[8 * 32], root_a[32], root_b[32], root[32];
    for (uint64_t i = 0; i < 8; i++)
    {
        leaf(&a[i * 32], i);
        leaf(&b[i * 32], 100 + i);
    }
    Merkle mk = merkle_new(8);
    merkle_commit(mk, a);
    merkle_get_root(root_a, mk);
    merkle_commit(mk, b);
    merkle_get_root(root_b, mk);
    reference_root(root, b, 8);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(root, root_b, 32);
    TEST_ASSERT_TRUE(memcmp(root_a, root_b, 32) != 0);
    merkle_commit(mk, a);
    merkle_get_root(root, mk);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(root_a, root, 32);
    merkle_free(mk);
}

/* Joined digest i is the digest of the concatenation of digest i of each part. */
void test_join_digests(void)
{
    uint8_t x[3 * 32], y[3 * 32], joined[3 * 32], expected[32], both[64];
    for (uint64_t i = 0; i < 3; i++)
    {
        leaf(&x[i * 32], i);
        leaf(&y[i * 32], 10 + i);
    }
    const uint8_t *parts[2] = {x, y};
    merkle_join_digests(joined, parts, 2, 3);
    for (uint64_t i = 0; i < 3; i++)
    {
        memcpy(both, &x[i * 32], 32);
        memcpy(&both[32], &y[i * 32], 32);
        hash_blake3(expected, both, 64);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, &joined[i * 32], 32);
    }
    const uint8_t *one[1] = {x};
    merkle_join_digests(joined, one, 1, 3);
    for (uint64_t i = 0; i < 3; i++)
    {
        hash_blake3(expected, &x[i * 32], 32);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, &joined[i * 32], 32);
    }
}

void test_empty_and_null(void)
{
    TEST_ASSERT_NULL(merkle_new(0));
    merkle_free(NULL);
    TEST_PASS();
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_one_leaf_is_its_own_root);
    RUN_TEST(test_roots_and_openings_for_every_small_size);
    RUN_TEST(test_tampering_is_rejected);
    RUN_TEST(test_recommit_reuses_the_tree);
    RUN_TEST(test_join_digests);
    RUN_TEST(test_empty_and_null);
    return UNITY_END();
}
