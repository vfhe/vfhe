// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// A binary Merkle tree over BLAKE3 digests. The caller hands merkle_commit the
// leaf digests, so the tree knows nothing of what a leaf is; this file is the
// digest arithmetic: the levels built bottom-up, the root, the siblings of
// one path, and a path replayed. merkle.h gives the level layout and the
// zero padding of leaf counts that are not a power of two.
#include "merkle.h"

#include <stdlib.h>
#include <string.h>

#include <blake3.h>

#include "alloc.h"

// The digest of left | right.
static void hash_pair(uint8_t *out, const uint8_t *left, const uint8_t *right)
{
    uint8_t pair[2 * BLAKE3_OUT_LEN];
    memcpy(pair, left, BLAKE3_OUT_LEN);
    memcpy(&pair[BLAKE3_OUT_LEN], right, BLAKE3_OUT_LEN);
    hash_blake3(out, pair, sizeof(pair));
}

void merkle_join_digests(uint8_t *out, const uint8_t *const *parts, uint64_t part_count,
                         uint64_t count)
{
    for (uint64_t i = 0; i < count; i++)
    {
        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        for (uint64_t p = 0; p < part_count; p++)
            blake3_hasher_update(&hasher, parts[p] + i * BLAKE3_OUT_LEN, BLAKE3_OUT_LEN);
        blake3_hasher_finalize(&hasher, out + i * BLAKE3_OUT_LEN, BLAKE3_OUT_LEN);
    }
}

Merkle merkle_new(uint64_t size)
{
    if (size < 1)
        return NULL;
    Merkle mk = (Merkle)safe_malloc(sizeof(*mk));
    mk->size = size;
    mk->log_size = 0; // ceil(log2(size))
    while ((1ULL << mk->log_size) < size)
        mk->log_size++;
    mk->levels = (uint8_t **)safe_malloc((mk->log_size + 1) * sizeof(uint8_t *));
    for (uint64_t i = 0; i <= mk->log_size; i++)
        mk->levels[i] = (uint8_t *)safe_aligned_malloc((1ULL << i) * BLAKE3_OUT_LEN);
    return mk;
}

void merkle_free(Merkle mk)
{
    if (mk == NULL)
        return;
    for (uint64_t i = 0; i <= mk->log_size; i++)
        free(mk->levels[i]);
    free(mk->levels);
    free(mk);
}

void merkle_commit(Merkle mk, const uint8_t *leaf_digests)
{
    uint8_t *leaves = mk->levels[mk->log_size];
    const uint64_t padded = 1ULL << mk->log_size;
    memcpy(leaves, leaf_digests, mk->size * BLAKE3_OUT_LEN);
    memset(&leaves[mk->size * BLAKE3_OUT_LEN], 0, (padded - mk->size) * BLAKE3_OUT_LEN);
    // Node j's two children are adjacent in the level below, so that level
    // is the batch: one 64-byte input per node, filling the SIMD lanes, an
    // order of magnitude over a hasher per node and most of what a large
    // tree costs.
    for (int64_t i = (int64_t)mk->log_size - 1; i >= 0; i--)
        hash_blake3_batch(mk->levels[i], mk->levels[i + 1], 1ULL << i, 2 * BLAKE3_OUT_LEN);
}

void merkle_get_root(uint8_t *out, Merkle mk) { memcpy(out, mk->levels[0], BLAKE3_OUT_LEN); }

void merkle_open(uint8_t *out, Merkle mk, uint64_t index)
{
    for (uint64_t i = mk->log_size; i > 0; i--)
    {
        memcpy(&out[(mk->log_size - i) * BLAKE3_OUT_LEN],
               &mk->levels[i][(index ^ 1) * BLAKE3_OUT_LEN], BLAKE3_OUT_LEN);
        index >>= 1;
    }
}

bool merkle_verify(const uint8_t *root, uint64_t index, const uint8_t *path, uint64_t path_len,
                   const uint8_t *leaf_digest)
{
    // A path of `path_len` siblings addresses 2^path_len leaves: an index
    // beyond them is no position of the tree, whatever its low bits replay.
    if (path_len < 64 && (index >> path_len) != 0)
        return false;
    uint8_t node[BLAKE3_OUT_LEN], next[BLAKE3_OUT_LEN];
    memcpy(node, leaf_digest, BLAKE3_OUT_LEN);
    for (uint64_t i = 0; i < path_len; i++)
    {
        const uint8_t *sibling = &path[i * BLAKE3_OUT_LEN];
        if (index & 1)
            hash_pair(next, sibling, node);
        else
            hash_pair(next, node, sibling);
        memcpy(node, next, BLAKE3_OUT_LEN);
        index >>= 1;
    }
    return memcmp(node, root, BLAKE3_OUT_LEN) == 0;
}
