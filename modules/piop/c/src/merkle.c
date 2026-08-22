// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "merkle.h"

#include <blake3.h>

#include "misc.h"

// -------------------------------------------------------------
// Binary Merkle tree over BLAKE3 digests
// -------------------------------------------------------------
// Leaf hashing happens outside this file: the caller hands `merkle_commit` the
// leaf digests, so the tree is oblivious to what a leaf is. Everything here is
// the digest arithmetic - build the levels bottom-up, read a root, copy the
// siblings of one path, and replay a path. See merkle.h for the level layout
// and the zero padding of non-power-of-two leaf counts.

static void merkle_hash_pair(uint8_t *out, const uint8_t *left, const uint8_t *right)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, left, MERKLE_DIGEST_LEN);
    blake3_hasher_update(&hasher, right, MERKLE_DIGEST_LEN);
    blake3_hasher_finalize(&hasher, out, MERKLE_DIGEST_LEN);
}

void merkle_hash(uint8_t *out, const uint8_t *in, uint64_t len)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, in, len);
    blake3_hasher_finalize(&hasher, out, MERKLE_DIGEST_LEN);
}

Merkle merkle_new(uint64_t size)
{
    if (size < 1)
        return NULL;
    Merkle mk = (Merkle)safe_malloc(sizeof(*mk));
    mk->size = size;
    // log_size = ceil(log2(size))
    mk->log_size = 0;
    while ((1ULL << mk->log_size) < size)
        mk->log_size++;
    mk->levels = (uint8_t **)safe_malloc((mk->log_size + 1) * sizeof(uint8_t *));
    for (uint64_t i = 0; i <= mk->log_size; i++)
    {
        mk->levels[i] = (uint8_t *)safe_aligned_malloc((1ULL << i) * MERKLE_DIGEST_LEN);
    }
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
    memcpy(leaves, leaf_digests, mk->size * MERKLE_DIGEST_LEN);
    memset(&leaves[mk->size * MERKLE_DIGEST_LEN], 0, (padded - mk->size) * MERKLE_DIGEST_LEN);

    for (int64_t i = (int64_t)mk->log_size - 1; i >= 0; i--)
    {
        const uint8_t *below = mk->levels[i + 1];
        uint8_t *level = mk->levels[i];
        for (uint64_t j = 0; j < (1ULL << i); j++)
        {
            merkle_hash_pair(&level[j * MERKLE_DIGEST_LEN], &below[(2 * j) * MERKLE_DIGEST_LEN],
                             &below[(2 * j + 1) * MERKLE_DIGEST_LEN]);
        }
    }
}

void merkle_get_root(uint8_t *out, Merkle mk) { memcpy(out, mk->levels[0], MERKLE_DIGEST_LEN); }

void merkle_open(uint8_t *out, Merkle mk, uint64_t index)
{
    for (uint64_t i = mk->log_size; i > 0; i--)
    {
        memcpy(&out[(mk->log_size - i) * MERKLE_DIGEST_LEN],
               &mk->levels[i][(index ^ 1) * MERKLE_DIGEST_LEN], MERKLE_DIGEST_LEN);
        index >>= 1;
    }
}

bool merkle_verify(const uint8_t *root, uint64_t index, const uint8_t *path, uint64_t path_len,
                   const uint8_t *leaf_digest)
{
    uint8_t node[MERKLE_DIGEST_LEN], next[MERKLE_DIGEST_LEN];
    memcpy(node, leaf_digest, MERKLE_DIGEST_LEN);
    for (uint64_t i = 0; i < path_len; i++)
    {
        const uint8_t *sibling = &path[i * MERKLE_DIGEST_LEN];
        if (index & 1)
            merkle_hash_pair(next, sibling, node);
        else
            merkle_hash_pair(next, node, sibling);
        memcpy(node, next, MERKLE_DIGEST_LEN);
        index >>= 1;
    }
    return memcmp(node, root, MERKLE_DIGEST_LEN) == 0;
}
