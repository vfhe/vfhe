// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <arith.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Binary Merkle tree over BLAKE3 digests. Defined in piop/c/src/merkle.c.
    //
    // Every node is a MERKLE_DIGEST_LEN-byte digest: a leaf's digest is
    // supplied by the caller (the Python layer hashes leaf objects, which may
    // be anything), an internal node is BLAKE3(left || right). Levels are
    // numbered from the root: level 0 holds the root, level `log_size` the
    // (padded) leaves, so level i holds 2^i digests. A leaf count that is not
    // a power of two is padded with zero digests up to 1 << log_size, which
    // makes the root well defined for any size (the leaf count is public
    // protocol data, so the padding is not a domain-separation concern).
    //
    // Digest buffers are plain byte arrays, never aliased between input and
    // output.

#define MERKLE_DIGEST_LEN 32

    typedef struct _Merkle
    {
        uint8_t **levels; // levels[i] = 2^i digests; levels[log_size] = leaves
        uint64_t size;    // leaves committed to (<= 1 << log_size)
        uint64_t log_size;
    } *Merkle;

    // The BLAKE3 digest of an arbitrary byte string - the hash the tree is
    // built from, exposed so callers can hash leaves with the same function.
    void merkle_hash(uint8_t *out, const uint8_t *in, uint64_t len);

    // A tree for `size` leaves (>= 1); allocates all levels once, so the same
    // object can be committed to repeatedly.
    Merkle merkle_new(uint64_t size);
    void merkle_free(Merkle mk);

    // Fill the leaf level with `size` digests (MERKLE_DIGEST_LEN bytes each,
    // contiguous) and rebuild every internal level.
    void merkle_commit(Merkle mk, const uint8_t *leaf_digests);
    void merkle_get_root(uint8_t *out, Merkle mk);

    // The opening of leaf `index` (< 1 << log_size): the `log_size` sibling
    // digests from the leaf's level upwards, so out[i] is the sibling of the
    // node on the path at height i. `out` holds log_size * MERKLE_DIGEST_LEN
    // bytes.
    void merkle_open(uint8_t *out, Merkle mk, uint64_t index);

    // Recompute the root from a leaf digest and its opening, and compare:
    // `path_len` sibling digests, `index` the leaf position they belong to.
    bool merkle_verify(const uint8_t *root, uint64_t index, const uint8_t *path,
                       uint64_t path_len, const uint8_t *leaf_digest);

#ifdef __cplusplus
}
#endif
