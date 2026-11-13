// SPDX-FileCopyrightText: 2026 The vfhe authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file merkle.h
 * @brief A binary Merkle tree over BLAKE3 digests.
 *
 * The caller supplies the leaf digests; an internal node is
 * `BLAKE3(left || right)`. Levels are numbered from the root: level 0 holds
 * the root, level `log_size` the leaves, so level `i` holds `2^i` digests. A
 * leaf count that is not a power of two is padded with zero digests, which
 * makes the root well defined for any size; the leaf count is public
 * protocol data, so the padding needs no domain separation. Digest buffers
 * never alias between input and output.
 *
 * @note Thread-safe between trees; one tree belongs to one thread at a time.
 */
#ifndef VFHE_MERKLE_H
#define VFHE_MERKLE_H

#include <stdbool.h>
#include <stdint.h>

#include "hash.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief A tree over `size` leaves. */
    typedef struct _Merkle
    {
        uint8_t **levels;  /**< `levels[i]` holds the `2^i` digests of level `i`. */
        uint64_t size;     /**< Leaves committed to, at most `1 << log_size`. */
        uint64_t log_size; /**< Height: the leaves sit at this level. */
    } *Merkle;

    /**
     * @brief Writes joined leaf digests: digest `i` of @p out is the digest
     *        of digest `i` of every array in @p parts, concatenated in order.
     *
     * A tree over @p out commits to every array's leaves at once, one leaf
     * per index. A large join runs on the library's threads, with the same
     * result at any number of them.
     *
     * @param[out] out        Receives @p count digests.
     * @param[in]  parts      @p part_count arrays of @p count packed digests.
     * @param[in]  part_count Arrays.
     * @param[in]  count      Digests per array.
     */
    void merkle_join_digests(uint8_t *out, const uint8_t *const *parts, uint64_t part_count,
                             uint64_t count);

    /**
     * @brief Allocates a tree for @p size leaves, every level at once, so the
     *        same tree can be committed to repeatedly.
     * @param[in] size Leaves, at least 1.
     * @return A tree to release with merkle_free(); NULL when @p size is 0.
     * @warning Exits the process if an allocation is refused.
     */
    Merkle merkle_new(uint64_t size);

    /**
     * @brief Releases a tree.
     * @param[in] mk The tree, or NULL.
     */
    void merkle_free(Merkle mk);

    /**
     * @brief Fills the leaf level and rebuilds every internal level, a large
     *        level on the library's threads, the same tree at any number of them.
     * @param[in,out] mk           The tree.
     * @param[in]     leaf_digests `size` packed digests.
     */
    void merkle_commit(Merkle mk, const uint8_t *leaf_digests);

    /**
     * @brief Writes the root of the last commit.
     * @param[out] out Receives the digest.
     * @param[in]  mk  A committed tree.
     */
    void merkle_get_root(uint8_t *out, Merkle mk);

    /**
     * @brief Writes the opening of a leaf: the `log_size` sibling digests
     *        from the leaf's level upwards, so `out[i]` is the sibling of the
     *        path node at height `i`.
     *
     * @param[out] out   Receives `log_size * BLAKE3_OUT_LEN` bytes.
     * @param[in]  mk    A committed tree.
     * @param[in]  index The leaf, below `1 << log_size`.
     */
    void merkle_open(uint8_t *out, Merkle mk, uint64_t index);
    /**
     * @brief Writes the openings of several leaves, each as merkle_open()
     *        writes it, end to end: the opening of `indices[k]` starts at
     *        `out + k * log_size * BLAKE3_OUT_LEN`.
     *
     * @param[out] out     Receives `count * log_size * BLAKE3_OUT_LEN` bytes.
     * @param[in]  mk      A committed tree.
     * @param[in]  indices @p count leaves, each below `1 << log_size`.
     * @param[in]  count   Leaves.
     */
    void merkle_open_many(uint8_t *out, Merkle mk, const uint64_t *indices, uint64_t count);

    /**
     * @brief Recomputes a root from a leaf digest and its opening and
     *        compares it with @p root.
     *
     * @param[in] root        The digest to match.
     * @param[in] index       The leaf position the opening belongs to.
     * @param[in] path        @p path_len packed sibling digests, as
     *                        merkle_open() writes them.
     * @param[in] path_len    Siblings.
     * @param[in] leaf_digest The leaf's digest.
     * @return true when the opening is valid.
     */
    bool merkle_verify(const uint8_t *root, uint64_t index, const uint8_t *path, uint64_t path_len,
                       const uint8_t *leaf_digest);
    /**
     * @brief Checks the openings of several leaves of one tree, each as
     *        merkle_verify() does.
     *
     * Leaf `k` sits at `indices[k]`, its digest at
     * `leaf_digests + k * BLAKE3_OUT_LEN`, and its opening is the @p path_len
     * siblings at `paths + k * path_len * BLAKE3_OUT_LEN`. @p path_len is not
     * checked against the tree's depth, which the caller knows.
     *
     * @param[in] root         The digest to match.
     * @param[in] indices      @p count leaf positions.
     * @param[in] paths        @p count openings, packed end to end.
     * @param[in] path_len     Siblings per opening.
     * @param[in] leaf_digests @p count packed leaf digests.
     * @param[in] count        Leaves.
     * @return -1 when every opening is valid, else the first `k` whose is not.
     */
    int64_t merkle_verify_many(const uint8_t *root, const uint64_t *indices, const uint8_t *paths,
                               uint64_t path_len, const uint8_t *leaf_digests, uint64_t count);

#ifdef __cplusplus
}
#endif

#endif // VFHE_MERKLE_H
