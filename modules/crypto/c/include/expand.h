// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file expand.h
 * @brief Seed expansion: AES-128 in counter mode, filtered to values below a bound.
 *
 * The sampler for bulk public randomness, such as a seeded MLWE mask. The
 * sequence is a file-format commitment: stored seeds are re-expanded through
 * it, so it never changes. Under a 16-byte `key`:
 *
 *     keystream block (j, t) = AES-128_key(LE64(j) || LE64(t))
 *     w = 4 if bound <= 2^32 else 8;  m = the smallest 2^k - 1 >= bound - 1
 *     value i = the first, over attempts t = 0, 1, ..., of
 *               (the w-byte little-endian word at byte i*w of attempt t's
 *                keystream) & m  that is below bound
 *
 * Values are exactly uniform in `[0, bound)`, value `i` depends on `i` alone,
 * and every engine computes the same sequence. The key is public: the AES
 * here is not constant time on the portable engine.
 *
 * @note Thread-safe; no state.
 */
#ifndef VFHE_EXPAND_H
#define VFHE_EXPAND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Derives the key a seed expands under: the first 16 bytes of
     *        BLAKE3 in derive-key mode with @p context, over @p seed followed
     *        by each word of @p label as 8 little-endian bytes.
     *
     * @param[out] key       Receives the key.
     * @param[in]  context   A fixed string literal per use.
     * @param[in]  seed      Any length.
     * @param[in]  seed_len  Bytes of @p seed.
     * @param[in]  label     Separates the streams drawn from one seed.
     * @param[in]  label_len Words of @p label.
     */
    void expand_key(uint8_t key[16], const char *context, const uint8_t *seed, uint64_t seed_len,
                    const uint64_t *label, uint64_t label_len);

    /**
     * @brief Writes values `[start, start + count)` of the sequence @p key
     *        defines, as 64-bit words.
     *
     * @param[out] out   Receives @p count values.
     * @param[in]  count Values.
     * @param[in]  start Index of the first.
     * @param[in]  bound Values are below it; at least 1.
     * @param[in]  key   From expand_key().
     * @pre @p bound is at least 1.
     */
    void expand_below(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                      const uint8_t key[16]);

    /**
     * @brief The same values as 32-bit words, for a @p bound up to 2^32.
     *
     * @param[out] out   Receives @p count values.
     * @param[in]  count Values.
     * @param[in]  start Index of the first.
     * @param[in]  bound Values are below it; in `[1, 2^32]`.
     * @param[in]  key   From expand_key().
     * @pre @p bound is in `[1, 2^32]`.
     */
    void expand_below32(uint32_t *out, uint64_t count, uint64_t start, uint64_t bound,
                        const uint8_t key[16]);

    /**
     * @brief Writes keystream blocks `first .. first + count - 1` of attempt
     *        @p attempt under @p key, 16 bytes each: the blocks the sequence
     *        above is read from.
     *
     * @param[out] out     Receives `16 * count` bytes.
     * @param[in]  count   Blocks.
     * @param[in]  first   Index of the first block.
     * @param[in]  attempt The attempt, `t` above.
     * @param[in]  key     Any 16 bytes.
     */
    void expand_aes128_ctr(uint8_t *out, uint64_t count, uint64_t first, uint64_t attempt,
                           const uint8_t key[16]);

#ifdef __cplusplus
}
#endif

#endif // VFHE_EXPAND_H
