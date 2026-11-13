// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file seeded.h
 * @brief The seeded sampler: values that are a pure function of a context and a seed.
 *
 * What a protocol uses for values both parties must agree on, such as
 * challenges derived from a transcript. The sequence is defined by its index:
 * attempt stream `k` is BLAKE3 in derive-key mode with `context` over
 * `seed || LE64(k)`, and value `i` is word `i` of stream 0 masked to the
 * bits of `bound - 1`, falling through to word `i` of streams 1, 2, ... while
 * the masked word is not below `bound`. Values are exactly uniform, identical
 * on every engine, and independent of how many are asked for.
 *
 * @note Thread-safe; no state. Not constant time in `bound`.
 */
#ifndef VFHE_SEEDED_H
#define VFHE_SEEDED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Writes values `[start, start + count)` of the sequence.
     *
     * Each position is located by its index, so a window costs what its own
     * values cost.
     *
     * @param[out] out      Receives @p count values.
     * @param[in]  count    Values.
     * @param[in]  start    Index of the first.
     * @param[in]  bound    Values are below it; at least 1.
     * @param[in]  context  A NUL-terminated domain-separation tag: a fixed
     *                      string literal per call site, never caller data.
     * @param[in]  seed     Any length.
     * @param[in]  seed_len Bytes of @p seed.
     * @pre @p bound is at least 1.
     */
    void seeded_below_from(uint64_t *out, uint64_t count, uint64_t start, uint64_t bound,
                           const char *context, const uint8_t *seed, uint64_t seed_len);

    /**
     * @brief Writes values `[0, count)` of the sequence: seeded_below_from()
     *        with `start = 0`.
     *
     * @param[out] out      Receives @p count values.
     * @param[in]  count    Values.
     * @param[in]  bound    Values are below it; at least 1.
     * @param[in]  context  As in seeded_below_from().
     * @param[in]  seed     Any length.
     * @param[in]  seed_len Bytes of @p seed.
     * @pre @p bound is at least 1.
     */
    void seeded_below(uint64_t *out, uint64_t count, uint64_t bound, const char *context,
                      const uint8_t *seed, uint64_t seed_len);

#ifdef __cplusplus
}
#endif

#endif // VFHE_SEEDED_H
