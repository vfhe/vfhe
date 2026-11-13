// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file random.h
 * @brief The source a sampler draws from: the entropy stream, or a stream a
 *        seed reproduces.
 *
 * Every sampler in vfhe takes a `Random`. `NULL` is the entropy stream: seeds
 * come from RDRAND where the engine has it and from `getentropy` otherwise,
 * and expand through AES-128 in counter mode on an AES-NI engine and through
 * BLAKE3 elsewhere, so the stream differs between engines; requests under 512
 * bytes come from the calling thread's pool, larger ones expand a fresh seed
 * of their own, and a forked child discards the pool it inherits. When the
 * seed source fails, the process prints the reason to stderr and exits with
 * `EXIT_FAILURE`: a predictable value is never returned. A seeded source is a
 * BLAKE3 XOF of its seed, the same bytes on every engine, read in order; it
 * belongs to one thread at a time, and two sources from one seed give the
 * same bytes in the same order.
 *
 * @note Thread-safe for `NULL`; a seeded source is not shared between threads.
 */
#ifndef VFHE_RANDOM_H
#define VFHE_RANDOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief A source of randomness; `NULL` stands for the entropy stream. */
    typedef struct Random *Random;

    /**
     * @brief Creates a source every draw of which is a function of @p seed.
     *
     * @param[in] seed     The seed; a secret, when the draws are.
     * @param[in] seed_len Its length in bytes.
     * @return A source to release with random_free().
     * @warning Exits the process with `EXIT_FAILURE` if the allocation is refused.
     */
    Random random_seeded(const uint8_t *seed, uint64_t seed_len);

    /**
     * @brief Releases a seeded source, wiping its state.
     * @param[in] source From random_seeded(), or `NULL`, which does nothing.
     */
    void random_free(Random source);

    /**
     * @brief Writes bytes from the source.
     * @param[in]  source A source, or `NULL` for the entropy stream.
     * @param[out] out    Receives @p amount bytes.
     * @param[in]  amount Bytes to write.
     * @warning Exits the process if the entropy seed source fails.
     */
    void random_bytes(Random source, uint8_t *out, uint64_t amount);

    /**
     * @brief Writes values exactly uniform in `[0, bound)`, by rejection.
     *
     * @param[in]  source A source, or `NULL` for the entropy stream.
     * @param[out] out    Receives @p count values.
     * @param[in]  count  Values to write.
     * @param[in]  bound  At least 1.
     */
    void random_below(Random source, uint64_t *out, uint64_t count, uint64_t bound);

#ifdef __cplusplus
}
#endif

#endif // VFHE_RANDOM_H
