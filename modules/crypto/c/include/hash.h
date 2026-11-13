// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hash.h
 * @brief BLAKE3: one-shot digests, a batch of equal-length inputs, an incremental hasher, and a
 * digest chain.
 *
 * Every digest is `BLAKE3_OUT_LEN` (32) bytes, the vendored library's
 * constant. No function allocates; the caller owns every buffer, and no
 * input may overlap an output.
 *
 * @note Thread-safe. A hasher state belongs to one thread at a time.
 */
#ifndef VFHE_HASH_H
#define VFHE_HASH_H

#include <stdbool.h>
#include <stdint.h>

#include <blake3.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Writes the digest of a byte string.
     *
     * @param[out] out Receives `BLAKE3_OUT_LEN` bytes.
     * @param[in]  in  The bytes to hash.
     * @param[in]  len Their count.
     */
    void hash_blake3(uint8_t *out, const uint8_t *in, uint64_t len);

    /**
     * @brief Reports whether hash_blake3_batch() takes inputs of `len` bytes across
     *        the SIMD lanes: a nonzero multiple of 64, at most 1024.
     *
     * Other lengths are still hashed, one input at a time.
     *
     * @param[in] len Bytes per input.
     * @return true when the batch runs lane-parallel.
     */
    bool hash_blake3_batch_fits(uint64_t len);

    /**
     * @brief Writes the digests of `count` independent inputs of `len` bytes
     *        each, laid out end to end.
     *
     * Each digest equals hash_blake3() of that input alone.
     *
     * @param[out] out   Receives `count * BLAKE3_OUT_LEN` bytes.
     * @param[in]  in    `count * len` bytes, input after input.
     * @param[in]  count Inputs.
     * @param[in]  len   Bytes per input.
     */
    void hash_blake3_batch(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len);

    /**
     * @brief Returns the bytes a hasher state takes, for hash_blake3_stream_init().
     * @return The state size in bytes.
     */
    uint64_t hash_blake3_stream_size(void);

    /**
     * @brief Starts an incremental hash in caller-owned memory.
     * @param[out] state hash_blake3_stream_size() bytes, any alignment.
     */
    void hash_blake3_stream_init(void *state);

    /**
     * @brief Feeds bytes to an incremental hash.
     * @param[in,out] state A state from hash_blake3_stream_init().
     * @param[in]     in    The bytes.
     * @param[in]     len   Their count.
     */
    void hash_blake3_stream_update(void *state, const uint8_t *in, uint64_t len);

    /**
     * @brief Writes the digest of everything fed so far; the state stays
     *        usable and more may follow.
     * @param[in]  state A state from hash_blake3_stream_init().
     * @param[out] out   Receives the digest.
     */
    void hash_blake3_stream_digest(const void *state, uint8_t out[BLAKE3_OUT_LEN]);

    /**
     * @brief Advances a digest chain over parts of one length:
     *        `state = H(state | part)`, part after part, in place.
     *
     * @param[in,out] state The chain's digest, replaced by the new one.
     * @param[in]     in    `count * len` bytes, part after part.
     * @param[in]     count Parts.
     * @param[in]     len   Bytes per part.
     */
    void hash_blake3_chain(uint8_t state[BLAKE3_OUT_LEN], const uint8_t *in, uint64_t count,
                           uint64_t len);

#ifdef __cplusplus
}
#endif

#endif // VFHE_HASH_H
