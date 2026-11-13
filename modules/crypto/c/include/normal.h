// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file normal.h
 * @brief Gaussian samples by Box-Muller, from a source or from a secret seed.
 *
 * The support is unbounded: a caller needing a tail bound clamps or
 * resamples. Box-Muller is not constant time in its input.
 *
 * @note Thread-safe.
 */
#ifndef VFHE_NORMAL_H
#define VFHE_NORMAL_H

#include <stdint.h>

#include "random.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Returns one sample of a zero-mean Gaussian with standard
     *        deviation @p sigma, from 16 bytes of @p source.
     * @param[in] source A source, or `NULL` for the entropy stream.
     * @param[in] sigma  The standard deviation.
     * @return The sample.
     */
    double normal_random(Random source, double sigma);

    /**
     * @brief Writes samples of the same distribution, each rounded to the
     *        nearest integer, from the BLAKE3 XOF of a secret seed.
     *
     * A function of @p seed alone, so a parallel kernel that gives each unit
     * of work a seed of its own draws the same noise on any number of
     * threads. The seed is as secret as the noise: draw it from the
     * source, never from a transcript.
     *
     * @param[out] out      Receives @p count samples.
     * @param[in]  count    Samples.
     * @param[in]  sigma    The standard deviation.
     * @param[in]  seed     Any length.
     * @param[in]  seed_len Bytes of @p seed.
     */
    void normal_from_seed(int64_t *out, uint64_t count, double sigma, const uint8_t *seed,
                          uint64_t seed_len);

#ifdef __cplusplus
}
#endif

#endif // VFHE_NORMAL_H
