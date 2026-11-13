// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file arith_gpu.h
 * @brief RNS polynomials held on a CUDA device, and the arithmetic that runs there.
 *
 * Part of the `_vfhe_cuda` extension, which gpu.h describes. An element is
 * `rows * N` residues as 64-bit words, row-major, each row reduced modulo its
 * prime: the layout polynomial_RNS_export_u64() writes. Every function returns
 * 0 on success, otherwise a CUDA error code for vfhe_gpu_error_string(), with
 * the output undefined.
 *
 * @note Thread-safe. Blocking: a call returns once the device has finished.
 */
#ifndef VFHE_ARITH_GPU_H
#define VFHE_ARITH_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Allocates an element on the device, holding no value yet.
     *
     * @param[out] out    Receives the element, to release with
     *                    arith_gpu_rns_free().
     * @param[in]  rows   Primes, one row each.
     * @param[in]  N      Residues per row.
     * @param[in]  primes The rows' moduli, @p rows of them, each below 2^63.
     */
    int arith_gpu_rns_new(void **out, uint64_t rows, uint64_t N, const uint64_t *primes);

    /**
     * @brief Releases an element.
     * @param[in] p The element, or NULL.
     */
    void arith_gpu_rns_free(void *p);

    /** @brief Copies `rows * N` words from the host into @p p. */
    int arith_gpu_rns_upload(void *p, const uint64_t *words);

    /** @brief Copies @p p's `rows * N` words to the host. */
    int arith_gpu_rns_download(uint64_t *words, const void *p);

    /** @brief `out = in`, on the device. Both have the same rows and N. */
    int arith_gpu_rns_copy(void *out, const void *in);

    /**
     * @brief `out = a + b` row-wise modulo each row's prime.
     *
     * All three have the same rows, N and primes; @p out may be @p a or
     * @p b.
     */
    int arith_gpu_rns_add(void *out, const void *a, const void *b);

    /** @brief `out = a - b`, as arith_gpu_rns_add(). */
    int arith_gpu_rns_sub(void *out, const void *a, const void *b);

    /** @brief `out = -in` modulo each row's prime; @p out may be @p in. */
    int arith_gpu_rns_negate(void *out, const void *in);

#ifdef __cplusplus
}
#endif

#endif // VFHE_ARITH_GPU_H
