// SPDX-License-Identifier: Apache-2.0
/**
 * @file rng.h
 * @brief Cryptographic randomness and sampling.
 *
 * This module is the engine's entropy boundary. It is organized in three
 * small layers (one source file each under `src/`):
 *
 *  - **seed.c**    hardware/OS entropy (RDRAND on x86, /dev/urandom elsewhere)
 *  - **stream.c**  buffered pseudorandom byte stream over a backend chosen at
 *                  compile time: AES-NI counter mode (`aes_ctr.c`, x86 builds
 *                  with `-maes`) or BLAKE3 in XOF mode (portable default)
 *  - **sampling.c** distributions built on the stream (Gaussian, sparse
 *                  ternary)
 *
 * Thread safety: each thread owns its own stream buffer (reseeded from the
 * entropy source independently), so ::rng_random_bytes and the samplers may
 * be called concurrently. Byte streams are non-deterministic by design --
 * there is no user-visible seeding API; deterministic replay belongs in the
 * (future) transcript layer, not here. (A build-flag-gated deterministic
 * override, ``rng_test_set_seed`` in rng_internal.h, exists solely for the
 * fuzz harnesses and reproducibility-sensitive tests; it is never compiled
 * into production or wheel builds.)
 */
#ifndef VFHE_RNG_H
#define VFHE_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Fill @p out with @p amount cryptographically random bytes.
     *
     * Small requests (< 512 bytes) are served from a per-thread buffer that is
     * refilled from the backend in 1 KiB batches; large requests stream directly.
     *
     * @param amount bytes requested
     * @param out    destination buffer of at least @p amount bytes
     */
    void rng_random_bytes(uint64_t amount, uint8_t *out);

    /**
     * One sample from a centered normal distribution N(0, sigma^2)
     * (Box-Muller over two fresh uniform words; always finite).
     *
     * @param sigma standard deviation
     * @return the sample
     */
    double rng_gaussian(double sigma);

    /**
     * Balanced sparse ternary vector mod q: exactly @p h nonzero entries at
     * uniformly random positions, half `+1` and half `-1` (encoded as `q - 1`),
     * so the entries sum to 0 mod q.
     *
     * @param out  vector of @p size residues (fully overwritten)
     * @param size vector length (power of two)
     * @param h    Hamming weight; must be even and <= size
     * @param q    modulus used to encode -1
     */
    void rng_sparse_ternary(uint64_t *out, uint64_t size, uint64_t h, uint64_t q);

#ifdef __cplusplus
}
#endif

#endif // VFHE_RNG_H
