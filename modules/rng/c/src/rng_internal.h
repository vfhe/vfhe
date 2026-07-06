// SPDX-License-Identifier: Apache-2.0
/**
 * @file rng_internal.h
 * @brief rng-private declarations shared between seed, stream, and backends.
 */
#ifndef VFHE_RNG_INTERNAL_H
#define VFHE_RNG_INTERNAL_H

#include <stdint.h>

/**
 * Fill @p p with 32 bytes from the platform entropy source
 * (RDRAND on x86 fast builds, /dev/urandom otherwise). Defined in seed.c.
 */
void rng_seed_bytes(uint64_t *p);

/**
 * Testing hook: replace the platform entropy source with a deterministic,
 * reproducible stream derived from @p seed. Call once before the first draw.
 *
 * Defined ONLY when the module is compiled with @c -DVFHE_RNG_TESTING (the
 * fuzz harnesses and reproducibility-sensitive tests); production and wheel
 * builds omit it entirely, so the entropy boundary is never weakened. See
 * seed.c and rng.h.
 */
void rng_test_set_seed(uint64_t seed);

/** Testing hook (stream.c): flush the per-thread buffer so the next draw
 *  reseeds. Called by rng_test_set_seed. Only under @c -DVFHE_RNG_TESTING. */
void rng_stream_reset(void);

#if !defined(PORTABLE_BUILD) && !defined(PORTABLE) && defined(__AES__)
/**
 * AES-128 counter-mode expansion (aes_ctr.c): generate @p outlen bytes with
 * the counter starting from the first 16 bytes of @p input. The AES key is a
 * process-global initialized from the entropy source on first use.
 * Requires `outlen >= 256` and `inlen >= 16`.
 */
void aes_prng(uint8_t *output, uint64_t outlen, const uint8_t *input, uint64_t inlen);
#endif

#endif // VFHE_RNG_INTERNAL_H
