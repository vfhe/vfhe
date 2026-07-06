// SPDX-License-Identifier: Apache-2.0
/**
 * @file stream.c
 * @brief Buffered pseudorandom byte stream (see rng.h).
 *
 * Front end of the module: picks the expansion backend at compile time
 * (AES-NI counter mode on x86 `-maes` builds, BLAKE3 XOF otherwise) and
 * amortizes small requests through a per-thread 1 KiB buffer. Every refill
 * draws a fresh seed from the platform entropy source, so the stream has no
 * long-lived secret state to protect.
 */
#include <stdint.h>
#include <string.h>

#include <blake3.h>

#include "rng.h"
#include "rng_internal.h"

/** Expand a fresh entropy seed into @p amount output bytes. */
static void stream_generate(uint64_t amount, uint8_t *out)
{
    uint64_t rnd[4];
    rng_seed_bytes(rnd);

#if !defined(PORTABLE_BUILD) && !defined(PORTABLE) && defined(__AES__)
    aes_prng(out, amount, (uint8_t *)rnd, 32);
#else
    // Portable backend: BLAKE3 as an XOF keyed by the seed.
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, (uint8_t *)rnd, 32);
    blake3_hasher_finalize(&hasher, out, amount);
#endif
}

#define STREAM_BUFFER_SIZE 1024

// Per-thread buffer for small requests. Starts "empty" (idx at the end) so the
// first draw triggers a refill.
static _Thread_local uint8_t stream_buffer[STREAM_BUFFER_SIZE] __attribute__((aligned(64)));
static _Thread_local uint64_t stream_idx = STREAM_BUFFER_SIZE;

/** Serve a small request from the per-thread buffer, refilling as needed. */
static void stream_take_buffered(uint64_t amount, uint8_t *out)
{
    if (amount > (STREAM_BUFFER_SIZE - stream_idx))
    {
        stream_idx = 0;
        stream_generate(STREAM_BUFFER_SIZE, stream_buffer);
    }
    memcpy(out, stream_buffer + stream_idx, (size_t)amount);
    stream_idx += amount;
}

#ifdef VFHE_RNG_TESTING
// Discard any buffered bytes so the next draw reseeds. Used by the deterministic
// testing override (rng_test_set_seed) to make each seed fully reproducible.
void rng_stream_reset(void) { stream_idx = STREAM_BUFFER_SIZE; }
#endif

void rng_random_bytes(uint64_t amount, uint8_t *out)
{
    if (amount < 512)
        stream_take_buffered(amount, out);
    else
        stream_generate(amount, out);
}
