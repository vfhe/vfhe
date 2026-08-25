// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "misc.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <blake3.h>

#ifdef USE_SHAKE
#include "../third-party/sha3/fips202.c"
#endif

// Forward declaration of aes_prng (only used if AES-NI is available and not in portable build)
#if !defined(PORTABLE_BUILD) && !defined(PORTABLE) && (defined(__x86_64__) || defined(_M_X64)) &&  \
    defined(__AES__)
void aes_prng(uint8_t *output, uint64_t outlen, const uint8_t *input, uint64_t inlen);
#endif

// --- Deterministic seed override (tests only) ----------------------------
// Reproducible FHE-bootstrap tests need a fixed RNG. When a deterministic seed
// is set (vfhe_prng_set_deterministic_seed), generate_rnd_seed yields a
// reproducible, non-repeating stream via splitmix64 instead of hardware
// entropy. Production never calls the setters, so it keeps using RDRAND /
// /dev/urandom. Not thread-safe; intended for single-threaded test use.
static int det_active = 0;
static uint64_t det_state = 0;

// Buffered entropy pool (lifted to file scope so the setters can discard it).
static uint8_t rnd_buffer[1024] __attribute__((aligned(64)));
static uint64_t rnd_buffer_idx = sizeof(rnd_buffer);

static uint64_t splitmix64_next(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void det_fill_seed(uint64_t *p)
{
    for (int i = 0; i < 4; i++)
        p[i] = splitmix64_next(&det_state);
}

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(PORTABLE) && !defined(PORTABLE_BUILD)
// Intel hardware RDRAND seed generator
void generate_rnd_seed(uint64_t *p)
{
    if (det_active)
    {
        det_fill_seed(p);
        return;
    }
    if (0 == _rdrand64_step((unsigned long long *)p) ||
        0 == _rdrand64_step((unsigned long long *)&(p[1])) ||
        0 == _rdrand64_step((unsigned long long *)&(p[2])) ||
        0 == _rdrand64_step((unsigned long long *)&(p[3])))
    {
        printf("Random Generation Failed\n");
        return;
    }
}
#else
// Fallback urandom seed generator
void generate_rnd_seed(uint64_t *p)
{
    if (det_active)
    {
        det_fill_seed(p);
        return;
    }
    FILE *fp = fopen("/dev/urandom", "r");
    const size_t read = fp ? fread(p, 1, 32, fp) : 0;
    if (fp)
        fclose(fp);
    if (read != 32)
    {
        printf("Random Generation Failed\n");
        return;
    }
}
#endif

void get_rnd_from_hash(uint64_t amount, uint8_t *pointer)
{
    uint64_t rnd[4];
    generate_rnd_seed(rnd);

#if !defined(PORTABLE_BUILD) && !defined(PORTABLE) && (defined(__x86_64__) || defined(_M_X64)) &&  \
    defined(__AES__)
    aes_prng(pointer, amount, (uint8_t *)rnd, 32);
#elif defined(USE_SHAKE)
    shake256(pointer, amount, (uint8_t *)rnd, 32);
#else
    // Default fallback is Blake3
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, (uint8_t *)rnd, 32);
    blake3_hasher_finalize(&hasher, pointer, amount);
#endif
}

void get_rnd_from_buffer(uint64_t amount, uint8_t *pointer)
{
    // A request the pool could never satisfy is served directly, so the copy
    // below always stays inside it.
    if (amount > sizeof(rnd_buffer))
    {
        get_rnd_from_hash(amount, pointer);
        return;
    }
    if (amount > (sizeof(rnd_buffer) - rnd_buffer_idx))
    {
        rnd_buffer_idx = 0;
        get_rnd_from_hash(sizeof(rnd_buffer), rnd_buffer);
    }
    memcpy(pointer, rnd_buffer + rnd_buffer_idx, (size_t)amount);
    rnd_buffer_idx += amount;
}

void generate_random_bytes(uint64_t amount, uint8_t *pointer)
{
    if (amount < 512)
        get_rnd_from_buffer(amount, pointer);
    else
        get_rnd_from_hash(amount, pointer);
}

// Test-only: pin the PRNG to a reproducible stream (see the note above). The
// setter also discards any buffered entropy so the next draw starts from the
// seed; clearing returns to hardware entropy.
void vfhe_prng_set_deterministic_seed(uint64_t seed)
{
    det_active = 1;
    det_state = seed;
    rnd_buffer_idx = sizeof(rnd_buffer);
}

void vfhe_prng_clear_deterministic_seed(void)
{
    det_active = 0;
    rnd_buffer_idx = sizeof(rnd_buffer);
}

// --- Gaussian sampling ---------------------------------------------------
// Box-Muller over the entropy-backed stream above, so it follows the
// deterministic-seed override too. Moved here from misc_tp.c: every generator
// in the library lives in this file.

double int2double(uint64_t x) { return ((double)x) / 18446744073709551616.0; }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double generate_normal_random(double sigma)
{
    uint64_t rnd[2];
    generate_random_bytes(16, (uint8_t *)rnd);
    return cos(2. * M_PI * int2double(rnd[0])) * sqrt(-2. * log(int2double(rnd[1]))) * sigma;
}

// --- Seeded sampling (deterministic, independent of the stream above) ----
// A *different* generator from everything above: the caller supplies the seed,
// so the result is a pure function of (context, seed) and the
// deterministic-seed override has nothing to do with it. Used where a value
// must be reproducible from a transcript rather than merely unpredictable.
//
// `context` is a domain-separation tag: two callers with the same seed and
// different tags get independent streams. Pass a fixed string literal per use
// site, never anything caller-controlled.

void prng_sample_below(uint64_t *out, uint64_t count, uint64_t bound, const char *context,
                       const uint8_t *seed, uint64_t seed_len)
{
    assert(bound > 0); // otherwise the rejection loop below never terminates

    blake3_hasher hasher;
    blake3_hasher_init_derive_key(&hasher, context);
    blake3_hasher_update(&hasher, seed, seed_len);

    // Smallest 2^k - 1 that covers `bound`, so rejection discards under half
    // the draws.
    uint64_t mask = bound;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;

    // BLAKE3 as an XOF: `blake3_hasher_finalize` is a pure function of hasher
    // state, so finalizing repeatedly without an intervening update returns the
    // same bytes. Seeking along the output stream is what makes successive
    // draws independent.
    uint64_t offset = 0;
    for (uint64_t i = 0; i < count; i++)
    {
        for (;;)
        {
            uint8_t buf[sizeof(uint64_t)];
            blake3_hasher_finalize_seek(&hasher, offset, buf, sizeof(buf));
            offset += sizeof(buf);

            uint64_t word;
            memcpy(&word, buf, sizeof(word)); // the buffer is not aligned for a cast
            const uint64_t sampled = word & mask;
            if (sampled < bound)
            {
                out[i] = sampled;
                break;
            }
        }
    }
}
