// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "crypto.h"

#include <math.h>

#include <blake3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double int2double(uint64_t x) { return ((double)x) / 18446744073709551616.0; }

// Box-Muller on two uniform words, keeping one of its two outputs.
static double box_muller(const uint64_t rnd[2], double sigma)
{
    return cos(2. * M_PI * int2double(rnd[0])) * sqrt(-2. * log(int2double(rnd[1]))) * sigma;
}

double generate_normal_random(double sigma)
{
    uint64_t rnd[2];
    generate_random_bytes(16, (uint8_t *)rnd);
    return box_muller(rnd, sigma);
}

// The derive-key context of the seeded draw. Never change it: define a new one.
static const char NORMAL_CONTEXT[] = "vfhe 2026-10-05 rounded Gaussian from a secret seed";

// Samples drawn per read of the keystream: 4 KiB of it, on the stack.
#define NORMAL_CHUNK 256

void prng_normal_seeded(int64_t *out, uint64_t count, double sigma, const uint8_t *seed,
                        uint64_t seed_len)
{
    blake3_hasher hasher;
    blake3_hasher_init_derive_key(&hasher, NORMAL_CONTEXT);
    blake3_hasher_update(&hasher, seed, seed_len);
    uint64_t rnd[2 * NORMAL_CHUNK];
    for (uint64_t done = 0; done < count; done += NORMAL_CHUNK)
    {
        const uint64_t n = count - done < NORMAL_CHUNK ? count - done : NORMAL_CHUNK;
        blake3_hasher_finalize_seek(&hasher, done * sizeof(rnd[0]) * 2, (uint8_t *)rnd,
                                    n * sizeof(rnd[0]) * 2);
        for (uint64_t i = 0; i < n; i++)
            out[done + i] = (int64_t)round(box_muller(&rnd[2 * i], sigma));
    }
}
