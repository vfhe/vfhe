// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Gaussian samples by Box-Muller, from a source or from a seed.
#include "normal.h"

#include <math.h>

#include <blake3.h>

#include "random.h"
#include <wipe.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// A word as a double in the open interval (0, 1): its top 53 bits plus a
// half, so the logarithm below never sees 0 and every sample is finite.
static double unit_open(uint64_t x)
{
    return ((double)(x >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// Box-Muller on two uniform words, keeping one of its two outputs.
static double box_muller(const uint64_t rnd[2], double sigma)
{
    return cos(2. * M_PI * unit_open(rnd[0])) * sqrt(-2. * log(unit_open(rnd[1]))) * sigma;
}

double normal_random(Random source, double sigma)
{
    uint64_t rnd[2];
    random_bytes(source, (uint8_t *)rnd, sizeof(rnd));
    const double sample = box_muller(rnd, sigma);
    wipe(rnd, sizeof(rnd));
    return sample;
}

// The derive-key context of the seeded draw names the mapping from seed to
// samples; a change to that mapping is a new context.
static const char NORMAL_CONTEXT[] = "vfhe 2026-11-04 rounded Gaussian from a secret seed";

// Samples drawn per seek of the stream: 4 KiB of it, on the stack.
#define NORMAL_CHUNK 256

void normal_from_seed(int64_t *out, uint64_t count, double sigma, const uint8_t *seed,
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
    wipe(rnd, sizeof(rnd));
    wipe(&hasher, sizeof(hasher));
}
