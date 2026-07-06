// SPDX-License-Identifier: Apache-2.0
/**
 * @file sampling.c
 * @brief Distributions built on the random byte stream (see rng.h).
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "base.h"
#include "rng.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/** Map a uniform 64-bit word to [0, 1). */
static double int2double(uint64_t x) { return ((double)x) / 18446744073709551616.0; }

double rng_gaussian(double sigma)
{
    // Box-Muller over two uniform words. Redraw the radius word if it is 0:
    // log(0) would yield -inf (probability 2^-64 per draw).
    uint64_t rnd[2];
    do
    {
        rng_random_bytes(16, (uint8_t *)rnd);
    } while (rnd[1] == 0);
    return cos(2. * M_PI * int2double(rnd[0])) * sqrt(-2. * log(int2double(rnd[1]))) * sigma;
}

void rng_sparse_ternary(uint64_t *out, uint64_t size, uint64_t h, uint64_t q)
{
    memset(out, 0, sizeof(uint64_t) * size);
    uint64_t hw = 0, val = 1, *rnd_buffer;
    // Rejection sampling over batches of uniform positions; h*10 candidates
    // per batch make a second batch unlikely for any h <= size/2.
    const uint64_t buffer_size = h * 10;
    rnd_buffer = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * buffer_size);
    while (hw < h)
    {
        rng_random_bytes(sizeof(uint64_t) * buffer_size, (uint8_t *)rnd_buffer);
        array_mod_switch_from_2k(rnd_buffer, rnd_buffer, size, size, buffer_size);
        uint64_t i = 0;
        while (i < buffer_size && hw < h)
        {
            const uint64_t idx = rnd_buffer[i++];
            if (out[idx])
                continue; // position already used: reject
            out[idx] = (uint64_t)((q + (int64_t)val) % (int64_t)q);
            val = -val; // alternate +1 / -1 to keep the vector balanced
            hw++;
        }
    }
    free(rnd_buffer);
#ifndef NDEBUG
    uint64_t hw_check = 0, sum_check = 0;
    for (uint64_t i = 0; i < size; i++)
    {
        sum_check += out[i];
        hw_check += (out[i] != 0);
    }
    assert(hw_check == h);
    assert((sum_check % q) == 0);
#endif
}
