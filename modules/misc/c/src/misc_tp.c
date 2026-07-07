#include "misc.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Function adapted from OpenFHE
// BSD 2-Clause License https://www.openfhe.org/
// the main rounding operation used in ModSwitch (as described in Section 3 of
// https://eprint.iacr.org/2014/816) The idea is that Round(x) = 0.5 + Floor(x)
uint64_t RoundqQ(uint64_t v, uint64_t q, uint64_t Q)
{
    return ((uint64_t)floor(0.5 + ((double)v) * ((double)q) / ((double)Q)) % q);
}

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

// Mem alloc

uint64_t _glb_mem_count = 0;
// safe_malloc
void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr && (size > 0))
    {
        perror("malloc failed!");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_aligned_malloc(size_t size)
{
    void *ptr;
    int err = posix_memalign(&ptr, 64, size);
    if (err != 0)
    {
        perror("posix_memalign failed!");
        exit(EXIT_FAILURE);
    }
    return ptr;
}
