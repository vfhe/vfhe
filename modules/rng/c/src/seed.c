// SPDX-License-Identifier: Apache-2.0
/**
 * @file seed.c
 * @brief Platform entropy source (see rng_internal.h).
 */
#include <stdio.h>

#include "rng_internal.h"

#if defined(VFHE_RNG_TESTING)

// Deterministic entropy override for reproducible tests and fuzzing. This
// replaces the platform source entirely, so every downstream draw (byte
// stream, Gaussian, sparse ternary) is a pure function of the installed seed.
// Compiled only under -DVFHE_RNG_TESTING; never in production/wheel builds.
static uint64_t g_test_counter = 0x9E3779B97F4A7C15ULL;

void rng_test_set_seed(uint64_t seed)
{
    g_test_counter = seed + 0x9E3779B97F4A7C15ULL;
    rng_stream_reset(); // drop buffered bytes so this seed is fully reproducible
}

// splitmix64 over a running counter -> 32 fresh bytes per refill.
void rng_seed_bytes(uint64_t *p)
{
    for (int i = 0; i < 4; i++)
    {
        uint64_t z = (g_test_counter += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        p[i] = z ^ (z >> 31);
    }
}

#elif (defined(__x86_64__) || defined(_M_X64)) && !defined(PORTABLE) && !defined(PORTABLE_BUILD)

#include <immintrin.h>

// Intel hardware RDRAND. The target attribute enables the rdrnd feature for
// just this function, so no global -mrdrnd compile flag is needed.
__attribute__((target("rdrnd"))) void rng_seed_bytes(uint64_t *p)
{
    if (0 == _rdrand64_step((unsigned long long *)p) ||
        0 == _rdrand64_step((unsigned long long *)&(p[1])) ||
        0 == _rdrand64_step((unsigned long long *)&(p[2])) ||
        0 == _rdrand64_step((unsigned long long *)&(p[3])))
    {
        // RDRAND exhaustion is transient and vanishingly rare; the caller
        // mixes this seed through a PRF, so a partial fill degrades entropy
        // rather than correctness. Flag it and continue.
        fprintf(stderr, "rng: RDRAND failed to return entropy\n");
    }
}

#else

// OS entropy fallback.
void rng_seed_bytes(uint64_t *p)
{
    FILE *fp = fopen("/dev/urandom", "r");
    if (fp)
    {
        size_t unused = fread(p, 1, 32, fp);
        (void)unused;
        fclose(fp);
    }
}

#endif
