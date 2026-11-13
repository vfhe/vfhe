// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "../gates.h"

#if VFHE_HAVE_RDRAND
#include <immintrin.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../seed.h"

// A step fails when the generator is momentarily drained; Intel's guidance is
// ten tries before treating it as a fault.
static bool rdrand64(uint64_t *out)
{
    for (int tries = 0; tries < 10; tries++)
        if (_rdrand64_step((unsigned long long *)out))
            return true;
    return false;
}

void seed_source(uint64_t p[4])
{
    for (int i = 0; i < 4; i++)
        if (!rdrand64(&p[i]))
        {
            fprintf(stderr, "vfhe: RDRAND failed\n"); /* GCOVR_EXCL_LINE */
            exit(EXIT_FAILURE);                       /* GCOVR_EXCL_LINE */
        }
}
#endif // VFHE_HAVE_RDRAND
