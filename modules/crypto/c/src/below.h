// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// What every sampler below a bound starts from.
#ifndef VFHE_CRYPTO_BELOW_H
#define VFHE_CRYPTO_BELOW_H

#include <stdint.h>

// The smallest 2^k - 1 covering every value below `bound`: a word masked by
// it lands below `bound` more than half the time, so rejection sampling
// discards under half the draws.
static inline uint64_t below_mask(uint64_t bound)
{
    uint64_t mask = bound - 1;
    for (unsigned s = 1; s < 64; s <<= 1)
        mask |= mask >> s;
    return mask;
}

#endif // VFHE_CRYPTO_BELOW_H
