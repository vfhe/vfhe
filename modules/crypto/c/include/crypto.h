/* SPDX-License-Identifier: Apache-2.0 */
#ifndef VFHE_CRYPTO_H
#define VFHE_CRYPTO_H
#include <stdint.h>

#define CRYPTO_LCG_MULT 6364136223846793005ULL
#define CRYPTO_LCG_INC 1442695040888963407ULL

uint64_t crypto_sample(uint64_t seed);

static inline uint64_t crypto_sample_twice(uint64_t seed)
{
    return crypto_sample(crypto_sample(seed));
}

#endif
