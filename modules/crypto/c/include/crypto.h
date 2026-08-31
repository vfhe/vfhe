// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void generate_random_bytes(uint64_t amount, uint8_t *pointer);
    double generate_normal_random(double sigma);

    // Test-only: pin the stream to a reproducible sequence so probabilistic FHE
    // tests are deterministic. Production uses hardware entropy.
    void vfhe_prng_set_deterministic_seed(uint64_t seed);
    void vfhe_prng_clear_deterministic_seed(void);

#ifdef __cplusplus
}
#endif
