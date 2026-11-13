// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the seeded sampler (crypto/c/src/seeded.c, prototyped
// in crypto/c/include/seeded.h); cffi accepts a `bytes` for a `const uint8_t *`.

void seeded_below(uint64_t *out, uint64_t count, uint64_t bound, const char *context,
                  const uint8_t *seed, uint64_t seed_len);
