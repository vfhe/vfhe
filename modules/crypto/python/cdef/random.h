// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the randomness source (crypto/c/src/random.c, prototyped
// in crypto/c/include/random.h). NULL is the entropy stream.

void *random_seeded(const uint8_t *seed, uint64_t seed_len);
void random_free(void *source);
void random_bytes(void *source, uint8_t *out, uint64_t amount);
void random_below(void *source, uint64_t *out, uint64_t count, uint64_t bound);
