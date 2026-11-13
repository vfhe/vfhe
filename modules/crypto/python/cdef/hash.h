// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the hashing (crypto/c/src/hash.c, prototyped in
// crypto/c/include/hash.h). A hasher state is a caller-allocated buffer of
// hash_blake3_stream_size() bytes.

void hash_blake3(uint8_t *out, const uint8_t *in, uint64_t len);
uint64_t hash_blake3_stream_size(void);
void hash_blake3_stream_init(void *state);
void hash_blake3_stream_digest(const void *state, uint8_t out[32]);
void hash_blake3_chain(uint8_t state[32], const uint8_t *in, uint64_t count, uint64_t len);
