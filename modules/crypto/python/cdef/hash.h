// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// Python-facing ABI of the incremental BLAKE3 hasher (crypto/c/src/hash.c,
// prototyped in crypto/c/include/crypto.h). The state is a caller-allocated
// buffer of hash_stream_size() bytes.

uint64_t hash_stream_size(void);
void hash_stream_init(void *state);
void hash_stream_update(void *state, const uint8_t *in, uint64_t len);
void hash_stream_digest(const void *state, uint8_t out[32]);
