// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// AES-128 in counter mode, the kernel behind expand.c and the x86 entropy
// stream. portable/aes128.c is a table AES and x86_64/aes128.c AES-NI, with
// VAES where the engine has it; both give the same bytes.
#ifndef VFHE_CRYPTO_KERNELS_AES128_H
#define VFHE_CRYPTO_KERNELS_AES128_H

#include <stdint.h>

// The 11 round keys, 16 bytes each, in the byte order XORed into the state.
void aes128_expand_key(uint8_t round_keys[176], const uint8_t key[16]);

// Keystream blocks [j0, j0 + count) of attempt `t`, 16 bytes each: AES-128
// of the counter block LE64(j) || LE64(t) under the round keys.
void aes128_ctr_keystream(uint8_t *out, const uint8_t round_keys[176], uint64_t j0, uint64_t t,
                          uint64_t count);

#endif // VFHE_CRYPTO_KERNELS_AES128_H
