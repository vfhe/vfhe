// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The entropy stream's expander: `amount` bytes from one fresh 32-byte seed.
// x86_64/stream.c keys AES-128 in counter mode with the seed and takes the
// counter's nonce from it; portable/stream.c takes the BLAKE3 XOF of the
// seed, since the table AES is not constant time and the seed is secret. The
// bytes differ between the two.
#ifndef VFHE_CRYPTO_KERNELS_STREAM_H
#define VFHE_CRYPTO_KERNELS_STREAM_H

#include <stdint.h>

void stream_from_seed(uint8_t *out, uint64_t amount, const uint64_t seed[4]);

#endif // VFHE_CRYPTO_KERNELS_STREAM_H
