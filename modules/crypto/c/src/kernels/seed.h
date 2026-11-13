// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The seed source of the entropy stream: 32 bytes of hardware entropy, from
// RDRAND (x86_64/seed.c) where the engine has it and getentropy
// (portable/seed.c) otherwise. On failure both print the reason to stderr and
// exit the process: a predictable seed is never returned.
#ifndef VFHE_CRYPTO_KERNELS_SEED_H
#define VFHE_CRYPTO_KERNELS_SEED_H

#include <stdint.h>

void seed_source(uint64_t out[4]);

#endif // VFHE_CRYPTO_KERNELS_SEED_H
