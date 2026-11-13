// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "../gates.h"

#if !VFHE_HAVE_AESNI
#include <blake3.h>

#include <wipe.h>
#include "../stream.h"

void stream_from_seed(uint8_t *out, uint64_t amount, const uint64_t seed[4])
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, (const uint8_t *)seed, 32);
    blake3_hasher_finalize(&hasher, out, amount);
    wipe(&hasher, sizeof(hasher));
}
#endif // !VFHE_HAVE_AESNI
