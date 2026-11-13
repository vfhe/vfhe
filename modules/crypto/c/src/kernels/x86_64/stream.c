// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "../gates.h"

#if VFHE_HAVE_AESNI
#include <string.h>

#include <wipe.h>
#include "../aes128.h"
#include "../stream.h"

// The first 16 bytes of the seed key the cipher and the third word is the
// counter's nonce, so 24 of the 32 bytes shape the stream.
void stream_from_seed(uint8_t *out, uint64_t amount, const uint64_t seed[4])
{
    uint8_t rk[176];
    aes128_expand_key(rk, (const uint8_t *)seed);
    const uint64_t blocks = amount / 16;
    aes128_ctr_keystream(out, rk, 0, seed[2], blocks);
    if (amount % 16)
    {
        uint8_t tail[16];
        aes128_ctr_keystream(tail, rk, blocks, seed[2], 1);
        memcpy(&out[16 * blocks], tail, amount % 16);
        wipe(tail, sizeof(tail));
    }
    wipe(rk, sizeof(rk));
}
#endif // VFHE_HAVE_AESNI
