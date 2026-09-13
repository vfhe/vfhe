// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <blake3.h>
#include <blake3_impl.h>

#include "crypto.h"

// -------------------------------------------------------------
// Many small independent inputs, hashed across the SIMD lanes
// -------------------------------------------------------------
// blake3_hash_many is BLAKE3's lane-parallel core: it is what the reference
// implementation uses to hash the chunks of one long message, and it does not
// care that the inputs come from one message rather than from many. Handed a
// batch of short independent inputs it is the same work with the lanes full --
// 8 wide under AVX2, 16 under AVX-512 -- where a hasher per input leaves them
// idle and pays an init and a finalize each.
//
// The flags are what make the result an ordinary BLAKE3 digest rather than an
// interior chaining value. An input of at most one chunk is its own root node,
// so its digest is the chunk compressed with CHUNK_START on the first block,
// CHUNK_END on the last, and ROOT with it; hash_many applies flags_start and
// flags_end exactly there and writes the first 32 bytes of the final state,
// which is the definition of the digest. Hence both limits in hash_batch_fits.

// Pointers per call into the lane-parallel core. It wants an array, and the
// inputs here are at a fixed stride, so this is the one thing that has to be
// materialized. Keeping the array small enough to stay in L1 costs under 2%
// against handing over one array for the whole batch, and saves an allocation
// proportional to the input count.
#define HASH_BATCH_POINTERS 256

bool hash_batch_fits(uint64_t len)
{
    return len > 0 && len % BLAKE3_BLOCK_LEN == 0 && len <= BLAKE3_CHUNK_LEN;
}

void hash_batch(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len)
{
    if (!hash_batch_fits(len))
    {
        for (uint64_t i = 0; i < count; i++)
        {
            blake3_hasher hasher;
            blake3_hasher_init(&hasher);
            blake3_hasher_update(&hasher, &in[i * len], len);
            blake3_hasher_finalize(&hasher, &out[i * BLAKE3_OUT_LEN], BLAKE3_OUT_LEN);
        }
        return;
    }

    const uint8_t *inputs[HASH_BATCH_POINTERS];
    for (uint64_t done = 0; done < count;)
    {
        uint64_t run = count - done;
        if (run > HASH_BATCH_POINTERS)
            run = HASH_BATCH_POINTERS;
        for (uint64_t k = 0; k < run; k++)
            inputs[k] = &in[(done + k) * len];
        // One call takes any count: the core walks its own lane widths down to
        // singles, so a run that is not a multiple of the width needs no tail
        // loop here.
        blake3_hash_many(inputs, run, len / BLAKE3_BLOCK_LEN, IV, 0, false, 0, CHUNK_START,
                         CHUNK_END | ROOT, &out[done * BLAKE3_OUT_LEN]);
        done += run;
    }
}
