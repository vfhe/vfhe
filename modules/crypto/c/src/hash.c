// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <string.h>

#include <blake3.h>
#include <blake3_impl.h>

#include "crypto.h"
#include "parallel.h"

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
// which is the definition of the digest. Hence the chunk limit in
// hash_batch_fits.
//
// hash_many compresses whole 64-byte blocks only: it has no argument for a
// final block's true length. An input that ends in a partial block is
// therefore split. Its whole blocks go through hash_many without CHUNK_END or
// ROOT, which leaves each input's chaining value, and the last block is one
// blake3_compress_in_place per input from that value, with the block's true
// length, CHUNK_END and ROOT (and CHUNK_START when it is the only block).

// Pointers per call into the lane-parallel core. It wants an array, and the
// inputs here are at a fixed stride, so this is the one thing that has to be
// materialized. Keeping the array small enough to stay in L1 costs under 2%
// against handing over one array for the whole batch, and saves an allocation
// proportional to the input count.
#define HASH_BATCH_POINTERS 256

// Input bytes a chunk of a batch hashes at least when the batch runs on
// several threads, so that a chunk's hashing outweighs handing it to a thread;
// a batch under two of these stays on the calling thread.
#define HASH_PARALLEL_BYTES (256u << 10)
// Chunks per thread, so that threads finishing early take another one.
#define HASH_CHUNKS_PER_THREAD 4

uint64_t hash_stream_size(void) { return sizeof(blake3_hasher); }

void hash_stream_init(void *state) { blake3_hasher_init((blake3_hasher *)state); }

void hash_stream_update(void *state, const uint8_t *in, uint64_t len)
{
    blake3_hasher_update((blake3_hasher *)state, in, len);
}

void hash_stream_digest(const void *state, uint8_t out[32])
{
    blake3_hasher_finalize((const blake3_hasher *)state, out, BLAKE3_OUT_LEN);
}

void hash_chain(uint8_t state[32], const uint8_t *in, uint64_t count, uint64_t len)
{
    if (len <= BLAKE3_BLOCK_LEN - BLAKE3_OUT_LEN)
    {
        // state | part fits one block. The BLAKE3 digest of a one-block
        // message is one compression of the zero-padded block, given its true
        // length and the CHUNK_START, CHUNK_END and ROOT flags.
        uint8_t block[BLAKE3_BLOCK_LEN] = {0};
        uint32_t cv[8];
        memcpy(block, state, BLAKE3_OUT_LEN);
        for (uint64_t i = 0; i < count; i++)
        {
            memcpy(&block[BLAKE3_OUT_LEN], &in[i * len], len);
            memcpy(cv, IV, sizeof(cv));
            blake3_compress_in_place(cv, block, (uint8_t)(BLAKE3_OUT_LEN + len), 0,
                                     CHUNK_START | CHUNK_END | ROOT);
            store_cv_words(block, cv);
        }
        memcpy(state, block, BLAKE3_OUT_LEN);
        return;
    }
    for (uint64_t i = 0; i < count; i++)
    {
        blake3_hasher hasher;
        blake3_hasher_init(&hasher);
        blake3_hasher_update(&hasher, state, BLAKE3_OUT_LEN);
        blake3_hasher_update(&hasher, &in[i * len], len);
        blake3_hasher_finalize(&hasher, state, BLAKE3_OUT_LEN);
    }
}

bool hash_batch_fits(uint64_t len) { return len > 0 && len <= BLAKE3_CHUNK_LEN; }

// The inputs of one run, at most HASH_BATCH_POINTERS of them, whose length ends
// in a partial block.
static void hash_batch_split(uint8_t *out, const uint8_t *const *inputs, uint64_t run, uint64_t len)
{
    const uint64_t whole = (len - 1) / BLAKE3_BLOCK_LEN;
    const uint64_t last = len - whole * BLAKE3_BLOCK_LEN;
    uint8_t cvs[HASH_BATCH_POINTERS * BLAKE3_OUT_LEN];

    if (whole)
        blake3_hash_many(inputs, run, whole, IV, 0, false, 0, CHUNK_START, 0, cvs);
    for (uint64_t k = 0; k < run; k++)
    {
        uint32_t cv[8];
        uint8_t block[BLAKE3_BLOCK_LEN] = {0};
        if (whole)
            load_key_words(&cvs[k * BLAKE3_OUT_LEN], cv);
        else
            memcpy(cv, IV, sizeof(cv));
        memcpy(block, inputs[k] + whole * BLAKE3_BLOCK_LEN, last);
        blake3_compress_in_place(cv, block, (uint8_t)last, 0,
                                 (whole ? 0 : CHUNK_START) | CHUNK_END | ROOT);
        store_cv_words(&out[k * BLAKE3_OUT_LEN], cv);
    }
}

// hash_batch on the calling thread.
static void hash_batch_serial(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len)
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
        if (len % BLAKE3_BLOCK_LEN)
            hash_batch_split(&out[done * BLAKE3_OUT_LEN], inputs, run, len);
        else
            blake3_hash_many(inputs, run, len / BLAKE3_BLOCK_LEN, IV, 0, false, 0, CHUNK_START,
                             CHUNK_END | ROOT, &out[done * BLAKE3_OUT_LEN]);
        done += run;
    }
}

typedef struct
{
    uint8_t *out;
    const uint8_t *in;
    uint64_t count, len, per_chunk;
} BatchJob;

static void hash_batch_chunk(void *ctx, uint64_t c)
{
    const BatchJob *job = (const BatchJob *)ctx;
    const uint64_t first = c * job->per_chunk;
    const uint64_t left = job->count - first;
    hash_batch_serial(&job->out[first * BLAKE3_OUT_LEN], &job->in[first * job->len],
                      left < job->per_chunk ? left : job->per_chunk, job->len);
}

void hash_batch(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len)
{
    const uint64_t most_chunks = count * len / HASH_PARALLEL_BYTES;
    const uint64_t threads = vfhe_threads_for(0, most_chunks);
    if (threads <= 1)
    {
        hash_batch_serial(out, in, count, len);
        return;
    }
    // Every digest depends on its own input only, so the chunks are
    // contiguous runs of inputs, each a whole number of the runs the serial
    // code hands the lane-parallel core.
    uint64_t chunks = threads * HASH_CHUNKS_PER_THREAD;
    if (chunks > most_chunks)
        chunks = most_chunks;
    uint64_t per_chunk = (count + chunks - 1) / chunks;
    per_chunk = (per_chunk + HASH_BATCH_POINTERS - 1) / HASH_BATCH_POINTERS * HASH_BATCH_POINTERS;
    BatchJob job = {out, in, count, len, per_chunk};
    vfhe_parallel_for((count + per_chunk - 1) / per_chunk, 0, hash_batch_chunk, &job);
}
