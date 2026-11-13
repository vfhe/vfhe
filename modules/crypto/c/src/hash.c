// SPDX-FileCopyrightText: 2026 The vfhe authors
// SPDX-License-Identifier: Apache-2.0
// BLAKE3 as the library uses it: one-shot digests, an incremental hasher
// behind an opaque state, a digest chain, and a batch of equal-length inputs
// hashed across the SIMD lanes and the library's threads.
#include "hash.h"

#include <string.h>

#include <blake3.h>
#include <blake3_impl.h>

#include "parallel.h"

void hash_blake3(uint8_t *out, const uint8_t *in, uint64_t len)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, in, len);
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

uint64_t hash_blake3_stream_size(void) { return sizeof(blake3_hasher); }

void hash_blake3_stream_init(void *state) { blake3_hasher_init((blake3_hasher *)state); }

void hash_blake3_stream_update(void *state, const uint8_t *in, uint64_t len)
{
    blake3_hasher_update((blake3_hasher *)state, in, len);
}

void hash_blake3_stream_digest(const void *state, uint8_t out[32])
{
    blake3_hasher_finalize((const blake3_hasher *)state, out, BLAKE3_OUT_LEN);
}

void hash_blake3_chain(uint8_t state[32], const uint8_t *in, uint64_t count, uint64_t len)
{
    if (len <= BLAKE3_BLOCK_LEN - BLAKE3_OUT_LEN)
    {
        // state | part fits one block, and the digest of a one-block message
        // is one compression of the zero-padded block under its true length
        // and the CHUNK_START, CHUNK_END and ROOT flags.
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

// --- The batch -------------------------------------------------------------------
// blake3_hash_many is BLAKE3's lane-parallel core, what the reference
// implementation hashes the chunks of one long message with; it is the same
// work with the lanes full (8 under AVX2, 16 under AVX-512) when the inputs
// are many short messages, where a hasher per input leaves them idle and
// pays an init and a finalize each. An input of at most one chunk is its own
// root node, so hash_many with CHUNK_START on the first block and CHUNK_END
// with ROOT on the last writes an ordinary digest: hence the chunk limit in
// hash_blake3_batch_fits.
//
// hash_many compresses whole 64-byte blocks only: it has no argument for a
// final block's true length. An input that ends in a partial block is
// therefore split. Its whole blocks go through hash_many without CHUNK_END or
// ROOT, which leaves each input's chaining value, and the last block is one
// blake3_compress_in_place per input from that value, with the block's true
// length, CHUNK_END and ROOT (and CHUNK_START when it is the only block).

// Input pointers per call into the core. The inputs are at a fixed stride,
// but the core wants an array; one that stays in L1 costs under 2% against
// one for the whole batch and saves an allocation proportional to the count.
#define BATCH_POINTERS 256

// Input bytes a chunk of a batch hashes at least when the batch runs on
// several threads, so that a chunk's hashing outweighs handing it to a thread;
// a batch under two of these stays on the calling thread.
#define BATCH_PARALLEL_BYTES (256u << 10)
// Chunks per thread, so that threads finishing early take another one.
#define BATCH_CHUNKS_PER_THREAD 4

bool hash_blake3_batch_fits(uint64_t len) { return len > 0 && len <= BLAKE3_CHUNK_LEN; }

// The inputs of one run, at most BATCH_POINTERS of them, whose length ends in
// a partial block.
static void batch_split(uint8_t *out, const uint8_t *const *inputs, uint64_t run, uint64_t len)
{
    const uint64_t whole = (len - 1) / BLAKE3_BLOCK_LEN;
    const uint64_t last = len - whole * BLAKE3_BLOCK_LEN;
    uint8_t cvs[BATCH_POINTERS * BLAKE3_OUT_LEN];

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

// The batch on the calling thread.
static void batch_serial(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len)
{
    if (!hash_blake3_batch_fits(len))
    {
        for (uint64_t i = 0; i < count; i++)
            hash_blake3(&out[i * BLAKE3_OUT_LEN], &in[i * len], len);
        return;
    }
    const uint8_t *inputs[BATCH_POINTERS];
    for (uint64_t done = 0; done < count;)
    {
        const uint64_t run = count - done < BATCH_POINTERS ? count - done : BATCH_POINTERS;
        for (uint64_t k = 0; k < run; k++)
            inputs[k] = &in[(done + k) * len];
        // The core walks its lane widths down to singles, so any count works.
        if (len % BLAKE3_BLOCK_LEN)
            batch_split(&out[done * BLAKE3_OUT_LEN], inputs, run, len);
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

static void batch_chunk(void *ctx, uint64_t c)
{
    const BatchJob *job = (const BatchJob *)ctx;
    const uint64_t first = c * job->per_chunk;
    const uint64_t left = job->count - first;
    batch_serial(&job->out[first * BLAKE3_OUT_LEN], &job->in[first * job->len],
                 left < job->per_chunk ? left : job->per_chunk, job->len);
}

void hash_blake3_batch(uint8_t *out, const uint8_t *in, uint64_t count, uint64_t len)
{
    const uint64_t most_chunks = count * len / BATCH_PARALLEL_BYTES;
    const uint64_t threads = vfhe_threads_for(0, most_chunks);
    if (threads <= 1)
    {
        batch_serial(out, in, count, len);
        return;
    }
    // Every digest depends on its own input only, so the chunks are
    // contiguous runs of inputs, each a whole number of the runs the serial
    // code hands the lane-parallel core.
    uint64_t chunks = threads * BATCH_CHUNKS_PER_THREAD;
    if (chunks > most_chunks)
        chunks = most_chunks;
    uint64_t per_chunk = (count + chunks - 1) / chunks;
    per_chunk = (per_chunk + BATCH_POINTERS - 1) / BATCH_POINTERS * BATCH_POINTERS;
    BatchJob job = {out, in, count, len, per_chunk};
    vfhe_parallel_for((count + per_chunk - 1) / per_chunk, 0, batch_chunk, &job);
}
