// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
// The randomness source: the entropy stream behind NULL, or a seeded BLAKE3
// XOF read in order.
#include "random.h"

#include <alloc.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <blake3.h>

#include "below.h"
#include "kernels/seed.h"
#include "kernels/stream.h"
#include <wipe.h>

// --- The entropy stream -------------------------------------------------------
// Bytes expanded from fresh hardware seeds, small requests served from a pool
// per thread.

// `amount` bytes from one fresh seed; the seed draw dominates small requests.
static void bytes_from_seed(uint8_t *out, uint64_t amount)
{
    uint64_t seed[4];
    seed_source(seed);
    stream_from_seed(out, amount, seed);
    wipe(seed, sizeof(seed));
}

// --- The pool ---------------------------------------------------------------
// One per thread, so concurrent callers share no mutable state. Bytes served
// are wiped as they go, so the pool never holds what a caller was given.
#define SMALL_REQUEST 512
static _Thread_local uint8_t pool[1024] __attribute__((aligned(64)));
static _Thread_local uint64_t pool_idx = sizeof(pool);
_Static_assert(SMALL_REQUEST <= sizeof(pool), "a small request fits a fresh pool");

// A forked child holds a copy of the forking thread's pool, the only one it
// inherits; serving it would repeat the bytes the parent serves next.
static void discard_pool_in_child(void)
{
    wipe(pool, sizeof(pool));
    pool_idx = sizeof(pool);
}

// Registered as the library loads, before anything can draw.
__attribute__((constructor)) static void watch_forks(void)
{
    if (pthread_atfork(NULL, NULL, discard_pool_in_child) != 0)
    {
        perror("vfhe: pthread_atfork"); /* GCOVR_EXCL_LINE */
        exit(EXIT_FAILURE);             /* GCOVR_EXCL_LINE */
    }
}

// `amount` bytes, at most SMALL_REQUEST, from the calling thread's pool,
// refilled from a fresh seed when what remains will not cover the request.
static void bytes_from_pool(uint8_t *out, uint64_t amount)
{
    if (amount > sizeof(pool) - pool_idx)
    {
        pool_idx = 0;
        bytes_from_seed(pool, sizeof(pool));
    }
    memcpy(out, pool + pool_idx, (size_t)amount);
    wipe(pool + pool_idx, (size_t)amount);
    pool_idx += amount;
}

// The stream: a small request from the pool, a large one from a seed of its own.
static void entropy_bytes(uint8_t *out, uint64_t amount)
{
    if (amount < SMALL_REQUEST)
        bytes_from_pool(out, amount);
    else
        bytes_from_seed(out, amount);
}

// --- The seeded stream --------------------------------------------------------

// The derive-key context names the mapping from seed to stream; a change to
// the mapping is a new context.
static const char RANDOM_CONTEXT[] = "vfhe 2026-11-10 random source";

struct Random
{
    blake3_hasher hasher;
    uint64_t offset;
};

Random random_seeded(const uint8_t *seed, uint64_t seed_len)
{
    Random source = (Random)safe_malloc(sizeof(*source));
    blake3_hasher_init_derive_key(&source->hasher, RANDOM_CONTEXT);
    blake3_hasher_update(&source->hasher, seed, seed_len);
    source->offset = 0;
    return source;
}

void random_free(Random source)
{
    if (source == NULL)
        return;
    wipe(source, sizeof(*source));
    free(source);
}

void random_bytes(Random source, uint8_t *out, uint64_t amount)
{
    if (source == NULL)
    {
        entropy_bytes(out, amount);
        return;
    }
    blake3_hasher_finalize_seek(&source->hasher, source->offset, out, amount);
    source->offset += amount;
}

void random_below(Random source, uint64_t *out, uint64_t count, uint64_t bound)
{
    assert(bound >= 1);
    const uint64_t mask = below_mask(bound);
    random_bytes(source, (uint8_t *)out, count * sizeof(uint64_t));
    for (uint64_t i = 0; i < count; i++)
    {
        uint64_t v = out[i] & mask;
        while (v >= bound)
        {
            random_bytes(source, (uint8_t *)&v, sizeof(v));
            v &= mask;
        }
        out[i] = v;
    }
}
