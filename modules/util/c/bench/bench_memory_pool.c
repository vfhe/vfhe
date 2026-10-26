// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file bench_memory_pool.c
 * @brief Taking a buffer from the memory pool and giving it back.
 *
 * After the first call the pool holds a buffer of the size asked for, so every timed call
 * is the pool's own bookkeeping: the lookup on the calling thread's free list and the
 * accounting on release. The buffer is not written.
 */
#include <memory_pool.h>

#include "bench.h"

typedef struct
{
    size_t bytes;
} Buffer;

/* arg[0] is the buffer's size in bytes. */
static void *setup(const BenchCase *c)
{
    static Buffer buffer;
    buffer.bytes = (size_t)c->arg[0];
    return &buffer;
}

static void cycle(void *state)
{
    const Buffer *b = state;
    void *p = mempool_aligned_malloc(b->bytes);
    BENCH_KEEP(p);
    mempool_free(p, b->bytes);
}

#define KIB(n) ((uint64_t)(n) << 10)
#define MIB(n) ((uint64_t)(n) << 20)

static const BenchCase cases[] = {
    {"mempool_cycle", "64KiB", 0, {KIB(64)}, 1, setup, cycle, NULL, NULL},
    {"mempool_cycle", "1MiB", 0, {MIB(1)}, 1, setup, cycle, NULL, NULL},
    {"mempool_cycle", "16MiB", 0, {MIB(16)}, 1, setup, cycle, NULL, NULL},
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, cases, sizeof cases / sizeof *cases);
}
