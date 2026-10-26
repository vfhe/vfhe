// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file bench_parallel.c
 * @brief The cost of one vfhe_parallel_for() call, at the thread limit the runner sets.
 *
 * With an empty body the time is the loop's own overhead: waking the pool's threads,
 * handing out the items and waiting for the last one. With a body of fixed work it shows
 * how much of that work the loop turns into speedup.
 */
#include <stdint.h>

#include <parallel.h>

#include "bench.h"

/* One slot per cache line, so bodies on different threads do not share a line. */
static volatile double sink[64 * 8];

typedef struct
{
    uint64_t items; /* items per call; 0 means one per thread of the limit */
    uint64_t work;  /* multiply-adds per item */
} Loop;

static void body(void *ctx, uint64_t i)
{
    const Loop *loop = ctx;
    double x = (double)i;
    for (uint64_t k = 0; k < loop->work; k++)
        x = x * 1.0000001 + 1e-9;
    sink[(i % 64) * 8] = x;
}

/* arg[0] is the items per call (0: one per thread), arg[1] the work per item. */
static void *setup(const BenchCase *c)
{
    static Loop loop;
    loop.items = c->arg[0];
    loop.work = c->arg[1];
    return &loop;
}

static void parallel_for(void *state)
{
    Loop *loop = state;
    const uint64_t items = loop->items ? loop->items : vfhe_num_threads();
    vfhe_parallel_for(items, 0, body, loop);
}

static const BenchCase cases[] = {
    {"parallel_for", "itemsT-work0", 1, {0, 0}, 1, setup, parallel_for, NULL, NULL},
    {"parallel_for", "items64-work0", 0, {64, 0}, 1, setup, parallel_for, NULL, NULL},
    {"parallel_for", "itemsT-work2000", 0, {0, 2000}, 1, setup, parallel_for, NULL, NULL},
    {"parallel_for", "items64-work2000", 0, {64, 2000}, 1, setup, parallel_for, NULL, NULL},
};

int main(int argc, char **argv)
{
    return bench_main(argc, argv, cases, sizeof cases / sizeof *cases);
}
