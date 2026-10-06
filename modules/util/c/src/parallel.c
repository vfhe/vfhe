// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "parallel.h"

#include <alloc.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

// 0 until first read or after a reset; the environment is read on first use.
static atomic_uint_fast64_t thread_limit = 0;

// Set while a thread (the caller's included) runs loop bodies, so a loop
// started inside a body runs serially.
static _Thread_local int inside_parallel_loop = 0;

typedef struct
{
    void (*body)(void *ctx, uint64_t i);
    void *ctx;
    uint64_t n;
    atomic_uint_fast64_t next;
} ParallelLoop;

static uint64_t default_thread_limit(void);

// Items are handed out one at a time, which balances uneven item costs.
static void run_loop(ParallelLoop *loop);

static void *run_loop_thread(void *loop);

uint64_t vfhe_num_threads(void)
{
    uint_fast64_t limit = atomic_load(&thread_limit);
    if (limit == 0)
    {
        uint_fast64_t unset = 0;
        atomic_compare_exchange_strong(&thread_limit, &unset, default_thread_limit());
        limit = atomic_load(&thread_limit);
    }
    return (uint64_t)limit;
}

void vfhe_set_num_threads(uint64_t n) { atomic_store(&thread_limit, n); }

uint64_t vfhe_threads_for(uint64_t requested, uint64_t n_items)
{
    if (inside_parallel_loop || n_items <= 1)
        return 1;
    const uint64_t limit = vfhe_num_threads();
    uint64_t n = (requested == 0 || requested > limit) ? limit : requested;
    return n < n_items ? n : n_items;
}

void vfhe_parallel_for(uint64_t n, uint64_t n_threads, void (*body)(void *ctx, uint64_t i),
                       void *ctx)
{
    const uint64_t threads = vfhe_threads_for(n_threads, n);
    if (threads <= 1)
    {
        for (uint64_t i = 0; i < n; i++)
            body(ctx, i);
        return;
    }
    ParallelLoop loop = {body, ctx, n, 0};
    pthread_t *helpers = (pthread_t *)safe_malloc((threads - 1) * sizeof(pthread_t));
    uint64_t started = 0;
    // If a thread cannot be created, the others (and the caller) pick up its
    // share.
    while (started < threads - 1 &&
           pthread_create(&helpers[started], NULL, run_loop_thread, &loop) == 0)
        started++;
    run_loop(&loop);
    for (uint64_t t = 0; t < started; t++)
        pthread_join(helpers[t], NULL);
    free(helpers);
}

static uint64_t default_thread_limit(void)
{
    const char *env = getenv("VFHE_NUM_THREADS");
    if (env != NULL && *env != '\0')
    {
        char *end;
        const unsigned long long value = strtoull(env, &end, 10);
        if (*end == '\0' && value > 0)
            return (uint64_t)value;
    }
    return 1;
}

static void run_loop(ParallelLoop *loop)
{
    inside_parallel_loop = 1;
    for (;;)
    {
        const uint64_t i = (uint64_t)atomic_fetch_add(&loop->next, 1);
        if (i >= loop->n)
            break;
        loop->body(loop->ctx, i);
    }
    inside_parallel_loop = 0;
}

static void *run_loop_thread(void *loop)
{
    run_loop((ParallelLoop *)loop);
    return NULL;
}
