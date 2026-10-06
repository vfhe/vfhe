// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "util.h"

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>

// 0 until first read or after a reset: the default is resolved lazily, so the
// environment is read on first use rather than at load.
static atomic_uint_fast64_t thread_limit = 0;

// Set while a thread (the caller's included) runs loop bodies, so a loop
// started inside a body runs serially instead of spawning more threads.
static _Thread_local int inside_parallel_loop = 0;

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

typedef struct ParallelLoop
{
    void (*body)(void *ctx, uint64_t i);
    void *ctx;
    uint64_t n;
    atomic_uint_fast64_t next;
    // Both guarded by pool.lock.
    uint64_t helpers_wanted; // workers that may still join
    uint64_t helpers_running;
    struct ParallelLoop *next_open;
} ParallelLoop;

// Workers are created on first need and then kept, asleep on work_posted
// between loops. Open loops are listed so that loops started concurrently by
// different callers share the workers.
static struct
{
    pthread_mutex_t lock;
    pthread_cond_t work_posted; // workers wait here for a loop to join
    pthread_cond_t helper_left; // callers wait here for their helpers
    ParallelLoop *open;
    uint64_t n_workers;
} pool = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, 0};

// Items are handed out one at a time, which balances uneven item costs.
static void draw_items(ParallelLoop *loop)
{
    for (;;)
    {
        const uint64_t i = (uint64_t)atomic_fetch_add(&loop->next, 1);
        if (i >= loop->n)
            break;
        loop->body(loop->ctx, i);
    }
}

static ParallelLoop *loop_to_join(void)
{
    for (ParallelLoop *loop = pool.open; loop != NULL; loop = loop->next_open)
        if (loop->helpers_wanted > 0 && atomic_load(&loop->next) < loop->n)
            return loop;
    return NULL;
}

static void *worker_main(void *unused)
{
    (void)unused;
    inside_parallel_loop = 1;
    pthread_mutex_lock(&pool.lock);
    for (;;)
    {
        ParallelLoop *loop = loop_to_join();
        if (loop == NULL)
        {
            pthread_cond_wait(&pool.work_posted, &pool.lock);
            continue;
        }
        loop->helpers_wanted--;
        loop->helpers_running++;
        pthread_mutex_unlock(&pool.lock);
        draw_items(loop);
        pthread_mutex_lock(&pool.lock);
        if (--loop->helpers_running == 0)
            pthread_cond_broadcast(&pool.helper_left);
    }
    return NULL;
}

// A fork copies only the forking thread: the child starts with no workers and
// no open loops, and creates workers again on first need.
static void before_fork(void) { pthread_mutex_lock(&pool.lock); }
static void after_fork_in_parent(void) { pthread_mutex_unlock(&pool.lock); }
static void after_fork_in_child(void)
{
    pthread_mutex_unlock(&pool.lock);
    pthread_cond_init(&pool.work_posted, NULL);
    pthread_cond_init(&pool.helper_left, NULL);
    pool.open = NULL;
    pool.n_workers = 0;
}

static void register_fork_handlers(void)
{
    pthread_atfork(before_fork, after_fork_in_parent, after_fork_in_child);
}

// Called with pool.lock held. Workers block every signal, so signals reach the
// application's own threads. A worker that cannot be created is left out: the
// loops that wanted it run on fewer threads.
static void grow_pool(uint64_t n_workers)
{
    static pthread_once_t fork_handlers_once = PTHREAD_ONCE_INIT;
    if (pool.n_workers >= n_workers)
        return;
    pthread_once(&fork_handlers_once, register_fork_handlers);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    sigset_t all, caller_mask;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &caller_mask);
    pthread_t worker;
    while (pool.n_workers < n_workers && pthread_create(&worker, &attr, worker_main, NULL) == 0)
        pool.n_workers++;
    pthread_sigmask(SIG_SETMASK, &caller_mask, NULL);
    pthread_attr_destroy(&attr);
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
    ParallelLoop loop = {body, ctx, n, 0, threads - 1, 0, NULL};
    pthread_mutex_lock(&pool.lock);
    grow_pool(threads - 1);
    loop.next_open = pool.open;
    pool.open = &loop;
    pthread_mutex_unlock(&pool.lock);
    // Woken after unlocking, so a woken worker does not block on the lock. All
    // of them: the ones that do not find a place go back to sleep.
    pthread_cond_broadcast(&pool.work_posted);

    inside_parallel_loop = 1;
    draw_items(&loop);
    inside_parallel_loop = 0;

    // Every item has been drawn, so no worker joins from here on; wait for the
    // ones still running theirs.
    pthread_mutex_lock(&pool.lock);
    while (loop.helpers_running > 0)
        pthread_cond_wait(&pool.helper_left, &pool.lock);
    ParallelLoop **link = &pool.open;
    while (*link != &loop)
        link = &(*link)->next_open;
    *link = loop.next_open;
    pthread_mutex_unlock(&pool.lock);
}
