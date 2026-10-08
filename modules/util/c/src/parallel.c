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

typedef struct Worker Worker;

typedef struct ParallelLoop
{
    void (*body)(void *ctx, uint64_t i);
    void *ctx;
    uint64_t n;
    atomic_uint_fast64_t next;
    // Guarded by pool.lock.
    uint64_t helpers_wanted;  // further helpers that may still join
    uint64_t helpers_running; // given the loop and not yet back
    // The workers the caller gave the loop to. They wake as a binary tree:
    // the caller wakes handed[0], and handed[k] wakes handed[2k+1] and
    // handed[2k+2] as it starts.
    Worker **handed;
    uint64_t n_handed;
    struct ParallelLoop *next_open;
} ParallelLoop;

// Workers are created on first need and then kept. An idle worker sleeps on
// its own condvar, so a loop wakes exactly the workers it gives itself to,
// the most recently idle first. Open loops are listed so that a worker
// finishing one loop joins another caller's loop that still wants helpers.
struct Worker
{
    pthread_cond_t wake;
    ParallelLoop *loop;  // given, not yet started; guarded by pool.lock
    uint64_t wake_index; // its place in loop->handed
    Worker *next_idle;
    Worker *next_all;
};

static struct
{
    pthread_mutex_t lock;
    pthread_cond_t helper_left; // callers wait here for their helpers
    Worker *idle;               // most recently idle first
    Worker *all;
    ParallelLoop *open;
    uint64_t n_workers;
} pool = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, NULL, NULL, NULL, 0};

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

static void *worker_main(void *arg)
{
    Worker *self = (Worker *)arg;
    inside_parallel_loop = 1;
    pthread_mutex_lock(&pool.lock);
    for (;;)
    {
        while (self->loop == NULL)
            pthread_cond_wait(&self->wake, &pool.lock);
        ParallelLoop *loop = self->loop;
        self->loop = NULL;
        pthread_mutex_unlock(&pool.lock);
        // `handed` stays valid: the caller waits for this worker before its
        // loop goes away. A child the caller has taken back meanwhile finds no
        // loop and sleeps again.
        for (uint64_t c = 2 * self->wake_index + 1; c <= 2 * self->wake_index + 2; c++)
            if (c < loop->n_handed)
                pthread_cond_signal(&loop->handed[c]->wake);
        for (;;)
        {
            draw_items(loop);
            pthread_mutex_lock(&pool.lock);
            if (--loop->helpers_running == 0)
                pthread_cond_broadcast(&pool.helper_left);
            loop = loop_to_join();
            if (loop == NULL)
                break;
            loop->helpers_wanted--;
            loop->helpers_running++;
            pthread_mutex_unlock(&pool.lock);
        }
        self->next_idle = pool.idle;
        pool.idle = self;
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
    pthread_cond_init(&pool.helper_left, NULL);
    for (Worker *w = pool.all, *next; w != NULL; w = next)
    {
        next = w->next_all;
        free(w);
    }
    pool.idle = NULL;
    pool.all = NULL;
    pool.open = NULL;
    pool.n_workers = 0;
}

static void register_fork_handlers(void)
{
    pthread_atfork(before_fork, after_fork_in_parent, after_fork_in_child);
}

// Called with pool.lock held. New workers start idle. Workers block every
// signal, so signals reach the application's own threads. A worker that cannot
// be created is left out: the loops that wanted it run on fewer threads.
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
    while (pool.n_workers < n_workers)
    {
        Worker *w = (Worker *)calloc(1, sizeof(Worker));
        if (w == NULL)
            break;
        pthread_cond_init(&w->wake, NULL);
        pthread_t thread;
        if (pthread_create(&thread, &attr, worker_main, w) != 0)
        {
            pthread_cond_destroy(&w->wake);
            free(w);
            break;
        }
        w->next_all = pool.all;
        pool.all = w;
        w->next_idle = pool.idle;
        pool.idle = w;
        pool.n_workers++;
    }
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
    ParallelLoop loop = {body, ctx, n, 0, threads - 1, 0, NULL, 0, NULL};
    pthread_mutex_lock(&pool.lock);
    grow_pool(threads - 1);
    const uint64_t most_handed = threads - 1 < pool.n_workers ? threads - 1 : pool.n_workers;
    Worker *handed[most_handed > 0 ? most_handed : 1];
    loop.handed = handed;
    while (loop.n_handed < most_handed && pool.idle != NULL)
    {
        Worker *w = pool.idle;
        pool.idle = w->next_idle;
        w->loop = &loop;
        w->wake_index = loop.n_handed;
        handed[loop.n_handed++] = w;
        loop.helpers_wanted--;
        loop.helpers_running++;
    }
    loop.next_open = pool.open;
    pool.open = &loop;
    pthread_mutex_unlock(&pool.lock);
    // Woken after unlocking, so the worker does not block on the lock.
    if (loop.n_handed > 0)
        pthread_cond_signal(&handed[0]->wake);

    inside_parallel_loop = 1;
    draw_items(&loop);
    inside_parallel_loop = 0;

    // Every item has been drawn, so no worker joins from here on. The ones
    // given the loop that have not started are taken back rather than waited
    // for; then wait for the ones still running their items.
    pthread_mutex_lock(&pool.lock);
    for (uint64_t k = 0; k < loop.n_handed; k++)
    {
        Worker *w = handed[k];
        if (w->loop == &loop)
        {
            w->loop = NULL;
            loop.helpers_running--;
            w->next_idle = pool.idle;
            pool.idle = w;
        }
    }
    while (loop.helpers_running > 0)
        pthread_cond_wait(&pool.helper_left, &pool.lock);
    ParallelLoop **link = &pool.open;
    while (*link != &loop)
        link = &(*link)->next_open;
    *link = loop.next_open;
    pthread_mutex_unlock(&pool.lock);
}
