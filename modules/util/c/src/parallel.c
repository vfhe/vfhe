// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "util.h"

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// 0 until first read or after a reset: the default is resolved lazily, so the
// environment is read on first use rather than at load.
static atomic_uint_fast64_t thread_limit = 0;

// The calling thread's local number of threads, 0 for none set. A loop sets
// it for the threads running its bodies to their share of the loop's threads,
// so a loop started inside a body divides that share instead of multiplying
// threads.
static _Thread_local uint64_t local_num_threads = 0;

// The smallest share of a loop's threads that its bodies are lent.
#define MIN_LENT_SHARE 4

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

uint64_t vfhe_set_local_num_threads(uint64_t n)
{
    const uint64_t previous = local_num_threads;
    local_num_threads = n;
    return previous;
}

uint64_t vfhe_local_num_threads(void) { return local_num_threads; }

// The threads a loop asking for `requested` may use, before its item count.
static uint64_t thread_budget(uint64_t requested)
{
    uint64_t n = vfhe_num_threads();
    if (local_num_threads != 0 && local_num_threads < n)
        n = local_num_threads;
    return (requested == 0 || requested > n) ? n : requested;
}

uint64_t vfhe_threads_for(uint64_t requested, uint64_t n_items)
{
    if (n_items <= 1)
        return 1;
    const uint64_t n = thread_budget(requested);
    return n < n_items ? n : n_items;
}

typedef struct Worker Worker;

typedef struct ParallelLoop
{
    void (*body)(void *ctx, uint64_t i);
    void *ctx;
    uint64_t n;
    uint64_t share; // the local number of threads of every thread running bodies
    atomic_uint_fast64_t next;
    // The threads that have not yet retired from the loop: the caller and
    // every helper. The last to retire sets `done`, under `done_lock`; the
    // caller waits for it before the loop goes away.
    atomic_uint_fast64_t running;
    pthread_mutex_t done_lock;
    pthread_cond_t done_cond;
    int done;
    // The workers the caller gave the loop to. They wake as a binary tree:
    // the caller wakes handed[0], and handed[k] wakes handed[2k+1] and
    // handed[2k+2] as it starts.
    Worker **handed;
    uint64_t n_handed;
    // Guarded by pool.open_lock. A loop that was given fewer workers than it
    // wants is open: a worker done with another loop may join it.
    uint64_t helpers_wanted;
    struct ParallelLoop *next_open;
} ParallelLoop;

// Workers are created on first need and then kept. An idle worker sleeps on
// its own lock and condvar until a caller puts a loop in its slot, so a
// hand-off involves the caller and the workers it gives the loop to, and no
// lock that every loop takes.
struct Worker
{
    _Alignas(64) pthread_mutex_t lock;
    pthread_cond_t wake;
    _Atomic(ParallelLoop *) loop; // given, not yet started
    uint64_t wake_index;          // its place in loop->handed
    _Atomic uint32_t next_idle;   // the next idle worker's index + 1, 0 for none
    uint32_t index;
};

// Workers live in chunks that never move, chunk c holding the 2^c workers
// from index 2^c - 1 on, so that an index finds its worker without a lock.
#define N_CHUNKS 32

static struct
{
    Worker *_Atomic chunks[N_CHUNKS];
    // The idle workers, most recently idle first: a stack whose head holds
    // the top worker's index + 1 (0 when empty) in its low half and a count
    // of changes in its high half. The count fails a pop whose view of the
    // top's successor has gone stale, even if the same worker is back on top.
    _Atomic uint64_t idle;
    atomic_uint_fast64_t n_workers;
    pthread_mutex_t grow_lock; // held while creating workers
    pthread_mutex_t open_lock;
    ParallelLoop *open; // guarded by open_lock
    atomic_uint_fast64_t n_open;
} pool = {{NULL}, 0, 0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, NULL, 0};

static Worker *worker_at(uint32_t index)
{
    const unsigned c = 31 - (unsigned)__builtin_clz(index + 1);
    return &atomic_load(&pool.chunks[c])[index + 1 - (1u << c)];
}

static void push_idle(Worker *w)
{
    uint64_t head = atomic_load(&pool.idle);
    do
        atomic_store_explicit(&w->next_idle, (uint32_t)head, memory_order_relaxed);
    while (!atomic_compare_exchange_weak(&pool.idle, &head,
                                         ((head >> 32) + 1) << 32 | (w->index + 1)));
}

static Worker *pop_idle(void)
{
    uint64_t head = atomic_load(&pool.idle);
    Worker *w;
    do
    {
        if ((uint32_t)head == 0)
            return NULL;
        w = worker_at((uint32_t)head - 1);
    } while (!atomic_compare_exchange_weak(
        &pool.idle, &head,
        ((head >> 32) + 1) << 32 | atomic_load_explicit(&w->next_idle, memory_order_relaxed)));
    return w;
}

// Called after setting the worker's slot. Taking its lock orders the signal
// after the worker has either seen the slot or started to wait.
static void wake(Worker *w)
{
    pthread_mutex_lock(&w->lock);
    pthread_mutex_unlock(&w->lock);
    pthread_cond_signal(&w->wake);
}

static void retire(ParallelLoop *loop)
{
    if (atomic_fetch_sub(&loop->running, 1) != 1)
        return;
    pthread_mutex_lock(&loop->done_lock);
    loop->done = 1;
    pthread_cond_signal(&loop->done_cond);
    pthread_mutex_unlock(&loop->done_lock);
}

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

// An open loop with items left, joined: the caller has not retired from an
// open loop, so it cannot finish before the joiner retires.
static ParallelLoop *loop_to_join(void)
{
    if (atomic_load(&pool.n_open) == 0)
        return NULL;
    pthread_mutex_lock(&pool.open_lock);
    ParallelLoop *loop = pool.open;
    while (loop != NULL && (loop->helpers_wanted == 0 || atomic_load(&loop->next) >= loop->n))
        loop = loop->next_open;
    if (loop != NULL)
    {
        loop->helpers_wanted--;
        atomic_fetch_add(&loop->running, 1);
    }
    pthread_mutex_unlock(&pool.open_lock);
    return loop;
}

static void *worker_main(void *arg)
{
    Worker *self = (Worker *)arg;
    for (;;)
    {
        ParallelLoop *loop;
        pthread_mutex_lock(&self->lock);
        while ((loop = atomic_load(&self->loop)) == NULL)
            pthread_cond_wait(&self->wake, &self->lock);
        pthread_mutex_unlock(&self->lock);
        // Its caller may have taken this worker back meanwhile.
        if (!atomic_compare_exchange_strong(&self->loop, &loop, NULL))
            continue;
        // `handed` stays valid: the caller waits for this worker before its
        // loop goes away. A child the caller has taken back meanwhile finds no
        // loop and sleeps again.
        for (uint64_t c = 2 * self->wake_index + 1; c <= 2 * self->wake_index + 2; c++)
            if (c < loop->n_handed)
                wake(loop->handed[c]);
        for (;;)
        {
            local_num_threads = loop->share;
            draw_items(loop);
            // Idle before retiring, so that the caller's next loop finds this
            // worker, the most recently idle, on top. It may be given that
            // loop before it retires from this one; it starts it after.
            ParallelLoop *joined = loop_to_join();
            if (joined == NULL)
                push_idle(self);
            retire(loop);
            if (joined == NULL)
                break;
            loop = joined;
        }
    }
    return NULL;
}

// A fork copies only the forking thread: the child starts with no workers and
// no open loops, and creates workers again on first need.
static void before_fork(void)
{
    pthread_mutex_lock(&pool.grow_lock);
    pthread_mutex_lock(&pool.open_lock);
}
static void after_fork_in_parent(void)
{
    pthread_mutex_unlock(&pool.open_lock);
    pthread_mutex_unlock(&pool.grow_lock);
}
static void after_fork_in_child(void)
{
    pthread_mutex_unlock(&pool.open_lock);
    pthread_mutex_unlock(&pool.grow_lock);
    for (unsigned c = 0; c < N_CHUNKS; c++)
    {
        free(atomic_load(&pool.chunks[c]));
        atomic_store(&pool.chunks[c], NULL);
    }
    atomic_store(&pool.idle, 0);
    atomic_store(&pool.n_workers, 0);
    pool.open = NULL;
    atomic_store(&pool.n_open, 0);
}

static void register_fork_handlers(void)
{
    pthread_atfork(before_fork, after_fork_in_parent, after_fork_in_child);
}

// New workers start idle. Workers block every signal, so signals reach the
// application's own threads. A worker that cannot be created is left out: the
// loops that wanted it run on fewer threads.
static void grow_pool(uint64_t n_workers)
{
    static pthread_once_t fork_handlers_once = PTHREAD_ONCE_INIT;
    if (atomic_load(&pool.n_workers) >= n_workers)
        return;
    pthread_mutex_lock(&pool.grow_lock);
    pthread_once(&fork_handlers_once, register_fork_handlers);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    sigset_t all, caller_mask;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &caller_mask);
    for (uint64_t count = atomic_load(&pool.n_workers); count < n_workers && count < UINT32_MAX - 1;
         count++)
    {
        const uint32_t index = (uint32_t)count;
        const unsigned c = 31 - (unsigned)__builtin_clz(index + 1);
        if (atomic_load(&pool.chunks[c]) == NULL)
        {
            Worker *chunk = (Worker *)aligned_alloc(_Alignof(Worker), sizeof(Worker) << c);
            if (chunk == NULL)
                break;
            memset(chunk, 0, sizeof(Worker) << c);
            atomic_store(&pool.chunks[c], chunk);
        }
        Worker *w = worker_at(index);
        w->index = index;
        pthread_mutex_init(&w->lock, NULL);
        pthread_cond_init(&w->wake, NULL);
        pthread_t thread;
        if (pthread_create(&thread, &attr, worker_main, w) != 0)
        {
            pthread_cond_destroy(&w->wake);
            pthread_mutex_destroy(&w->lock);
            break;
        }
        push_idle(w);
        atomic_store(&pool.n_workers, count + 1);
    }
    pthread_sigmask(SIG_SETMASK, &caller_mask, NULL);
    pthread_attr_destroy(&attr);
    pthread_mutex_unlock(&pool.grow_lock);
}

void vfhe_parallel_for(uint64_t n, uint64_t n_threads, void (*body)(void *ctx, uint64_t i),
                       void *ctx)
{
    // The loop's threads split its budget: each body may use budget / threads,
    // so a loop over few items lends the rest to loops inside its bodies. A
    // share below MIN_LENT_SHARE is not lent: so few threads on one body cost
    // more in hand-offs than they return.
    const uint64_t budget = thread_budget(n_threads);
    const uint64_t threads = n <= 1 ? 1 : (budget < n ? budget : n);
    const uint64_t caller_local = local_num_threads;
    if (threads <= 1)
    {
        local_num_threads = budget;
        for (uint64_t i = 0; i < n; i++)
            body(ctx, i);
        local_num_threads = caller_local;
        return;
    }
    const uint64_t share = budget / threads >= MIN_LENT_SHARE ? budget / threads : 1;
    ParallelLoop loop = {body, ctx, n, share, 0, 0};
    pthread_mutex_init(&loop.done_lock, NULL);
    pthread_cond_init(&loop.done_cond, NULL);
    grow_pool(threads - 1);
    const uint64_t n_workers = atomic_load(&pool.n_workers);
    const uint64_t most_handed = threads - 1 < n_workers ? threads - 1 : n_workers;
    Worker *handed[most_handed > 0 ? most_handed : 1];
    loop.handed = handed;
    // All are picked before any is given the loop, so that a worker starting
    // at once finds `handed` complete.
    while (loop.n_handed < most_handed)
    {
        Worker *w = pop_idle();
        if (w == NULL)
            break;
        w->wake_index = loop.n_handed;
        handed[loop.n_handed++] = w;
    }
    atomic_store(&loop.running, loop.n_handed + 1);
    for (uint64_t k = 0; k < loop.n_handed; k++)
        atomic_store(&handed[k]->loop, &loop);
    const int open = loop.n_handed < threads - 1;
    if (open)
    {
        pthread_mutex_lock(&pool.open_lock);
        loop.helpers_wanted = threads - 1 - loop.n_handed;
        loop.next_open = pool.open;
        pool.open = &loop;
        atomic_fetch_add(&pool.n_open, 1);
        pthread_mutex_unlock(&pool.open_lock);
    }
    if (loop.n_handed > 0)
        wake(handed[0]);

    local_num_threads = loop.share;
    draw_items(&loop);
    local_num_threads = caller_local;

    // Every item has been drawn, so no worker joins from here on. The ones
    // given the loop that have not started are taken back rather than waited
    // for; then wait for the ones still running their items.
    if (open)
    {
        pthread_mutex_lock(&pool.open_lock);
        ParallelLoop **link = &pool.open;
        while (*link != &loop)
            link = &(*link)->next_open;
        *link = loop.next_open;
        atomic_fetch_sub(&pool.n_open, 1);
        pthread_mutex_unlock(&pool.open_lock);
    }
    for (uint64_t k = 0; k < loop.n_handed; k++)
    {
        ParallelLoop *given = &loop;
        if (atomic_compare_exchange_strong(&handed[k]->loop, &given, NULL))
        {
            push_idle(handed[k]);
            retire(&loop);
        }
    }
    retire(&loop);
    pthread_mutex_lock(&loop.done_lock);
    while (!loop.done)
        pthread_cond_wait(&loop.done_cond, &loop.done_lock);
    pthread_mutex_unlock(&loop.done_lock);
    pthread_cond_destroy(&loop.done_cond);
    pthread_mutex_destroy(&loop.done_lock);
}
