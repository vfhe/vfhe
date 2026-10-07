// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "util.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SANITIZE_ADDRESS__)
#define MEMPOOL_BYPASS 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MEMPOOL_BYPASS 1
#endif
#endif
#ifndef MEMPOOL_BYPASS
#define MEMPOOL_BYPASS 0
#endif

// Smaller buffers go straight to the C library, whose small-chunk caches
// already recycle them.
#define MEMPOOL_MIN_BYTES 4096u
// Distinct buffer sizes one thread retains at a time.
#define MEMPOOL_BUCKETS 32

// A retained buffer's first word links it to the next one of its size.
typedef struct FreeBuffer
{
    struct FreeBuffer *next;
} FreeBuffer;

typedef struct
{
    size_t bytes; // 0: unused
    FreeBuffer *head;
    uint64_t last_use;
} Bucket;

// One per thread that has used the pool. Its owner is the only thread that
// takes the lock in normal use; release_all, statistics and fork take it from
// outside.
typedef struct ThreadStorage
{
    pthread_mutex_t lock;
    Bucket buckets[MEMPOOL_BUCKETS];
    uint64_t clock;
    uint64_t hits, misses;
    struct ThreadStorage *next;
} ThreadStorage;

// Lock order: registry.lock before any storage's lock.
static struct
{
    pthread_mutex_t lock;
    ThreadStorage *threads;
    // Counts of threads that have exited, so the statistics are cumulative.
    uint64_t exited_hits, exited_misses;
} registry = {PTHREAD_MUTEX_INITIALIZER, NULL, 0, 0};

// Bytes sitting in free lists, over all threads; the capacity bounds it.
static atomic_uint_fast64_t retained = 0;
// Bytes handed out and not yet returned through mempool_free, and its highest
// value so far, which the automatic capacity follows.
static atomic_int_fast64_t outstanding = 0;
static atomic_uint_fast64_t peak_outstanding = 0;
// MEMPOOL_CAPACITY_DEFAULT until first read or after a reset: the environment
// is read on first use rather than at load.
static atomic_uint_fast64_t capacity_setting = MEMPOOL_CAPACITY_DEFAULT;

static _Thread_local ThreadStorage *own_storage = NULL;
static pthread_key_t thread_exit_key;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
// The MALLOC_PERTURB_ byte, 0 when unset: like the C library, the pool fills
// a buffer with its complement when handing it out and with the byte itself
// when taking it back, so reads of memory nobody wrote show up in tests.
static int perturb_byte = 0;

// Zeroes in a way the compiler may not drop, even though the buffer may be
// freed right after.
static void wipe(void *ptr, size_t bytes)
{
    memset(ptr, 0, bytes);
    __asm__ __volatile__("" : : "r"(ptr) : "memory");
}

static uint64_t release_storage(ThreadStorage *s)
{
    uint64_t released = 0;
    for (int k = 0; k < MEMPOOL_BUCKETS; k++)
    {
        Bucket *b = &s->buckets[k];
        while (b->head != NULL)
        {
            FreeBuffer *buffer = b->head;
            b->head = buffer->next;
            free(buffer);
            released += b->bytes;
        }
    }
    atomic_fetch_sub(&retained, released);
    return released;
}

static void unlink_storage(ThreadStorage *s)
{
    ThreadStorage **link = &registry.threads;
    while (*link != NULL && *link != s)
        link = &(*link)->next;
    if (*link == s)
        *link = s->next;
}

// A thread's storage outlives nothing: what it retained goes back to the C
// library when the thread exits.
static void on_thread_exit(void *arg)
{
    ThreadStorage *s = (ThreadStorage *)arg;
    pthread_mutex_lock(&registry.lock);
    unlink_storage(s);
    registry.exited_hits += s->hits;
    registry.exited_misses += s->misses;
    pthread_mutex_unlock(&registry.lock);
    release_storage(s);
    pthread_mutex_destroy(&s->lock);
    free(s);
    own_storage = NULL;
}

// A fork copies only the forking thread. Every storage is locked across it,
// so the child finds each free list whole; there, the other threads' storage
// is released and dropped, as their exit destructors will never run.
static void before_fork(void)
{
    pthread_mutex_lock(&registry.lock);
    for (ThreadStorage *s = registry.threads; s != NULL; s = s->next)
        pthread_mutex_lock(&s->lock);
}

static void after_fork_in_parent(void)
{
    for (ThreadStorage *s = registry.threads; s != NULL; s = s->next)
        pthread_mutex_unlock(&s->lock);
    pthread_mutex_unlock(&registry.lock);
}

static void after_fork_in_child(void)
{
    ThreadStorage *s = registry.threads;
    registry.threads = NULL;
    while (s != NULL)
    {
        ThreadStorage *next = s->next;
        if (s == own_storage)
        {
            pthread_mutex_init(&s->lock, NULL);
            s->next = registry.threads;
            registry.threads = s;
        }
        else
        {
            registry.exited_hits += s->hits;
            registry.exited_misses += s->misses;
            release_storage(s);
            free(s);
        }
        s = next;
    }
    pthread_mutex_init(&registry.lock, NULL);
}

static void init_pool(void)
{
    pthread_key_create(&thread_exit_key, on_thread_exit);
    pthread_atfork(before_fork, after_fork_in_parent, after_fork_in_child);
    const char *env = getenv("MALLOC_PERTURB_");
    if (env != NULL && *env != '\0')
        perturb_byte = atoi(env) & 0xff;
}

static ThreadStorage *this_thread_storage(void)
{
    if (own_storage != NULL)
        return own_storage;
    pthread_once(&init_once, init_pool);
    ThreadStorage *s = (ThreadStorage *)safe_malloc(sizeof(*s));
    memset(s, 0, sizeof(*s));
    pthread_mutex_init(&s->lock, NULL);
    pthread_mutex_lock(&registry.lock);
    s->next = registry.threads;
    registry.threads = s;
    pthread_mutex_unlock(&registry.lock);
    pthread_setspecific(thread_exit_key, s);
    own_storage = s;
    return s;
}

static uint64_t default_capacity(void)
{
    const char *env = getenv("VFHE_MEMPOOL_CAPACITY");
    if (env != NULL && *env != '\0')
    {
        char *end;
        const unsigned long long value = strtoull(env, &end, 10);
        if (*end == '\0' && value < MEMPOOL_CAPACITY_DEFAULT)
            return (uint64_t)value;
    }
    return MEMPOOL_CAPACITY_AUTO;
}

uint64_t mempool_capacity(void)
{
    uint_fast64_t setting = atomic_load(&capacity_setting);
    if (setting == MEMPOOL_CAPACITY_DEFAULT)
    {
        uint_fast64_t unset = MEMPOOL_CAPACITY_DEFAULT;
        atomic_compare_exchange_strong(&capacity_setting, &unset, default_capacity());
        setting = atomic_load(&capacity_setting);
    }
    return (uint64_t)setting;
}

// The most the pool may retain right now.
static uint64_t retention_limit(void)
{
    const uint64_t setting = mempool_capacity();
    return setting == MEMPOOL_CAPACITY_AUTO ? (uint64_t)atomic_load(&peak_outstanding) : setting;
}

void mempool_set_capacity(uint64_t bytes)
{
    atomic_store(&capacity_setting, bytes);
    if (atomic_load(&retained) > retention_limit())
        mempool_release_all();
}

// Reserves room for `bytes` more under the limit; false when there is none.
static int reserve_retention(size_t bytes)
{
    const uint64_t limit = retention_limit();
    uint_fast64_t current = atomic_load(&retained);
    do
    {
        if (current + bytes > limit)
            return 0;
    } while (!atomic_compare_exchange_weak(&retained, &current, current + bytes));
    return 1;
}

static void note_handed_out(size_t bytes)
{
    const int_fast64_t now =
        atomic_fetch_add(&outstanding, (int_fast64_t)bytes) + (int_fast64_t)bytes;
    if (now <= 0)
        return;
    uint_fast64_t peak = atomic_load(&peak_outstanding);
    while ((uint_fast64_t)now > peak &&
           !atomic_compare_exchange_weak(&peak_outstanding, &peak, (uint_fast64_t)now))
        ;
}

static Bucket *find_bucket(ThreadStorage *s, size_t bytes)
{
    for (int k = 0; k < MEMPOOL_BUCKETS; k++)
        if (s->buckets[k].bytes == bytes)
            return &s->buckets[k];
    return NULL;
}

// A bucket for a size this thread holds none of: an unused or empty one if
// there is one, otherwise the least recently used, emptied first.
static Bucket *claim_bucket(ThreadStorage *s, size_t bytes)
{
    Bucket *victim = &s->buckets[0];
    for (int k = 0; k < MEMPOOL_BUCKETS; k++)
    {
        Bucket *b = &s->buckets[k];
        if (b->bytes == 0 || b->head == NULL)
        {
            victim = b;
            break;
        }
        if (b->last_use < victim->last_use)
            victim = b;
    }
    uint64_t released = 0;
    while (victim->head != NULL)
    {
        FreeBuffer *buffer = victim->head;
        victim->head = buffer->next;
        free(buffer);
        released += victim->bytes;
    }
    atomic_fetch_sub(&retained, released);
    victim->bytes = bytes;
    return victim;
}

void *mempool_aligned_malloc(size_t bytes)
{
    if (MEMPOOL_BYPASS || bytes < MEMPOOL_MIN_BYTES)
        return safe_aligned_malloc(bytes);
    ThreadStorage *s = this_thread_storage();
    FreeBuffer *buffer = NULL;
    pthread_mutex_lock(&s->lock);
    Bucket *b = find_bucket(s, bytes);
    if (b != NULL && b->head != NULL)
    {
        buffer = b->head;
        b->head = buffer->next;
        b->last_use = ++s->clock;
        s->hits++;
    }
    else
        s->misses++;
    pthread_mutex_unlock(&s->lock);
    note_handed_out(bytes);
    if (buffer == NULL)
        return safe_aligned_malloc(bytes);
    atomic_fetch_sub(&retained, bytes);
    if (perturb_byte)
        memset(buffer, perturb_byte ^ 0xff, bytes);
    return buffer;
}

void mempool_free(void *ptr, size_t bytes)
{
    if (ptr == NULL)
        return;
    if (MEMPOOL_BYPASS || bytes < MEMPOOL_MIN_BYTES)
    {
        free(ptr);
        return;
    }
    atomic_fetch_sub(&outstanding, (int_fast64_t)bytes);
    if (!reserve_retention(bytes))
    {
        free(ptr);
        return;
    }
    if (perturb_byte)
        memset(ptr, perturb_byte, bytes);
    ThreadStorage *s = this_thread_storage();
    pthread_mutex_lock(&s->lock);
    Bucket *b = find_bucket(s, bytes);
    if (b == NULL)
        b = claim_bucket(s, bytes);
    FreeBuffer *buffer = (FreeBuffer *)ptr;
    buffer->next = b->head;
    b->head = buffer;
    b->last_use = ++s->clock;
    pthread_mutex_unlock(&s->lock);
}

void mempool_free_and_wipe(void *ptr, size_t bytes)
{
    if (ptr == NULL)
        return;
    wipe(ptr, bytes);
    mempool_free(ptr, bytes);
}

void mempool_release_all(void)
{
    pthread_mutex_lock(&registry.lock);
    for (ThreadStorage *s = registry.threads; s != NULL; s = s->next)
    {
        pthread_mutex_lock(&s->lock);
        release_storage(s);
        pthread_mutex_unlock(&s->lock);
    }
    pthread_mutex_unlock(&registry.lock);
}

void mempool_statistics(MempoolStatistics *out)
{
    memset(out, 0, sizeof(*out));
    pthread_mutex_lock(&registry.lock);
    out->hits = registry.exited_hits;
    out->misses = registry.exited_misses;
    for (ThreadStorage *s = registry.threads; s != NULL; s = s->next)
    {
        pthread_mutex_lock(&s->lock);
        out->hits += s->hits;
        out->misses += s->misses;
        pthread_mutex_unlock(&s->lock);
    }
    pthread_mutex_unlock(&registry.lock);
    const int_fast64_t now = atomic_load(&outstanding);
    out->retained_bytes = (uint64_t)atomic_load(&retained);
    out->outstanding_bytes = now > 0 ? (uint64_t)now : 0;
    out->peak_outstanding_bytes = (uint64_t)atomic_load(&peak_outstanding);
    out->retention_limit_bytes = retention_limit();
}
