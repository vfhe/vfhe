// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// The memory pool's contract: aligned buffers, reuse of a released buffer for
// the next request of its size, retention bounded by the capacity, release
// reaching every thread, buffers moving between threads, no buffer handed to
// two holders at once, wiping, thread exit, and fork.

#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unity.h>
#include <util.h>

#if defined(__SANITIZE_ADDRESS__)
#define BYPASSED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define BYPASSED 1
#endif
#endif
#ifndef BYPASSED
#define BYPASSED 0
#endif

#define SIZE (64u * 1024u)

static MempoolStatistics stats(void)
{
    MempoolStatistics s;
    mempool_statistics(&s);
    return s;
}

void setUp(void)
{
    mempool_set_capacity(MEMPOOL_CAPACITY_AUTO);
    mempool_release_all();
    if (BYPASSED)
        TEST_IGNORE_MESSAGE("a sanitized build bypasses the pool");
}
void tearDown(void)
{
    mempool_set_capacity(MEMPOOL_CAPACITY_DEFAULT);
    mempool_release_all();
}

static void buffers_are_aligned_and_reused(void)
{
    void *first = mempool_aligned_malloc(SIZE);
    TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)((uintptr_t)first % 64));
    memset(first, 0xA5, SIZE);
    mempool_free(first, SIZE);
    TEST_ASSERT_EQUAL_UINT64(SIZE, stats().retained_bytes);

    const uint64_t hits = stats().hits;
    void *second = mempool_aligned_malloc(SIZE);
    TEST_ASSERT_EQUAL_PTR(first, second);
    TEST_ASSERT_EQUAL_UINT64(hits + 1, stats().hits);
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
    mempool_free(second, SIZE);
}

static void a_different_size_is_not_served_from_another_size(void)
{
    void *a = mempool_aligned_malloc(SIZE);
    mempool_free(a, SIZE);
    const uint64_t misses = stats().misses;
    void *b = mempool_aligned_malloc(2 * SIZE);
    TEST_ASSERT_EQUAL_UINT64(misses + 1, stats().misses);
    TEST_ASSERT_EQUAL_UINT64(SIZE, stats().retained_bytes);
    mempool_free(b, 2 * SIZE);
}

static void small_buffers_go_to_the_c_library(void)
{
    void *p = mempool_aligned_malloc(256);
    TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)((uintptr_t)p % 64));
    mempool_free(p, 256);
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
}

static void capacity_zero_retains_nothing(void)
{
    mempool_set_capacity(0);
    TEST_ASSERT_EQUAL_UINT64(0, mempool_capacity());
    void *p = mempool_aligned_malloc(SIZE);
    mempool_free(p, SIZE);
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
}

#define MANY 8

static void a_fixed_capacity_bounds_what_is_retained(void)
{
    mempool_set_capacity(3 * SIZE);
    void *p[MANY];
    for (int i = 0; i < MANY; i++)
        p[i] = mempool_aligned_malloc(SIZE);
    for (int i = 0; i < MANY; i++)
        mempool_free(p[i], SIZE);
    TEST_ASSERT_EQUAL_UINT64(3 * SIZE, stats().retained_bytes);
    TEST_ASSERT_EQUAL_UINT64(3 * SIZE, stats().retention_limit_bytes);
}

static void lowering_the_capacity_releases(void)
{
    void *p[MANY];
    for (int i = 0; i < MANY; i++)
        p[i] = mempool_aligned_malloc(SIZE);
    for (int i = 0; i < MANY; i++)
        mempool_free(p[i], SIZE);
    TEST_ASSERT_EQUAL_UINT64(MANY * SIZE, stats().retained_bytes);
    mempool_set_capacity(SIZE);
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
}

// A workload that repeats its live set is served entirely from the pool.
static void auto_capacity_keeps_the_peak_live_set(void)
{
    void *p[MANY];
    for (int i = 0; i < MANY; i++)
        p[i] = mempool_aligned_malloc(SIZE);
    TEST_ASSERT_GREATER_OR_EQUAL_UINT64(MANY * SIZE, stats().peak_outstanding_bytes);
    for (int i = 0; i < MANY; i++)
        mempool_free(p[i], SIZE);
    TEST_ASSERT_EQUAL_UINT64(MANY * SIZE, stats().retained_bytes);

    const uint64_t misses = stats().misses;
    for (int i = 0; i < MANY; i++)
        p[i] = mempool_aligned_malloc(SIZE);
    TEST_ASSERT_EQUAL_UINT64(misses, stats().misses);
    for (int i = 0; i < MANY; i++)
        mempool_free(p[i], SIZE);
}

// More distinct sizes than a thread keeps buckets for: the oldest sizes are
// dropped, and the total retained stays consistent.
static void many_sizes_evict_the_least_recently_used(void)
{
    enum
    {
        SIZES = 40
    };
    void *p[SIZES];
    for (int i = 0; i < SIZES; i++)
        p[i] = mempool_aligned_malloc(SIZE + 64u * (unsigned)i);
    for (int i = 0; i < SIZES; i++)
        mempool_free(p[i], SIZE + 64u * (unsigned)i);
    uint64_t newest = 0;
    for (int i = SIZES - 32; i < SIZES; i++)
        newest += SIZE + 64u * (unsigned)i;
    TEST_ASSERT_EQUAL_UINT64(newest, stats().retained_bytes);
    mempool_release_all();
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
}

// The secret pattern differs from every fill the pool or the C library may use.
static void a_wiped_buffer_reaches_the_next_holder_without_its_contents(void)
{
    const char *env = getenv("MALLOC_PERTURB_");
    const int perturb = env ? atoi(env) & 0xff : 0;
    uint8_t secret = 0x11;
    while (secret == perturb || secret == (perturb ^ 0xff))
        secret += 0x11;

    uint8_t *buf = (uint8_t *)mempool_aligned_malloc(SIZE);
    memset(buf, secret, SIZE);
    mempool_free_and_wipe(buf, SIZE);
    uint8_t *next = (uint8_t *)mempool_aligned_malloc(SIZE);
    TEST_ASSERT_EQUAL_PTR(buf, next);
    for (size_t i = 0; i < SIZE; i++)
        if (next[i] == secret)
            TEST_FAIL_MESSAGE("a secret byte survived the wipe");
    mempool_free(next, SIZE);
}

static void a_size_header_buffer_is_released_by_pointer_alone(void)
{
    uint8_t *p = (uint8_t *)mempool_aligned_malloc_with_size_header(SIZE);
    TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)((uintptr_t)p % 64));
    memset(p, 0x5A, SIZE);
    mempool_free_with_size_header(p);
    TEST_ASSERT_EQUAL_UINT64(SIZE + 64, stats().retained_bytes);
    uint8_t *q = (uint8_t *)mempool_aligned_malloc_with_size_header(SIZE);
    TEST_ASSERT_EQUAL_PTR(p, q);
    mempool_free_with_size_header(q);
    mempool_free_with_size_header(NULL);
}

static void wipe_on_release_zeroes_every_released_buffer(void)
{
    const char *env = getenv("MALLOC_PERTURB_");
    const int perturb = env ? atoi(env) & 0xff : 0;
    uint8_t secret = 0x11;
    while (secret == perturb || secret == (perturb ^ 0xff))
        secret += 0x11;

    TEST_ASSERT_EQUAL_INT(0, mempool_wipe_on_release());
    mempool_set_wipe_on_release(1);
    uint8_t *buf = (uint8_t *)mempool_aligned_malloc(SIZE);
    memset(buf, secret, SIZE);
    mempool_free(buf, SIZE);
    uint8_t *next = (uint8_t *)mempool_aligned_malloc(SIZE);
    TEST_ASSERT_EQUAL_PTR(buf, next);
    for (size_t i = 0; i < SIZE; i++)
        if (next[i] == secret)
            TEST_FAIL_MESSAGE("a byte survived the release");
    mempool_free(next, SIZE);
    mempool_set_wipe_on_release(-1);
    TEST_ASSERT_EQUAL_INT(0, mempool_wipe_on_release());
}

typedef struct
{
    pthread_barrier_t retained, released;
    void *handed;
} Holder;

static void *retain_then_wait(void *arg)
{
    Holder *h = (Holder *)arg;
    void *p = mempool_aligned_malloc(SIZE);
    mempool_free(p, SIZE);
    pthread_barrier_wait(&h->retained);
    pthread_barrier_wait(&h->released);
    return NULL;
}

// The thread that retained the buffer is alive but doing nothing, like a
// worker asleep in the thread pool.
static void release_all_reaches_a_waiting_thread(void)
{
    Holder h;
    pthread_barrier_init(&h.retained, NULL, 2);
    pthread_barrier_init(&h.released, NULL, 2);
    pthread_t t;
    pthread_create(&t, NULL, retain_then_wait, &h);
    pthread_barrier_wait(&h.retained);
    TEST_ASSERT_EQUAL_UINT64(SIZE, stats().retained_bytes);
    mempool_release_all();
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
    pthread_barrier_wait(&h.released);
    pthread_join(t, NULL);
    pthread_barrier_destroy(&h.retained);
    pthread_barrier_destroy(&h.released);
}

static void *allocate_only(void *arg)
{
    *(void **)arg = mempool_aligned_malloc(SIZE);
    return NULL;
}

static void a_buffer_released_on_another_thread_is_kept_there(void)
{
    void *p;
    pthread_t t;
    pthread_create(&t, NULL, allocate_only, &p);
    pthread_join(t, NULL);
    mempool_free(p, SIZE);
    TEST_ASSERT_EQUAL_UINT64(SIZE, stats().retained_bytes);
    void *again = mempool_aligned_malloc(SIZE);
    TEST_ASSERT_EQUAL_PTR(p, again);
    mempool_free(again, SIZE);
}

static void *retain_and_exit(void *arg)
{
    (void)arg;
    void *p = mempool_aligned_malloc(SIZE);
    mempool_free(p, SIZE);
    return NULL;
}

static void a_thread_exit_returns_its_buffers(void)
{
    const uint64_t hits = stats().hits, misses = stats().misses;
    pthread_t t;
    pthread_create(&t, NULL, retain_and_exit, NULL);
    pthread_join(t, NULL);
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
    TEST_ASSERT_EQUAL_UINT64(hits + misses + 1, stats().hits + stats().misses);
}

#define WORKERS 8
#define ROUNDS 2000
#define HELD 4

static atomic_int corrupted;

// Each worker stamps the buffers it holds with its id and checks the stamp
// before releasing: a buffer handed to two holders at once breaks a stamp.
static void *churn(void *arg)
{
    const uint64_t id = (uint64_t)(uintptr_t)arg;
    static const size_t sizes[] = {SIZE, 2 * SIZE, 3 * SIZE};
    uint64_t *held[HELD] = {0};
    size_t held_size[HELD] = {0};
    uint64_t state = id * 0x9E3779B97F4A7C15ull + 1;
    for (int round = 0; round < ROUNDS; round++)
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const int slot = (int)((state >> 33) % HELD);
        if (held[slot] != NULL)
        {
            const size_t words = held_size[slot] / sizeof(uint64_t);
            if (held[slot][0] != id || held[slot][words - 1] != id)
                atomic_fetch_add(&corrupted, 1);
            mempool_free(held[slot], held_size[slot]);
            held[slot] = NULL;
        }
        else
        {
            held_size[slot] = sizes[(state >> 40) % 3];
            held[slot] = (uint64_t *)mempool_aligned_malloc(held_size[slot]);
            const size_t words = held_size[slot] / sizeof(uint64_t);
            held[slot][0] = held[slot][words - 1] = id;
        }
    }
    for (int slot = 0; slot < HELD; slot++)
        if (held[slot] != NULL)
            mempool_free(held[slot], held_size[slot]);
    return NULL;
}

static void concurrent_use_hands_no_buffer_to_two_holders(void)
{
    atomic_store(&corrupted, 0);
    const uint64_t outstanding = stats().outstanding_bytes;
    pthread_t t[WORKERS];
    for (uint64_t w = 0; w < WORKERS; w++)
        pthread_create(&t[w], NULL, churn, (void *)(uintptr_t)(w + 1));
    for (int w = 0; w < WORKERS; w++)
        pthread_join(t[w], NULL);
    TEST_ASSERT_EQUAL_INT(0, atomic_load(&corrupted));
    TEST_ASSERT_EQUAL_UINT64(outstanding, stats().outstanding_bytes);
    TEST_ASSERT_EQUAL_UINT64(0, stats().retained_bytes);
}

static void a_forked_child_uses_the_pool(void)
{
    Holder h;
    pthread_barrier_init(&h.retained, NULL, 2);
    pthread_barrier_init(&h.released, NULL, 2);
    pthread_t t;
    pthread_create(&t, NULL, retain_then_wait, &h);
    pthread_barrier_wait(&h.retained);
    void *mine = mempool_aligned_malloc(2 * SIZE);
    mempool_free(mine, 2 * SIZE);

    const pid_t pid = fork();
    if (pid == 0)
    {
        // Only this thread's buffer survives into the child.
        MempoolStatistics s;
        mempool_statistics(&s);
        int ok = s.retained_bytes == 2 * SIZE;
        void *p = mempool_aligned_malloc(2 * SIZE);
        ok &= p == mine;
        mempool_free(p, 2 * SIZE);
        mempool_release_all();
        mempool_statistics(&s);
        ok &= s.retained_bytes == 0;
        _exit(ok ? 0 : 1);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    TEST_ASSERT_TRUE(WIFEXITED(status));
    TEST_ASSERT_EQUAL_INT(0, WEXITSTATUS(status));
    TEST_ASSERT_EQUAL_UINT64(3 * SIZE, stats().retained_bytes);
    pthread_barrier_wait(&h.released);
    pthread_join(t, NULL);
    pthread_barrier_destroy(&h.retained);
    pthread_barrier_destroy(&h.released);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(buffers_are_aligned_and_reused);
    RUN_TEST(a_different_size_is_not_served_from_another_size);
    RUN_TEST(small_buffers_go_to_the_c_library);
    RUN_TEST(capacity_zero_retains_nothing);
    RUN_TEST(a_fixed_capacity_bounds_what_is_retained);
    RUN_TEST(lowering_the_capacity_releases);
    RUN_TEST(auto_capacity_keeps_the_peak_live_set);
    RUN_TEST(many_sizes_evict_the_least_recently_used);
    RUN_TEST(a_wiped_buffer_reaches_the_next_holder_without_its_contents);
    RUN_TEST(wipe_on_release_zeroes_every_released_buffer);
    RUN_TEST(a_size_header_buffer_is_released_by_pointer_alone);
    RUN_TEST(release_all_reaches_a_waiting_thread);
    RUN_TEST(a_buffer_released_on_another_thread_is_kept_there);
    RUN_TEST(a_thread_exit_returns_its_buffers);
    RUN_TEST(concurrent_use_hands_no_buffer_to_two_holders);
    RUN_TEST(a_forked_child_uses_the_pool);
    return UNITY_END();
}
