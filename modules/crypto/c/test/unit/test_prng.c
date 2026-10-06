// SPDX-FileCopyrightText: 2026 Michał Osadnik <micosa97@gmail.com>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_prng.c
 * @brief The entropy-backed generators under threads and fork: no two
 * streams may meet.
 */
#include <crypto.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) { vfhe_prng_clear_deterministic_seed(); }

#define THREADS 8
#define SMALL_DRAWS 64 /* 16 bytes each, through the pool */
#define LARGE_DRAWS 4  /* a fresh seed apiece */
#define LARGE_BYTES 4096
#define WORDS_PER_THREAD ((SMALL_DRAWS * 16 + LARGE_DRAWS * LARGE_BYTES) / 8)

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Both paths a draw can take. */
static void *draw_both_ways(void *arg)
{
    uint8_t *out = (uint8_t *)arg;
    for (int i = 0; i < SMALL_DRAWS; i++, out += 16)
        generate_random_bytes(16, out);
    for (int i = 0; i < LARGE_DRAWS; i++, out += LARGE_BYTES)
        generate_random_bytes(LARGE_BYTES, out);
    return NULL;
}

/* Every 64-bit word drawn by every thread at once: a repeat means two threads
 * were served the same bytes. */
static void assert_threads_never_repeat(void)
{
    uint64_t *words = (uint64_t *)malloc(THREADS * WORDS_PER_THREAD * sizeof(uint64_t));
    pthread_t threads[THREADS];
    for (int t = 0; t < THREADS; t++)
        TEST_ASSERT_EQUAL_INT(
            0, pthread_create(&threads[t], NULL, draw_both_ways, &words[t * WORDS_PER_THREAD]));
    for (int t = 0; t < THREADS; t++)
        pthread_join(threads[t], NULL);

    qsort(words, THREADS * WORDS_PER_THREAD, sizeof(uint64_t), cmp_u64);
    for (uint64_t i = 1; i < THREADS * WORDS_PER_THREAD; i++)
        TEST_ASSERT_NOT_EQUAL_UINT64(words[i - 1], words[i]);
    free(words);
}

void test_threads_never_share_a_stream(void) { assert_threads_never_repeat(); }

/* The pinned stream is one counter for every thread: it must not hand two
 * threads the same seed. */
void test_pinned_threads_never_share_a_stream(void)
{
    vfhe_prng_set_deterministic_seed(7);
    assert_threads_never_repeat();
}

/* The child inherits its parent's pool; both must still draw different bytes
 * after the fork, on either path. */
void test_a_forked_child_does_not_repeat_its_parent(void)
{
    uint8_t warm[16], parent[32 + LARGE_BYTES], child[32 + LARGE_BYTES];
    generate_random_bytes(sizeof warm, warm); /* leaves most of a pool to inherit */

    int fds[2];
    TEST_ASSERT_EQUAL_INT(0, pipe(fds));
    const pid_t pid = fork();
    TEST_ASSERT_TRUE(pid >= 0);
    if (pid == 0)
    {
        close(fds[0]);
        generate_random_bytes(32, child);
        generate_random_bytes(LARGE_BYTES, child + 32);
        for (size_t sent = 0; sent < sizeof child;)
        {
            const ssize_t n = write(fds[1], child + sent, sizeof child - sent);
            if (n <= 0)
                _exit(1);
            sent += (size_t)n;
        }
        _exit(0);
    }
    close(fds[1]); /* so that a child that dies early ends the read below */
    generate_random_bytes(32, parent);
    generate_random_bytes(LARGE_BYTES, parent + 32);

    size_t got = 0;
    for (ssize_t n = 1; got < sizeof child && n > 0; got += n > 0 ? (size_t)n : 0)
        n = read(fds[0], child + got, sizeof child - got);
    close(fds[0]);
    int status;
    waitpid(pid, &status, 0);
    TEST_ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    TEST_ASSERT_EQUAL_UINT64(sizeof child, got);
    TEST_ASSERT_TRUE(memcmp(parent, child, 32) != 0);
    TEST_ASSERT_TRUE(memcmp(parent + 32, child + 32, LARGE_BYTES) != 0);
}

/* A thread that fills its pool, waits for the test to pin the stream, then
 * draws again. */
typedef struct
{
    pthread_mutex_t lock;
    pthread_cond_t moved;
    int stage;
    uint8_t after_pin[16];
} AcrossAPin;

static void wait_for(AcrossAPin *r, int stage)
{
    while (r->stage != stage)
        pthread_cond_wait(&r->moved, &r->lock);
}

static void move_to(AcrossAPin *r, int stage)
{
    r->stage = stage;
    pthread_cond_broadcast(&r->moved);
}

static void *draw_across_a_pin(void *arg)
{
    AcrossAPin *r = (AcrossAPin *)arg;
    uint8_t warm[16];
    generate_random_bytes(sizeof warm, warm);
    pthread_mutex_lock(&r->lock);
    move_to(r, 1);
    wait_for(r, 2);
    pthread_mutex_unlock(&r->lock);
    generate_random_bytes(sizeof r->after_pin, r->after_pin);
    return NULL;
}

/* Pinning discards the pool of every thread, not only the pinning one's: a
 * thread that drew before the pin draws from the seed after it. */
void test_pinning_discards_every_pool(void)
{
    AcrossAPin r = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, {0}};
    pthread_t thread;
    TEST_ASSERT_EQUAL_INT(0, pthread_create(&thread, NULL, draw_across_a_pin, &r));
    pthread_mutex_lock(&r.lock);
    wait_for(&r, 1);
    vfhe_prng_set_deterministic_seed(3);
    move_to(&r, 2);
    pthread_mutex_unlock(&r.lock);
    pthread_join(thread, NULL);

    uint8_t expected[16];
    vfhe_prng_set_deterministic_seed(3);
    generate_random_bytes(sizeof expected, expected);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, r.after_pin, sizeof expected);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_threads_never_share_a_stream);
    RUN_TEST(test_pinned_threads_never_share_a_stream);
    RUN_TEST(test_a_forked_child_does_not_repeat_its_parent);
    RUN_TEST(test_pinning_discards_every_pool);
    return UNITY_END();
}
