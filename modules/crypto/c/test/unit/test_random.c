// SPDX-FileCopyrightText: 2026 Michał Osadnik <micosa97@gmail.com>
/* SPDX-License-Identifier: Apache-2.0 */
/**
 * @file test_random.c
 * @brief The entropy stream behind a NULL source, under threads and fork: no
 * two streams may meet; and the seeded source, a function of its seed.
 */
#include <random.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

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
        random_bytes(NULL, out, 16);
    for (int i = 0; i < LARGE_DRAWS; i++, out += LARGE_BYTES)
        random_bytes(NULL, out, LARGE_BYTES);
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

/* The child inherits its parent's pool; both must still draw different bytes
 * after the fork, on either path. */
void test_a_forked_child_does_not_repeat_its_parent(void)
{
    uint8_t warm[16], parent[32 + LARGE_BYTES], child[32 + LARGE_BYTES];
    random_bytes(NULL, warm, sizeof warm); /* leaves most of a pool to inherit */

    int fds[2];
    TEST_ASSERT_EQUAL_INT(0, pipe(fds));
    const pid_t pid = fork();
    TEST_ASSERT_TRUE(pid >= 0);
    if (pid == 0)
    {
        close(fds[0]);
        random_bytes(NULL, child, 32);
        random_bytes(NULL, child + 32, LARGE_BYTES);
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
    random_bytes(NULL, parent, 32);
    random_bytes(NULL, parent + 32, LARGE_BYTES);

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

/* Every size around the pool's thresholds comes back the right length and
 * unlike the draw before it: 0 and 1 byte, both sides of the 512-byte path
 * switch, the pool's whole 1024 and past it. */
void test_bytes_of_every_size_around_the_pool(void)
{
    const uint64_t sizes[] = {0, 1, 7, 511, 512, 513, 1023, 1024, 1025, 4096};
    uint8_t a[4096], b[4096];
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++)
    {
        memset(a, 0, sizeof a);
        memset(b, 0, sizeof b);
        random_bytes(NULL, a, sizes[s]);
        random_bytes(NULL, b, sizes[s]);
        if (sizes[s] >= 8)
            TEST_ASSERT_TRUE(memcmp(a, b, sizes[s]) != 0);
    }
    /* exactly what a pool holds, then one more: a refill in between */
    random_bytes(NULL, a, 1000);
    random_bytes(NULL, b, 100);
    TEST_ASSERT_TRUE(memcmp(a, b, 100) != 0);
}

/* Values below a bound stay below it, cover a small range, and come out as
 * zeros below 1. */
void test_uniform_below_stays_in_range(void)
{
    const uint64_t bounds[] = {1, 2, 3, 256, 257, (1ULL << 32) + 1, (1ULL << 61) - 1, UINT64_MAX};
    uint64_t out[512];
    for (size_t b = 0; b < sizeof bounds / sizeof bounds[0]; b++)
    {
        random_below(NULL, out, 512, bounds[b]);
        for (int i = 0; i < 512; i++)
            TEST_ASSERT_TRUE(out[i] < bounds[b]);
    }
    random_below(NULL, out, 512, 1);
    for (int i = 0; i < 512; i++)
        TEST_ASSERT_EQUAL_UINT64(0, out[i]);
    int seen[3] = {0};
    random_below(NULL, out, 512, 3);
    for (int i = 0; i < 512; i++)
        seen[out[i]] = 1;
    TEST_ASSERT_TRUE(seen[0] && seen[1] && seen[2]);
    random_below(NULL, out, 0, 5); /* writes nothing */
}

/* A seeded source is a function of its seed: the same seed replays every draw,
 * another seed does not, and NULL does not. */
void test_a_seeded_source_replays(void)
{
    uint8_t a[700], b[700];
    uint64_t ua[16], ub[16];
    Random first = random_seeded((const uint8_t *)"seed", 4);
    Random again = random_seeded((const uint8_t *)"seed", 4);
    Random other = random_seeded((const uint8_t *)"other", 5);
    random_bytes(first, a, 100);
    random_bytes(first, a + 100, 600);
    random_below(first, ua, 16, 1000);
    random_bytes(again, b, 100);
    random_bytes(again, b + 100, 600);
    random_below(again, ub, 16, 1000);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(a, b, sizeof a);
    TEST_ASSERT_EQUAL_UINT64_ARRAY(ua, ub, 16);
    random_bytes(other, b, 100);
    TEST_ASSERT_TRUE(memcmp(a, b, 100) != 0);
    random_bytes(NULL, b, 100);
    TEST_ASSERT_TRUE(memcmp(a, b, 100) != 0);
    random_free(first);
    random_free(again);
    random_free(other);
    random_free(NULL);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_bytes_of_every_size_around_the_pool);
    RUN_TEST(test_uniform_below_stays_in_range);
    RUN_TEST(test_threads_never_share_a_stream);
    RUN_TEST(test_a_forked_child_does_not_repeat_its_parent);
    RUN_TEST(test_a_seeded_source_replays);
    return UNITY_END();
}
