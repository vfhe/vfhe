// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// parallel.h's contract: every index once, at most the limit's threads, and a
// loop inside a body runs on the thread that calls it.

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>
#include <unity.h>
#include <parallel.h>

#define ITEMS 4096
#define MAX_SEEN 64

typedef struct
{
    atomic_int hits[ITEMS];
    pthread_t seen[MAX_SEEN];
    atomic_int n_seen;
    pthread_mutex_t lock;
} Record;

static void note_thread(Record *r)
{
    pthread_t self = pthread_self();
    pthread_mutex_lock(&r->lock);
    int known = 0;
    for (int k = 0; k < atomic_load(&r->n_seen); k++)
        known |= pthread_equal(r->seen[k], self);
    if (!known && atomic_load(&r->n_seen) < MAX_SEEN)
        r->seen[atomic_fetch_add(&r->n_seen, 1)] = self;
    pthread_mutex_unlock(&r->lock);
}

static void count_body(void *ctx, uint64_t i)
{
    Record *r = (Record *)ctx;
    atomic_fetch_add(&r->hits[i], 1);
    note_thread(r);
    // Long enough that the helpers get to draw items too.
    usleep(i % 64 == 0 ? 200 : 0);
}

static Record record;

static void reset_record(void)
{
    for (int i = 0; i < ITEMS; i++)
        atomic_store(&record.hits[i], 0);
    atomic_store(&record.n_seen, 0);
    pthread_mutex_init(&record.lock, NULL);
}

void setUp(void)
{
    vfhe_set_num_threads(0);
    reset_record();
}
void tearDown(void) { vfhe_set_num_threads(0); }

static void every_index_runs_once(void)
{
    vfhe_set_num_threads(4);
    vfhe_parallel_for(ITEMS, 0, count_body, &record);
    for (int i = 0; i < ITEMS; i++)
        TEST_ASSERT_EQUAL_INT(1, atomic_load(&record.hits[i]));
}

static void the_limit_bounds_the_threads(void)
{
    vfhe_set_num_threads(3);
    vfhe_parallel_for(ITEMS, 8, count_body, &record);
    TEST_ASSERT_LESS_OR_EQUAL_INT(3, atomic_load(&record.n_seen));
    TEST_ASSERT_EQUAL_UINT64(2, vfhe_threads_for(2, 100));
    TEST_ASSERT_EQUAL_UINT64(3, vfhe_threads_for(0, 100));
    TEST_ASSERT_EQUAL_UINT64(3, vfhe_threads_for(8, 100));
    TEST_ASSERT_EQUAL_UINT64(1, vfhe_threads_for(0, 1));
}

static void one_thread_is_the_callers(void)
{
    vfhe_set_num_threads(1);
    vfhe_parallel_for(ITEMS, 0, count_body, &record);
    TEST_ASSERT_EQUAL_INT(1, atomic_load(&record.n_seen));
    TEST_ASSERT_TRUE(pthread_equal(record.seen[0], pthread_self()));
}

static atomic_int nested_threads_seen;

static void nested_body(void *ctx, uint64_t i)
{
    (void)ctx;
    (void)i;
    if (vfhe_threads_for(0, 1000) != 1)
        atomic_fetch_add(&nested_threads_seen, 1);
}

static void outer_body(void *ctx, uint64_t i)
{
    (void)i;
    vfhe_parallel_for(16, 0, nested_body, ctx);
}

static void a_nested_loop_stays_on_its_thread(void)
{
    vfhe_set_num_threads(4);
    atomic_store(&nested_threads_seen, 0);
    vfhe_parallel_for(16, 0, outer_body, NULL);
    TEST_ASSERT_EQUAL_INT(0, atomic_load(&nested_threads_seen));
    // And the caller is back outside a loop afterwards.
    TEST_ASSERT_EQUAL_UINT64(4, vfhe_threads_for(0, 100));
}

static void zero_restores_the_default(void)
{
    const uint64_t original = vfhe_num_threads();
    vfhe_set_num_threads(5);
    TEST_ASSERT_EQUAL_UINT64(5, vfhe_num_threads());
    vfhe_set_num_threads(0);
    TEST_ASSERT_EQUAL_UINT64(original, vfhe_num_threads());
    TEST_ASSERT_GREATER_OR_EQUAL_UINT64(1, vfhe_num_threads());
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(every_index_runs_once);
    RUN_TEST(the_limit_bounds_the_threads);
    RUN_TEST(one_thread_is_the_callers);
    RUN_TEST(a_nested_loop_stays_on_its_thread);
    RUN_TEST(zero_restores_the_default);
    return UNITY_END();
}
