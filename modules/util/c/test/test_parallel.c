// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
// vfhe_parallel_for's contract: every index once, never more threads than the
// limit, a loop inside a body runs on the thread that calls it, loops started
// by several callers at once all complete, and a forked child can run loops.

#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unity.h>
#include <util.h>

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

static atomic_uint_fast64_t share_seen[2];
static atomic_int nested_items_run;

static void counted_body(void *ctx, uint64_t i)
{
    (void)ctx;
    (void)i;
    atomic_fetch_add(&nested_items_run, 1);
}

static void lending_body(void *ctx, uint64_t i)
{
    (void)ctx;
    atomic_store(&share_seen[i], vfhe_threads_for(0, 1000));
    vfhe_parallel_for(64, 0, counted_body, NULL);
}

static void a_loop_over_few_items_lends_its_threads(void)
{
    vfhe_set_num_threads(8);
    atomic_store(&nested_items_run, 0);
    vfhe_parallel_for(2, 0, lending_body, NULL);
    TEST_ASSERT_EQUAL_UINT64(4, atomic_load(&share_seen[0]));
    TEST_ASSERT_EQUAL_UINT64(4, atomic_load(&share_seen[1]));
    TEST_ASSERT_EQUAL_INT(128, atomic_load(&nested_items_run));
    // A single item runs on the caller with the whole budget.
    vfhe_parallel_for(1, 0, lending_body, NULL);
    TEST_ASSERT_EQUAL_UINT64(8, atomic_load(&share_seen[0]));
    // A loop asking for fewer threads lends only what it was allowed, and a
    // share below 4 is not lent at all.
    vfhe_set_num_threads(16);
    vfhe_parallel_for(2, 8, lending_body, NULL);
    TEST_ASSERT_EQUAL_UINT64(4, atomic_load(&share_seen[0]));
    vfhe_parallel_for(2, 6, lending_body, NULL);
    TEST_ASSERT_EQUAL_UINT64(1, atomic_load(&share_seen[0]));
    vfhe_set_num_threads(8);
    TEST_ASSERT_EQUAL_UINT64(8, vfhe_threads_for(0, 100));
}

static void the_local_number_bounds_the_callers_loops(void)
{
    vfhe_set_num_threads(8);
    TEST_ASSERT_EQUAL_UINT64(0, vfhe_set_local_num_threads(3));
    TEST_ASSERT_EQUAL_UINT64(3, vfhe_local_num_threads());
    TEST_ASSERT_EQUAL_UINT64(3, vfhe_threads_for(0, 100));
    TEST_ASSERT_EQUAL_UINT64(2, vfhe_threads_for(2, 100));
    atomic_store(&record.n_seen, 0);
    vfhe_parallel_for(ITEMS, 0, count_body, &record);
    TEST_ASSERT_LESS_OR_EQUAL_INT(3, atomic_load(&record.n_seen));
    TEST_ASSERT_EQUAL_UINT64(3, vfhe_set_local_num_threads(0));
    TEST_ASSERT_EQUAL_UINT64(8, vfhe_threads_for(0, 100));
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

// Kernel thread ids: unlike pthread_t values, not reused as soon as a thread
// is joined.
#define MAX_TIDS 1024
static pid_t tids[MAX_TIDS];
static atomic_int n_tids;
static pthread_mutex_t tids_lock = PTHREAD_MUTEX_INITIALIZER;

static void tid_body(void *ctx, uint64_t i)
{
    (void)ctx;
    const pid_t self = gettid();
    pthread_mutex_lock(&tids_lock);
    int known = 0;
    for (int k = 0; k < atomic_load(&n_tids); k++)
        known |= tids[k] == self;
    if (!known && atomic_load(&n_tids) < MAX_TIDS)
        tids[atomic_fetch_add(&n_tids, 1)] = self;
    pthread_mutex_unlock(&tids_lock);
    usleep(i % 16 == 0 ? 100 : 0);
}

static void the_threads_are_kept_between_loops(void)
{
    vfhe_set_num_threads(4);
    atomic_store(&n_tids, 0);
    for (int round = 0; round < 50; round++)
        vfhe_parallel_for(256, 0, tid_body, NULL);
    TEST_ASSERT_GREATER_THAN_INT(1, atomic_load(&n_tids));
    TEST_ASSERT_LESS_OR_EQUAL_INT(4, atomic_load(&n_tids));
}

#define CALLERS 4
#define CALLER_ITEMS 512

typedef struct
{
    atomic_int hits[CALLER_ITEMS];
} CallerRecord;

static void caller_body(void *ctx, uint64_t i)
{
    atomic_fetch_add(&((CallerRecord *)ctx)->hits[i], 1);
    usleep(i % 32 == 0 ? 100 : 0);
}

static void *caller_main(void *ctx)
{
    for (int round = 0; round < 20; round++)
        vfhe_parallel_for(CALLER_ITEMS, 0, caller_body, ctx);
    return NULL;
}

static void concurrent_callers_share_the_threads(void)
{
    vfhe_set_num_threads(4);
    static CallerRecord records[CALLERS];
    pthread_t callers[CALLERS];
    for (int c = 0; c < CALLERS; c++)
    {
        for (int i = 0; i < CALLER_ITEMS; i++)
            atomic_store(&records[c].hits[i], 0);
        pthread_create(&callers[c], NULL, caller_main, &records[c]);
    }
    for (int c = 0; c < CALLERS; c++)
        pthread_join(callers[c], NULL);
    for (int c = 0; c < CALLERS; c++)
        for (int i = 0; i < CALLER_ITEMS; i++)
            TEST_ASSERT_EQUAL_INT(20, atomic_load(&records[c].hits[i]));
}

static void a_forked_child_runs_loops(void)
{
    vfhe_set_num_threads(4);
    vfhe_parallel_for(ITEMS, 0, count_body, &record);
    const pid_t child = fork();
    TEST_ASSERT_NOT_EQUAL_INT(-1, child);
    if (child == 0)
    {
        reset_record();
        vfhe_parallel_for(ITEMS, 0, count_body, &record);
        int ok = atomic_load(&record.n_seen) > 1;
        for (int i = 0; i < ITEMS; i++)
            ok &= atomic_load(&record.hits[i]) == 1;
        _exit(ok ? 0 : 1);
    }
    int status;
    TEST_ASSERT_EQUAL_INT(child, waitpid(child, &status, 0));
    TEST_ASSERT_TRUE(WIFEXITED(status));
    TEST_ASSERT_EQUAL_INT(0, WEXITSTATUS(status));
    // And the parent's threads still serve it.
    reset_record();
    vfhe_parallel_for(ITEMS, 0, count_body, &record);
    for (int i = 0; i < ITEMS; i++)
        TEST_ASSERT_EQUAL_INT(1, atomic_load(&record.hits[i]));
}

static atomic_int signals_on_workers;
static pthread_t signal_target;

static void note_signal(int sig)
{
    (void)sig;
    if (!pthread_equal(pthread_self(), signal_target))
        atomic_fetch_add(&signals_on_workers, 1);
}

static void signals_reach_the_callers_threads(void)
{
    vfhe_set_num_threads(4);
    vfhe_parallel_for(ITEMS, 0, count_body, &record);
    signal_target = pthread_self();
    atomic_store(&signals_on_workers, 0);
    struct sigaction action = {0}, previous;
    action.sa_handler = note_signal;
    sigaction(SIGUSR1, &action, &previous);
    // A process-directed signal goes to a thread that does not block it.
    for (int k = 0; k < 20; k++)
        kill(getpid(), SIGUSR1);
    sigaction(SIGUSR1, &previous, NULL);
    TEST_ASSERT_EQUAL_INT(0, atomic_load(&signals_on_workers));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(every_index_runs_once);
    RUN_TEST(the_limit_bounds_the_threads);
    RUN_TEST(one_thread_is_the_callers);
    RUN_TEST(a_nested_loop_stays_on_its_thread);
    RUN_TEST(a_loop_over_few_items_lends_its_threads);
    RUN_TEST(the_local_number_bounds_the_callers_loops);
    RUN_TEST(zero_restores_the_default);
    RUN_TEST(the_threads_are_kept_between_loops);
    RUN_TEST(concurrent_callers_share_the_threads);
    RUN_TEST(a_forked_child_runs_loops);
    RUN_TEST(signals_reach_the_callers_threads);
    return UNITY_END();
}
