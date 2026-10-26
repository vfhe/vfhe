// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file bench.h
 * @brief The main loop of a C benchmark executable: selecting cases, timing them, and
 *        writing their records.
 *
 * A benchmark file, `modules/<module>/c/bench/bench_<topic>.c`, defines an array of
 * BenchCase and returns bench_main() from main(). For every selected case and thread
 * limit, bench_main() builds the state with `setup`, calls `op` once and passes the state
 * to `check`, then times `op`: each of `reps` repetitions times as many calls as fill the
 * window, and the record keeps every repetition's seconds per call, their minimum and
 * their median.
 *
 * Records go to standard output as JSON lines; anything else the executable prints goes
 * to standard error. `tools/bench/run.py` starts the executable and reads them.
 *
 * Arguments, all optional: `--threads 1,8`, `--tier headline|full`, `--bench GLOB` on case
 * names, `--reps N`, `--window SECONDS`, and `--smoke`, which runs the first case of each
 * name once, with one call timed.
 *
 * The file includes this header once, and its functions are static.
 */
#ifndef VFHE_BENCH_H
#define VFHE_BENCH_H

#include <fnmatch.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <parallel.h>

/**
 * @brief Keeps the compiler from deleting the computation that produced `p`.
 *
 * An `op` whose output nothing reads may otherwise be removed whole; pass it the output
 * buffer after the computation.
 */
#define BENCH_KEEP(p) __asm__ volatile("" : : "r"(p) : "memory")

typedef struct BenchCase BenchCase;

/** @brief One benchmark at one parameter set. */
struct BenchCase
{
    const char *name;  /**< the benchmark, e.g. "ntt_forward" */
    const char *shape; /**< the parameter set's label, e.g. "n4096-q50"; cases sharing a name
                            list theirs smallest first */
    int headline;      /**< nonzero when the case belongs to the headline tier */
    uint64_t arg[4];   /**< parameters for `setup` */
    uint64_t items;    /**< operations one call of `op` performs; 0 counts as 1 */
    void *(*setup)(const BenchCase *c); /**< builds the state; NULL on failure */
    void (*op)(void *state);            /**< the timed call */
    int (*check)(void *state);          /**< optional: 0 when the state after one `op` is right */
    void (*teardown)(void *state);      /**< optional: releases the state */
};

#define BENCH_MAX_REPS 1000
#define BENCH_MAX_THREADS 64
#define BENCH_MAX_INNER (UINT64_C(1) << 30)

static double bench_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static long bench_minor_faults(void)
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_minflt;
}

/* Calls op `inner` times; returns the seconds, and the minor page faults in *faults. */
static double bench_timed(const BenchCase *c, void *state, uint64_t inner, long *faults)
{
    long before = bench_minor_faults();
    double start = bench_now();
    for (uint64_t i = 0; i < inner; i++)
        c->op(state);
    double elapsed = bench_now() - start;
    *faults = bench_minor_faults() - before;
    return elapsed;
}

/* Returns how many calls of op take at least `window` seconds; the calls warm up too. */
static uint64_t bench_calibrate(const BenchCase *c, void *state, double window)
{
    uint64_t inner = 1;
    for (;;)
    {
        long faults;
        double elapsed = bench_timed(c, state, inner, &faults);
        if (elapsed >= window || inner >= BENCH_MAX_INNER)
            return inner;
        double grown = 1.2 * (double)inner * window / (elapsed > 1e-9 ? elapsed : 1e-9);
        uint64_t next = grown > (double)BENCH_MAX_INNER ? BENCH_MAX_INNER : (uint64_t)grown;
        inner = next > 2 * inner ? next : 2 * inner;
        if (inner > BENCH_MAX_INNER)
            inner = BENCH_MAX_INNER;
    }
}

static int bench_compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double bench_median(double *values, int n)
{
    qsort(values, (size_t)n, sizeof *values, bench_compare_doubles);
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static void bench_error(FILE *records, const BenchCase *c, uint64_t threads, const char *what)
{
    fprintf(records,
            "{\"kind\": \"error\", \"layer\": \"c\", \"bench\": \"%s\", \"shape\": \"%s\", "
            "\"threads\": %" PRIu64 ", \"error\": \"%s\"}\n",
            c->name, c->shape, threads, what);
    fflush(records);
}

/* Returns 1 when an earlier case has the same name, so the smoke run skips this one. */
static int bench_seen(const BenchCase *cases, size_t i)
{
    for (size_t j = 0; j < i; j++)
        if (strcmp(cases[j].name, cases[i].name) == 0)
            return 1;
    return 0;
}

/* Times one case at every thread limit; returns the number of failures. */
static int bench_run_case(FILE *records, const BenchCase *c, const uint64_t *threads, int n_threads,
                          int reps, double window, int smoke)
{
    void *state = c->setup(c);
    if (state == NULL)
    {
        bench_error(records, c, threads[0], "setup failed");
        return 1;
    }
    int failures = 0;
    for (int t = 0; t < n_threads; t++)
    {
        vfhe_set_num_threads(threads[t]);
        c->op(state);
        if (c->check != NULL && c->check(state) != 0)
        {
            bench_error(records, c, threads[t], "check failed");
            failures++;
            continue;
        }
        uint64_t inner = smoke ? 1 : bench_calibrate(c, state, window);
        int n = smoke ? 1 : reps;
        double seconds[BENCH_MAX_REPS], faults[BENCH_MAX_REPS];
        for (int r = 0; r < n; r++)
        {
            long flt;
            seconds[r] = bench_timed(c, state, inner, &flt) / (double)inner;
            faults[r] = (double)flt / (double)inner;
        }
        double min = seconds[0];
        for (int r = 1; r < n; r++)
            min = seconds[r] < min ? seconds[r] : min;
        fprintf(records,
                "{\"kind\": \"result\", \"layer\": \"c\", \"bench\": \"%s\", \"shape\": \"%s\", "
                "\"tier\": \"%s\", \"threads\": %" PRIu64 ", \"items\": %" PRIu64 ", "
                "\"min_s\": %.9g, \"times_s\": [",
                c->name, c->shape, c->headline ? "headline" : "full", threads[t],
                c->items ? c->items : 1, min);
        for (int r = 0; r < n; r++)
            fprintf(records, "%s%.9g", r ? ", " : "", seconds[r]);
        /* bench_median sorts its argument, so the times are printed first. */
        fprintf(records,
                "], \"median_s\": %.9g, \"reps\": %d, \"inner\": %" PRIu64 ", \"faults\": %.6g}\n",
                bench_median(seconds, n), n, inner, bench_median(faults, n));
        fflush(records);
    }
    vfhe_set_num_threads(0);
    if (c->teardown != NULL)
        c->teardown(state);
    return failures;
}

/* Parses "1,8,32" into threads; returns the count, or -1 when malformed. */
static int bench_parse_threads(const char *text, uint64_t *threads)
{
    int n = 0;
    while (*text != '\0')
    {
        char *end;
        unsigned long long value = strtoull(text, &end, 10);
        if (end == text || value == 0 || n == BENCH_MAX_THREADS || (*end != ',' && *end != '\0'))
            return -1;
        threads[n++] = value;
        text = *end == ',' ? end + 1 : end;
    }
    return n > 0 ? n : -1;
}

static int bench_usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [--threads 1,8] [--tier headline|full] [--bench GLOB] [--reps N] "
            "[--window SECONDS] [--smoke]\n",
            program);
    return 2;
}

/**
 * @brief Runs the selected cases and writes their records.
 * @param[in] argc, argv The executable's arguments.
 * @param[in] cases The cases, those sharing a name listed smallest shape first.
 * @param[in] n_cases How many there are.
 * @return 0 when every selected case ran and passed its check, 1 otherwise, 2 on a usage
 *         error.
 */
static int bench_main(int argc, char **argv, const BenchCase *cases, size_t n_cases)
{
    uint64_t threads[BENCH_MAX_THREADS] = {1};
    int n_threads = 1, reps = 7, smoke = 0, headline_only = 1;
    double window = 0.05;
    const char *pattern = "*";
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--smoke") == 0)
        {
            smoke = 1;
            continue;
        }
        if (i + 1 == argc)
            return bench_usage(argv[0]);
        const char *flag = argv[i], *value = argv[++i];
        if (strcmp(flag, "--threads") == 0)
            n_threads = bench_parse_threads(value, threads);
        else if (strcmp(flag, "--tier") == 0)
            headline_only = strcmp(value, "headline") == 0 ? 1
                            : strcmp(value, "full") == 0   ? 0
                                                           : -1;
        else if (strcmp(flag, "--bench") == 0)
            pattern = value;
        else if (strcmp(flag, "--reps") == 0)
            reps = atoi(value);
        else if (strcmp(flag, "--window") == 0)
            window = atof(value);
        else
            return bench_usage(argv[0]);
    }
    if (n_threads < 0 || headline_only < 0 || reps < 1 || reps > BENCH_MAX_REPS || window < 0)
        return bench_usage(argv[0]);

    /* Records keep the original standard output; anything else written there goes to
       standard error. */
    FILE *records = fdopen(dup(STDOUT_FILENO), "w");
    if (records == NULL || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
    {
        perror("bench: redirecting standard output");
        return 1;
    }

    int failures = 0;
    for (size_t i = 0; i < n_cases; i++)
    {
        const BenchCase *c = &cases[i];
        if ((headline_only && !smoke && !c->headline) || fnmatch(pattern, c->name, 0) != 0)
            continue;
        if (smoke && bench_seen(cases, i))
            continue;
        failures += bench_run_case(records, c, threads, n_threads, reps, window, smoke);
    }
    fclose(records);
    return failures ? 1 : 0;
}

#endif
