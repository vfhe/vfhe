// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "fhe.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <util.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif

/* ------------------------------------------------------------------------------------------------
 * The blind rotation is a sequence of steps, one per `unfolding` mask coefficients. A step adds
 * to the accumulator one term per bit pattern j of its key bits [BMMP18, Alg. 1]:
 *
 *   acc <- acc + sum_j (X^e_j - 1) * (sum_t D_t (.) BK_j[t]),   D_t the gadget digits of acc,
 *
 * where t runs over the (r + 1) * ell rows of an MGSW key. Each term is the inner product of the
 * digits with that pattern's key, multiplied by its factor X^e_j - 1 afterwards, so the digits
 * are computed once per step and every key is read once. A step over a single coefficient
 * decomposes (X^a - 1) * acc instead and has no factor: the [CGGI16] step. Terms with e_j = 0
 * vanish, and so do steps where every term does.
 *
 * Each step has three stages, every one a set of independent tasks:
 *
 *   0  decomposition:  the digits D_t, and the factors X^e_j - 1 in the mul domain;
 *   1  inner products: for each term j, component i of the accumulator and chunk k of the
 *                      digits, sum over t in the chunk of D_t (.) BK_j[t].component(i);
 *   2  update:         component i of acc plus sum_j (X^e_j - 1) * sum_k (the chunks), brought
 *                      back to the canonical domain and rescaled to the accumulator's ring.
 *
 * One vfhe_parallel_for runs the whole rotation, every worker following all the stages: it takes
 * tasks of the current stage until none is left, the worker finishing the last one opens the next
 * stage, and the others wait for that (spinning briefly, then sleeping). A worker only waits for
 * tasks another running worker has taken, so the rotation completes whatever the number of
 * threads -- including when the loop runs its items one after the other, and item 0 does it all.
 * ------------------------------------------------------------------------------------------------
 */

// How long a worker spins on the stage counter before sleeping.
#define WAIT_SPINS 4096

// The terms one step adds.
typedef struct
{
    uint64_t terms;
    uint64_t *key;      // index in bk of each term's key
    uint64_t *exponent; // its e_j
    int single;         // a step over one coefficient: decompose (X^a - 1) * acc
} Step;

typedef struct
{
    RNSc_MLWE acc;
    ArithRing ring, key_ring; // the accumulator's, and the keys'
    RNS_MLWE *const *bk;
    uint64_t r, ell, log_base;
    uint64_t components; // r + 1
    uint64_t digits;     // components * ell
    uint64_t chunks;     // the digit sum of an inner product is split into this many tasks
    uint64_t max_terms;

    Step *step;
    uint64_t steps;
    ArithElement one;            // canonical 1 over the key's ring
    ArithElement *digit;         // [digits]
    ArithElement *factor;        // [max_terms]
    ArithElement *inner_product; // [max_terms][components][chunks]

    uint64_t stages; // 3 per step
    uint64_t *tasks;
    atomic_uint_fast64_t *taken, *done; // per stage
    atomic_uint_fast64_t stage;         // the stage open to workers
    pthread_mutex_t lock;
    pthread_cond_t stage_opened;
} Rotation;

static ArithElement *component(RNS_MLWE c, uint64_t i, uint64_t r)
{
    return i < r ? &c->a[i] : &c->b;
}

static ArithElement *inner_product_at(Rotation *R, uint64_t j, uint64_t i, uint64_t k)
{
    return &R->inner_product[(j * R->components + i) * R->chunks + k];
}

// Step s covers mask coefficients s*unfolding onwards, the last step fewer when `unfolding` does
// not divide n; its keys are those of patterns j = 1 .. 2^size - 1, in that order.
static void prepare_steps(Rotation *R, const uint64_t *a, uint64_t n, uint64_t unfolding)
{
    const uint64_t two_n = 2 * R->ring->N, count = (n + unfolding - 1) / unfolding;
    R->step = (Step *)safe_malloc(count * sizeof(Step));
    R->steps = 0;
    R->max_terms = 1;
    uint64_t first_key = 0;
    for (uint64_t s = 0; s < count; s++)
    {
        const uint64_t first = s * unfolding;
        const uint64_t size = n - first < unfolding ? n - first : unfolding;
        const uint64_t patterns = 1ULL << size;
        Step *S = &R->step[R->steps];
        S->terms = 0;
        S->single = size == 1;
        S->key = (uint64_t *)safe_malloc(patterns * sizeof(uint64_t));
        S->exponent = (uint64_t *)safe_malloc(patterns * sizeof(uint64_t));
        for (uint64_t j = 1; j < patterns; j++)
        {
            uint64_t e = 0;
            for (uint64_t t = 0; t < size; t++)
                if ((j >> t) & 1)
                    e += a[first + t];
            e %= two_n;
            if (e == 0)
                continue;
            S->key[S->terms] = first_key + j - 1;
            S->exponent[S->terms] = e;
            S->terms++;
        }
        first_key += patterns - 1;
        if (S->terms == 0)
        {
            free(S->key);
            free(S->exponent);
            continue;
        }
        if (S->terms > R->max_terms)
            R->max_terms = S->terms;
        R->steps++;
    }
}

static uint64_t stage_tasks(const Rotation *R, uint64_t stage)
{
    const Step *S = &R->step[stage / 3];
    switch (stage % 3)
    {
    case 0:
        return R->digits + (S->single ? 0 : S->terms);
    case 1:
        return S->terms * R->components * R->chunks;
    default:
        return R->components;
    }
}

// Task t of the decomposition: digit t, or for t past the digits a factor. `rotated` is the
// worker's scratch element over the accumulator's ring.
static void decompose_accumulator(Rotation *R, const Step *S, uint64_t t, ArithElement *rotated)
{
    if (t < R->digits)
    {
        const uint64_t c = t / R->ell, d = t % R->ell;
        const ArithElement *source = component(R->acc, c, R->r);
        if (S->single)
        {
            arith_mul_by_monomial(R->ring, rotated, source, S->exponent[0], 1);
            source = rotated;
        }
        gadget_decompose_digit(&R->digit[t], R->bk[0], source, d, R->log_base);
        return;
    }
    const uint64_t j = t - R->digits;
    arith_mul_by_monomial(R->key_ring, &R->factor[j], &R->one, S->exponent[j], 1);
    arith_to_mul(R->key_ring, &R->factor[j]);
}

static void compute_inner_product(Rotation *R, const Step *S, uint64_t t)
{
    const uint64_t k = t % R->chunks, i = (t / R->chunks) % R->components;
    const uint64_t j = t / (R->chunks * R->components);
    const uint64_t from = k * R->digits / R->chunks, to = (k + 1) * R->digits / R->chunks;
    RNS_MLWE *key = R->bk[S->key[j]];
    ArithElement *out = inner_product_at(R, j, i, k);
    for (uint64_t d = from; d < to; d++)
    {
        const ArithElement *row = component(key[d], i, R->r);
        if (d == from)
            arith_mul(R->key_ring, out, &R->digit[d], row);
        else
            arith_mul_addto(R->key_ring, out, &R->digit[d], row);
    }
}

static void update_accumulator(Rotation *R, const Step *S, uint64_t i)
{
    ArithRing ring = R->key_ring;
    ArithElement sum;
    arith_new(ring, &sum);
    arith_zero_in(ring, &sum, arith_mul_domain(ring));
    for (uint64_t j = 0; j < S->terms; j++)
    {
        ArithElement *y = inner_product_at(R, j, i, 0);
        for (uint64_t k = 1; k < R->chunks; k++)
            arith_add(ring, y, y, inner_product_at(R, j, i, k));
        if (S->single)
            arith_add(ring, &sum, &sum, y);
        else
            arith_mul_addto(ring, &sum, y, &R->factor[j]);
    }
    arith_to_canonical(ring, &sum);
    arith_round_division(ring, &sum, R->ring);
    ArithElement *acc = component(R->acc, i, R->r);
    arith_add(R->ring, acc, acc, &sum);
    arith_free(ring, &sum);
}

static void run_task(Rotation *R, uint64_t stage, uint64_t t, ArithElement *rotated)
{
    const Step *S = &R->step[stage / 3];
    switch (stage % 3)
    {
    case 0:
        decompose_accumulator(R, S, t, rotated);
        break;
    case 1:
        compute_inner_product(R, S, t);
        break;
    default:
        update_accumulator(R, S, t);
        break;
    }
}

static void open_stage(Rotation *R, uint64_t stage)
{
    atomic_store(&R->stage, stage);
    pthread_mutex_lock(&R->lock);
    pthread_cond_broadcast(&R->stage_opened);
    pthread_mutex_unlock(&R->lock);
}

static void wait_for_stage(Rotation *R, uint64_t stage)
{
    for (int spin = 0; spin < WAIT_SPINS; spin++)
    {
        if (atomic_load(&R->stage) >= stage)
            return;
        CPU_RELAX();
    }
    pthread_mutex_lock(&R->lock);
    while (atomic_load(&R->stage) < stage)
        pthread_cond_wait(&R->stage_opened, &R->lock);
    pthread_mutex_unlock(&R->lock);
}

static void rotation_worker(void *ctx, uint64_t item)
{
    (void)item;
    Rotation *R = (Rotation *)ctx;
    ArithElement rotated;
    arith_new(R->ring, &rotated);
    uint64_t stage = 0;
    for (;;)
    {
        // A worker that starts late joins at the stage already open.
        const uint64_t open = (uint64_t)atomic_load(&R->stage);
        if (open > stage)
            stage = open;
        if (stage >= R->stages)
            break;
        for (;;)
        {
            const uint64_t t = (uint64_t)atomic_fetch_add(&R->taken[stage], 1);
            if (t >= R->tasks[stage])
                break;
            run_task(R, stage, t, &rotated);
            if ((uint64_t)atomic_fetch_add(&R->done[stage], 1) + 1 == R->tasks[stage])
                open_stage(R, stage + 1);
        }
        wait_for_stage(R, stage + 1);
        stage++;
    }
    arith_free(R->ring, &rotated);
}

static void rotate(RNSc_MLWE acc, const uint64_t *a, uint64_t n, RNS_MLWE *const *bk,
                   uint64_t unfolding, uint64_t ell, uint64_t log_base, uint64_t threads)
{
    if (n == 0)
        return;
    Rotation R;
    R.acc = acc;
    R.ring = acc->ring;
    R.key_ring = bk[0][0]->ring;
    R.bk = bk;
    R.r = acc->r;
    R.ell = ell;
    R.log_base = log_base;
    R.components = R.r + 1;
    R.digits = R.components * ell;
    prepare_steps(&R, a, n, unfolding);
    if (R.steps == 0)
    {
        free(R.step);
        return;
    }
    // Enough inner-product tasks for every thread, twice over, when there are few terms.
    const uint64_t per_chunk = R.components * R.max_terms;
    const uint64_t chunks = (2 * threads + per_chunk - 1) / per_chunk;
    R.chunks = chunks > R.digits ? R.digits : chunks;

    ArithRing ring = R.key_ring;
    const uint64_t value = 1;
    arith_new(ring, &R.one);
    arith_from_int_array(ring, &R.one, &value, 1);
    arith_to_canonical(ring, &R.one);
    const uint64_t products = R.max_terms * R.components * R.chunks;
    R.digit = (ArithElement *)safe_malloc(R.digits * sizeof(ArithElement));
    R.factor = (ArithElement *)safe_malloc(R.max_terms * sizeof(ArithElement));
    R.inner_product = (ArithElement *)safe_malloc(products * sizeof(ArithElement));
    for (uint64_t i = 0; i < R.digits; i++)
        arith_new(ring, &R.digit[i]);
    for (uint64_t i = 0; i < R.max_terms; i++)
        arith_new(ring, &R.factor[i]);
    for (uint64_t i = 0; i < products; i++)
        arith_new(ring, &R.inner_product[i]);

    R.stages = 3 * R.steps;
    R.tasks = (uint64_t *)safe_malloc(R.stages * sizeof(uint64_t));
    R.taken = (atomic_uint_fast64_t *)safe_malloc(R.stages * sizeof(atomic_uint_fast64_t));
    R.done = (atomic_uint_fast64_t *)safe_malloc(R.stages * sizeof(atomic_uint_fast64_t));
    for (uint64_t s = 0; s < R.stages; s++)
    {
        R.tasks[s] = stage_tasks(&R, s);
        atomic_init(&R.taken[s], 0);
        atomic_init(&R.done[s], 0);
    }
    atomic_init(&R.stage, 0);
    pthread_mutex_init(&R.lock, NULL);
    pthread_cond_init(&R.stage_opened, NULL);

    vfhe_parallel_for(threads, threads, rotation_worker, &R);

    pthread_mutex_destroy(&R.lock);
    pthread_cond_destroy(&R.stage_opened);
    free(R.tasks);
    free(R.taken);
    free(R.done);
    for (uint64_t i = 0; i < R.digits; i++)
        arith_free(ring, &R.digit[i]);
    for (uint64_t i = 0; i < R.max_terms; i++)
        arith_free(ring, &R.factor[i]);
    for (uint64_t i = 0; i < products; i++)
        arith_free(ring, &R.inner_product[i]);
    free(R.digit);
    free(R.factor);
    free(R.inner_product);
    arith_free(ring, &R.one);
    for (uint64_t s = 0; s < R.steps; s++)
    {
        free(R.step[s].key);
        free(R.step[s].exponent);
    }
    free(R.step);
}

void cggi16_blind_rotate(RNSc_MLWE acc, const uint64_t *a, uint64_t n, RNS_MLWE *const *bk,
                         uint64_t unfolding, uint64_t ell, uint64_t log_base, uint64_t n_threads)
{
    rotate(acc, a, n, bk, unfolding, ell, log_base, vfhe_threads_for(n_threads, UINT64_MAX));
}

typedef struct
{
    RNSc_MLWE *acc;
    const uint64_t *a;
    uint64_t n, unfolding, ell, log_base;
    RNS_MLWE *const *bk;
} Batch;

static void rotate_one_of_batch(void *ctx, uint64_t k)
{
    const Batch *B = (const Batch *)ctx;
    rotate(B->acc[k], B->a + k * B->n, B->n, B->bk, B->unfolding, B->ell, B->log_base, 1);
}

void cggi16_blind_rotate_batch(RNSc_MLWE *acc, const uint64_t *a, uint64_t count, uint64_t n,
                               RNS_MLWE *const *bk, uint64_t unfolding, uint64_t ell,
                               uint64_t log_base, uint64_t n_threads)
{
    Batch B = {acc, a, n, unfolding, ell, log_base, bk};
    vfhe_parallel_for(count, n_threads, rotate_one_of_batch, &B);
}
