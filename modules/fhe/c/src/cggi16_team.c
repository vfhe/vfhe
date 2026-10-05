// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "cggi16_team.h"

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
 * The data-parallel blind rotation. A group's step is computed without its combined key:
 *
 *   acc <- [acc +] sum_j f_j * (sum_t D_t (.) BK_j[t]),   D_t the gadget digits of acc,
 *
 * where t runs over the (r + 1) * ell rows of an MGSW key and f_j is X^e_j [ZYL+17] or
 * X^e_j - 1 [BMMP18], applied to the inner products rather than folded into the keys. The digits
 * are computed once, every key is read once, and the steps of one group split into three phases
 * of independent tasks:
 *
 *   0  digits:   D_t for every t, and (evaluation) the transformed factors f_j;
 *   1  partials: P[j][i][k] = sum over chunk k of the digits of D_t (.) BK_j[t].component(i);
 *   2  finalize: component i of acc from sum_j f_j * sum_k P[j][i][k], back to canonical, rescaled
 *                to the accumulator's ring, plus acc (BMMP18).
 *
 * A [BMMP18] group of one coefficient decomposes (X^a - 1) * acc instead and has no factor, as
 * the sequential rotation does, so the result is the same bit for bit. Groups whose every factor
 * vanishes are left out of the schedule.
 *
 * The team is one vfhe_parallel_for whose items all run the whole schedule. A participant claims
 * tasks of the current phase until none is left, the one finishing the last task advances the
 * phase, and the others wait for that (spinning briefly, then sleeping). A waiter only waits on
 * tasks a running participant has claimed, so the schedule completes whatever the number of
 * threads -- including a loop that runs its items one after the other, where item 0 does it all.
 * ------------------------------------------------------------------------------------------------
 */

#define TEAM_SPINS 4096

typedef struct
{
    uint64_t group, count;
    uint64_t *key;      // index of each active pattern's key in bk
    uint64_t *exponent; // its e_j
    int rotate_input;   // a [BMMP18] group of one coefficient: decompose (X^a - 1) * acc
} GroupPlan;

typedef struct
{
    const UnfoldedRotation *R;
    RNSc_MLWE acc;
    ArithRing ring; // the accumulator's
    uint64_t comps, digits, chunks, max_patterns;
    int evaluate_factors;

    GroupPlan *plan;
    uint64_t scheduled;
    ArithElement one;     // canonical 1 over the key's ring
    ArithElement *digit;  // [digits]
    ArithElement *factor; // [max_patterns]
    ArithElement *partial;

    uint64_t phases;
    uint64_t *tasks;
    atomic_uint_fast64_t *next, *done;
    atomic_uint_fast64_t phase;
    pthread_mutex_t lock;
    pthread_cond_t advanced;
} Team;

static ArithElement *component(RNS_MLWE c, uint64_t i, uint64_t r)
{
    return i < r ? &c->a[i] : &c->b;
}

static ArithElement *partial_at(Team *T, uint64_t p, uint64_t i, uint64_t k)
{
    return &T->partial[(p * T->comps + i) * T->chunks + k];
}

static void plan_groups(Team *T)
{
    const UnfoldedRotation *R = T->R;
    const uint64_t skip = R->all_patterns ? 0 : 1;
    T->plan = (GroupPlan *)safe_malloc(R->groups * sizeof(GroupPlan));
    T->scheduled = 0;
    T->max_patterns = 1;
    for (uint64_t g = 0; g < R->groups; g++)
    {
        GroupPlan *G = &T->plan[T->scheduled];
        const uint64_t patterns = 1ULL << zyl17_group_size(R, g);
        G->group = g;
        G->count = 0;
        G->rotate_input = !zyl17_group_is_combined(R, g);
        G->key = (uint64_t *)safe_malloc(patterns * sizeof(uint64_t));
        G->exponent = (uint64_t *)safe_malloc(patterns * sizeof(uint64_t));
        for (uint64_t j = skip; j < patterns; j++)
        {
            const uint64_t e = zyl17_pattern_exponent(R, g, j);
            if (e == 0 && !R->all_patterns)
                continue;
            G->key[G->count] = g * R->keys_per_group + (j - skip);
            G->exponent[G->count] = e;
            G->count++;
        }
        if (G->count == 0)
        {
            free(G->key);
            free(G->exponent);
            continue;
        }
        if (G->count > T->max_patterns)
            T->max_patterns = G->count;
        T->scheduled++;
    }
}

static uint64_t phase_tasks(const Team *T, uint64_t phase)
{
    const GroupPlan *G = &T->plan[phase / 3];
    switch (phase % 3)
    {
    case 0:
        return T->digits + (T->evaluate_factors && !G->rotate_input ? G->count : 0);
    case 1:
        return G->count * T->comps * T->chunks;
    default:
        return T->comps;
    }
}

// `rotated` is the participant's scratch element over the accumulator's ring.
static void run_digits(Team *T, const GroupPlan *G, uint64_t t, ArithElement *rotated)
{
    const UnfoldedRotation *R = T->R;
    if (t < T->digits)
    {
        const uint64_t c = t / R->ell, d = t % R->ell;
        const ArithElement *source = component(T->acc, c, R->r);
        if (G->rotate_input)
        {
            arith_mul_by_monomial(T->ring, rotated, source, G->exponent[0], 1);
            source = rotated;
        }
        gadget_decompose_digit(&T->digit[t], R->bk[0], source, d, R->log_base);
        return;
    }
    const uint64_t p = t - T->digits, e = G->exponent[p];
    if (e == 0)
        return; // a unit factor, applied as an addition
    arith_mul_by_monomial(R->key_ring, &T->factor[p], &T->one, e, !R->all_patterns);
    arith_to_mul(R->key_ring, &T->factor[p]);
}

static void run_partial(Team *T, const GroupPlan *G, uint64_t t)
{
    const UnfoldedRotation *R = T->R;
    const uint64_t k = t % T->chunks, i = (t / T->chunks) % T->comps;
    const uint64_t p = t / (T->chunks * T->comps);
    const uint64_t from = k * T->digits / T->chunks, to = (k + 1) * T->digits / T->chunks;
    RNS_MLWE *key = R->bk[G->key[p]];
    ArithElement *out = partial_at(T, p, i, k);
    for (uint64_t d = from; d < to; d++)
    {
        const ArithElement *row = component(key[d], i, R->r);
        if (d == from)
            arith_mul(R->key_ring, out, &T->digit[d], row);
        else
            arith_mul_addto(R->key_ring, out, &T->digit[d], row);
    }
}

// wide += f * y, with f = X^e or X^e - 1 in the mul domain.
static void add_factor_times(Team *T, ArithElement *wide, const ArithElement *y, uint64_t p,
                             uint64_t e)
{
    const UnfoldedRotation *R = T->R;
    ArithRing ring = R->key_ring;
    const uint64_t N = ring->N;
    if (T->evaluate_factors)
    {
        if (e == 0)
            arith_add(ring, wide, wide, y);
        else
            arith_mul_addto(ring, wide, y, &T->factor[p]);
        return;
    }
    // X^e = sign * X^m from the table; the -1 is a subtraction of y.
    const uint64_t m = e % N;
    const int negate = e >= N;
    if (m == 0)
    {
        if (negate)
            arith_sub(ring, wide, wide, y);
        else
            arith_add(ring, wide, wide, y);
    }
    else if (negate)
        arith_mul_subto(ring, wide, y, zyl17_monomial(R->monomials, m));
    else
        arith_mul_addto(ring, wide, y, zyl17_monomial(R->monomials, m));
    if (!R->all_patterns)
        arith_sub(ring, wide, wide, y);
}

static void run_finalize(Team *T, const GroupPlan *G, uint64_t i)
{
    const UnfoldedRotation *R = T->R;
    ArithRing ring = R->key_ring;
    ArithElement wide;
    arith_new(ring, &wide);
    arith_zero_in(ring, &wide, arith_mul_domain(ring));
    for (uint64_t p = 0; p < G->count; p++)
    {
        ArithElement *y = partial_at(T, p, i, 0);
        for (uint64_t k = 1; k < T->chunks; k++)
            arith_add(ring, y, y, partial_at(T, p, i, k));
        if (G->rotate_input)
            arith_add(ring, &wide, &wide, y);
        else
            add_factor_times(T, &wide, y, p, G->exponent[p]);
    }
    arith_to_canonical(ring, &wide);
    arith_round_division(ring, &wide, T->ring);
    ArithElement *acc = component(T->acc, i, R->r);
    if (R->all_patterns)
        arith_copy(T->ring, acc, &wide);
    else
        arith_add(T->ring, acc, acc, &wide);
    arith_free(ring, &wide);
}

static void run_task(Team *T, uint64_t phase, uint64_t t, ArithElement *rotated)
{
    const GroupPlan *G = &T->plan[phase / 3];
    switch (phase % 3)
    {
    case 0:
        run_digits(T, G, t, rotated);
        break;
    case 1:
        run_partial(T, G, t);
        break;
    default:
        run_finalize(T, G, t);
        break;
    }
}

static void advance(Team *T, uint64_t phase)
{
    atomic_store(&T->phase, phase + 1);
    pthread_mutex_lock(&T->lock);
    pthread_cond_broadcast(&T->advanced);
    pthread_mutex_unlock(&T->lock);
}

static void wait_past(Team *T, uint64_t phase)
{
    for (int spin = 0; spin < TEAM_SPINS; spin++)
    {
        if (atomic_load(&T->phase) > phase)
            return;
        CPU_RELAX();
    }
    pthread_mutex_lock(&T->lock);
    while (atomic_load(&T->phase) <= phase)
        pthread_cond_wait(&T->advanced, &T->lock);
    pthread_mutex_unlock(&T->lock);
}

static void team_body(void *ctx, uint64_t item)
{
    (void)item;
    Team *T = (Team *)ctx;
    ArithElement rotated;
    arith_new(T->ring, &rotated);
    uint64_t p = 0;
    for (;;)
    {
        const uint64_t current = (uint64_t)atomic_load(&T->phase);
        if (current > p)
            p = current;
        if (p >= T->phases)
            break;
        for (;;)
        {
            const uint64_t t = (uint64_t)atomic_fetch_add(&T->next[p], 1);
            if (t >= T->tasks[p])
                break;
            run_task(T, p, t, &rotated);
            if ((uint64_t)atomic_fetch_add(&T->done[p], 1) + 1 == T->tasks[p])
                advance(T, p);
        }
        wait_past(T, p);
        p++;
    }
    arith_free(T->ring, &rotated);
}

void cggi16_blind_rotate_team(const UnfoldedRotation *R, RNSc_MLWE acc, uint64_t threads)
{
    Team T;
    T.R = R;
    T.acc = acc;
    T.ring = acc->ring;
    T.comps = R->r + 1;
    T.digits = T.comps * R->ell;
    T.evaluate_factors = !(R->combination == ZYL17_COMBINE_PRECOMPUTED && R->monomials);
    plan_groups(&T);
    if (T.scheduled == 0)
    {
        free(T.plan);
        return;
    }
    // Enough partial tasks for every thread, twice over, when there are few patterns.
    uint64_t chunks = (2 * threads + T.comps * T.max_patterns - 1) / (T.comps * T.max_patterns);
    T.chunks = chunks < 1 ? 1 : (chunks > T.digits ? T.digits : chunks);

    ArithRing ring = R->key_ring;
    const uint64_t value = 1;
    arith_new(ring, &T.one);
    arith_from_int_array(ring, &T.one, &value, 1);
    arith_to_canonical(ring, &T.one);
    const uint64_t n_partial = T.max_patterns * T.comps * T.chunks;
    T.digit = (ArithElement *)safe_malloc(T.digits * sizeof(ArithElement));
    T.factor = (ArithElement *)safe_malloc(T.max_patterns * sizeof(ArithElement));
    T.partial = (ArithElement *)safe_malloc(n_partial * sizeof(ArithElement));
    for (uint64_t i = 0; i < T.digits; i++)
        arith_new(ring, &T.digit[i]);
    for (uint64_t i = 0; i < T.max_patterns; i++)
        arith_new(ring, &T.factor[i]);
    for (uint64_t i = 0; i < n_partial; i++)
        arith_new(ring, &T.partial[i]);

    T.phases = 3 * T.scheduled;
    T.tasks = (uint64_t *)safe_malloc(T.phases * sizeof(uint64_t));
    T.next = (atomic_uint_fast64_t *)safe_malloc(T.phases * sizeof(atomic_uint_fast64_t));
    T.done = (atomic_uint_fast64_t *)safe_malloc(T.phases * sizeof(atomic_uint_fast64_t));
    for (uint64_t p = 0; p < T.phases; p++)
    {
        T.tasks[p] = phase_tasks(&T, p);
        atomic_init(&T.next[p], 0);
        atomic_init(&T.done[p], 0);
    }
    atomic_init(&T.phase, 0);
    pthread_mutex_init(&T.lock, NULL);
    pthread_cond_init(&T.advanced, NULL);

    vfhe_parallel_for(threads, threads, team_body, &T);

    pthread_mutex_destroy(&T.lock);
    pthread_cond_destroy(&T.advanced);
    free(T.tasks);
    free(T.next);
    free(T.done);
    for (uint64_t i = 0; i < T.digits; i++)
        arith_free(ring, &T.digit[i]);
    for (uint64_t i = 0; i < T.max_patterns; i++)
        arith_free(ring, &T.factor[i]);
    for (uint64_t i = 0; i < n_partial; i++)
        arith_free(ring, &T.partial[i]);
    free(T.digit);
    free(T.factor);
    free(T.partial);
    arith_free(ring, &T.one);
    for (uint64_t g = 0; g < T.scheduled; g++)
    {
        free(T.plan[g].key);
        free(T.plan[g].exponent);
    }
    free(T.plan);
}
