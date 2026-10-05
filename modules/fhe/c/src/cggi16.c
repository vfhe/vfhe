// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "fhe.h"
#include "cggi16_team.h"
#include "zyl17.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <util.h>

static void rotation_init(UnfoldedRotation *R, RNSc_MLWE acc, const uint64_t *a, uint64_t n,
                          RNS_MLWE *const *bk, uint64_t unfolding, int all_patterns,
                          ZYL17_Combination combination, ZYL17_MonomialTable monomials,
                          uint64_t ell, uint64_t log_base)
{
    R->a = a;
    R->n = n;
    R->unfolding = unfolding;
    R->groups = (n + unfolding - 1) / unfolding;
    R->keys_per_group = (1ULL << unfolding) - (all_patterns ? 0 : 1);
    R->two_n = 2 * acc->ring->N;
    R->bk = bk;
    R->all_patterns = all_patterns;
    R->combination = combination;
    R->monomials = monomials;
    R->ell = ell;
    R->log_base = log_base;
    R->r = acc->r;
    R->key_ring = n ? bk[0][0]->ring : acc->ring;
}

// The accumulator and its two scratch samples; `cur` holds the value and may
// be either of the caller's sample and `spare` after a step.
typedef struct
{
    RNSc_MLWE caller, cur, spare, rotated;
} Accumulator;

static void accumulator_init(Accumulator *A, RNSc_MLWE acc)
{
    A->caller = A->cur = acc;
    A->spare = mlwe_alloc_sample(acc->ring, acc->r);
    A->rotated = mlwe_alloc_sample(acc->ring, acc->r);
}

static void accumulator_finish(Accumulator *A)
{
    if (A->cur != A->caller)
    {
        mlwe_copy_RNS_sample(A->caller, A->cur);
        free_mlwe_RNS_sample(A->cur);
    }
    else
        free_mlwe_RNS_sample(A->spare);
    free_mlwe_RNS_sample(A->rotated);
}

static void apply_group(const UnfoldedRotation *R, uint64_t g, const CombinedKey *ck,
                        Accumulator *A)
{
    if (zyl17_group_is_combined(R, g))
    {
        if (ck->zero)
            return;
        mgsw_external_product_canonical(A->spare, ck->rows, A->cur, R->ell, R->log_base);
    }
    else
    {
        const uint64_t e = R->a[g * R->unfolding] % R->two_n;
        if (e == 0)
            return;
        mlwe_RNSc_mul_by_xai_minus1(A->rotated, A->cur, e);
        mgsw_external_product_canonical(A->spare, R->bk[g * R->keys_per_group], A->rotated, R->ell,
                                        R->log_base);
    }
    if (!R->all_patterns)
        mlwe_addto_RNSc_sample(A->spare, A->cur);
    RNSc_MLWE t = A->cur;
    A->cur = A->spare;
    A->spare = t;
}

static void rotate_sequential(const UnfoldedRotation *R, RNSc_MLWE acc)
{
    const int combine = zyl17_needs_combination(R);
    CombinedKey ck = {0};
    CombineScratch s = {0};
    if (combine)
    {
        zyl17_combined_key_init(&ck, R);
        zyl17_scratch_init(&s, R);
    }
    Accumulator A;
    accumulator_init(&A, acc);
    for (uint64_t g = 0; g < R->groups; g++)
    {
        zyl17_combine_group(R, g, &ck, &s);
        apply_group(R, g, &ck, &A);
    }
    accumulator_finish(&A);
    if (combine)
    {
        zyl17_combined_key_free(&ck, R);
        zyl17_scratch_free(&s, R);
    }
}

/* ------------------------------------------------------------------------------------------------
 * The pipeline. Item 0 of the parallel loop runs the external products in group order; every
 * other item claims the next unbuilt group once its slot g % slots is free -- the external product
 * of the group that last used it has finished -- and combines its key there. When the next group
 * has not been claimed yet, item 0 claims and combines it itself, so it waits only on a group a
 * running helper is combining, and a helper waits only on item 0's progress: no thread count,
 * including a loop that runs its items one after the other, can deadlock it.
 *
 * A waiting thread sleeps rather than spins (a spinning helper on a sibling hardware thread slows
 * the external products down), and each finished group wakes one helper, since it frees one slot.
 * ------------------------------------------------------------------------------------------------
 */
typedef struct
{
    const UnfoldedRotation *R;
    RNSc_MLWE acc;
    uint64_t slots;
    CombinedKey *slot;
    atomic_uint_fast64_t *slot_group; // g + 1 once slot g % slots holds group g
    pthread_mutex_t lock;
    pthread_cond_t room;  // a slot was freed
    pthread_cond_t ready; // a slot was filled
    uint64_t next_claim;  // first group no thread has claimed, under `lock`
    uint64_t consumed;    // groups whose external product is done, under `lock`
} Pipeline;

static int slot_holds(Pipeline *P, uint64_t g)
{
    return atomic_load_explicit(&P->slot_group[g % P->slots], memory_order_acquire) == g + 1;
}

// The next group for a helper, once its slot is free; `groups` when none is left.
static uint64_t claim_for_helper(Pipeline *P)
{
    const uint64_t groups = P->R->groups;
    pthread_mutex_lock(&P->lock);
    while (P->next_claim < groups && P->next_claim >= P->consumed + P->slots)
        pthread_cond_wait(&P->room, &P->lock);
    const uint64_t g = P->next_claim < groups ? P->next_claim++ : groups;
    pthread_mutex_unlock(&P->lock);
    return g;
}

static void pipeline_combine(Pipeline *P, CombineScratch *s)
{
    const UnfoldedRotation *R = P->R;
    for (;;)
    {
        const uint64_t g = claim_for_helper(P);
        if (g >= R->groups)
            return;
        zyl17_combine_group(R, g, &P->slot[g % P->slots], s);
        atomic_store_explicit(&P->slot_group[g % P->slots], g + 1, memory_order_release);
        pthread_mutex_lock(&P->lock);
        pthread_cond_signal(&P->ready);
        pthread_mutex_unlock(&P->lock);
    }
}

static void pipeline_apply(Pipeline *P, CombineScratch *s)
{
    const UnfoldedRotation *R = P->R;
    Accumulator A;
    accumulator_init(&A, P->acc);
    for (uint64_t g = 0; g < R->groups; g++)
    {
        const uint64_t k = g % P->slots;
        if (!slot_holds(P, g))
        {
            pthread_mutex_lock(&P->lock);
            const int unclaimed = P->next_claim == g;
            if (unclaimed)
                P->next_claim++;
            else
                while (!slot_holds(P, g))
                    pthread_cond_wait(&P->ready, &P->lock);
            pthread_mutex_unlock(&P->lock);
            if (unclaimed)
                zyl17_combine_group(R, g, &P->slot[k], s);
        }
        apply_group(R, g, &P->slot[k], &A);
        pthread_mutex_lock(&P->lock);
        P->consumed = g + 1;
        if (P->consumed == R->groups)
            pthread_cond_broadcast(&P->room);
        else
            pthread_cond_signal(&P->room);
        pthread_mutex_unlock(&P->lock);
    }
    accumulator_finish(&A);
}

static void pipeline_body(void *ctx, uint64_t item)
{
    Pipeline *P = (Pipeline *)ctx;
    CombineScratch s;
    zyl17_scratch_init(&s, P->R);
    if (item == 0)
        pipeline_apply(P, &s);
    else
        pipeline_combine(P, &s);
    zyl17_scratch_free(&s, P->R);
}

void cggi16_blind_rotate(RNSc_MLWE acc, const uint64_t *a, uint64_t n, RNS_MLWE *const *bk,
                         uint64_t unfolding, int all_patterns, ZYL17_Combination combination,
                         ZYL17_MonomialTable monomials, uint64_t ell, uint64_t log_base,
                         CGGI16_Parallelism parallelism, uint64_t n_threads)
{
    UnfoldedRotation R;
    rotation_init(&R, acc, a, n, bk, unfolding, all_patterns, combination, monomials, ell,
                  log_base);
    // On one thread too, since it reads the keys in its own domain.
    if (parallelism == CGGI16_DATA_PARALLEL)
    {
        cggi16_blind_rotate_team(&R, acc, vfhe_threads_for(n_threads, UINT64_MAX));
        return;
    }
    const uint64_t threads =
        zyl17_needs_combination(&R) ? vfhe_threads_for(n_threads, R.groups) : 1;
    if (threads <= 1)
    {
        rotate_sequential(&R, acc);
        return;
    }

    Pipeline P;
    P.R = &R;
    P.acc = acc;
    P.slots = threads + 1;
    P.slot = (CombinedKey *)safe_malloc(P.slots * sizeof(CombinedKey));
    P.slot_group = (atomic_uint_fast64_t *)safe_malloc(P.slots * sizeof(atomic_uint_fast64_t));
    for (uint64_t k = 0; k < P.slots; k++)
    {
        zyl17_combined_key_init(&P.slot[k], &R);
        atomic_init(&P.slot_group[k], 0);
    }
    P.next_claim = 0;
    P.consumed = 0;
    pthread_mutex_init(&P.lock, NULL);
    pthread_cond_init(&P.room, NULL);
    pthread_cond_init(&P.ready, NULL);

    vfhe_parallel_for(threads, threads, pipeline_body, &P);

    pthread_mutex_destroy(&P.lock);
    pthread_cond_destroy(&P.room);
    pthread_cond_destroy(&P.ready);
    for (uint64_t k = 0; k < P.slots; k++)
        zyl17_combined_key_free(&P.slot[k], &R);
    free(P.slot);
    free(P.slot_group);
}

typedef struct
{
    const UnfoldedRotation *R;
    RNSc_MLWE *acc;
} Batch;

static void batch_body(void *ctx, uint64_t k)
{
    const Batch *B = (const Batch *)ctx;
    UnfoldedRotation R = *B->R;
    R.a = B->R->a + k * R.n;
    rotate_sequential(&R, B->acc[k]);
}

void cggi16_blind_rotate_batch(RNSc_MLWE *acc, const uint64_t *a, uint64_t count, uint64_t n,
                               RNS_MLWE *const *bk, uint64_t unfolding, int all_patterns,
                               ZYL17_Combination combination, ZYL17_MonomialTable monomials,
                               uint64_t ell, uint64_t log_base, uint64_t n_threads)
{
    if (count == 0)
        return;
    UnfoldedRotation R;
    rotation_init(&R, acc[0], a, n, bk, unfolding, all_patterns, combination, monomials, ell,
                  log_base);
    Batch B = {&R, acc};
    vfhe_parallel_for(count, n_threads, batch_body, &B);
}
