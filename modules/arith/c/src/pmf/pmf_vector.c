// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// Vectors of pseudo-Mersenne elements, stored as L limb planes.
//
// A single element fills one AVX-512 register with its own limbs, which puts
// every carry on the critical path across lanes. A vector turns the layout
// inside out: plane j holds limb j of every element, so one register holds one
// limb of eight DIFFERENT elements, carries move between planes at a fixed lane,
// and no step ever needs lanes to interact. That is the whole reason this file
// exists, and it is why the arithmetic here is worth more than a loop over the
// element kernels.
//
// The kernels are written once, against the `pv_t` word in pmf_vec_ops.h, and
// compile to eight-wide IFMA or to one-wide scalar. They mirror the algorithms
// in pmf.c -- the same fold, the same passes, the same bounds -- without sharing
// its code: those are the oracle these are tested against.
//
// Each kernel walks the vector one group of PMF_VEC_GROUP elements at a time,
// holding that group's limbs and its whole reduction in registers and storing
// once. Working a group at a time rather than a plane at a time is what keeps
// the reduction tail -- a chain along the limbs of one element -- in registers
// instead of spilling 2L intermediate planes to memory, and it is also what
// makes `out` free to alias any input.
#include <arith.h>
#include <assert.h>
#include <blake3.h>
#include <crypto.h>
#include <stdlib.h>
#include <string.h>

#include "arith_internal.h"
#include "alloc.h"
#include "pmf_vec_group.h"

uint64_t pmf_vec_padded_length(uint64_t n)
{
    const uint64_t unit = PMF_LANES;
    return ((n + unit - 1) / unit) * unit;
}

// The three entry-point shapes below differ only in where the right-hand operand
// comes from -- the other vector, or one element broadcast -- so each is a walk
// over the groups around one of the cores in pmf_vec_group.h. Every kernel in
// this file is a body taking the params, behind an entry point that dispatches
// it with PMF_WITH_LIMBS.

#define PMF_VEC_BINARY(name, core)                                                                 \
    PMF_INLINE void name##_body(PMFVector out, const PMFVector a, const PMFVector b, PMFParams P)  \
    {                                                                                              \
        const uint64_t L = P->limbs;                                                               \
        pv_t av[PMF_MAX_LIMBS], bv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];                           \
        for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)                               \
        {                                                                                          \
            pmf_vec_load_group(av, a, i, L);                                                       \
            pmf_vec_load_group(bv, b, i, L);                                                       \
            core(T, av, bv, P);                                                                    \
            pmf_vec_store_group(out, i, T, L);                                                     \
        }                                                                                          \
    }                                                                                              \
    void name(PMFVector out, const PMFVector a, const PMFVector b)                                 \
    {                                                                                              \
        PMF_WITH_LIMBS(a->params, name##_body(out, a, b, &P_));                                    \
    }

#define PMF_VEC_BROADCAST(name, core)                                                              \
    PMF_INLINE void name##_body(PMFVector out, const PMFVector a, const uint64_t *s, PMFParams P)  \
    {                                                                                              \
        const uint64_t L = P->limbs;                                                               \
        pv_t av[PMF_MAX_LIMBS], sv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];                           \
        pmf_vec_load_scalar(sv, s, L);                                                             \
        for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)                               \
        {                                                                                          \
            pmf_vec_load_group(av, a, i, L);                                                       \
            core(T, av, sv, P);                                                                    \
            pmf_vec_store_group(out, i, T, L);                                                     \
        }                                                                                          \
    }                                                                                              \
    void name(PMFVector out, const PMFVector a, const uint64_t *s)                                 \
    {                                                                                              \
        PMF_WITH_LIMBS(a->params, name##_body(out, a, s, &P_));                                    \
    }

PMF_VEC_BINARY(pmf_vec_add, pmf_vec_add_group)
PMF_VEC_BINARY(pmf_vec_sub, pmf_vec_sub_group)
PMF_VEC_BINARY(pmf_vec_mul, pmf_vec_mul_group)

PMF_VEC_BROADCAST(pmf_vec_add_scalar, pmf_vec_add_group)
PMF_VEC_BROADCAST(pmf_vec_sub_scalar, pmf_vec_sub_group)
PMF_VEC_BROADCAST(pmf_vec_scale, pmf_vec_mul_group)

PMF_INLINE void pmf_vec_scalar_sub_body(PMFVector out, const uint64_t *s, const PMFVector a,
                                        PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t av[PMF_MAX_LIMBS], sv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(sv, s, L);
    for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_group(av, a, i, L);
        pmf_vec_sub_group(T, sv, av, P);
        pmf_vec_store_group(out, i, T, L);
    }
}

void pmf_vec_scalar_sub(PMFVector out, const uint64_t *s, const PMFVector a)
{
    PMF_WITH_LIMBS(a->params, pmf_vec_scalar_sub_body(out, s, a, &P_));
}

PMF_INLINE void pmf_vec_neg_body(PMFVector out, const PMFVector a, PMFParams P)
{
    const uint64_t L = P->limbs;
    const pv_t mask = pv_mask52();
    pv_t av[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS];

    for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_group(av, a, i, L);

        // p - a, except for zero, whose negation is zero rather than p.
        pv_t any = pv_zero();
        for (uint64_t k = 0; k < L; k++)
            any = pv_or(any, av[k]);
        // A lane is nonzero exactly when `any` or its negation has bit 63 set.
        const pv_t nonzero = pv_mask_from_flag(pv_srl(pv_or(any, pv_sub(pv_zero(), any)), 63));

        pv_t borrow = pv_zero();
        for (uint64_t k = 0; k < L; k++)
        {
            const pv_t difference = pv_sub(pv_sub(pv_bcast(P->p[k]), av[k]), borrow);
            borrow = pv_borrow(difference);
            T[k] = pv_and(pv_and(difference, mask), nonzero);
        }
        assert(pv_is_zero(borrow)); // a < p, so p - a never borrows out

        pmf_vec_store_group(out, i, T, L);
    }
}

void pmf_vec_neg(PMFVector out, const PMFVector a)
{
    PMF_WITH_LIMBS(a->params, pmf_vec_neg_body(out, a, &P_));
}

void pmf_vec_sum(uint64_t *out, const PMFVector a)
{
    const uint64_t L = a->params->limbs;
    // Every live element is canonical, so each limb is below 2^52 and 255 of
    // them add to less than 2^60 -- the most the reduction tail accepts.
    const uint64_t BATCH = 255;
    pv_t acc[PMF_MAX_LIMBS + 1];
    pv_t av[PMF_MAX_LIMBS];
    _Alignas(64) uint64_t partial[PMF_LANES * PMF_MAX_LIMBS];
    uint64_t element[PMF_LANES];

    for (uint64_t k = 0; k <= L; k++)
        acc[k] = pv_zero();

    // Whole groups of live elements only: the padding is canonical but carries
    // no value, so it must not reach the total.
    const uint64_t whole = (a->n / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    uint64_t since_reduce = 0;
    for (uint64_t i = 0; i < whole; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_group(av, a, i, L);
        for (uint64_t k = 0; k < L; k++)
            acc[k] = pv_add(acc[k], av[k]);
        if (++since_reduce == BATCH)
        {
            acc[L] = pv_zero();
            pmf_vec_reduce_wide(acc, a->params);
            since_reduce = 0;
        }
    }
    acc[L] = pv_zero();
    pmf_vec_reduce_wide(acc, a->params);

    // The lanes hold PMF_VEC_GROUP separate partial sums; fold them, and the
    // elements past the last whole group, with the element kernel.
    for (uint64_t k = 0; k < L; k++)
        pv_store(partial + k * PMF_LANES, acc[k]);

    for (uint64_t k = 0; k < PMF_LANES; k++)
        out[k] = 0;
    // The limbs at or past L are zero whatever the lane, so set them once;
    // reading them would address past partial's L * PMF_LANES live entries.
    for (uint64_t k = L; k < PMF_LANES; k++)
        element[k] = 0;
    for (uint64_t lane = 0; lane < PMF_VEC_GROUP; lane++)
    {
        for (uint64_t k = 0; k < L; k++)
            element[k] = partial[k * PMF_LANES + lane];
        pmf_add(out, out, element, a->params);
    }
    for (uint64_t i = whole; i < a->n; i++)
    {
        pmf_vec_get_element(element, a, i);
        pmf_add(out, out, element, a->params);
    }
}

// --- adjacent pairs -------------------------------------------------------

// The even- and odd-indexed elements among the 2 * PMF_VEC_GROUP input elements
// from 2 * i0 on, as one group each. Where the input's allocation ends both
// read as zero, which keeps whatever is computed from them canonical. One of
// the two places in this file lanes are rearranged (the other is
// pmf_vec_store_pairs): a pair sits in adjacent lanes and its two halves have
// to end up in the same lane of two registers.
PMF_INLINE void pmf_vec_load_pairs(pv_t *E, pv_t *O, const PMFVector a, uint64_t i0, uint64_t L)
{
    const uint64_t first = 2 * i0;
#if VFHE_HAVE_AVX512IFMA
    if (first + 2 * PMF_VEC_GROUP <= a->allocated_n)
    {
        const __m512i evens = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        const __m512i odds = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        for (uint64_t k = 0; k < L; k++)
        {
            const pv_t lo = pv_load(a->limbs[k] + first);
            const pv_t hi = pv_load(a->limbs[k] + first + PMF_VEC_GROUP);
            E[k] = _mm512_permutex2var_epi64(lo, evens, hi);
            O[k] = _mm512_permutex2var_epi64(lo, odds, hi);
        }
        return;
    }
    // The last group of an output whose padded half-length runs past the
    // input's allocation: gather the pairs that exist, zero the rest.
    const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i at =
        _mm512_add_epi64(_mm512_slli_epi64(lanes, 1), _mm512_set1_epi64((long long)first));
    const uint64_t present = a->allocated_n > first ? (a->allocated_n - first) / 2 : 0;
    const __mmask8 live = (__mmask8)((1u << present) - 1);
    for (uint64_t k = 0; k < L; k++)
    {
        E[k] = _mm512_mask_i64gather_epi64(pv_zero(), live, at, (const void *)a->limbs[k], 8);
        O[k] =
            _mm512_mask_i64gather_epi64(pv_zero(), live, _mm512_add_epi64(at, _mm512_set1_epi64(1)),
                                        (const void *)a->limbs[k], 8);
    }
#else
    const int present = first + 1 < a->allocated_n;
    for (uint64_t k = 0; k < L; k++)
    {
        E[k] = present ? a->limbs[k][first] : 0;
        O[k] = present ? a->limbs[k][first + 1] : 0;
    }
#endif
}

// The fold of the pairs from 2i on, for the output group at i: E + r (O - E).
PMF_INLINE void pmf_vec_fold_group(pv_t *T, const PMFVector a, uint64_t i, const pv_t *R,
                                   PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t E[PMF_MAX_LIMBS], O[PMF_MAX_LIMBS], D[PMF_MAX_LIMBS + 1];

    pmf_vec_load_pairs(E, O, a, i, L);
    pmf_vec_sub_group(D, O, E, P);
    pmf_vec_mul_group(T, D, R, P);
    pmf_vec_add_group(T, E, T, P);
}

PMF_INLINE void pmf_vec_fold_body(PMFVector out, const PMFVector a, const uint64_t *r, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t R[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(R, r, L);
    for (uint64_t i = 0; i < out->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_fold_group(T, a, i, R, P);
        pmf_vec_store_group(out, i, T, L);
    }
}

void pmf_vec_fold(PMFVector out, const PMFVector a, const uint64_t *r)
{
    PMF_WITH_LIMBS(a->params, pmf_vec_fold_body(out, a, r, &P_));
}

// --- inner products of folds ----------------------------------------------
//
// Over the pairs (u, v) of one vector and (x, y) of another, the sums of u x,
// v y and (v - u)(y - x). The first two are q(0) and q(1) for the quadratic
// q(t) = sum (u + t (v - u)) (x + t (y - x)), the third is its leading
// coefficient, so q(2) = 2 q(1) - q(0) + 2 sum (v - u)(y - x).

// The three sums, lane by lane, with limbs added unreduced in between.
typedef struct
{
    pv_t acc[3][PMF_MAX_LIMBS + 1];
    uint64_t since_reduce;
} PMFInnerSums;

PMF_INLINE void pmf_vec_inner_sums_init(PMFInnerSums *S, uint64_t L)
{
    for (uint64_t s = 0; s < 3; s++)
        for (uint64_t k = 0; k <= L; k++)
            S->acc[s][k] = pv_zero();
    S->since_reduce = 0;
}

// acc += T over the first L limbs.
PMF_INLINE void pmf_vec_accumulate_group(pv_t *acc, const pv_t *T, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        acc[k] = pv_add(acc[k], T[k]);
}

// One group of pairs into the sums; V and Y are overwritten. Canonical limbs
// are below 2^52, so 255 groups add to less than 2^60, the most
// pmf_vec_reduce_wide accepts; the sums are reduced that often.
PMF_INLINE void pmf_vec_inner_sums_add(PMFInnerSums *S, const pv_t *U, pv_t *V, const pv_t *X,
                                       pv_t *Y, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t T[PMF_MAX_LIMBS + 1];

    pmf_vec_mul_group(T, U, X, P);
    pmf_vec_accumulate_group(S->acc[0], T, L);
    pmf_vec_mul_group(T, V, Y, P);
    pmf_vec_accumulate_group(S->acc[1], T, L);
    pmf_vec_sub_group(V, V, U, P);
    pmf_vec_sub_group(Y, Y, X, P);
    pmf_vec_mul_group(T, V, Y, P);
    pmf_vec_accumulate_group(S->acc[2], T, L);
    if (++S->since_reduce == 255)
    {
        for (uint64_t s = 0; s < 3; s++)
        {
            S->acc[s][L] = pv_zero();
            pmf_vec_reduce_wide(S->acc[s], P);
        }
        S->since_reduce = 0;
    }
}

// The lanes of the sums added up into the elements sums[0..2], and then the
// pairs `from` .. a->n / 2 - 1 of a and b, which no whole group covers, added
// one element at a time.
PMF_INLINE void pmf_vec_inner_sums_finish(uint64_t *const *sums, PMFInnerSums *S, const PMFVector a,
                                          const PMFVector b, uint64_t from, PMFParams P)
{
    const uint64_t L = P->limbs;
    _Alignas(64) uint64_t partial[PMF_LANES * PMF_MAX_LIMBS];
    uint64_t element[PMF_LANES] = {0};

    for (uint64_t s = 0; s < 3; s++)
    {
        for (uint64_t k = 0; k < PMF_LANES; k++)
            sums[s][k] = 0;
        S->acc[s][L] = pv_zero();
        pmf_vec_reduce_wide(S->acc[s], P);
        for (uint64_t k = 0; k < L; k++)
            pv_store(partial + k * PMF_LANES, S->acc[s][k]);
        for (uint64_t lane = 0; lane < PMF_VEC_GROUP; lane++)
        {
            for (uint64_t k = 0; k < L; k++)
                element[k] = partial[k * PMF_LANES + lane];
            pmf_add(sums[s], sums[s], element, P);
        }
    }

    uint64_t u[PMF_LANES], v[PMF_LANES], x[PMF_LANES], y[PMF_LANES], t[PMF_LANES];
    for (uint64_t i = from; i < a->n / 2; i++)
    {
        pmf_vec_get_element(u, a, 2 * i);
        pmf_vec_get_element(v, a, 2 * i + 1);
        pmf_vec_get_element(x, b, 2 * i);
        pmf_vec_get_element(y, b, 2 * i + 1);
        pmf_mul(t, u, x, P);
        pmf_add(sums[0], sums[0], t, P);
        pmf_mul(t, v, y, P);
        pmf_add(sums[1], sums[1], t, P);
        pmf_sub(v, v, u, P);
        pmf_sub(y, y, x, P);
        pmf_mul(t, v, y, P);
        pmf_add(sums[2], sums[2], t, P);
    }
}

// q2 = 2 q1 - q0 + 2 q2: the leading coefficient in q2 becomes q(2).
static void pmf_vec_inner_sums_extrapolate(uint64_t *q0, uint64_t *q1, uint64_t *q2,
                                           PMFParams params)
{
    pmf_add(q2, q2, q2, params);
    pmf_add(q2, q2, q1, params);
    pmf_add(q2, q2, q1, params);
    pmf_sub(q2, q2, q0, params);
}

// The pairs in whole groups go through the lanes; the rest, whose lanes would
// reach the padding, through the element kernels.
PMF_INLINE void pmf_vec_fold_inner_products_body(uint64_t *const *sums, const PMFVector a,
                                                 const PMFVector b, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], X[PMF_MAX_LIMBS], Y[PMF_MAX_LIMBS];
    PMFInnerSums S;

    pmf_vec_inner_sums_init(&S, L);
    const uint64_t whole = (a->n / 2 / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    for (uint64_t i = 0; i < whole; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_pairs(U, V, a, i, L);
        pmf_vec_load_pairs(X, Y, b, i, L);
        pmf_vec_inner_sums_add(&S, U, V, X, Y, P);
    }
    pmf_vec_inner_sums_finish(sums, &S, a, b, whole, P);
}

void pmf_vec_fold_inner_products(uint64_t *q0, uint64_t *q1, uint64_t *q2, const PMFVector a,
                                 const PMFVector b)
{
    assert(a->n % 2 == 0 && b->n == a->n);
    uint64_t *const sums[3] = {q0, q1, q2};
    PMF_WITH_LIMBS(a->params, pmf_vec_fold_inner_products_body(sums, a, b, &P_));
    pmf_vec_inner_sums_extrapolate(q0, q1, q2, a->params);
}

// The pairs of the 2 * PMF_VEC_GROUP consecutive elements held in F0 and F1, as
// one group each of first (E) and second (O) members: the lanes of
// pmf_vec_load_pairs, from registers.
PMF_INLINE void pmf_vec_pairs_of(pv_t *E, pv_t *O, const pv_t *F0, const pv_t *F1, uint64_t L)
{
#if VFHE_HAVE_AVX512IFMA
    const __m512i evens = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i odds = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    for (uint64_t k = 0; k < L; k++)
    {
        E[k] = _mm512_permutex2var_epi64(F0[k], evens, F1[k]);
        O[k] = _mm512_permutex2var_epi64(F0[k], odds, F1[k]);
    }
#else
    for (uint64_t k = 0; k < L; k++)
    {
        E[k] = F0[k];
        O[k] = F1[k];
    }
#endif
}

// Each pass of the main loop folds two output groups of each vector, stores
// them, and takes the pairs of the next fold from the registers it has just
// stored: the 2 * PMF_VEC_GROUP output elements from 2j are the pairs j .. j +
// PMF_VEC_GROUP - 1 of the folded vector. The remaining output groups, padding
// included, are folded alone, and the pairs among them summed one at a time.
PMF_INLINE void pmf_vec_fold_and_inner_products_body(PMFVector a_out, PMFVector b_out,
                                                     uint64_t *const *sums, const PMFVector a,
                                                     const PMFVector b, const uint64_t *r,
                                                     PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t R[PMF_MAX_LIMBS];
    pv_t A0[PMF_MAX_LIMBS + 1], A1[PMF_MAX_LIMBS + 1], B0[PMF_MAX_LIMBS + 1], B1[PMF_MAX_LIMBS + 1];
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], X[PMF_MAX_LIMBS], Y[PMF_MAX_LIMBS];
    PMFInnerSums S;

    pmf_vec_load_scalar(R, r, L);
    pmf_vec_inner_sums_init(&S, L);
    const uint64_t whole = (a_out->n / 2 / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    for (uint64_t j = 0; j < whole; j += PMF_VEC_GROUP)
    {
        const uint64_t i = 2 * j;
        pmf_vec_fold_group(A0, a, i, R, P);
        pmf_vec_fold_group(A1, a, i + PMF_VEC_GROUP, R, P);
        pmf_vec_fold_group(B0, b, i, R, P);
        pmf_vec_fold_group(B1, b, i + PMF_VEC_GROUP, R, P);
        pmf_vec_store_group(a_out, i, A0, L);
        pmf_vec_store_group(a_out, i + PMF_VEC_GROUP, A1, L);
        pmf_vec_store_group(b_out, i, B0, L);
        pmf_vec_store_group(b_out, i + PMF_VEC_GROUP, B1, L);
        pmf_vec_pairs_of(U, V, A0, A1, L);
        pmf_vec_pairs_of(X, Y, B0, B1, L);
        pmf_vec_inner_sums_add(&S, U, V, X, Y, P);
    }
    for (uint64_t i = 2 * whole; i < a_out->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_fold_group(A0, a, i, R, P);
        pmf_vec_fold_group(B0, b, i, R, P);
        pmf_vec_store_group(a_out, i, A0, L);
        pmf_vec_store_group(b_out, i, B0, L);
    }
    pmf_vec_inner_sums_finish(sums, &S, a_out, b_out, whole, P);
}

void pmf_vec_fold_and_inner_products(PMFVector a_out, PMFVector b_out, uint64_t *q0, uint64_t *q1,
                                     uint64_t *q2, const PMFVector a, const PMFVector b,
                                     const uint64_t *r)
{
    assert(a->n % 4 == 0 && b->n == a->n && 2 * a_out->n == a->n && 2 * b_out->n == a->n);
    uint64_t *const sums[3] = {q0, q1, q2};
    PMF_WITH_LIMBS(a->params,
                   pmf_vec_fold_and_inner_products_body(a_out, b_out, sums, a, b, r, &P_));
    pmf_vec_inner_sums_extrapolate(q0, q1, q2, a->params);
}

// One element, 1/2 = (p + 1) / 2: p is odd. PMF_LANES words.
static void pmf_vec_half(uint64_t *half, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t sum[PMF_MAX_LIMBS + 1] = {0}, carry = 1;
    for (uint64_t k = 0; k < L; k++)
    {
        const uint64_t s = params->p[k] + carry;
        sum[k] = s & PMF_LIMB_MASK;
        carry = s >> PMF_LIMB_BITS;
    }
    sum[L] = carry;
    memset(half, 0, PMF_LANES * sizeof(*half));
    for (uint64_t k = 0; k < L; k++)
        half[k] = (sum[k] >> 1) | ((sum[k + 1] & 1) << (PMF_LIMB_BITS - 1));
}

// out = v + (u - v) (untwist r + 1/2) over each pair (u, v): the documented
// (u + v)/2 + (u - v) untwist r with two products a pair.
PMF_INLINE void pmf_vec_fold_twisted_body(PMFVector out, const PMFVector word,
                                          const PMFVector untwist, const uint64_t *r,
                                          const uint64_t *half, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS], R[PMF_MAX_LIMBS], H[PMF_MAX_LIMBS];
    pv_t C[PMF_MAX_LIMBS + 1], D[PMF_MAX_LIMBS + 1], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(R, r, L);
    pmf_vec_load_scalar(H, half, L);
    for (uint64_t i = 0; i < out->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_pairs(U, V, word, i, L);
        pmf_vec_load_group(W, untwist, i, L);
        pmf_vec_mul_group(C, W, R, P);
        pmf_vec_add_group(C, C, H, P);
        pmf_vec_sub_group(D, U, V, P);
        pmf_vec_mul_group(T, D, C, P);
        pmf_vec_add_group(D, V, T, P);
        pmf_vec_store_group(out, i, D, L);
    }
}

void pmf_vec_fold_twisted(PMFVector out, const PMFVector word, const PMFVector untwist,
                          const uint64_t *r)
{
    uint64_t half[PMF_LANES];
    assert(word->n == 2 * out->n && untwist->n == out->n);
    pmf_vec_half(half, word->params);
    PMF_WITH_LIMBS(word->params, pmf_vec_fold_twisted_body(out, word, untwist, r, half, &P_));
}

// The table entries for the group at i, read cyclically from a table of
// t->n: a plain load where the group lies inside one period at a group
// boundary (a period that is a whole number of groups, or the operands' whole
// length), lane by lane otherwise. Lanes past the operands read whatever the
// period gives: what they produce lands in the output's padding, which only
// has to stay canonical.
PMF_INLINE void pmf_vec_load_cyclic(pv_t *W, const PMFVector t, uint64_t i, uint64_t n, uint64_t L)
{
    const uint64_t period = t->n;
    if (period == n)
    {
        pmf_vec_load_group(W, t, i, L);
        return;
    }
    if (period % PMF_VEC_GROUP == 0)
    {
        pmf_vec_load_group(W, t, i % period, L);
        return;
    }
#if VFHE_HAVE_AVX512IFMA
    long long index[PMF_VEC_GROUP];
    for (uint64_t l = 0; l < PMF_VEC_GROUP; l++)
        index[l] = (long long)((i + l) % period);
    const __m512i at = _mm512_loadu_si512((const void *)index);
    for (uint64_t k = 0; k < L; k++)
        W[k] = _mm512_i64gather_epi64(at, (const void *)t->limbs[k], 8);
#endif
}

// E and O as the pairs (E[l], O[l]) of the 2 * PMF_VEC_GROUP output elements
// from 2i. The second place in this file lanes are rearranged: lane l of the
// two registers goes to adjacent positions. Pairs past the output's
// allocation are not stored: the input's padding runs to a whole group, and
// twice that can exceed the output's.
PMF_INLINE void pmf_vec_store_pairs(PMFVector out, uint64_t i, const pv_t *E, const pv_t *O,
                                    uint64_t L)
{
#if VFHE_HAVE_AVX512IFMA
    const __m512i first = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i second = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const int both = 2 * i + 2 * PMF_VEC_GROUP <= out->allocated_n;
    for (uint64_t k = 0; k < L; k++)
    {
        pv_store(out->limbs[k] + 2 * i, _mm512_permutex2var_epi64(E[k], first, O[k]));
        if (both)
            pv_store(out->limbs[k] + 2 * i + PMF_VEC_GROUP,
                     _mm512_permutex2var_epi64(E[k], second, O[k]));
    }
#else
    if (2 * i + 1 >= out->allocated_n)
        return;
    for (uint64_t k = 0; k < L; k++)
    {
        out->limbs[k][2 * i] = E[k];
        out->limbs[k][2 * i + 1] = O[k];
    }
#endif
}

PMF_INLINE void pmf_vec_lift_twisted_body(PMFVector out, const PMFVector a, const PMFVector b,
                                          const PMFVector t, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t A[PMF_MAX_LIMBS], B[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS];
    pv_t M[PMF_MAX_LIMBS + 1], E[PMF_MAX_LIMBS + 1], O[PMF_MAX_LIMBS + 1];

    for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_group(A, a, i, L);
        pmf_vec_load_group(B, b, i, L);
        pmf_vec_load_cyclic(W, t, i, a->n, L);
        pmf_vec_mul_group(M, B, W, P);
        pmf_vec_add_group(E, A, M, P);
        pmf_vec_sub_group(O, A, M, P);
        pmf_vec_store_pairs(out, i, E, O, L);
    }
}

void pmf_vec_lift_twisted(PMFVector out, const PMFVector a, const PMFVector b, const PMFVector t)
{
    assert(b->n == a->n && out->n == 2 * a->n && t->n > 0 && a->n % t->n == 0);
    PMF_WITH_LIMBS(a->params, pmf_vec_lift_twisted_body(out, a, b, t, &P_));
}

// --- movement -------------------------------------------------------------

void pmf_vec_get_element(uint64_t *out, const PMFVector a, uint64_t index)
{
    const uint64_t L = a->params->limbs;
    for (uint64_t k = 0; k < L; k++)
        out[k] = a->limbs[k][index];
    for (uint64_t k = L; k < PMF_LANES; k++)
        out[k] = 0;
}

void pmf_vec_set_element(PMFVector out, uint64_t index, const uint64_t *value)
{
    for (uint64_t k = 0; k < out->params->limbs; k++)
        out->limbs[k][index] = value[k];
}

void pmf_vec_set_range(PMFVector out, uint64_t start, const uint64_t *values, uint64_t count)
{
    const uint64_t L = out->params->limbs;
    for (uint64_t k = 0; k < L; k++)
    {
        uint64_t *plane = out->limbs[k] + start;
        for (uint64_t i = 0; i < count; i++)
            plane[i] = values[i * PMF_LANES + k];
    }
}

void pmf_vec_get_range(uint64_t *out, const PMFVector a, uint64_t start, uint64_t count)
{
    const uint64_t L = a->params->limbs;
    for (uint64_t i = 0; i < count; i++)
    {
        for (uint64_t k = L; k < PMF_LANES; k++)
            out[i * PMF_LANES + k] = 0;
    }
    for (uint64_t k = 0; k < L; k++)
    {
        const uint64_t *plane = a->limbs[k] + start;
        for (uint64_t i = 0; i < count; i++)
            out[i * PMF_LANES + k] = plane[i];
    }
}

static inline void pmf_vec_set_limbs(uint64_t *limb0, uint64_t *limb1, uint64_t i, uint64_t value)
{
    limb0[i] = value & PMF_LIMB_MASK;
    limb1[i] = value >> PMF_LIMB_BITS;
}

void pmf_vec_set_uint64(PMFVector out, uint64_t start, const uint64_t *low, const uint64_t *high,
                        uint64_t count, uint32_t shift, uint32_t width)
{
    assert(start + count <= out->n && shift < (high ? 128u : 64u) && width >= 1 && width <= 64);
    const uint64_t mask = width == 64 ? ~0ULL : (1ULL << width) - 1;
    uint64_t *limb0 = out->limbs[0] + start, *limb1 = out->limbs[1] + start;
    // One loop per case, so none of them branches per element.
    if (!high || shift == 0)
        for (uint64_t i = 0; i < count; i++)
            pmf_vec_set_limbs(limb0, limb1, i, (low[i] >> shift) & mask);
    else if (shift >= 64)
        for (uint64_t i = 0; i < count; i++)
            pmf_vec_set_limbs(limb0, limb1, i, (high[i] >> (shift - 64)) & mask);
    else
        for (uint64_t i = 0; i < count; i++)
            pmf_vec_set_limbs(limb0, limb1, i,
                              ((low[i] >> shift) | (high[i] << (64 - shift))) & mask);
    for (uint64_t k = 2; k < out->params->limbs; k++)
        memset(out->limbs[k] + start, 0, count * sizeof(uint64_t));
}

bool pmf_vec_get_uint64(uint64_t *out, const PMFVector a)
{
    // Limbs are 52 bits, so a canonical element is below 2^64 exactly when its
    // limbs from the third on are zero and the second is below 2^12.
    const uint64_t *low = a->limbs[0], *high = a->limbs[1];
    for (uint64_t i = 0; i < a->n; i++)
    {
        uint64_t above = high[i] >> (64 - PMF_LIMB_BITS);
        for (uint64_t k = 2; k < a->params->limbs; k++)
            above |= a->limbs[k][i];
        if (above)
            return false;
        out[i] = low[i] | high[i] << PMF_LIMB_BITS;
    }
    return true;
}

void pmf_vec_copy(PMFVector out, const PMFVector a)
{
    for (uint64_t k = 0; k < a->params->limbs; k++)
        memcpy(out->limbs[k], a->limbs[k], a->allocated_n * sizeof(uint64_t));
}

void pmf_vec_split_even_odd(PMFVector even, PMFVector odd, const PMFVector a)
{
    const uint64_t half = a->n / 2;
    for (uint64_t k = 0; k < a->params->limbs; k++)
    {
        const uint64_t *plane = a->limbs[k];
        for (uint64_t i = 0; i < half; i++)
        {
            even->limbs[k][i] = plane[2 * i];
            odd->limbs[k][i] = plane[2 * i + 1];
        }
    }
}

void pmf_vec_interleave(PMFVector out, const PMFVector even, const PMFVector odd)
{
    for (uint64_t k = 0; k < out->params->limbs; k++)
        vec_interleave_u64(out->limbs[k], even->limbs[k], odd->limbs[k], even->n);
}

void pmf_vec_concat(PMFVector out, const PMFVector *parts, uint64_t count)
{
    uint64_t at = 0;
    for (uint64_t p = 0; p < count; p++)
    {
        for (uint64_t k = 0; k < out->params->limbs; k++)
            memcpy(out->limbs[k] + at, parts[p]->limbs[k], parts[p]->n * sizeof(uint64_t));
        at += parts[p]->n;
    }
}

bool pmf_vec_gather(PMFVector out, const PMFVector a, const uint64_t *indices, uint64_t count)
{
    // Validated in its own pass, for the reasons field_vec_gather gives.
    for (uint64_t i = 0; i < count; i++)
        if (indices[i] >= a->n)
            return false;
    for (uint64_t k = 0; k < a->params->limbs; k++)
        for (uint64_t i = 0; i < count; i++)
            out->limbs[k][i] = a->limbs[k][indices[i]];
    return true;
}

int pmf_vec_is_equal(const PMFVector a, const PMFVector b)
{
    if (a->n != b->n || a->params->limbs != b->params->limbs)
        return 0;
    for (uint64_t k = 0; k < a->params->limbs; k++)
    {
        if (memcmp(a->limbs[k], b->limbs[k], a->n * sizeof(uint64_t)) != 0)
            return 0;
    }
    return 1;
}

// --- sampling and hashing -------------------------------------------------
//
// Both go through the canonical encoding of one element at a time. Hashing
// stages the encodings and hands BLAKE3 whole buffers or whole batches of
// windows, since a 20-40 byte update per element costs more than the hashing.

void pmf_vec_sample_random(PMFVector out, const uint8_t *seed, uint64_t seed_len)
{
    blake3_hasher hasher;
    uint64_t offset = 0;
    uint64_t element[PMF_LANES];

    // One stream for the whole vector: a fresh derivation per element, from the
    // one seed, would make every element the same value.
    blake3_hasher_init_derive_key(&hasher, "pmf_vec_sample");
    blake3_hasher_update(&hasher, seed, seed_len);
    for (uint64_t i = 0; i < out->n; i++)
    {
        pmf_sample_stream(element, &hasher, &offset, out->params);
        pmf_vec_set_element(out, i, element);
    }
}

// The canonical encoding of element `index`, the bytes pmf_to_bytes writes:
// the value's nbytes bytes, most significant first. The limbs are shifted into
// 64-bit words, least significant first, which are then written out from the
// top byte down.
static void pmf_vec_encode(uint8_t *out, const PMFVector a, uint64_t index)
{
    const uint64_t L = a->params->limbs, nbytes = a->params->nbytes;
    uint64_t words[PMF_MAX_LIMBS] = {0};
    unsigned __int128 acc = 0;
    unsigned bits = 0, w = 0;

    for (uint64_t k = 0; k < L; k++)
    {
        acc |= (unsigned __int128)a->limbs[k][index] << bits;
        bits += PMF_LIMB_BITS;
        if (bits >= 64)
        {
            words[w++] = (uint64_t)acc;
            acc >>= 64;
            bits -= 64;
        }
    }
    if (bits)
        words[w] = (uint64_t)acc;
    for (uint64_t i = 0; i < nbytes; i++)
        out[nbytes - 1 - i] = (uint8_t)(words[i / 8] >> (8 * (i % 8)));
}

// Encodings staged per call into BLAKE3, as a byte budget.
#define PMF_VEC_HASH_STAGE_BYTES (32 * 1024)

// BLAKE3 of the encodings of `count` elements from `start`, staged so the
// hasher takes them a buffer at a time and runs its multi-chunk path.
static void pmf_vec_hash_span(uint8_t *out, const PMFVector a, uint64_t start, uint64_t count)
{
    const uint64_t nbytes = a->params->nbytes, per = PMF_VEC_HASH_STAGE_BYTES / nbytes;
    uint8_t *staged = (uint8_t *)safe_aligned_malloc(per * nbytes);
    blake3_hasher hasher;

    blake3_hasher_init(&hasher);
    for (uint64_t i = 0; i < count;)
    {
        const uint64_t run = count - i < per ? count - i : per;
        for (uint64_t j = 0; j < run; j++)
            pmf_vec_encode(&staged[j * nbytes], a, start + i + j);
        blake3_hasher_update(&hasher, staged, run * nbytes);
        i += run;
    }
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    free(staged);
}

void pmf_vec_hash(uint8_t *out, const PMFVector a) { pmf_vec_hash_span(out, a, 0, a->n); }

void pmf_vec_clear_padding(PMFVector a)
{
    if (a->allocated_n <= a->n)
        return;
    const size_t tail = (size_t)(a->allocated_n - a->n) * sizeof(uint64_t);
    for (uint64_t k = 0; k < a->params->limbs; k++)
        memset(a->limbs[k] + a->n, 0, tail);
}

uint64_t pmf_vec_hash_count(const PMFVector a, uint64_t group, uint64_t stride)
{
    if (group == 0 || stride == 0 || a->n < group)
        return 0;
    return (a->n - group) / stride + 1;
}

void pmf_vec_hash_elements(uint8_t *out, const PMFVector a, uint64_t group, uint64_t stride)
{
    const uint64_t count = pmf_vec_hash_count(a, group, stride);
    const uint64_t nbytes = a->params->nbytes;
    const uint64_t leaf = group * nbytes;
    if (count == 0)
        return;

    // A window past a BLAKE3 chunk, which hash_batch cannot take: one hasher
    // each. The width is the *encoded* one, nbytes per element, not a machine
    // word.
    if (!hash_batch_fits(leaf))
    {
        for (uint64_t k = 0; k < count; k++)
            pmf_vec_hash_span(&out[k * BLAKE3_OUT_LEN], a, k * stride, group);
        return;
    }

    // Staged because a digest covers canonical encodings, which nothing holds.
    // Windows are batched by the same byte budget as a span; see field_vector.c,
    // which sizes its own staging the same way and for the same reasons.
    // hash_batch_fits caps `leaf` at a chunk, so a batch is at least 32.
    const uint64_t batch = PMF_VEC_HASH_STAGE_BYTES / leaf;
    uint8_t *staged = (uint8_t *)safe_aligned_malloc(batch * leaf);
    for (uint64_t k = 0; k < count;)
    {
        uint64_t run = count - k;
        if (run > batch)
            run = batch;
        for (uint64_t j = 0; j < run; j++)
            for (uint64_t i = 0; i < group; i++)
                pmf_vec_encode(&staged[(j * group + i) * nbytes], a, (k + j) * stride + i);
        hash_batch(&out[k * BLAKE3_OUT_LEN], staged, run, leaf);
        k += run;
    }
    free(staged);
}
