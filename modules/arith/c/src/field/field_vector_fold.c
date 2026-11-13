// SPDX-FileCopyrightText: 2026 The vfhe authors
// SPDX-License-Identifier: Apache-2.0
//
// Sums of products of folded tables over the extension field F_p[x]/(x^d - w):
// the native bodies of FieldVector's fold_inner_products,
// fold_and_inner_products, fold_product_sums and fold_and_product_sums for
// extension-field vectors. arith.h documents the four entry points.
//
// What they compute. A table is read in adjacent pairs (u_i, v_i) =
// (table[2i], table[2i + 1]). Folding it at a point t replaces each pair by
// u_i + t (v_i - u_i): for the table of a multilinear polynomial's values, in
// least-significant-bit-first order, that is the table with its first
// variable set to t. For a few small t, the nodes 0, 1, 2, ..., these
// functions add up, over all pairs, signed products of several tables' folds,
// optionally weighted per pair:
//
//     S(t) = sum_i w_i sum_(products) sign * prod_(j in product) (u_ji + t (v_ji - u_ji))
//
// The "fold and" variants first fold every table at a given element r, write
// those folded tables out, and compute the sums over them, reading each input
// table once.
//
// How. The work is one pass over the pairs, split into chunks for the
// library's threads (field_fold_run). Each chunk adds up its own pairs, and
// the chunks' results are added in chunk order, so the result does not depend
// on the number of threads. Within a chunk, a pair's folds at the nodes are
// formed once, f(t) = f(t - 1) + (v - u) from t = 2 on, and every product is
// built from them.
//
// Lazy reduction. The product of two elements is first a polynomial in x of
// degree 2d - 2, its "unreduced" form; reducing it means taking its
// coefficients modulo q and the polynomial modulo x^d - w. The sums are
// linear, so the last multiplication on the way into a sum -- by the weight,
// or by the product's last factor when there are no weights -- is left
// unreduced: its 2d - 1 coefficients are added as they are into the sum's
// accumulators, and the accumulators are reduced once, at the end of the
// chunk. Only the multiplications between a product's earlier factors are
// reduced.
//
// Three bodies compute the same values, word for word, since every result is
// fully reduced:
// - IFMA (avx512ifma engine, q < 2^50 and d <= FIELD_MAX_FUSED_D, the gate of
//   field_fused_applies): eight pairs at a time, in 52-bit lanes;
// - element-wise (avx512ifma engine, any other q or d): the engine's
//   element-wise kernels (mod_eltwise_*) over a cache-sized block of pairs at
//   a time;
// - scalar (the other engines): one pair at a time, in 128-bit arithmetic.
// Each body is compiled for the degrees 1, 2, 4 and 8, and once for a degree
// known only at run time (FIELD_FOLD_CALL).
//
// The files. This one decodes the shape, runs the chunks on the library's
// threads, picks the body and holds the entry points; the bodies are in
// field_vector_fold_ifma.c, field_vector_fold_eltwise.c and
// field_vector_fold_scalar.c, each compiled for its engine only; and
// field_vector_fold.h holds what they share.
#include <arith.h>
#include <assert.h>
#include <string.h>
#include "arith_internal.h"
#include "alloc.h"
#include "field_vector_fold.h"

// A chunk covers at least FIELD_FOLD_PAIR_GRAIN pairs divided by the number of
// tables read, so that a thread's share outweighs the cost of handing it out.
#define FIELD_FOLD_PAIR_GRAIN (1u << 13)

// --- the shape --------------------------------------------------------------

// Decode the shape word array of field_vec_fold_product_sums (arith.h describes
// it) into `shape`, asserting the FIELD_FOLD_MAX_* bounds.
static void field_fold_shape(FieldProductShape *shape, const uint64_t *words, uint64_t tables,
                             uint64_t nodes)
{
    assert(tables >= 1 && tables <= FIELD_FOLD_MAX_TABLES);
    assert(nodes >= 1 && nodes <= FIELD_FOLD_MAX_NODES);
    memset(shape->is_used, 0, sizeof(shape->is_used));
    shape->nodes = nodes;
    shape->sums = *words++;
    assert(shape->sums >= 1 && shape->sums <= FIELD_FOLD_MAX_SUMS);
    uint64_t p = 0;
    for (uint64_t s = 0; s < shape->sums; s++)
    {
        shape->first_product[s] = p;
        const uint64_t products = *words++;
        assert(products >= 1 && p + products <= FIELD_FOLD_MAX_PRODUCTS);
        for (uint64_t end = p + products; p < end; p++)
        {
            shape->negated[p] = *words++;
            shape->factor_count[p] = *words++;
            assert(shape->negated[p] <= 1);
            assert(shape->factor_count[p] >= 1 && shape->factor_count[p] <= FIELD_FOLD_MAX_FACTORS);
            for (uint64_t f = 0; f < shape->factor_count[p]; f++)
            {
                shape->factors[p][f] = *words++;
                assert(shape->factors[p][f] < tables);
                shape->is_used[shape->factors[p][f]] = true;
            }
        }
    }
    shape->first_product[shape->sums] = p;
    shape->used_count = 0;
    for (uint64_t j = 0; j < tables; j++)
        if (shape->is_used[j])
            shape->used[shape->used_count++] = j;
}

// --- running over chunks of pairs -------------------------------------------

// One chunk of job->range, into the chunk's own slots of job->partials.
static void field_fold_chunk(void *ctx, uint64_t chunk)
{
    const FieldFoldJob *job = (const FieldFoldJob *)ctx;
    const uint64_t d = job->tables[0]->d;
    uint64_t *sums[FIELD_FOLD_MAX_SUMS * FIELD_FOLD_MAX_NODES];
    for (uint64_t s = 0; s < job->slots; s++)
        sums[s] = job->partials + (chunk * job->slots + s) * d;
    job->range(job, vec_chunk_begin(&job->chunks, chunk), vec_chunk_end(&job->chunks, chunk), sums);
}

// Run job->range over all the pairs, [0, job->total), and write the sums into
// results. With one chunk the body writes them directly; with several, each
// chunk writes its own partial sums and those are added in chunk order.
static void field_fold_run(FieldFoldJob *job, uint64_t *const *results, uint64_t grain)
{
    const FieldVector first = job->tables[0];
    const uint64_t d = first->d;
    job->slots = job->shape->sums * job->shape->nodes;
    if (!vec_chunks_plan(&job->chunks, job->total, grain, FIELD_FOLD_GROUP))
    {
        job->range(job, 0, job->total, results);
        return;
    }
    const uint64_t count = job->chunks.count;
    job->partials = (uint64_t *)safe_malloc(count * job->slots * d * sizeof(uint64_t));
    vfhe_parallel_for(count, 0, field_fold_chunk, job);
    for (uint64_t s = 0; s < job->slots; s++)
    {
        memcpy(results[s], job->partials + s * d, d * sizeof(uint64_t));
        for (uint64_t c = 1; c < count; c++)
            field_ext_add(results[s], results[s], job->partials + (c * job->slots + s) * d, d,
                          first->mod->q);
    }
    free(job->partials);
}

// --- picking the body -------------------------------------------------------
//
// The range of a call, as FieldFoldJob.range: on the avx512ifma engine the IFMA
// body where it applies and the element-wise body otherwise, on the other engines
// the scalar body.

static void field_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                  uint64_t *const *sums)
{
#if VFHE_HAVE_AVX512IFMA
    if (field_fold_ifma_applies(job->tables[0]))
        field_fold_ifma_sums_range(job, begin, end, sums);
    else
        field_fold_eltwise_sums_range(job, begin, end, sums);
#else
    field_fold_scalar_sums_range(job, begin, end, sums);
#endif
}

static void field_fold_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                       uint64_t *const *sums)
{
#if VFHE_HAVE_AVX512IFMA
    if (field_fold_ifma_applies(job->tables[0]))
        field_fold_ifma_fold_sums_range(job, begin, end, sums);
    else
        field_fold_eltwise_fold_sums_range(job, begin, end, sums);
#else
    field_fold_scalar_fold_sums_range(job, begin, end, sums);
#endif
}

// --- entry points -----------------------------------------------------------

void field_vec_fold_product_sums(uint64_t *const *sums, const FieldVector *tables, uint64_t count,
                                 const uint64_t *shape, uint64_t nodes, const FieldVector weights)
{
    FieldProductShape decoded;
    field_fold_shape(&decoded, shape, count, nodes);
    const FieldVector first = tables[0];
    assert(first->n % 2 == 0 && first->d >= 1 && first->d <= FIELD_FOLD_MAX_DEGREE);
    for (uint64_t j = 0; j < count; j++)
        assert(tables[j]->n == first->n && tables[j]->d == first->d);
    assert(!weights || (2 * weights->n == first->n && weights->d == first->d));
    FieldFoldJob job = {.range = field_fold_sums_range,
                        .total = first->n / 2,
                        .tables = tables,
                        .count = count,
                        .weights = weights,
                        .shape = &decoded};
    field_fold_run(&job, sums, FIELD_FOLD_PAIR_GRAIN / count);
}

void field_vec_fold_and_product_sums(FieldVector *outs, uint64_t *const *sums,
                                     const FieldVector *tables, uint64_t count, const uint64_t *r,
                                     const uint64_t *shape, uint64_t nodes,
                                     const FieldVector weights)
{
    FieldProductShape decoded;
    field_fold_shape(&decoded, shape, count, nodes);
    const FieldVector first = tables[0];
    assert(first->n % 4 == 0 && first->d >= 1 && first->d <= FIELD_FOLD_MAX_DEGREE);
    for (uint64_t j = 0; j < count; j++)
        assert(tables[j]->n == first->n && tables[j]->d == first->d && 2 * outs[j]->n == first->n &&
               outs[j]->allocated_n == outs[0]->allocated_n);
    assert(!weights || (4 * weights->n == first->n && weights->d == first->d));
    FieldFoldJob job = {.range = field_fold_fold_sums_range,
                        .total = first->n / 4,
                        .tables = tables,
                        .count = count,
                        .outs = (const FieldVector *)outs,
                        .weights = weights,
                        .r = r,
                        .shape = &decoded};
    field_fold_run(&job, sums, FIELD_FOLD_PAIR_GRAIN / (2 * count));
}

// The one product f_a f_b at the nodes 0, 1 and 2: q(t) = sum_i (u_i + t (v_i -
// u_i)) (x_i + t (y_i - x_i)), as a shape of one sum and one product.
void field_vec_fold_inner_products(uint64_t *q0, uint64_t *q1, uint64_t *q2, const FieldVector a,
                                   const FieldVector b)
{
    const bool same = a == b;
    const uint64_t shape[] = {1, 1, 0, 2, 0, same ? 0 : 1};
    const FieldVector tables[2] = {a, b};
    uint64_t *const sums[3] = {q0, q1, q2};
    assert(b->n == a->n);
    field_vec_fold_product_sums(sums, tables, same ? 1 : 2, shape, 3, NULL);
}

void field_vec_fold_and_inner_products(FieldVector a_out, FieldVector b_out, uint64_t *q0,
                                       uint64_t *q1, uint64_t *q2, const FieldVector a,
                                       const FieldVector b, const uint64_t *r)
{
    const uint64_t shape[] = {1, 1, 0, 2, 0, 1};
    const FieldVector tables[2] = {a, b};
    FieldVector outs[2] = {a_out, b_out};
    uint64_t *const sums[3] = {q0, q1, q2};
    assert(b->n == a->n);
    field_vec_fold_and_product_sums(outs, sums, tables, 2, r, shape, 3, NULL);
}
