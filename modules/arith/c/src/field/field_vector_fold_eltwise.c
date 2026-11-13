// SPDX-FileCopyrightText: 2026 The vfhe authors
// SPDX-License-Identifier: Apache-2.0
//
// The element-wise body of the extension-field fold kernels, on the avx512ifma
// engine for the moduli and degrees the IFMA body does not take.
// field_vector_fold.c describes the kernels.
#include <arith.h>
#include <assert.h>
#include <string.h>
#include "arith_internal.h"
#include "alloc.h"
#include "field_vector_fold.h"

#if VFHE_HAVE_AVX512IFMA

// --- element-wise body: the engine's kernels over a block of pairs -----------
//
// For a modulus or degree the IFMA body does not take, on the avx512ifma
// engine. A chunk goes through its pairs a block at a time. A block's pairs
// are split into planes of first and second members, one plane per
// coefficient, so that every step is a whole-plane call to one of the engine's
// element-wise kernels (mod_eltwise_*), which are vectorized for every
// modulus. The last multiplication into each sum is added into accumulator
// planes, one per coefficient of the unreduced product, each kept reduced
// modulo q. All of a block's planes together fit in a core's own cache
// (FIELD_FOLD_BLOCK_BYTES); the accumulators are added up over their
// positions and reduced modulo x^d - w once per chunk.

// The scratch for one chunk's planes, in bytes; it sets the block's length.
#define FIELD_FOLD_BLOCK_BYTES (256u << 10)

// One chunk's planes, `block_len` words each, 64-byte aligned, in one
// allocation (`storage`):
// - F[(x * stride + t) * d + k]: coefficient k of the folds at node t of the
//   x-th table the shape reads (table_index[j] is x for table j); there is room
//   for nodes 0 and 1 whatever the node count, since the pairs are loaded
//   there;
// - acc[(s * nodes + t) * (2d - 1) + k]: coefficient k of the unreduced
//   polynomial of sum s at node t;
// - running_product, signed_sum and diff: d planes each, as in the scalar body;
// - schoolbook: 2d - 1 planes for one product before its reduction modulo
//   x^d - w;
// - quarters and halves: a table's input split four ways, and the two halves
//   of its fold, for the "fold and" variant.
typedef struct
{
    uint64_t d, nodes, stride, block_len;
    uint64_t *storage;
    uint64_t **planes;
    uint64_t **F, **acc, **running_product, **signed_sum, **schoolbook, **diff, **quarters,
        **halves;
    uint64_t table_index[FIELD_FOLD_MAX_TABLES]; // x for table j
} FieldFoldEltwiseScratch;

static void field_fold_eltwise_open(FieldFoldEltwiseScratch *scratch, const FieldFoldJob *job,
                                    uint64_t length)
{
    const FieldProductShape *shape = job->shape;
    const uint64_t d = job->tables[0]->d, nodes = shape->nodes, n_unreduced = 2 * d - 1;
    const uint64_t stride = nodes < 2 ? 2 : nodes;
    const uint64_t f_planes = shape->used_count * stride * d;
    const uint64_t acc_planes = shape->sums * nodes * n_unreduced;
    const uint64_t count = f_planes + acc_planes + 3 * d + n_unreduced + 6 * d;
    uint64_t block_len = FIELD_FOLD_BLOCK_BYTES / (count * sizeof(uint64_t));
    block_len -= block_len % MOD_MIN_VECTOR_LEN;
    if (block_len < MOD_MIN_VECTOR_LEN)
        block_len = MOD_MIN_VECTOR_LEN;
    if (block_len > field_vec_padded_length(length))
        block_len = field_vec_padded_length(length);

    scratch->d = d;
    scratch->nodes = nodes;
    scratch->stride = stride;
    scratch->block_len = block_len;
    scratch->storage = (uint64_t *)safe_aligned_malloc(count * block_len * sizeof(uint64_t));
    scratch->planes = (uint64_t **)safe_malloc(count * sizeof(uint64_t *));
    for (uint64_t p = 0; p < count; p++)
        scratch->planes[p] = scratch->storage + p * block_len;
    scratch->F = scratch->planes;
    scratch->acc = scratch->F + f_planes;
    scratch->running_product = scratch->acc + acc_planes;
    scratch->signed_sum = scratch->running_product + d;
    scratch->diff = scratch->signed_sum + d;
    scratch->schoolbook = scratch->diff + d;
    scratch->quarters = scratch->schoolbook + n_unreduced;
    scratch->halves = scratch->quarters + 4 * d;
    memset(scratch->acc[0], 0, acc_planes * block_len * sizeof(uint64_t));
    for (uint64_t x = 0; x < shape->used_count; x++)
        scratch->table_index[shape->used[x]] = x;
}

static void field_fold_eltwise_close(FieldFoldEltwiseScratch *scratch)
{
    free(scratch->planes);
    free(scratch->storage);
}

// The d planes of the folds of table j at node t.
static uint64_t **field_fold_eltwise_fold_planes(const FieldFoldEltwiseScratch *scratch, uint64_t j,
                                                 uint64_t t)
{
    return scratch->F + (scratch->table_index[j] * scratch->stride + t) * scratch->d;
}

// c = a * b in F_p[x]/(x^d - w), fully reduced, at the first n positions of
// the planes, through the schoolbook planes. c may be a or b.
static void field_fold_eltwise_mul(uint64_t *const *c, uint64_t *const *a, uint64_t *const *b,
                                   const FieldFoldEltwiseScratch *scratch, uint64_t n,
                                   uint64_t w_mod_q, Modulus mod)
{
    const uint64_t d = scratch->d;
    memset(scratch->schoolbook[0], 0, (2 * d - 1) * scratch->block_len * sizeof(uint64_t));
    for (uint64_t i = 0; i < d; i++)
        for (uint64_t j = 0; j < d; j++)
            mod_eltwise_mul_addto(scratch->schoolbook[i + j], a[i], b[j], n, mod);
    for (uint64_t k = 2 * d - 1; k-- > d;)
        mod_eltwise_fma(scratch->schoolbook[k - d], scratch->schoolbook[k], w_mod_q, n, mod);
    for (uint64_t k = 0; k < d; k++)
        memcpy(c[k], scratch->schoolbook[k], n * sizeof(uint64_t));
}

// Fill the folds at the nodes 2 .. nodes - 1 from those at nodes 0 and 1 (the
// pairs' first and second members, already in place).
static void field_fold_eltwise_nodes(uint64_t *const *F0, const FieldFoldEltwiseScratch *scratch,
                                     uint64_t n, Modulus mod)
{
    const uint64_t d = scratch->d;
    if (scratch->nodes <= 2)
        return;
    for (uint64_t k = 0; k < d; k++)
        mod_eltwise_sub(scratch->diff[k], F0[d + k], F0[k], n, mod);
    for (uint64_t t = 2; t < scratch->nodes; t++)
        for (uint64_t k = 0; k < d; k++)
            mod_eltwise_add(F0[t * d + k], F0[(t - 1) * d + k], scratch->diff[k], n, mod);
}

// Add the first n positions of a block into the accumulators, as
// field_fold_scalar_add_pair does for one pair, a plane at a time. `weight`
// is the weights' planes at the block, or NULL for weights of one.
static void field_fold_eltwise_add(const FieldFoldEltwiseScratch *scratch, uint64_t *const *weight,
                                   const FieldProductShape *shape, uint64_t n, uint64_t w_mod_q,
                                   Modulus mod)
{
    const uint64_t d = scratch->d, n_unreduced = 2 * d - 1;
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            uint64_t *const *target = scratch->acc + (s * shape->nodes + t) * n_unreduced;
            for (uint64_t p = shape->first_product[s]; p < shape->first_product[s + 1]; p++)
            {
                const uint64_t *factors = shape->factors[p];
                const uint64_t last = shape->factor_count[p] - (weight ? 0 : 1);
                uint64_t *const *product = field_fold_eltwise_fold_planes(scratch, factors[0], t);
                if (last > 1)
                {
                    field_fold_eltwise_mul(scratch->running_product, product,
                                           field_fold_eltwise_fold_planes(scratch, factors[1], t),
                                           scratch, n, w_mod_q, mod);
                    for (uint64_t f = 2; f < last; f++)
                        field_fold_eltwise_mul(
                            scratch->running_product, scratch->running_product,
                            field_fold_eltwise_fold_planes(scratch, factors[f], t), scratch, n,
                            w_mod_q, mod);
                    product = scratch->running_product;
                }
                if (weight)
                {
                    const bool first = p == shape->first_product[s];
                    for (uint64_t k = 0; k < d; k++)
                    {
                        if (first && shape->negated[p])
                            mod_eltwise_negate(scratch->signed_sum[k], product[k], n, mod);
                        else if (first)
                            memcpy(scratch->signed_sum[k], product[k], n * sizeof(uint64_t));
                        else if (shape->negated[p])
                            mod_eltwise_sub(scratch->signed_sum[k], scratch->signed_sum[k],
                                            product[k], n, mod);
                        else
                            mod_eltwise_add(scratch->signed_sum[k], scratch->signed_sum[k],
                                            product[k], n, mod);
                    }
                    continue;
                }
                if (shape->factor_count[p] == 1)
                {
                    for (uint64_t k = 0; k < d; k++)
                        if (shape->negated[p])
                            mod_eltwise_sub(target[k], target[k], product[k], n, mod);
                        else
                            mod_eltwise_add(target[k], target[k], product[k], n, mod);
                    continue;
                }
                uint64_t *const *factor = field_fold_eltwise_fold_planes(scratch, factors[last], t);
                for (uint64_t i = 0; i < d; i++)
                    for (uint64_t j = 0; j < d; j++)
                        if (shape->negated[p])
                            mod_eltwise_mul_subto(target[i + j], product[i], factor[j], n, mod);
                        else
                            mod_eltwise_mul_addto(target[i + j], product[i], factor[j], n, mod);
            }
            if (weight)
                for (uint64_t i = 0; i < d; i++)
                    for (uint64_t j = 0; j < d; j++)
                        mod_eltwise_mul_addto(target[i + j], scratch->signed_sum[i], weight[j], n,
                                              mod);
        }
}

// Reduce each sum's accumulator planes into the element sums[s * nodes + t]:
// every coefficient added up over the block's positions, then the polynomial
// modulo x^d - w.
static void field_fold_eltwise_finish(uint64_t *const *sums, const FieldFoldEltwiseScratch *scratch,
                                      const FieldProductShape *shape, uint64_t w_mod_q, Modulus mod)
{
    const uint64_t d = scratch->d, n_unreduced = 2 * d - 1, q = mod->q;
    uint64_t poly[FIELD_FOLD_MAX_UNREDUCED];
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            uint64_t *const *target = scratch->acc + (s * shape->nodes + t) * n_unreduced;
            for (uint64_t k = 0; k < n_unreduced; k++)
            {
                const uint64_t *plane = target[k];
                uint64_t total = 0;
                for (uint64_t i = 0; i < scratch->block_len; i++)
                    total = add_modq(total, plane[i], q);
                poly[k] = total;
            }
            uint64_t *out = sums[s * shape->nodes + t];
            for (uint64_t k = 0; k < d; k++)
                out[k] =
                    k + 1 < d ? add_modq(poly[k], mul_modq(w_mod_q, poly[k + d], mod), q) : poly[k];
        }
}

// Zero positions [live, n) of each of the `count` planes.
static void field_fold_eltwise_clear(uint64_t *const *planes, uint64_t count, uint64_t live,
                                     uint64_t n)
{
    for (uint64_t p = 0; live < n && p < count; p++)
        memset(planes[p] + live, 0, (n - live) * sizeof(uint64_t));
}

// The sums of the pairs [begin, end) of the tables, into sums[0 .. slots - 1],
// a block at a time.
static void field_fold_eltwise_sums(uint64_t *const *sums, const FieldFoldJob *job, uint64_t begin,
                                    uint64_t end)
{
    const FieldProductShape *shape = job->shape;
    const FieldVector first = job->tables[0];
    const Modulus mod = first->mod;
    const uint64_t d = first->d, w_mod_q = modq(first->w, mod);
    FieldFoldEltwiseScratch scratch;
    uint64_t *weight[FIELD_FOLD_MAX_DEGREE];

    field_fold_eltwise_open(&scratch, job, end - begin);
    for (uint64_t start = begin; start < end; start += scratch.block_len)
    {
        const uint64_t live = end - start < scratch.block_len ? end - start : scratch.block_len;
        const uint64_t n = field_vec_padded_length(live);
        for (uint64_t x = 0; x < shape->used_count; x++)
        {
            const uint64_t j = shape->used[x];
            uint64_t *const *F0 = field_fold_eltwise_fold_planes(&scratch, j, 0);
            for (uint64_t k = 0; k < d; k++)
            {
                uint64_t *const halves[2] = {F0[k], F0[d + k]};
                vec_deinterleave_u64(halves, job->tables[j]->coeffs[k] + 2 * start, 2, live);
            }
            field_fold_eltwise_clear(F0, 2 * d, live, n);
            field_fold_eltwise_nodes(F0, &scratch, n, mod);
        }
        // Positions past `live` read the weights' padding, whatever it holds;
        // their folds are zero, so they add nothing.
        if (job->weights)
            for (uint64_t k = 0; k < d; k++)
                weight[k] = job->weights->coeffs[k] + start;
        field_fold_eltwise_add(&scratch, job->weights ? weight : NULL, shape, n, w_mod_q, mod);
    }
    field_fold_eltwise_finish(sums, &scratch, shape, w_mod_q, mod);
    field_fold_eltwise_close(&scratch);
}

// The "fold and" variant over the pairs [begin, end) of the folded tables, a
// block at a time, as field_fold_scalar_fold_sums. Folded pair i comes from the
// input's four elements 4i .. 4i + 3, so the input is split four ways
// (quarters) and folded pair by pair into two halves. The chunk that ends the
// range also zeroes the outputs' padding.
static void field_fold_eltwise_fold_sums(uint64_t *const *sums, const FieldFoldJob *job,
                                         uint64_t begin, uint64_t end)
{
    const FieldProductShape *shape = job->shape;
    const FieldVector first = job->tables[0];
    const Modulus mod = first->mod;
    const uint64_t d = first->d, w_mod_q = modq(first->w, mod);
    FieldFoldEltwiseScratch scratch;
    uint64_t r[FIELD_FOLD_MAX_DEGREE], wr[FIELD_FOLD_MAX_DEGREE];
    uint64_t *weight[FIELD_FOLD_MAX_DEGREE];

    for (uint64_t k = 0; k < d; k++)
    {
        r[k] = modq(job->r[k], mod);
        wr[k] = mul_modq(w_mod_q, r[k], mod);
    }
    field_fold_eltwise_open(&scratch, job, end - begin);
    for (uint64_t start = begin; start < end; start += scratch.block_len)
    {
        const uint64_t live = end - start < scratch.block_len ? end - start : scratch.block_len;
        const uint64_t n = field_vec_padded_length(live);
        for (uint64_t j = 0; j < job->count; j++)
        {
            uint64_t *const *F0 =
                shape->is_used[j] ? field_fold_eltwise_fold_planes(&scratch, j, 0) : scratch.halves;
            for (uint64_t k = 0; k < d; k++)
                vec_deinterleave_u64(scratch.quarters + 4 * k,
                                     job->tables[j]->coeffs[k] + 4 * start, 4, live);
            field_fold_eltwise_clear(scratch.quarters, 4 * d, live, n);
            // Half h is the fold of quarters 2h and 2h + 1: u + r (v - u).
            for (uint64_t h = 0; h < 2; h++)
            {
                for (uint64_t k = 0; k < d; k++)
                {
                    mod_eltwise_sub(scratch.diff[k], scratch.quarters[4 * k + 2 * h + 1],
                                    scratch.quarters[4 * k + 2 * h], n, mod);
                    memcpy(F0[h * d + k], scratch.quarters[4 * k + 2 * h], n * sizeof(uint64_t));
                }
                for (uint64_t k = 0; k < d; k++)
                    for (uint64_t i = 0; i < d; i++)
                        mod_eltwise_fma(F0[h * d + k], scratch.diff[i],
                                        i > k ? wr[k + d - i] : r[k - i], n, mod);
            }
            for (uint64_t k = 0; k < d; k++)
                vec_interleave_u64(job->outs[j]->coeffs[k] + 2 * start, F0[k], F0[d + k], live);
            if (shape->is_used[j])
                field_fold_eltwise_nodes(F0, &scratch, n, mod);
        }
        if (job->weights)
            for (uint64_t k = 0; k < d; k++)
                weight[k] = job->weights->coeffs[k] + start;
        field_fold_eltwise_add(&scratch, job->weights ? weight : NULL, shape, n, w_mod_q, mod);
    }
    if (end == job->total)
        for (uint64_t j = 0; j < job->count; j++)
        {
            const FieldVector out = job->outs[j];
            for (uint64_t k = 0; k < d; k++)
                memset(out->coeffs[k] + 2 * end, 0,
                       (out->allocated_n - 2 * end) * sizeof(uint64_t));
        }
    field_fold_eltwise_finish(sums, &scratch, shape, w_mod_q, mod);
    field_fold_eltwise_close(&scratch);
}

// --- this body's ranges, for field_vector_fold.c ---------------------------

void field_fold_eltwise_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                   uint64_t *const *sums)
{
    field_fold_eltwise_sums(sums, job, begin, end);
}

void field_fold_eltwise_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                        uint64_t *const *sums)
{
    field_fold_eltwise_fold_sums(sums, job, begin, end);
}

#endif // VFHE_HAVE_AVX512IFMA
