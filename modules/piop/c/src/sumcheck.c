// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"
#include "arith_generic.h"
#include <arith_trace.h>
#include <parallel.h>
#include <stdbool.h>
#include <stdlib.h>

// -------------------------------------------------------------
// Sumcheck prover kernels over dense-MLE tables of ring elements
// -------------------------------------------------------------
// Libra-style evaluation-basis prover: per round, accumulate the round
// polynomial's evaluations from the table pairs (lo, hi) = (f|x=0, f|x=1)
// of the round variable. The round variable may sit at any position: the
// entry points only differ in where the pair sits — adjacent entries for
// the LSB variable (pairs), the two table halves for the MSB variable
// (halves), and stride-computed indices for any position (the generic
// fallback taking eval_var_idx). All three are the stride computation of
// pair_low with the variable's index. The Python layer picks by the
// variable's position, mirroring the mle.c binding kernels. `size` is the
// live table size (2^num_vars); entries must be in NTT (RNS) form. Folding
// with the challenge is binding the round variable — mle_dense_poly_evaluate*
// in mle.c, not duplicated here.
//
// A round over many pairs is split into contiguous chunks of pairs run on
// the library's threads. Chunk 0 accumulates into the message itself, every
// other chunk into accumulators of its own, and the caller adds those into
// the message in chunk order once the loop is done. The arithmetic is exact,
// so the message is the same at any number of chunks. While the execution
// tape records, or the generic path is forced, a round runs on the calling
// thread, so the tape sees every operation in order.

// Rounds reading fewer bytes of table entries than this run on one thread:
// the work is a few ring operations per entry read, and below this it does
// not pay for handing chunks to other threads.
#define ROUND_PARALLEL_BYTES (1u << 20)
// The fewest pairs one chunk covers, so that its accumulators (one element
// per node, added into the message after the loop) and temporaries stay
// small next to its own work.
#define ROUND_MIN_CHUNK_PAIRS 8
// Chunks per thread: a few, so that chunks of uneven cost balance out.
#define ROUND_CHUNKS_PER_THREAD 4

// Index of the low entry of pair i of the variable at eval_var_idx: the
// entries differing only in bit eval_var_idx, in the folded table's order.
static inline uint64_t pair_low(uint64_t i, uint64_t eval_var_idx)
{
    const uint64_t stride = 1ULL << eval_var_idx;
    return (i & (stride - 1)) + ((i >> eval_var_idx) << (eval_var_idx + 1));
}

// The index of the last variable of a table of `size` entries.
static uint64_t msb_index(uint64_t size)
{
    uint64_t eval_var_idx = 0;
    while ((2ULL << eval_var_idx) < size)
    {
        eval_var_idx++;
    }
    return eval_var_idx;
}

// Bytes of one element of `ring`: a word per coefficient and prime for an
// RNS ring, counted as one word per coefficient for any other.
static uint64_t element_bytes(ArithRing ring)
{
    const uint64_t rows = ring->impl == ARITH_IMPL_RNS
                              ? (uint64_t)__builtin_popcountll(arith_rns_ring_mask(ring))
                              : 1;
    return rows * ring->N * sizeof(uint64_t);
}

// Accumulates the pairs [begin, end) of one round into g[0 .. nodes), which
// hold zero on entry. `round` is the kernel's own description of the round.
typedef void (*RoundRange)(const void *round, ArithElement *g, uint64_t begin, uint64_t end);

typedef struct
{
    ArithRing ring;
    RoundRange range;
    const void *round;
    ArithElement *g_out;       // chunk 0's accumulators: the message
    ArithElement *partial;     // chunk c > 0's start at (c - 1) * nodes
    const ArithElement *model; // an entry, the shape of the accumulators
    uint64_t nodes, pairs, chunks;
} RoundChunks;

static void round_chunk(void *ctx, uint64_t c)
{
    const RoundChunks *job = (const RoundChunks *)ctx;
    ArithElement *g = job->g_out;
    if (c > 0)
    {
        g = &job->partial[(c - 1) * job->nodes];
        for (uint64_t t = 0; t < job->nodes; t++)
        {
            arith_zero_in(job->ring, &g[t], arith_mul_domain(job->ring));
        }
    }
    const uint64_t begin = c * job->pairs / job->chunks;
    const uint64_t end = (c + 1) * job->pairs / job->chunks;
    job->range(job->round, g, begin, end);
}

// How many chunks a round of `pairs` pairs, each reading `entries_per_pair`
// table entries, is split into; 1 runs it on the calling thread.
static uint64_t round_chunk_count(ArithRing ring, uint64_t pairs, uint64_t entries_per_pair)
{
    if (ARITH_GENERIC_PATH())
    {
        return 1;
    }
    if (pairs * entries_per_pair * element_bytes(ring) < ROUND_PARALLEL_BYTES)
    {
        return 1;
    }
    const uint64_t most = pairs / ROUND_MIN_CHUNK_PAIRS;
    const uint64_t threads = vfhe_threads_for(0, most);
    if (threads == 1)
    {
        return 1;
    }
    const uint64_t chunks = threads * ROUND_CHUNKS_PER_THREAD;
    return chunks < most ? chunks : most;
}

// Zeroes g_out[0 .. nodes) and accumulates every pair of the round into it,
// over chunks on the library's threads when the round is large enough.
static void round_over_pairs(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                             const ArithElement *model, uint64_t size, uint64_t entries_per_pair,
                             RoundRange range, const void *round)
{
    for (uint64_t t = 0; t < nodes; t++)
    {
        arith_zero_in(ring, &g_out[t], arith_mul_domain(ring));
    }
    const uint64_t pairs = size / 2;
    const uint64_t chunks = round_chunk_count(ring, pairs, entries_per_pair);
    if (chunks == 1)
    {
        range(round, g_out, 0, pairs);
        return;
    }
    const uint64_t n_partial = (chunks - 1) * nodes;
    ArithElement *partial = (ArithElement *)malloc(n_partial * sizeof(ArithElement));
    for (uint64_t k = 0; k < n_partial; k++)
    {
        arith_new_like(ring, model, &partial[k]);
    }
    RoundChunks job = {ring, range, round, g_out, partial, model, nodes, pairs, chunks};
    vfhe_parallel_for(chunks, 0, round_chunk, &job);
    for (uint64_t c = 1; c < chunks; c++)
    {
        for (uint64_t t = 0; t < nodes; t++)
        {
            arith_add(ring, &g_out[t], &g_out[t], &partial[(c - 1) * nodes + t]);
        }
    }
    for (uint64_t k = 0; k < n_partial; k++)
    {
        arith_free(ring, &partial[k]);
    }
    free(partial);
}

// --- degree 1 ----------------------------------------------------------------

typedef struct
{
    ArithRing ring;
    const ArithElement *table;
    uint64_t eval_var_idx;
} LinearRound;

static void linear_round_range(const void *round, ArithElement *g, uint64_t begin, uint64_t end)
{
    const LinearRound *job = (const LinearRound *)round;
    const uint64_t stride = 1ULL << job->eval_var_idx;
    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t idx0 = pair_low(i, job->eval_var_idx);
        arith_add(job->ring, &g[0], &g[0], &job->table[idx0]);
        arith_add(job->ring, &g[1], &g[1], &job->table[idx0 + stride]);
    }
}

static void linear_round(ArithRing ring, ArithElement *g0, ArithElement *g1,
                         const ArithElement *table, uint64_t size, uint64_t eval_var_idx)
{
    // The message as one array, as the chunks take it; the elements' domains
    // are copied back once it is written.
    ArithElement g[2] = {*g0, *g1};
    LinearRound job = {ring, table, eval_var_idx};
    round_over_pairs(ring, g, 2, &table[0], size, 2, linear_round_range, &job);
    *g0 = g[0];
    *g1 = g[1];
}

void sumcheck_round_pairs(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                          uint64_t size)
{
    linear_round(ring, g0, g1, table, size, 0);
}

void sumcheck_round_halves(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                           uint64_t size)
{
    linear_round(ring, g0, g1, table, size, msb_index(size));
}

void sumcheck_round(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                    uint64_t size, uint64_t eval_var_idx)
{
    linear_round(ring, g0, g1, table, size, eval_var_idx);
}

// --- degree 2 ----------------------------------------------------------------

// Degree-2 accumulation for one pair of each factor: g_out[0] += lo_f*lo_g,
// g_out[1] += hi_f*hi_g, g_out[2] += (2*hi_f - lo_f)*(2*hi_g - lo_g) (the
// evaluation at t = 2, extrapolated division-free).
static void sumcheck_prod2_accumulate(ArithRing ring, ArithElement *g_out, const ArithElement *f_lo,
                                      const ArithElement *f_hi, const ArithElement *g_lo,
                                      const ArithElement *g_hi, ArithElement *tmp1,
                                      ArithElement *tmp2)
{
    arith_mul_addto(ring, &g_out[0], f_lo, g_lo);
    arith_mul_addto(ring, &g_out[1], f_hi, g_hi);
    arith_scale_int(ring, tmp1, f_hi, 2);
    arith_sub(ring, tmp1, tmp1, f_lo);
    arith_scale_int(ring, tmp2, g_hi, 2);
    arith_sub(ring, tmp2, tmp2, g_lo);
    arith_mul_addto(ring, &g_out[2], tmp1, tmp2);
}

typedef struct
{
    ArithRing ring;
    const ArithElement *tf, *tg;
    uint64_t eval_var_idx;
} Prod2Round;

static void prod2_round_range(const void *round, ArithElement *g, uint64_t begin, uint64_t end)
{
    const Prod2Round *job = (const Prod2Round *)round;
    const uint64_t stride = 1ULL << job->eval_var_idx;
    const ArithElement *tf = job->tf, *tg = job->tg;
    ArithElement tmp1, tmp2;
    arith_new_like(job->ring, &tf[0], &tmp1);
    arith_new_like(job->ring, &tg[0], &tmp2);
    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t idx0 = pair_low(i, job->eval_var_idx);
        sumcheck_prod2_accumulate(job->ring, g, &tf[idx0], &tf[idx0 + stride], &tg[idx0],
                                  &tg[idx0 + stride], &tmp1, &tmp2);
    }
    arith_free(job->ring, &tmp1);
    arith_free(job->ring, &tmp2);
}

static void prod2_round(ArithRing ring, ArithElement *g_out, const ArithElement *tf,
                        const ArithElement *tg, uint64_t size, uint64_t eval_var_idx)
{
    Prod2Round job = {ring, tf, tg, eval_var_idx};
    round_over_pairs(ring, g_out, 3, &tf[0], size, 4, prod2_round_range, &job);
}

void sumcheck_prod2_round_pairs(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                ArithElement *tg, uint64_t size)
{
    prod2_round(ring, g_out, tf, tg, size, 0);
}

void sumcheck_prod2_round_halves(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                 ArithElement *tg, uint64_t size)
{
    prod2_round(ring, g_out, tf, tg, size, msb_index(size));
}

void sumcheck_prod2_round(ArithRing ring, ArithElement *g_out, ArithElement *tf, ArithElement *tg,
                          uint64_t size, uint64_t eval_var_idx)
{
    prod2_round(ring, g_out, tf, tg, size, eval_var_idx);
}

// --- degree 3 ----------------------------------------------------------------

// Degree-3 accumulation for one pair of each of three factors: g_out[t] +=
// f_t * g_t * h_t for t in {0, 1, 2, 3}, a factor's value at t = 2 and t = 3
// extrapolated division-free from its pair as hi + (hi - lo) and
// hi + 2 (hi - lo). `diff` and `ext` hold one temporary per factor.
static void sumcheck_prod3_accumulate(ArithRing ring, ArithElement *g_out,
                                      const ArithElement *const lo[3],
                                      const ArithElement *const hi[3], ArithElement *diff,
                                      ArithElement *ext, ArithElement *prod)
{
    arith_mul(ring, prod, lo[0], lo[1]);
    arith_mul_addto(ring, &g_out[0], prod, lo[2]);
    arith_mul(ring, prod, hi[0], hi[1]);
    arith_mul_addto(ring, &g_out[1], prod, hi[2]);
    for (int j = 0; j < 3; j++)
    {
        arith_sub(ring, &diff[j], hi[j], lo[j]);
        arith_add(ring, &ext[j], hi[j], &diff[j]);
    }
    arith_mul(ring, prod, &ext[0], &ext[1]);
    arith_mul_addto(ring, &g_out[2], prod, &ext[2]);
    for (int j = 0; j < 3; j++)
    {
        arith_add(ring, &ext[j], &ext[j], &diff[j]);
    }
    arith_mul(ring, prod, &ext[0], &ext[1]);
    arith_mul_addto(ring, &g_out[3], prod, &ext[2]);
}

typedef struct
{
    ArithRing ring;
    const ArithElement *tf, *tg, *th;
    uint64_t eval_var_idx;
} Prod3Round;

static void prod3_round_range(const void *round, ArithElement *g, uint64_t begin, uint64_t end)
{
    const Prod3Round *job = (const Prod3Round *)round;
    const uint64_t stride = 1ULL << job->eval_var_idx;
    const ArithElement *tf = job->tf, *tg = job->tg, *th = job->th;
    const ArithElement *tables[3] = {tf, tg, th};
    ArithElement diff[3], ext[3], prod;
    for (int j = 0; j < 3; j++)
    {
        arith_new_like(job->ring, &tables[j][0], &diff[j]);
        arith_new_like(job->ring, &tables[j][0], &ext[j]);
    }
    arith_new_like(job->ring, &tf[0], &prod);

    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t idx0 = pair_low(i, job->eval_var_idx);
        const ArithElement *lo[3] = {&tf[idx0], &tg[idx0], &th[idx0]};
        const ArithElement *hi[3] = {&tf[idx0 + stride], &tg[idx0 + stride], &th[idx0 + stride]};
        sumcheck_prod3_accumulate(job->ring, g, lo, hi, diff, ext, &prod);
    }

    for (int j = 0; j < 3; j++)
    {
        arith_free(job->ring, &diff[j]);
        arith_free(job->ring, &ext[j]);
    }
    arith_free(job->ring, &prod);
}

// The degree-3 round over the pairs of the variable at `eval_var_idx`.
static void prod3_round(ArithRing ring, ArithElement *g_out, const ArithElement *tf,
                        const ArithElement *tg, const ArithElement *th, uint64_t size,
                        uint64_t eval_var_idx)
{
    Prod3Round job = {ring, tf, tg, th, eval_var_idx};
    round_over_pairs(ring, g_out, 4, &tf[0], size, 6, prod3_round_range, &job);
}

void sumcheck_prod3_round_pairs(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                ArithElement *tg, ArithElement *th, uint64_t size)
{
    prod3_round(ring, g_out, tf, tg, th, size, 0);
}

void sumcheck_prod3_round_halves(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                 ArithElement *tg, ArithElement *th, uint64_t size)
{
    prod3_round(ring, g_out, tf, tg, th, size, msb_index(size));
}

void sumcheck_prod3_round(ArithRing ring, ArithElement *g_out, ArithElement *tf, ArithElement *tg,
                          ArithElement *th, uint64_t size, uint64_t eval_var_idx)
{
    prod3_round(ring, g_out, tf, tg, th, size, eval_var_idx);
}

// --- any number of factors, any number of nodes -------------------------------

typedef struct
{
    ArithRing ring;
    ArithElement *const *tables;
    uint64_t factors, nodes, eval_var_idx;
} ProductRound;

// acc += value[0] * ... * value[factors - 1]. `prod` holds two temporaries,
// used for three factors or more: the partial products alternate between
// them, as a product may not be written over one of its operands.
static void product_addto(ArithRing ring, ArithElement *acc, const ArithElement *const *value,
                          uint64_t factors, ArithElement *prod)
{
    if (factors == 1)
    {
        arith_add(ring, acc, acc, value[0]);
        return;
    }
    if (factors == 2)
    {
        arith_mul_addto(ring, acc, value[0], value[1]);
        return;
    }
    uint64_t cur = 0;
    arith_mul(ring, &prod[cur], value[0], value[1]);
    for (uint64_t j = 2; j + 1 < factors; j++)
    {
        arith_mul(ring, &prod[cur ^ 1], &prod[cur], value[j]);
        cur ^= 1;
    }
    arith_mul_addto(ring, acc, &prod[cur], value[factors - 1]);
}

// For each pair and each node t, every factor's value at t is lo at t = 0,
// hi at t = 1, and from t = 2 on the previous node's plus hi - lo; the
// product of the values is added into g[t].
static void product_round_range(const void *round, ArithElement *g, uint64_t begin, uint64_t end)
{
    const ProductRound *job = (const ProductRound *)round;
    const ArithRing ring = job->ring;
    const uint64_t k = job->factors, m = job->nodes;
    const uint64_t stride = 1ULL << job->eval_var_idx;
    ArithElement *const *tables = job->tables;
    const ArithElement **value = (const ArithElement **)malloc(k * sizeof(*value));
    // diff[j] = hi - lo and ext[j], the value at the node, per factor; then
    // the two product temporaries.
    ArithElement *scratch = (ArithElement *)malloc((2 * k + 2) * sizeof(ArithElement));
    ArithElement *diff = scratch, *ext = scratch + k, *prod = scratch + 2 * k;
    const bool extrapolates = m > 2;
    if (extrapolates)
    {
        for (uint64_t j = 0; j < k; j++)
        {
            arith_new_like(ring, &tables[j][0], &diff[j]);
            arith_new_like(ring, &tables[j][0], &ext[j]);
        }
    }
    if (k > 2)
    {
        arith_new_like(ring, &tables[0][0], &prod[0]);
        arith_new_like(ring, &tables[0][0], &prod[1]);
    }

    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t lo = pair_low(i, job->eval_var_idx), hi = lo + stride;
        for (uint64_t j = 0; j < k; j++)
        {
            value[j] = &tables[j][lo];
        }
        product_addto(ring, &g[0], value, k, prod);
        if (m < 2)
        {
            continue;
        }
        for (uint64_t j = 0; j < k; j++)
        {
            value[j] = &tables[j][hi];
        }
        product_addto(ring, &g[1], value, k, prod);
        for (uint64_t t = 2; t < m; t++)
        {
            for (uint64_t j = 0; j < k; j++)
            {
                if (t == 2)
                {
                    arith_sub(ring, &diff[j], &tables[j][hi], &tables[j][lo]);
                    arith_add(ring, &ext[j], &tables[j][hi], &diff[j]);
                }
                else
                {
                    arith_add(ring, &ext[j], &ext[j], &diff[j]);
                }
                value[j] = &ext[j];
            }
            product_addto(ring, &g[t], value, k, prod);
        }
    }

    if (extrapolates)
    {
        for (uint64_t j = 0; j < k; j++)
        {
            arith_free(ring, &diff[j]);
            arith_free(ring, &ext[j]);
        }
    }
    if (k > 2)
    {
        arith_free(ring, &prod[0]);
        arith_free(ring, &prod[1]);
    }
    free(scratch);
    free(value);
}

static void product_round(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                          ArithElement *const *tables, uint64_t factors, uint64_t size,
                          uint64_t eval_var_idx)
{
    ProductRound job = {ring, tables, factors, nodes, eval_var_idx};
    round_over_pairs(ring, g_out, nodes, &tables[0][0], size, 2 * factors, product_round_range,
                     &job);
}

void sumcheck_prod_round_pairs(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                               ArithElement *const *tables, uint64_t factors, uint64_t size)
{
    product_round(ring, g_out, nodes, tables, factors, size, 0);
}

void sumcheck_prod_round_halves(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                                ArithElement *const *tables, uint64_t factors, uint64_t size)
{
    product_round(ring, g_out, nodes, tables, factors, size, msb_index(size));
}

void sumcheck_prod_round(ArithRing ring, ArithElement *g_out, uint64_t nodes,
                         ArithElement *const *tables, uint64_t factors, uint64_t size,
                         uint64_t eval_var_idx)
{
    product_round(ring, g_out, nodes, tables, factors, size, eval_var_idx);
}
