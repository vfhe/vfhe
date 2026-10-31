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
#include "../vec_chunks.h"

uint64_t pmf_vec_padded_length(uint64_t n)
{
    const uint64_t unit = PMF_LANES;
    return ((n + unit - 1) / unit) * unit;
}

// --- running a kernel on the library's threads -----------------------------
//
// A kernel whose groups are independent is a loop over a range of its vector
// with that range as parameters. On one thread it runs once over the whole
// range; otherwise once per chunk (vec_chunks.h), each chunk a multiple of
// PMF_LANES elements, so chunks never share a group. A reduction keeps one
// partial result per chunk and adds them up in chunk order; every partial is
// canonical, so the total is the serial one, representation included.

// The fewest elements a chunk covers, by what an element costs.
// One addition or subtraction an element, bounded by memory traffic.
#define PMF_VEC_ADD_GRAIN (1u << 15)
// A field product or more an element: products, folds, sums of products and
// the inversion's sweeps.
#define PMF_VEC_PRODUCT_GRAIN (1u << 13)
// Copies and layout changes, in bytes of planes written: bandwidth-bound and
// cheap per byte, so a chunk is large.
#define PMF_VEC_MOVE_GRAIN_BYTES (4u << 20)
// Hashing, in bytes of encodings: an encoding per element and a compression
// per 64 bytes.
#define PMF_VEC_HASH_GRAIN_BYTES (256u << 10)

typedef struct PMFVecJob PMFVecJob;

// A kernel's operands, and its loop over [begin, end) of the range it splits;
// `chunk` indexes the partial result of a reduction. Which operand fields are
// used is up to the kernel.
struct PMFVecJob
{
    void (*range)(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk);
    VecChunks chunks;
    uint64_t total; // the range split, [0, total)
    PMFParams params;
    PMFVector out, out2, a, b, t;
    const uint64_t *s, *s2;
    PMFVector *outs;
    const PMFVector *tables;
    uint64_t count;
    const void *shape;
    uint64_t rows, width;
    uint64_t start, shift, mask;
    uint64_t group, stride;
    uint64_t *words;
    uint8_t *bytes;
    uint64_t *partials; // a reduction's `slots` elements per chunk
    uint64_t slots;
};

static void pmf_vec_chunk(void *ctx, uint64_t chunk)
{
    const PMFVecJob *job = (const PMFVecJob *)ctx;
    job->range(job, vec_chunk_begin(&job->chunks, chunk), vec_chunk_end(&job->chunks, chunk),
               chunk);
}

// Plans the chunks of [0, total); false when the kernel gets one thread.
static bool pmf_vec_split(PMFVecJob *job, uint64_t total, uint64_t grain)
{
    job->total = total;
    return vec_chunks_plan(&job->chunks, total, grain, PMF_LANES);
}

static void pmf_vec_run_chunks(PMFVecJob *job)
{
    vfhe_parallel_for(job->chunks.count, 0, pmf_vec_chunk, job);
}

// job->range over [0, total), split across threads when they are allowed.
static void pmf_vec_run(PMFVecJob *job, uint64_t total, uint64_t grain)
{
    if (pmf_vec_split(job, total, grain))
        pmf_vec_run_chunks(job);
    else
        job->range(job, 0, total, 0);
}

// The elements a chunk of a data-movement kernel covers at least.
static uint64_t pmf_vec_move_grain(PMFParams params)
{
    return PMF_VEC_MOVE_GRAIN_BYTES / (params->limbs * sizeof(uint64_t));
}

// The `slots` result elements of chunk `chunk`, PMF_LANES words each.
static uint64_t *pmf_vec_partial(const PMFVecJob *job, uint64_t chunk, uint64_t slot)
{
    return job->partials + (chunk * job->slots + slot) * PMF_LANES;
}

// Runs a planned split of a reduction: every chunk writes its `slots` partial
// results, which are then added up into results[0 .. slots - 1] in chunk
// order.
static void pmf_vec_reduce_chunks(PMFVecJob *job, uint64_t *const *results, uint64_t slots)
{
    const uint64_t count = job->chunks.count;
    job->slots = slots;
    job->partials = (uint64_t *)safe_malloc(count * slots * PMF_LANES * sizeof(uint64_t));
    pmf_vec_run_chunks(job);
    for (uint64_t s = 0; s < slots; s++)
    {
        memcpy(results[s], pmf_vec_partial(job, 0, s), PMF_LANES * sizeof(uint64_t));
        for (uint64_t c = 1; c < count; c++)
            pmf_add(results[s], results[s], pmf_vec_partial(job, c, s), job->params);
    }
    free(job->partials);
}

// The three entry-point shapes below differ only in where the right-hand operand
// comes from -- the other vector, or one element broadcast -- so each is a walk
// over the groups around one of the cores in pmf_vec_group.h. Every kernel in
// this file is a body taking the params, behind an entry point that dispatches
// it with PMF_WITH_LIMBS.

#define PMF_VEC_BINARY(name, core, grain)                                                          \
    PMF_INLINE void name##_body(PMFVector out, const PMFVector a, const PMFVector b,               \
                                uint64_t begin, uint64_t end, PMFParams P)                         \
    {                                                                                              \
        const uint64_t L = P->limbs;                                                               \
        pv_t av[PMF_MAX_LIMBS], bv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];                           \
        for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)                                      \
        {                                                                                          \
            pmf_vec_load_group(av, a, i, L);                                                       \
            pmf_vec_load_group(bv, b, i, L);                                                       \
            core(T, av, bv, P);                                                                    \
            pmf_vec_store_group(out, i, T, L);                                                     \
        }                                                                                          \
    }                                                                                              \
    static void name##_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)   \
    {                                                                                              \
        (void)chunk;                                                                               \
        PMF_WITH_LIMBS(job->params, name##_body(job->out, job->a, job->b, begin, end, &P_));       \
    }                                                                                              \
    void name(PMFVector out, const PMFVector a, const PMFVector b)                                 \
    {                                                                                              \
        PMFVecJob job = {.range = name##_range, .params = a->params, .out = out, .a = a, .b = b};  \
        pmf_vec_run(&job, a->allocated_n, grain);                                                  \
    }

#define PMF_VEC_BROADCAST(name, core, grain)                                                       \
    PMF_INLINE void name##_body(PMFVector out, const PMFVector a, const uint64_t *s,               \
                                uint64_t begin, uint64_t end, PMFParams P)                         \
    {                                                                                              \
        const uint64_t L = P->limbs;                                                               \
        pv_t av[PMF_MAX_LIMBS], sv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];                           \
        pmf_vec_load_scalar(sv, s, L);                                                             \
        for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)                                      \
        {                                                                                          \
            pmf_vec_load_group(av, a, i, L);                                                       \
            core(T, av, sv, P);                                                                    \
            pmf_vec_store_group(out, i, T, L);                                                     \
        }                                                                                          \
    }                                                                                              \
    static void name##_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)   \
    {                                                                                              \
        (void)chunk;                                                                               \
        PMF_WITH_LIMBS(job->params, name##_body(job->out, job->a, job->s, begin, end, &P_));       \
    }                                                                                              \
    void name(PMFVector out, const PMFVector a, const uint64_t *s)                                 \
    {                                                                                              \
        PMFVecJob job = {.range = name##_range, .params = a->params, .out = out, .a = a, .s = s};  \
        pmf_vec_run(&job, a->allocated_n, grain);                                                  \
    }

PMF_VEC_BINARY(pmf_vec_add, pmf_vec_add_group, PMF_VEC_ADD_GRAIN)
PMF_VEC_BINARY(pmf_vec_sub, pmf_vec_sub_group, PMF_VEC_ADD_GRAIN)
PMF_VEC_BINARY(pmf_vec_mul, pmf_vec_mul_group, PMF_VEC_PRODUCT_GRAIN)

PMF_VEC_BROADCAST(pmf_vec_add_scalar, pmf_vec_add_group, PMF_VEC_ADD_GRAIN)
PMF_VEC_BROADCAST(pmf_vec_sub_scalar, pmf_vec_sub_group, PMF_VEC_ADD_GRAIN)
PMF_VEC_BROADCAST(pmf_vec_scale, pmf_vec_mul_group, PMF_VEC_PRODUCT_GRAIN)

PMF_INLINE void pmf_vec_scalar_sub_body(PMFVector out, const uint64_t *s, const PMFVector a,
                                        uint64_t begin, uint64_t end, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t av[PMF_MAX_LIMBS], sv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(sv, s, L);
    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_group(av, a, i, L);
        pmf_vec_sub_group(T, sv, av, P);
        pmf_vec_store_group(out, i, T, L);
    }
}

static void pmf_vec_scalar_sub_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                     uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params, pmf_vec_scalar_sub_body(job->out, job->s, job->a, begin, end, &P_));
}

void pmf_vec_scalar_sub(PMFVector out, const uint64_t *s, const PMFVector a)
{
    PMFVecJob job = {
        .range = pmf_vec_scalar_sub_range, .params = a->params, .out = out, .a = a, .s = s};
    pmf_vec_run(&job, a->allocated_n, PMF_VEC_ADD_GRAIN);
}

PMF_INLINE void pmf_vec_neg_body(PMFVector out, const PMFVector a, uint64_t begin, uint64_t end,
                                 PMFParams P)
{
    const uint64_t L = P->limbs;
    const pv_t mask = pv_mask52();
    pv_t av[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS];

    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
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

static void pmf_vec_neg_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params, pmf_vec_neg_body(job->out, job->a, begin, end, &P_));
}

void pmf_vec_neg(PMFVector out, const PMFVector a)
{
    PMFVecJob job = {.range = pmf_vec_neg_range, .params = a->params, .out = out, .a = a};
    pmf_vec_run(&job, a->allocated_n, PMF_VEC_ADD_GRAIN);
}

// The sum of elements [begin, end) of `a`, both multiples of PMF_VEC_GROUP,
// through the lanes, plus elements end .. to - 1 one at a time, into one
// canonical element.
static void pmf_vec_sum_span(uint64_t *out, const PMFVector a, uint64_t begin, uint64_t end,
                             uint64_t to)
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

    uint64_t since_reduce = 0;
    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
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
    for (uint64_t i = end; i < to; i++)
    {
        pmf_vec_get_element(element, a, i);
        pmf_add(out, out, element, a->params);
    }
}

// The last chunk also takes the elements past the last whole group.
static void pmf_vec_sum_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)
{
    const uint64_t to = end == job->total ? job->a->n : end;
    pmf_vec_sum_span(pmf_vec_partial(job, chunk, 0), job->a, begin, end, to);
}

void pmf_vec_sum(uint64_t *out, const PMFVector a)
{
    // Whole groups of live elements only go through the lanes: the padding is
    // canonical but carries no value, so it must not reach the total.
    const uint64_t whole = (a->n / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    PMFVecJob job = {.range = pmf_vec_sum_range, .params = a->params, .a = a};
    uint64_t *const results[1] = {out};
    if (pmf_vec_split(&job, whole, PMF_VEC_ADD_GRAIN))
        pmf_vec_reduce_chunks(&job, results, 1);
    else
        pmf_vec_sum_span(out, a, 0, whole, a->n);
}

// --- inversion ------------------------------------------------------------

// Rows the inversion cuts a vector into. Montgomery's trick is a serial chain:
// each prefix product needs the one before it. Cut the vector into rows of
// `width` elements and every column is a chain of its own, so a step of either
// sweep is a group multiply per group of a row, none waiting on another, and a
// chain is only as long as there are rows. The column totals are then inverted
// together by the trick over single elements; 256 rows keep those a small
// fraction of the work.
#define PMF_VEC_INV_ROWS 256

// Montgomery's trick over `count >= 1` elements laid out PMF_LANES words
// apart, on the element kernels: inv[i] = 1 / x[i]. Returns 0, with `inv`
// holding no inverses, if one of the elements is zero. `inv` may not be `x`.
static int pmf_inv_serial(uint64_t *inv, const uint64_t *x, uint64_t count, PMFParams params)
{
    // inv[i] holds the prefix product x[0] ... x[i] until the sweep back
    // overwrites it.
    memcpy(inv, x, PMF_LANES * sizeof(uint64_t));
    for (uint64_t i = 1; i < count; i++)
        pmf_mul(inv + i * PMF_LANES, inv + (i - 1) * PMF_LANES, x + i * PMF_LANES, params);

    // The product is canonical, so it is zero exactly when every limb is.
    const uint64_t *total = inv + (count - 1) * PMF_LANES;
    uint64_t any = 0;
    for (uint64_t k = 0; k < params->limbs; k++)
        any |= total[k];
    if (!any)
        return 0;

    uint64_t running[PMF_LANES];
    pmf_ref_inverse(running, total, params);
    for (uint64_t i = count; i-- > 1;)
    {
        pmf_mul(inv + i * PMF_LANES, running, inv + (i - 1) * PMF_LANES, params);
        pmf_mul(running, running, x + i * PMF_LANES, params);
    }
    memcpy(inv, running, sizeof(running));
    return 1;
}

// The forward sweep over columns [c0, c1) of the first rows * width elements
// of `a`: prefix row t is the product of rows 0..t, column by column.
PMF_INLINE void pmf_vec_inv_forward_body(PMFVector prefix, const PMFVector a, uint64_t rows,
                                         uint64_t width, uint64_t c0, uint64_t c1, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t pv[PMF_MAX_LIMBS], av[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    for (uint64_t k = 0; k < L; k++)
        memcpy(prefix->limbs[k] + c0, a->limbs[k] + c0, (c1 - c0) * sizeof(uint64_t));
    for (uint64_t t = 1; t < rows; t++)
        for (uint64_t g = c0; g < c1; g += PMF_VEC_GROUP)
        {
            const uint64_t i = t * width + g;
            pmf_vec_load_group(pv, prefix, i - width, L);
            pmf_vec_load_group(av, a, i, L);
            pmf_vec_mul_group(T, pv, av, P);
            pmf_vec_store_group(prefix, i, T, L);
        }
}

// The backward sweep over columns [c0, c1). The last prefix row holds the
// running inverses, one per column, starting from the inverses of the column
// totals: for t from the last row down to 1, out row t is the running inverse
// times prefix row t - 1, and the running inverse then absorbs row t of `a`.
// Out row 0 is what is left. Each group of `a` is loaded before the same group
// of `out` is stored, so `out` may be `a`.
PMF_INLINE void pmf_vec_inv_backward_body(PMFVector out, const PMFVector a, PMFVector prefix,
                                          uint64_t rows, uint64_t width, uint64_t c0, uint64_t c1,
                                          PMFParams P)
{
    const uint64_t L = P->limbs, last = (rows - 1) * width;
    pv_t rv[PMF_MAX_LIMBS], pv[PMF_MAX_LIMBS], av[PMF_MAX_LIMBS];
    pv_t T[PMF_MAX_LIMBS + 1], U[PMF_MAX_LIMBS + 1];

    for (uint64_t t = rows; t-- > 1;)
    {
        for (uint64_t g = c0; g < c1; g += PMF_VEC_GROUP)
        {
            const uint64_t i = t * width + g;
            pmf_vec_load_group(rv, prefix, last + g, L);
            pmf_vec_load_group(pv, prefix, i - width, L);
            pmf_vec_load_group(av, a, i, L);
            pmf_vec_mul_group(T, rv, pv, P);
            pmf_vec_mul_group(U, rv, av, P);
            pmf_vec_store_group(out, i, T, L);
            pmf_vec_store_group(prefix, last + g, U, L);
        }
    }
    for (uint64_t k = 0; k < L; k++)
        memcpy(out->limbs[k] + c0, prefix->limbs[k] + last + c0, (c1 - c0) * sizeof(uint64_t));
}

// The columns are independent chains, so the sweeps split by columns: job->t
// is the prefix vector, job->rows and job->width its shape.
static void pmf_vec_inv_forward_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                      uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params, pmf_vec_inv_forward_body(job->t, job->a, job->rows, job->width,
                                                         begin, end, &P_));
}

static void pmf_vec_inv_backward_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                       uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params, pmf_vec_inv_backward_body(job->out, job->a, job->t, job->rows,
                                                          job->width, begin, end, &P_));
}

int pmf_vec_inv(PMFVector out, const PMFVector a)
{
    const uint64_t n = a->n, L = a->params->limbs;
    if (n == 0)
        return 1;

    // A row is a whole number of runs of PMF_LANES elements, so every row
    // starts aligned. Below PMF_VEC_INV_ROWS such runs there are no rows, and
    // every element goes through the trick over single elements.
    const uint64_t width = (n / PMF_VEC_INV_ROWS) & ~(uint64_t)(PMF_LANES - 1);
    const uint64_t rows = width ? n / width : 0, body = rows * width;

    // Inverted one at a time: the column totals, then the elements past the
    // last whole row.
    const uint64_t singles = width + (n - body);
    uint64_t *x = (uint64_t *)safe_malloc(2 * singles * PMF_LANES * sizeof(uint64_t));
    uint64_t *inv = x + singles * PMF_LANES;

    uint64_t *planes[PMF_MAX_LIMBS];
    uint64_t *block = NULL;
    struct _PMFVector prefix = *a;
    if (width)
    {
        block = (uint64_t *)safe_aligned_malloc(L * body * sizeof(uint64_t));
        for (uint64_t k = 0; k < L; k++)
            planes[k] = block + k * body;
        prefix.limbs = planes;
        prefix.n = prefix.allocated_n = body;
        PMFVecJob job = {.range = pmf_vec_inv_forward_range,
                         .params = a->params,
                         .a = a,
                         .t = &prefix,
                         .rows = rows,
                         .width = width};
        pmf_vec_run(&job, width, PMF_VEC_PRODUCT_GRAIN / rows);
        pmf_vec_get_range(x, &prefix, body - width, width);
    }
    pmf_vec_get_range(x + width * PMF_LANES, a, body, n - body);

    const int status = pmf_inv_serial(inv, x, singles, a->params);
    if (status)
    {
        if (width)
        {
            pmf_vec_set_range(&prefix, body - width, inv, width);
            PMFVecJob job = {.range = pmf_vec_inv_backward_range,
                             .params = a->params,
                             .out = out,
                             .a = a,
                             .t = &prefix,
                             .rows = rows,
                             .width = width};
            pmf_vec_run(&job, width, PMF_VEC_PRODUCT_GRAIN / rows);
        }
        pmf_vec_set_range(out, body, inv + width * PMF_LANES, n - body);
    }
    free(block);
    free(x);
    return status;
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

PMF_INLINE void pmf_vec_fold_body(PMFVector out, const PMFVector a, const uint64_t *r,
                                  uint64_t begin, uint64_t end, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t R[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(R, r, L);
    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
    {
        pmf_vec_fold_group(T, a, i, R, P);
        pmf_vec_store_group(out, i, T, L);
    }
}

static void pmf_vec_fold_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params, pmf_vec_fold_body(job->out, job->a, job->s, begin, end, &P_));
}

void pmf_vec_fold(PMFVector out, const PMFVector a, const uint64_t *r)
{
    PMFVecJob job = {.range = pmf_vec_fold_range, .params = a->params, .out = out, .a = a, .s = r};
    pmf_vec_run(&job, out->allocated_n, PMF_VEC_PRODUCT_GRAIN);
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

// The pairs [begin, end) go through the lanes, and the pairs from `from` on,
// which no whole group covers and whose lanes would reach the padding,
// through the element kernels.
PMF_INLINE void pmf_vec_fold_inner_products_body(uint64_t *const *sums, const PMFVector a,
                                                 const PMFVector b, uint64_t begin, uint64_t end,
                                                 uint64_t from, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], X[PMF_MAX_LIMBS], Y[PMF_MAX_LIMBS];
    PMFInnerSums S;

    pmf_vec_inner_sums_init(&S, L);
    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_pairs(U, V, a, i, L);
        pmf_vec_load_pairs(X, Y, b, i, L);
        pmf_vec_inner_sums_add(&S, U, V, X, Y, P);
    }
    pmf_vec_inner_sums_finish(sums, &S, a, b, from, P);
}

// The last chunk also takes the pairs past the last whole group.
static void pmf_vec_fold_inner_products_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                              uint64_t chunk)
{
    uint64_t *const sums[3] = {pmf_vec_partial(job, chunk, 0), pmf_vec_partial(job, chunk, 1),
                               pmf_vec_partial(job, chunk, 2)};
    const uint64_t from = end == job->total ? end : job->a->n / 2;
    PMF_WITH_LIMBS(job->params,
                   pmf_vec_fold_inner_products_body(sums, job->a, job->b, begin, end, from, &P_));
}

void pmf_vec_fold_inner_products(uint64_t *q0, uint64_t *q1, uint64_t *q2, const PMFVector a,
                                 const PMFVector b)
{
    assert(a->n % 2 == 0 && b->n == a->n);
    uint64_t *const sums[3] = {q0, q1, q2};
    const uint64_t whole = (a->n / 2 / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    PMFVecJob job = {
        .range = pmf_vec_fold_inner_products_range, .params = a->params, .a = a, .b = b};
    if (pmf_vec_split(&job, whole, PMF_VEC_PRODUCT_GRAIN))
        pmf_vec_reduce_chunks(&job, sums, 3);
    else
        PMF_WITH_LIMBS(a->params,
                       pmf_vec_fold_inner_products_body(sums, a, b, 0, whole, whole, &P_));
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
//
// [begin, end) is the range of j, the pairs of the folded vector the main loop
// covers; `last` adds the rest: the output groups past them and the pairs among
// those.
PMF_INLINE void pmf_vec_fold_and_inner_products_body(PMFVector a_out, PMFVector b_out,
                                                     uint64_t *const *sums, const PMFVector a,
                                                     const PMFVector b, const uint64_t *r,
                                                     uint64_t begin, uint64_t end, int last,
                                                     PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t R[PMF_MAX_LIMBS];
    pv_t A0[PMF_MAX_LIMBS + 1], A1[PMF_MAX_LIMBS + 1], B0[PMF_MAX_LIMBS + 1], B1[PMF_MAX_LIMBS + 1];
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], X[PMF_MAX_LIMBS], Y[PMF_MAX_LIMBS];
    PMFInnerSums S;

    pmf_vec_load_scalar(R, r, L);
    pmf_vec_inner_sums_init(&S, L);
    for (uint64_t j = begin; j < end; j += PMF_VEC_GROUP)
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
    if (!last)
    {
        pmf_vec_inner_sums_finish(sums, &S, a_out, b_out, a_out->n / 2, P);
        return;
    }
    for (uint64_t i = 2 * end; i < a_out->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_fold_group(A0, a, i, R, P);
        pmf_vec_fold_group(B0, b, i, R, P);
        pmf_vec_store_group(a_out, i, A0, L);
        pmf_vec_store_group(b_out, i, B0, L);
    }
    pmf_vec_inner_sums_finish(sums, &S, a_out, b_out, end, P);
}

static void pmf_vec_fold_and_inner_products_range(const PMFVecJob *job, uint64_t begin,
                                                  uint64_t end, uint64_t chunk)
{
    uint64_t *const sums[3] = {pmf_vec_partial(job, chunk, 0), pmf_vec_partial(job, chunk, 1),
                               pmf_vec_partial(job, chunk, 2)};
    PMF_WITH_LIMBS(job->params, pmf_vec_fold_and_inner_products_body(job->out, job->out2, sums,
                                                                     job->a, job->b, job->s, begin,
                                                                     end, end == job->total, &P_));
}

void pmf_vec_fold_and_inner_products(PMFVector a_out, PMFVector b_out, uint64_t *q0, uint64_t *q1,
                                     uint64_t *q2, const PMFVector a, const PMFVector b,
                                     const uint64_t *r)
{
    assert(a->n % 4 == 0 && b->n == a->n && 2 * a_out->n == a->n && 2 * b_out->n == a->n);
    uint64_t *const sums[3] = {q0, q1, q2};
    const uint64_t whole = (a_out->n / 2 / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    PMFVecJob job = {.range = pmf_vec_fold_and_inner_products_range,
                     .params = a->params,
                     .out = a_out,
                     .out2 = b_out,
                     .a = a,
                     .b = b,
                     .s = r};
    if (pmf_vec_split(&job, whole, PMF_VEC_PRODUCT_GRAIN / 2))
        pmf_vec_reduce_chunks(&job, sums, 3);
    else
        PMF_WITH_LIMBS(a->params, pmf_vec_fold_and_inner_products_body(a_out, b_out, sums, a, b, r,
                                                                       0, whole, 1, &P_));
    pmf_vec_inner_sums_extrapolate(q0, q1, q2, a->params);
}

// --- weighted sums of products of folds -----------------------------------
//
// Every table's pairs (u, v) are folded at the nodes once per group, f(t) = u +
// t (v - u) as f(t - 1) + (v - u) from t = 2 on, and the products and sums of
// the shape are formed from those folds.

// The shape word array of pmf_vec_fold_product_sums, decoded. The products of
// sum s are first_product[s] .. first_product[s + 1] - 1.
typedef struct
{
    uint64_t sums;
    uint64_t nodes;
    uint64_t first_product[PMF_FOLD_MAX_SUMS + 1];
    uint64_t negated[PMF_FOLD_MAX_PRODUCTS];
    uint64_t factor_count[PMF_FOLD_MAX_PRODUCTS];
    uint64_t factors[PMF_FOLD_MAX_PRODUCTS][PMF_FOLD_MAX_FACTORS];
} PMFProductShape;

static void pmf_vec_product_shape(PMFProductShape *shape, const uint64_t *words, uint64_t tables,
                                  uint64_t nodes)
{
    assert(tables >= 1 && tables <= PMF_FOLD_MAX_TABLES);
    assert(nodes >= 1 && nodes <= PMF_FOLD_MAX_NODES);
    shape->nodes = nodes;
    shape->sums = *words++;
    assert(shape->sums >= 1 && shape->sums <= PMF_FOLD_MAX_SUMS);
    uint64_t p = 0;
    for (uint64_t s = 0; s < shape->sums; s++)
    {
        shape->first_product[s] = p;
        const uint64_t products = *words++;
        assert(products >= 1 && p + products <= PMF_FOLD_MAX_PRODUCTS);
        for (uint64_t end = p + products; p < end; p++)
        {
            shape->negated[p] = *words++;
            shape->factor_count[p] = *words++;
            assert(shape->negated[p] <= 1);
            assert(shape->factor_count[p] >= 1 && shape->factor_count[p] <= PMF_FOLD_MAX_FACTORS);
            for (uint64_t f = 0; f < shape->factor_count[p]; f++)
            {
                shape->factors[p][f] = *words++;
                assert(shape->factors[p][f] < tables);
            }
        }
    }
    shape->first_product[shape->sums] = p;
}

// The folds of one group of pairs (U, V) at the shape's nodes.
PMF_INLINE void pmf_vec_fold_nodes(pv_t (*F)[PMF_MAX_LIMBS], const pv_t *U, const pv_t *V,
                                   uint64_t nodes, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t D[PMF_MAX_LIMBS + 1], T[PMF_MAX_LIMBS + 1];

    for (uint64_t k = 0; k < L; k++)
        F[0][k] = U[k];
    if (nodes > 1)
        for (uint64_t k = 0; k < L; k++)
            F[1][k] = V[k];
    if (nodes > 2)
        pmf_vec_sub_group(D, V, U, P);
    for (uint64_t t = 2; t < nodes; t++)
    {
        pmf_vec_add_group(T, F[t - 1], D, P);
        for (uint64_t k = 0; k < L; k++)
            F[t][k] = T[k];
    }
}

// Lanes `live` and up of X set to zero, the rest kept.
PMF_INLINE void pmf_vec_keep_lanes(pv_t *X, uint64_t live, uint64_t L)
{
    _Alignas(64) uint64_t flags[PMF_LANES];
    for (uint64_t lane = 0; lane < PMF_LANES; lane++)
        flags[lane] = lane < live;
    const pv_t keep = pv_mask_from_flag(pv_load(flags));
    for (uint64_t k = 0; k < L; k++)
        X[k] = pv_and(X[k], keep);
}

// The sums at every node, lane by lane, with limbs added unreduced in between.
typedef struct
{
    pv_t acc[PMF_FOLD_MAX_SUMS][PMF_FOLD_MAX_NODES][PMF_MAX_LIMBS + 1];
    uint64_t since_reduce;
} PMFProductSums;

PMF_INLINE void pmf_vec_product_sums_init(PMFProductSums *S, const PMFProductShape *shape,
                                          uint64_t L)
{
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
            for (uint64_t k = 0; k <= L; k++)
                S->acc[s][t][k] = pv_zero();
    S->since_reduce = 0;
}

// One group of pairs into the sums, from the folds F of every table and the
// group's weights W (NULL for weights of one). Each sum adds one canonical
// element a group to its accumulator, so 255 groups stay below 2^60, the most
// pmf_vec_reduce_wide accepts; the sums are reduced that often.
PMF_INLINE void pmf_vec_product_sums_add(PMFProductSums *S,
                                         pv_t (*F)[PMF_FOLD_MAX_NODES][PMF_MAX_LIMBS],
                                         const pv_t *W, const PMFProductShape *shape, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t G[PMF_MAX_LIMBS + 1], T[PMF_MAX_LIMBS + 1], Z[PMF_MAX_LIMBS];

    for (uint64_t k = 0; k < L; k++)
        Z[k] = pv_zero();
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            for (uint64_t p = shape->first_product[s]; p < shape->first_product[s + 1]; p++)
            {
                const uint64_t *factors = shape->factors[p];
                for (uint64_t k = 0; k < L; k++)
                    T[k] = F[factors[0]][t][k];
                for (uint64_t f = 1; f < shape->factor_count[p]; f++)
                    pmf_vec_mul_group(T, T, F[factors[f]][t], P);
                const pv_t *base = p == shape->first_product[s] ? Z : G;
                if (shape->negated[p])
                    pmf_vec_sub_group(G, base, T, P);
                else
                    pmf_vec_add_group(G, base, T, P);
            }
            if (W)
                pmf_vec_mul_group(G, G, W, P);
            pmf_vec_accumulate_group(S->acc[s][t], G, L);
        }
    if (++S->since_reduce == 255)
    {
        for (uint64_t s = 0; s < shape->sums; s++)
            for (uint64_t t = 0; t < shape->nodes; t++)
            {
                S->acc[s][t][L] = pv_zero();
                pmf_vec_reduce_wide(S->acc[s][t], P);
            }
        S->since_reduce = 0;
    }
}

// The lanes of each sum added up into the element sums[s * nodes + t].
PMF_INLINE void pmf_vec_product_sums_finish(uint64_t *const *sums, PMFProductSums *S,
                                            const PMFProductShape *shape, PMFParams P)
{
    const uint64_t L = P->limbs;
    _Alignas(64) uint64_t partial[PMF_LANES * PMF_MAX_LIMBS];
    uint64_t element[PMF_LANES] = {0};

    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            uint64_t *out = sums[s * shape->nodes + t];
            for (uint64_t k = 0; k < PMF_LANES; k++)
                out[k] = 0;
            S->acc[s][t][L] = pv_zero();
            pmf_vec_reduce_wide(S->acc[s][t], P);
            for (uint64_t k = 0; k < L; k++)
                pv_store(partial + k * PMF_LANES, S->acc[s][t][k]);
            for (uint64_t lane = 0; lane < PMF_VEC_GROUP; lane++)
            {
                for (uint64_t k = 0; k < L; k++)
                    element[k] = partial[k * PMF_LANES + lane];
                pmf_add(out, out, element, P);
            }
        }
}

// The pairs of every table from `i` on into the sums, with the lanes at or
// past `pairs` zeroed: a product with a zero factor adds nothing, whatever
// its weight.
PMF_INLINE void pmf_vec_product_sums_load_add(PMFProductSums *S,
                                              pv_t (*F)[PMF_FOLD_MAX_NODES][PMF_MAX_LIMBS],
                                              const PMFVector *tables, uint64_t count,
                                              const PMFVector weights, uint64_t i, uint64_t pairs,
                                              const PMFProductShape *shape, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS];

    for (uint64_t j = 0; j < count; j++)
    {
        pmf_vec_load_pairs(U, V, tables[j], i, L);
        if (pairs - i < PMF_VEC_GROUP)
        {
            pmf_vec_keep_lanes(U, pairs - i, L);
            pmf_vec_keep_lanes(V, pairs - i, L);
        }
        pmf_vec_fold_nodes(F[j], U, V, shape->nodes, P);
    }
    if (weights)
        pmf_vec_load_group(W, weights, i, L);
    pmf_vec_product_sums_add(S, F, weights ? W : NULL, shape, P);
}

// The pairs [begin, end) of the tables, `begin` a multiple of PMF_VEC_GROUP.
PMF_INLINE void pmf_vec_fold_product_sums_body(uint64_t *const *sums, const PMFVector *tables,
                                               uint64_t count, const PMFVector weights,
                                               const PMFProductShape *shape, uint64_t begin,
                                               uint64_t end, PMFParams P)
{
    pv_t F[PMF_FOLD_MAX_TABLES][PMF_FOLD_MAX_NODES][PMF_MAX_LIMBS];
    PMFProductSums S;

    pmf_vec_product_sums_init(&S, shape, P->limbs);
    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
        pmf_vec_product_sums_load_add(&S, F, tables, count, weights, i, end, shape, P);
    pmf_vec_product_sums_finish(sums, &S, shape, P);
}

// Pointers to the sums * nodes results of one chunk.
static void pmf_vec_product_partials(uint64_t **sums, const PMFVecJob *job, uint64_t chunk)
{
    for (uint64_t s = 0; s < job->slots; s++)
        sums[s] = pmf_vec_partial(job, chunk, s);
}

static void pmf_vec_fold_product_sums_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                            uint64_t chunk)
{
    uint64_t *sums[PMF_FOLD_MAX_SUMS * PMF_FOLD_MAX_NODES];
    pmf_vec_product_partials(sums, job, chunk);
    PMF_WITH_LIMBS(job->params, pmf_vec_fold_product_sums_body(
                                    sums, job->tables, job->count, job->b,
                                    (const PMFProductShape *)job->shape, begin, end, &P_));
}

void pmf_vec_fold_product_sums(uint64_t *const *sums, const PMFVector *tables, uint64_t count,
                               const uint64_t *shape, uint64_t nodes, const PMFVector weights)
{
    PMFProductShape decoded;
    pmf_vec_product_shape(&decoded, shape, count, nodes);
    for (uint64_t j = 0; j < count; j++)
        assert(tables[j]->n % 2 == 0 && tables[j]->n == tables[0]->n);
    assert(!weights || 2 * weights->n == tables[0]->n);
    const uint64_t pairs = tables[0]->n / 2;
    PMFVecJob job = {.range = pmf_vec_fold_product_sums_range,
                     .params = tables[0]->params,
                     .b = weights,
                     .tables = tables,
                     .count = count,
                     .shape = &decoded};
    if (pmf_vec_split(&job, pairs, PMF_VEC_PRODUCT_GRAIN / count))
        pmf_vec_reduce_chunks(&job, sums, decoded.sums * decoded.nodes);
    else
        PMF_WITH_LIMBS(
            tables[0]->params,
            pmf_vec_fold_product_sums_body(sums, tables, count, weights, &decoded, 0, pairs, &P_));
}

// As pmf_vec_fold_and_inner_products: each pass of the main loop folds two
// output groups of every table, stores them, and takes the pairs of the next
// round from the registers. The remaining output groups, padding included, are
// folded alone, and the pairs among them summed from the stored folds.
//
// [begin, end) is the range of j, as in pmf_vec_fold_and_inner_products_body,
// and `last` adds the rest.
PMF_INLINE void pmf_vec_fold_and_product_sums_body(PMFVector *outs, uint64_t *const *sums,
                                                   const PMFVector *tables, uint64_t count,
                                                   const uint64_t *r, const PMFVector weights,
                                                   const PMFProductShape *shape, uint64_t begin,
                                                   uint64_t end, int last, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t R[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS];
    pv_t A0[PMF_MAX_LIMBS + 1], A1[PMF_MAX_LIMBS + 1];
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS];
    pv_t F[PMF_FOLD_MAX_TABLES][PMF_FOLD_MAX_NODES][PMF_MAX_LIMBS];
    PMFProductSums S;

    pmf_vec_load_scalar(R, r, L);
    pmf_vec_product_sums_init(&S, shape, L);
    const uint64_t pairs = outs[0]->n / 2;
    for (uint64_t j = begin; j < end; j += PMF_VEC_GROUP)
    {
        const uint64_t i = 2 * j;
        for (uint64_t k = 0; k < count; k++)
        {
            pmf_vec_fold_group(A0, tables[k], i, R, P);
            pmf_vec_fold_group(A1, tables[k], i + PMF_VEC_GROUP, R, P);
            pmf_vec_store_group(outs[k], i, A0, L);
            pmf_vec_store_group(outs[k], i + PMF_VEC_GROUP, A1, L);
            pmf_vec_pairs_of(U, V, A0, A1, L);
            pmf_vec_fold_nodes(F[k], U, V, shape->nodes, P);
        }
        if (weights)
            pmf_vec_load_group(W, weights, j, L);
        pmf_vec_product_sums_add(&S, F, weights ? W : NULL, shape, P);
    }
    if (last)
    {
        for (uint64_t i = 2 * end; i < outs[0]->allocated_n; i += PMF_VEC_GROUP)
            for (uint64_t k = 0; k < count; k++)
            {
                pmf_vec_fold_group(A0, tables[k], i, R, P);
                pmf_vec_store_group(outs[k], i, A0, L);
            }
        if (end < pairs)
            pmf_vec_product_sums_load_add(&S, F, (const PMFVector *)outs, count, weights, end,
                                          pairs, shape, P);
    }
    pmf_vec_product_sums_finish(sums, &S, shape, P);
}

static void pmf_vec_fold_and_product_sums_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                                uint64_t chunk)
{
    uint64_t *sums[PMF_FOLD_MAX_SUMS * PMF_FOLD_MAX_NODES];
    pmf_vec_product_partials(sums, job, chunk);
    PMF_WITH_LIMBS(job->params,
                   pmf_vec_fold_and_product_sums_body(
                       job->outs, sums, job->tables, job->count, job->s, job->b,
                       (const PMFProductShape *)job->shape, begin, end, end == job->total, &P_));
}

void pmf_vec_fold_and_product_sums(PMFVector *outs, uint64_t *const *sums, const PMFVector *tables,
                                   uint64_t count, const uint64_t *r, const uint64_t *shape,
                                   uint64_t nodes, const PMFVector weights)
{
    PMFProductShape decoded;
    pmf_vec_product_shape(&decoded, shape, count, nodes);
    for (uint64_t j = 0; j < count; j++)
        assert(tables[j]->n % 4 == 0 && tables[j]->n == tables[0]->n &&
               2 * outs[j]->n == tables[0]->n);
    assert(!weights || 4 * weights->n == tables[0]->n);
    const uint64_t whole = (outs[0]->n / 2 / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    PMFVecJob job = {.range = pmf_vec_fold_and_product_sums_range,
                     .params = tables[0]->params,
                     .b = weights,
                     .s = r,
                     .outs = outs,
                     .tables = tables,
                     .count = count,
                     .shape = &decoded};
    if (pmf_vec_split(&job, whole, PMF_VEC_PRODUCT_GRAIN / (2 * count)))
        pmf_vec_reduce_chunks(&job, sums, decoded.sums * decoded.nodes);
    else
        PMF_WITH_LIMBS(tables[0]->params,
                       pmf_vec_fold_and_product_sums_body(outs, sums, tables, count, r, weights,
                                                          &decoded, 0, whole, 1, &P_));
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
                                          const uint64_t *half, uint64_t begin, uint64_t end,
                                          PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS], R[PMF_MAX_LIMBS], H[PMF_MAX_LIMBS];
    pv_t C[PMF_MAX_LIMBS + 1], D[PMF_MAX_LIMBS + 1], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(R, r, L);
    pmf_vec_load_scalar(H, half, L);
    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
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

// job->a is the word, job->t the untwisting table, job->s and job->s2 are r
// and 1/2.
static void pmf_vec_fold_twisted_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                       uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params, pmf_vec_fold_twisted_body(job->out, job->a, job->t, job->s, job->s2,
                                                          begin, end, &P_));
}

void pmf_vec_fold_twisted(PMFVector out, const PMFVector word, const PMFVector untwist,
                          const uint64_t *r)
{
    uint64_t half[PMF_LANES];
    assert(word->n == 2 * out->n && untwist->n == out->n);
    pmf_vec_half(half, word->params);
    PMFVecJob job = {.range = pmf_vec_fold_twisted_range,
                     .params = word->params,
                     .out = out,
                     .a = word,
                     .t = untwist,
                     .s = r,
                     .s2 = half};
    pmf_vec_run(&job, out->allocated_n, PMF_VEC_PRODUCT_GRAIN);
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
                                          const PMFVector t, uint64_t begin, uint64_t end,
                                          PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t A[PMF_MAX_LIMBS], B[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS];
    pv_t M[PMF_MAX_LIMBS + 1], E[PMF_MAX_LIMBS + 1], O[PMF_MAX_LIMBS + 1];

    for (uint64_t i = begin; i < end; i += PMF_VEC_GROUP)
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

static void pmf_vec_lift_twisted_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                       uint64_t chunk)
{
    (void)chunk;
    PMF_WITH_LIMBS(job->params,
                   pmf_vec_lift_twisted_body(job->out, job->a, job->b, job->t, begin, end, &P_));
}

void pmf_vec_lift_twisted(PMFVector out, const PMFVector a, const PMFVector b, const PMFVector t)
{
    assert(b->n == a->n && out->n == 2 * a->n && t->n > 0 && a->n % t->n == 0);
    PMFVecJob job = {.range = pmf_vec_lift_twisted_range,
                     .params = a->params,
                     .out = out,
                     .a = a,
                     .b = b,
                     .t = t};
    pmf_vec_run(&job, a->allocated_n, PMF_VEC_PRODUCT_GRAIN);
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

// Values [begin, end) of pmf_vec_set_uint64, job->s and job->s2 being `low`
// and `high`.
static void pmf_vec_set_uint64_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                     uint64_t chunk)
{
    (void)chunk;
    const uint64_t *low = job->s + begin, *high = job->s2 ? job->s2 + begin : NULL;
    const uint64_t shift = job->shift, mask = job->mask, count = end - begin;
    const uint64_t at = job->start + begin;
    uint64_t *limb0 = job->out->limbs[0] + at, *limb1 = job->out->limbs[1] + at;
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
    for (uint64_t k = 2; k < job->params->limbs; k++)
        memset(job->out->limbs[k] + at, 0, count * sizeof(uint64_t));
}

void pmf_vec_set_uint64(PMFVector out, uint64_t start, const uint64_t *low, const uint64_t *high,
                        uint64_t count, uint32_t shift, uint32_t width)
{
    assert(start + count <= out->n && shift < (high ? 128u : 64u) && width >= 1 && width <= 64);
    PMFVecJob job = {.range = pmf_vec_set_uint64_range,
                     .params = out->params,
                     .out = out,
                     .s = low,
                     .s2 = high,
                     .start = start,
                     .shift = shift,
                     .mask = width == 64 ? ~0ULL : (1ULL << width) - 1};
    pmf_vec_run(&job, count, pmf_vec_move_grain(out->params));
}

// Columns [begin, end) of pmf_vec_set_radix_decomposition, job->s and job->s2 being `low`
// and `high`, job->rows and job->width the matrix's shape, job->count the
// digits and job->shift the digit width. Eight columns at a time, so that the
// reads of a row are one run and the writes of a column stay sequential.
static void pmf_vec_set_radix_decomposition_range(const PMFVecJob *job, uint64_t begin,
                                                  uint64_t end, uint64_t chunk)
{
    (void)chunk;
    const uint64_t rows = job->rows, columns = job->width, digits = job->count;
    const uint64_t block = rows * columns, width = job->shift;
    const uint64_t mask = width == 64 ? ~0ULL : (1ULL << width) - 1;
    const uint64_t *low = job->s, *high = job->s2;
    uint64_t *limb0 = job->out->limbs[0], *limb1 = job->out->limbs[1];

    for (uint64_t k0 = begin; k0 < end; k0 += 8)
    {
        const uint64_t k1 = end - k0 < 8 ? end : k0 + 8;
        for (uint64_t i = 0; i < rows; i++)
            for (uint64_t k = k0; k < k1; k++)
            {
                const uint64_t at = i * columns + k;
                unsigned __int128 v = (unsigned __int128)(high ? high[at] : 0) << 64 | low[at];
                uint64_t position = k * rows + i;
                for (uint64_t j = 0; j < digits; j++, position += block)
                {
                    pmf_vec_set_limbs(limb0, limb1, position, (uint64_t)v & mask);
                    v >>= width;
                }
            }
    }
    for (uint64_t k = 2; k < job->params->limbs; k++)
        for (uint64_t j = 0; j < digits; j++)
            memset(job->out->limbs[k] + j * block + begin * rows, 0,
                   (end - begin) * rows * sizeof(uint64_t));
}

void pmf_vec_set_radix_decomposition(PMFVector out, const uint64_t *low, const uint64_t *high,
                                     uint64_t rows, uint64_t columns, uint32_t width,
                                     uint64_t digits)
{
    assert(digits * rows * columns <= out->n && width >= 1 && width <= 64);
    PMFVecJob job = {.range = pmf_vec_set_radix_decomposition_range,
                     .params = out->params,
                     .out = out,
                     .s = low,
                     .s2 = high,
                     .rows = rows,
                     .width = columns,
                     .count = digits,
                     .shift = width};
    const uint64_t per_column = digits * rows > 0 ? digits * rows : 1;
    pmf_vec_run(&job, columns, pmf_vec_move_grain(out->params) / per_column);
}

// Limbs are 52 bits, so a canonical element is below 2^64 exactly when its
// limbs from the third on are zero and the second is below 2^12. Elements
// [begin, end); false at the first that is not.
static bool pmf_vec_get_uint64_span(uint64_t *out, const PMFVector a, uint64_t begin, uint64_t end)
{
    const uint64_t *low = a->limbs[0], *high = a->limbs[1];
    for (uint64_t i = begin; i < end; i++)
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

// job->partials holds one flag per chunk: whether its elements fit.
static void pmf_vec_get_uint64_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                     uint64_t chunk)
{
    job->partials[chunk] = pmf_vec_get_uint64_span(job->words, job->a, begin, end);
}

bool pmf_vec_get_uint64(uint64_t *out, const PMFVector a)
{
    PMFVecJob job = {.range = pmf_vec_get_uint64_range, .params = a->params, .a = a, .words = out};
    if (!pmf_vec_split(&job, a->n, pmf_vec_move_grain(a->params)))
        return pmf_vec_get_uint64_span(out, a, 0, a->n);
    job.partials = (uint64_t *)safe_malloc(job.chunks.count * sizeof(uint64_t));
    pmf_vec_run_chunks(&job);
    bool fits = true;
    for (uint64_t c = 0; c < job.chunks.count; c++)
        fits = fits && job.partials[c];
    free(job.partials);
    return fits;
}

static void pmf_vec_copy_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)
{
    (void)chunk;
    for (uint64_t k = 0; k < job->params->limbs; k++)
        memcpy(job->out->limbs[k] + begin, job->a->limbs[k] + begin,
               (end - begin) * sizeof(uint64_t));
}

void pmf_vec_copy(PMFVector out, const PMFVector a)
{
    PMFVecJob job = {.range = pmf_vec_copy_range, .params = a->params, .out = out, .a = a};
    pmf_vec_run(&job, a->allocated_n, pmf_vec_move_grain(a->params));
}

// Elements [begin, end) of the two halves; job->out and job->out2 are the even
// and odd ones.
static void pmf_vec_split_even_odd_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                         uint64_t chunk)
{
    (void)chunk;
    for (uint64_t k = 0; k < job->params->limbs; k++)
    {
        const uint64_t *plane = job->a->limbs[k];
        uint64_t *even = job->out->limbs[k], *odd = job->out2->limbs[k];
        for (uint64_t i = begin; i < end; i++)
        {
            even[i] = plane[2 * i];
            odd[i] = plane[2 * i + 1];
        }
    }
}

void pmf_vec_split_even_odd(PMFVector even, PMFVector odd, const PMFVector a)
{
    PMFVecJob job = {.range = pmf_vec_split_even_odd_range,
                     .params = a->params,
                     .out = even,
                     .out2 = odd,
                     .a = a};
    pmf_vec_run(&job, a->n / 2, pmf_vec_move_grain(a->params) / 2);
}

// Elements [begin, end) of each half; job->a and job->b are the even and odd
// ones.
static void pmf_vec_interleave_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                     uint64_t chunk)
{
    (void)chunk;
    for (uint64_t k = 0; k < job->params->limbs; k++)
        vec_interleave_u64(job->out->limbs[k] + 2 * begin, job->a->limbs[k] + begin,
                           job->b->limbs[k] + begin, end - begin);
}

void pmf_vec_interleave(PMFVector out, const PMFVector even, const PMFVector odd)
{
    PMFVecJob job = {
        .range = pmf_vec_interleave_range, .params = out->params, .out = out, .a = even, .b = odd};
    pmf_vec_run(&job, even->n, pmf_vec_move_grain(out->params) / 2);
}

// Elements [begin, end) of the output, from whichever parts they lie in.
static void pmf_vec_concat_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)
{
    (void)chunk;
    uint64_t at = 0;
    for (uint64_t p = 0; p < job->count && at < end; p++)
    {
        const PMFVector part = job->tables[p];
        const uint64_t from = begin > at ? begin : at;
        const uint64_t to = at + part->n < end ? at + part->n : end;
        if (from < to)
            for (uint64_t k = 0; k < job->params->limbs; k++)
                memcpy(job->out->limbs[k] + from, part->limbs[k] + (from - at),
                       (to - from) * sizeof(uint64_t));
        at += part->n;
    }
}

void pmf_vec_concat(PMFVector out, const PMFVector *parts, uint64_t count)
{
    uint64_t total = 0;
    for (uint64_t p = 0; p < count; p++)
        total += parts[p]->n;
    PMFVecJob job = {.range = pmf_vec_concat_range,
                     .params = out->params,
                     .out = out,
                     .tables = parts,
                     .count = count};
    pmf_vec_run(&job, total, pmf_vec_move_grain(out->params));
}

// job->s holds the indices; job->partials one flag per chunk, whether all of
// its indices are in range.
static void pmf_vec_gather_check_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                       uint64_t chunk)
{
    uint64_t valid = 1;
    for (uint64_t i = begin; i < end; i++)
        valid &= job->s[i] < job->a->n;
    job->partials[chunk] = valid;
}

static void pmf_vec_gather_range(const PMFVecJob *job, uint64_t begin, uint64_t end, uint64_t chunk)
{
    (void)chunk;
    const uint64_t *indices = job->s;
    for (uint64_t k = 0; k < job->params->limbs; k++)
    {
        uint64_t *to = job->out->limbs[k];
        const uint64_t *from = job->a->limbs[k];
        for (uint64_t i = begin; i < end; i++)
            to[i] = from[indices[i]];
    }
}

bool pmf_vec_gather(PMFVector out, const PMFVector a, const uint64_t *indices, uint64_t count)
{
    PMFVecJob job = {
        .range = pmf_vec_gather_check_range, .params = a->params, .out = out, .a = a, .s = indices};
    if (!pmf_vec_split(&job, count, pmf_vec_move_grain(a->params)))
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
    // Every index is checked, by all the chunks, before any is used.
    job.partials = (uint64_t *)safe_malloc(job.chunks.count * sizeof(uint64_t));
    pmf_vec_run_chunks(&job);
    bool valid = true;
    for (uint64_t c = 0; c < job.chunks.count; c++)
        valid = valid && job.partials[c];
    free(job.partials);
    if (valid)
    {
        job.range = pmf_vec_gather_range;
        pmf_vec_run_chunks(&job);
    }
    return valid;
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

// Digests [begin, end) of pmf_vec_hash_elements, into job->bytes. A window
// past a BLAKE3 chunk, which hash_batch cannot take, gets one hasher each.
// Otherwise windows are staged because a digest covers canonical encodings,
// which nothing holds, and batched by the same byte budget as a span; see
// field_vector.c, which sizes its own staging the same way and for the same
// reasons. hash_batch_fits caps a window at a chunk, so a batch is at least
// 32.
static void pmf_vec_hash_elements_range(const PMFVecJob *job, uint64_t begin, uint64_t end,
                                        uint64_t chunk)
{
    (void)chunk;
    const PMFVector a = job->a;
    const uint64_t group = job->group, stride = job->stride;
    const uint64_t nbytes = a->params->nbytes, leaf = group * nbytes;
    uint8_t *out = job->bytes;

    if (!hash_batch_fits(leaf))
    {
        for (uint64_t k = begin; k < end; k++)
            pmf_vec_hash_span(&out[k * BLAKE3_OUT_LEN], a, k * stride, group);
        return;
    }

    const uint64_t batch = PMF_VEC_HASH_STAGE_BYTES / leaf;
    uint8_t *staged = (uint8_t *)safe_aligned_malloc(batch * leaf);
    for (uint64_t k = begin; k < end;)
    {
        uint64_t run = end - k;
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

void pmf_vec_hash_elements(uint8_t *out, const PMFVector a, uint64_t group, uint64_t stride)
{
    const uint64_t count = pmf_vec_hash_count(a, group, stride);
    if (count == 0)
        return;
    // The width is the *encoded* one, nbytes per element, not a machine word.
    const uint64_t leaf = group * a->params->nbytes;
    PMFVecJob job = {.range = pmf_vec_hash_elements_range,
                     .params = a->params,
                     .a = a,
                     .group = group,
                     .stride = stride,
                     .bytes = out};
    pmf_vec_run(&job, count, PMF_VEC_HASH_GRAIN_BYTES / leaf);
}
