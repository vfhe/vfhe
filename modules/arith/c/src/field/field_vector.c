// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// Vectors of extension-field elements, stored as d coefficient planes.
//
// The plane layout is the whole point: with the coefficients of position j of
// every element contiguous, the arithmetic here is a fixed number of
// mod_eltwise_* calls -- the engine-tuned kernels -- over length-allocated_n
// runs, independent of how many elements the vector holds. The contract those
// planes satisfy is stated at the declaration in arith.h.
#include <arith.h>
#include <blake3.h>
#include <string.h>
#include "arith_internal.h"
#include "alloc.h"
#include "../vec_chunks.h"
#include <crypto.h>

uint64_t field_vec_padded_length(uint64_t n)
{
    const uint64_t unit = MOD_MIN_VECTOR_LEN;
    return ((n + unit - 1) / unit) * unit;
}

void field_vec_clear_padding(FieldVector a)
{
    if (a->allocated_n <= a->n)
        return;
    const size_t tail = (size_t)(a->allocated_n - a->n) * sizeof(uint64_t);
    for (uint64_t j = 0; j < a->d; j++)
        memset(a->coeffs[j] + a->n, 0, tail);
}

// --- running a kernel on the library's threads -----------------------------
//
// An elementwise kernel is a loop over a range of positions; a windowed one a
// loop over windows. On one thread it runs once over the whole range;
// otherwise once per chunk (vec_chunks.h), each chunk a multiple of
// MOD_MIN_VECTOR_LEN positions (or of the window), so every plane offset stays
// 64-byte aligned and the eltwise kernels keep their no-tail contract.

// The fewest positions a chunk covers, by what a position costs.
// One addition or subtraction a coefficient, bounded by memory traffic.
#define FIELD_VEC_ADD_GRAIN (1u << 15)
// An extension product or more a position.
#define FIELD_VEC_PRODUCT_GRAIN (1u << 13)
// Hashing, in bytes of elements hashed: a gather and a compression per 64
// bytes.
#define FIELD_VEC_HASH_GRAIN_BYTES (256u << 10)

typedef struct FieldVecJob FieldVecJob;

// A kernel's operands, and its loop over [begin, end) of the range it splits.
// Which operand fields are used is up to the kernel.
struct FieldVecJob
{
    void (*range)(const FieldVecJob *job, uint64_t begin, uint64_t end);
    VecChunks chunks;
    FieldVector out, a, b, c, c2;
    const uint64_t *s, *s2;
    uint64_t window, period;
    uint64_t group, window_step, element_step;
    uint8_t *bytes;
};

static void field_vec_chunk(void *ctx, uint64_t chunk)
{
    const FieldVecJob *job = (const FieldVecJob *)ctx;
    job->range(job, vec_chunk_begin(&job->chunks, chunk), vec_chunk_end(&job->chunks, chunk));
}

// Plans the chunks of [0, total), a multiple of `unit` each; false when the
// kernel gets one thread.
static bool field_vec_split(FieldVecJob *job, uint64_t total, uint64_t grain, uint64_t unit)
{
    return vec_chunks_plan(&job->chunks, total, grain, unit);
}

static void field_vec_run_chunks(FieldVecJob *job)
{
    vfhe_parallel_for(job->chunks.count, 0, field_vec_chunk, job);
}

// job->range over [0, total), split across threads when they are allowed.
static void field_vec_run(FieldVecJob *job, uint64_t total, uint64_t grain, uint64_t unit)
{
    if (field_vec_split(job, total, grain, unit))
        field_vec_run_chunks(job);
    else
        job->range(job, 0, total);
}

// Positions [begin, end) of each operand of an elementwise kernel, as vectors
// of their own over `planes` (5 * d pointers): views[0..4] are out, a, b, c and
// c2, those that are NULL left unset.
static void field_vec_views(struct _FieldVector *views, uint64_t **planes, const FieldVecJob *job,
                            uint64_t begin, uint64_t end)
{
    const FieldVector operands[5] = {job->out, job->a, job->b, job->c, job->c2};
    for (uint64_t v = 0; v < 5; v++)
    {
        const FieldVector x = operands[v];
        if (x == NULL)
            continue;
        views[v] = *x;
        views[v].coeffs = planes + v * job->a->d;
        for (uint64_t j = 0; j < x->d; j++)
            views[v].coeffs[j] = x->coeffs[j] + begin;
        views[v].n = views[v].allocated_n = end - begin;
    }
}

// The plane pointers field_vec_views needs, from the stack up to
// FIELD_MAX_STACK_D.
#define FIELD_VEC_VIEW_PLANES(planes, d)                                                           \
    uint64_t *planes##_stack[5 * FIELD_MAX_STACK_D];                                               \
    uint64_t **planes = (d) <= FIELD_MAX_STACK_D                                                   \
                            ? planes##_stack                                                       \
                            : (uint64_t **)safe_malloc(5 * (d) * sizeof(uint64_t *))
#define FIELD_VEC_FREE_VIEW_PLANES(planes, d)                                                      \
    do                                                                                             \
    {                                                                                              \
        if ((d) > FIELD_MAX_STACK_D)                                                               \
            free(planes);                                                                          \
    } while (0)

static void field_vec_add_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
        mod_eltwise_add(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin,
                        job->b->coeffs[j] + begin, end - begin, job->a->mod);
}

void field_vec_add(FieldVector out, const FieldVector a, const FieldVector b)
{
    FieldVecJob job = {.range = field_vec_add_range, .out = out, .a = a, .b = b};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_ADD_GRAIN, MOD_MIN_VECTOR_LEN);
}

static void field_vec_sub_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
        mod_eltwise_sub(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin,
                        job->b->coeffs[j] + begin, end - begin, job->a->mod);
}

void field_vec_sub(FieldVector out, const FieldVector a, const FieldVector b)
{
    FieldVecJob job = {.range = field_vec_sub_range, .out = out, .a = a, .b = b};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_ADD_GRAIN, MOD_MIN_VECTOR_LEN);
}

static void field_vec_neg_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
        mod_eltwise_negate(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin, end - begin,
                           job->a->mod);
}

void field_vec_neg(FieldVector out, const FieldVector a)
{
    FieldVecJob job = {.range = field_vec_neg_range, .out = out, .a = a};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_ADD_GRAIN, MOD_MIN_VECTOR_LEN);
}

static void field_vec_add_scalar_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
        mod_eltwise_add_scalar(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin, job->s[j],
                               end - begin, job->a->mod);
}

void field_vec_add_scalar(FieldVector out, const FieldVector a, const uint64_t *s)
{
    FieldVecJob job = {.range = field_vec_add_scalar_range, .out = out, .a = a, .s = s};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_ADD_GRAIN, MOD_MIN_VECTOR_LEN);
}

static void field_vec_sub_scalar_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
        mod_eltwise_sub_scalar(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin, job->s[j],
                               end - begin, job->a->mod);
}

void field_vec_sub_scalar(FieldVector out, const FieldVector a, const uint64_t *s)
{
    FieldVecJob job = {.range = field_vec_sub_scalar_range, .out = out, .a = a, .s = s};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_ADD_GRAIN, MOD_MIN_VECTOR_LEN);
}

// s - a = -(a - s): the eltwise layer has no reversed-operand kernel.
static void field_vec_scalar_sub_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
    {
        uint64_t *out = job->out->coeffs[j] + begin;
        mod_eltwise_sub_scalar(out, job->a->coeffs[j] + begin, job->s[j], end - begin, job->a->mod);
        mod_eltwise_negate(out, out, end - begin, job->a->mod);
    }
}

void field_vec_scalar_sub(FieldVector out, const uint64_t *s, const FieldVector a)
{
    FieldVecJob job = {.range = field_vec_scalar_sub_range, .out = out, .a = a, .s = s};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_ADD_GRAIN, MOD_MIN_VECTOR_LEN);
}

// The schoolbook product of two degree-(d-1) polynomials, one plane at a time,
// followed by the fold on x^d == w. Same shape as field_ext_mul, with each
// coefficient-times-coefficient replaced by a whole-plane kernel call.
//
// Done over the whole vector at once the 2d^2 + d - 1 passes would each sweep a
// (2d-1)-plane accumulator, so beyond the last private cache every pass reloads
// it from memory and the rate is set by bandwidth rather than by the kernels.
// The product is therefore tiled: the same passes run over FIELD_VEC_MUL_TILE
// elements at a time out of one scratch that stays resident, which holds the
// in-cache rate at any length. The tile is a multiple of the eltwise vector
// width and the planes are 64-byte aligned, so every tile meets the kernels'
// length and alignment preconditions -- including the last, since allocated_n
// is itself a whole number of vector widths.
#define FIELD_VEC_MUL_TILE 4096

static void field_vec_mul_generic(FieldVector out, const FieldVector a, const void *b_or_scalar,
                                  int b_is_vector)
{
    const uint64_t d = a->d, len = a->allocated_n;
    const uint64_t wide = 2 * d - 1;
    const uint64_t tile = len < FIELD_VEC_MUL_TILE ? len : FIELD_VEC_MUL_TILE;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(wide * tile * sizeof(uint64_t));

    for (uint64_t base = 0; base < len; base += tile)
    {
        const uint64_t run = len - base < tile ? len - base : tile;
        for (uint64_t k = 0; k < wide; k++)
            memset(scratch + k * tile, 0, run * sizeof(uint64_t));

        for (uint64_t i = 0; i < d; i++)
        {
            for (uint64_t j = 0; j < d; j++)
            {
                uint64_t *acc = scratch + (i + j) * tile;
                if (b_is_vector)
                {
                    const FieldVector b = (const FieldVector)b_or_scalar;
                    mod_eltwise_mul_addto(acc, a->coeffs[i] + base, b->coeffs[j] + base, run,
                                          a->mod);
                }
                else
                {
                    const uint64_t *s = (const uint64_t *)b_or_scalar;
                    mod_eltwise_fma(acc, a->coeffs[i] + base, s[j], run, a->mod);
                }
            }
        }

        for (uint64_t i = wide; i-- > d;)
            mod_eltwise_fma(scratch + (i - d) * tile, scratch + i * tile, a->w, run, a->mod);

        // Copied only now, so `out` may alias either input: this tile's inputs
        // are read before it is overwritten, and later tiles are untouched.
        for (uint64_t j = 0; j < d; j++)
            memcpy(out->coeffs[j] + base, scratch + j * tile, run * sizeof(uint64_t));
    }
    free(scratch);
}

static void field_vec_mul_whole(FieldVector out, const FieldVector a, const FieldVector b)
{
    if (field_fused_applies(a->mod, a->d, a->allocated_n))
    {
        field_fused_mul(out->coeffs, a->coeffs, b->coeffs, a->allocated_n, a->d, a->w, a->mod);
        return;
    }
    field_vec_mul_generic(out, a, (const void *)b, 1);
}

static void field_vec_scale_whole(FieldVector out, const FieldVector a, const uint64_t *s)
{
    if (field_fused_applies(a->mod, a->d, a->allocated_n))
    {
        field_fused_scale(out->coeffs, a->coeffs, s, a->allocated_n, a->d, a->w, a->mod);
        return;
    }
    field_vec_mul_generic(out, a, (const void *)s, 0);
}

// The products, and the fused multiply-adds below, split by positions: each
// chunk runs the whole-vector kernel on views of its positions.
static void field_vec_mul_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    struct _FieldVector v[5];
    FIELD_VEC_VIEW_PLANES(planes, job->a->d);
    field_vec_views(v, planes, job, begin, end);
    field_vec_mul_whole(&v[0], &v[1], &v[2]);
    FIELD_VEC_FREE_VIEW_PLANES(planes, job->a->d);
}

void field_vec_mul(FieldVector out, const FieldVector a, const FieldVector b)
{
    FieldVecJob job = {.range = field_vec_mul_range, .out = out, .a = a, .b = b};
    if (field_vec_split(&job, a->allocated_n, FIELD_VEC_PRODUCT_GRAIN, MOD_MIN_VECTOR_LEN))
        field_vec_run_chunks(&job);
    else
        field_vec_mul_whole(out, a, b);
}

static void field_vec_scale_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    struct _FieldVector v[5];
    FIELD_VEC_VIEW_PLANES(planes, job->a->d);
    field_vec_views(v, planes, job, begin, end);
    field_vec_scale_whole(&v[0], &v[1], job->s);
    FIELD_VEC_FREE_VIEW_PLANES(planes, job->a->d);
}

void field_vec_scale(FieldVector out, const FieldVector a, const uint64_t *s)
{
    FieldVecJob job = {.range = field_vec_scale_range, .out = out, .a = a, .s = s};
    if (field_vec_split(&job, a->allocated_n, FIELD_VEC_PRODUCT_GRAIN, MOD_MIN_VECTOR_LEN))
        field_vec_run_chunks(&job);
    else
        field_vec_scale_whole(out, a, s);
}

// out = a + b * c, in one pass where the fused kernel applies. Everywhere else
// it is the product into a scratch vector and then the add -- the same answer,
// and the allocation this exists to avoid.
static void field_vec_fma_generic(FieldVector out, const FieldVector a, const FieldVector b,
                                  const void *c_or_scalar, int c_is_vector)
{
    const uint64_t d = a->d, len = a->allocated_n;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(d * len * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)malloc(d * sizeof(uint64_t *));
    struct _FieldVector prod = *out;

    prod.coeffs = planes;
    for (uint64_t j = 0; j < d; j++)
        planes[j] = scratch + j * len;

    if (c_is_vector)
        field_vec_mul(&prod, b, (const FieldVector)c_or_scalar);
    else
        field_vec_scale(&prod, b, (const uint64_t *)c_or_scalar);
    field_vec_add(out, a, &prod);

    free(planes);
    free(scratch);
}

static void field_vec_fma_whole(FieldVector out, const FieldVector a, const FieldVector b,
                                const FieldVector c)
{
    if (field_fused_applies(a->mod, a->d, a->allocated_n))
    {
        field_fused_fma(out->coeffs, a->coeffs, b->coeffs, c->coeffs, a->allocated_n, a->d, a->w,
                        a->mod);
        return;
    }
    field_vec_fma_generic(out, a, b, (const void *)c, 1);
}

static void field_vec_fma_scalar_whole(FieldVector out, const FieldVector a, const FieldVector b,
                                       const uint64_t *s)
{
    if (field_fused_applies(a->mod, a->d, a->allocated_n))
    {
        field_fused_fma_scalar(out->coeffs, a->coeffs, b->coeffs, s, a->allocated_n, a->d, a->w,
                               a->mod);
        return;
    }
    field_vec_fma_generic(out, a, b, (const void *)s, 0);
}

static void field_vec_fma_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    struct _FieldVector v[5];
    FIELD_VEC_VIEW_PLANES(planes, job->a->d);
    field_vec_views(v, planes, job, begin, end);
    field_vec_fma_whole(&v[0], &v[1], &v[2], &v[3]);
    FIELD_VEC_FREE_VIEW_PLANES(planes, job->a->d);
}

void field_vec_fma(FieldVector out, const FieldVector a, const FieldVector b, const FieldVector c)
{
    FieldVecJob job = {.range = field_vec_fma_range, .out = out, .a = a, .b = b, .c = c};
    if (field_vec_split(&job, a->allocated_n, FIELD_VEC_PRODUCT_GRAIN, MOD_MIN_VECTOR_LEN))
        field_vec_run_chunks(&job);
    else
        field_vec_fma_whole(out, a, b, c);
}

static void field_vec_fma_scalar_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    struct _FieldVector v[5];
    FIELD_VEC_VIEW_PLANES(planes, job->a->d);
    field_vec_views(v, planes, job, begin, end);
    field_vec_fma_scalar_whole(&v[0], &v[1], &v[2], job->s);
    FIELD_VEC_FREE_VIEW_PLANES(planes, job->a->d);
}

void field_vec_fma_scalar(FieldVector out, const FieldVector a, const FieldVector b,
                          const uint64_t *s)
{
    FieldVecJob job = {.range = field_vec_fma_scalar_range, .out = out, .a = a, .b = b, .s = s};
    if (field_vec_split(&job, a->allocated_n, FIELD_VEC_PRODUCT_GRAIN, MOD_MIN_VECTOR_LEN))
        field_vec_run_chunks(&job);
    else
        field_vec_fma_scalar_whole(out, a, b, s);
}

// Scratch for the two halves `field_vec_fma_interleave` produces, sized to
// stay in a core's own cache between being written and being read back.
#define FMA_INTERLEAVE_SCRATCH_BYTES (256u * 1024u)

// The windows from `begin` up to `end` of field_vec_fma_interleave: job->c
// and job->s are c_even and s_even, job->c2 and job->s2 c_odd and s_odd.
static void field_vec_fma_interleave_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    const FieldVector out = job->out, a = job->a, b = job->b, c_even = job->c, c_odd = job->c2;
    const uint64_t *s_even = job->s, *s_odd = job->s2;
    const uint64_t d = a->d, n = a->n, padded = end, window = job->window;

    // One window's two halves, plus the operand views that read it.
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(2 * d * window * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)safe_malloc(6 * d * sizeof(uint64_t *));
    struct _FieldVector even = *a, odd = *a, av = *a, bv = *a, cev = *a, cov = *a;

    even.coeffs = planes;
    odd.coeffs = planes + d;
    av.coeffs = planes + 2 * d;
    bv.coeffs = planes + 3 * d;
    cev.coeffs = planes + 4 * d;
    cov.coeffs = planes + 5 * d;
    for (uint64_t j = 0; j < d; j++)
    {
        even.coeffs[j] = scratch + j * window;
        odd.coeffs[j] = scratch + (d + j) * window;
    }

    for (uint64_t start = begin; start < padded; start += window)
    {
        const uint64_t len = (padded - start < window) ? padded - start : window;
        for (uint64_t j = 0; j < d; j++)
        {
            av.coeffs[j] = a->coeffs[j] + start;
            bv.coeffs[j] = b->coeffs[j] + start;
            if (c_even != NULL)
                cev.coeffs[j] = c_even->coeffs[j] + start;
            if (c_odd != NULL)
                cov.coeffs[j] = c_odd->coeffs[j] + start;
        }
        even.n = odd.n = av.n = bv.n = cev.n = cov.n = len;
        even.allocated_n = odd.allocated_n = av.allocated_n = len;
        bv.allocated_n = cev.allocated_n = cov.allocated_n = len;

        if (c_even != NULL)
            field_vec_fma(&even, &av, &bv, &cev);
        else
            field_vec_fma_scalar(&even, &av, &bv, s_even);
        if (c_odd != NULL)
            field_vec_fma(&odd, &av, &bv, &cov);
        else
            field_vec_fma_scalar(&odd, &av, &bv, s_odd);

        // Only the elements the vector really holds are interleaved: `out` is
        // padded for 2n, which is not the same as two n-element paddings.
        if (start < n)
        {
            const uint64_t take = (n - start < len) ? n - start : len;
            for (uint64_t j = 0; j < d; j++)
                vec_interleave_u64(out->coeffs[j] + 2 * start, even.coeffs[j], odd.coeffs[j], take);
        }
    }

    free(planes);
    free(scratch);
}

// The window of a windowed kernel over `padded` positions keeping `planes`
// planes of d words a position in FMA_INTERLEAVE_SCRATCH_BYTES.
static uint64_t field_vec_scratch_window(uint64_t planes, uint64_t d, uint64_t padded)
{
    uint64_t window = FMA_INTERLEAVE_SCRATCH_BYTES / (planes * d * sizeof(uint64_t));
    window -= window % MOD_MIN_VECTOR_LEN;
    if (window < MOD_MIN_VECTOR_LEN)
        window = MOD_MIN_VECTOR_LEN;
    if (window > padded)
        window = padded;
    return window;
}

// Windows are independent and each carries its own scratch, so the windowed
// kernels split by runs of whole windows.
// Whether a plane of `out` shares memory with a plane of `in`. A fold
// written over its own word reads, at each window, pairs past the positions
// earlier windows wrote, so it is correct only one window after another.
static bool field_vec_planes_overlap(const FieldVector out, const FieldVector in)
{
    for (uint64_t i = 0; i < out->d; i++)
        for (uint64_t j = 0; j < in->d; j++)
        {
            const uint64_t *o = out->coeffs[i], *w = in->coeffs[j];
            if (o < w + in->allocated_n && w < o + out->allocated_n)
                return true;
        }
    return false;
}

static void field_vec_run_windows(FieldVecJob *job, uint64_t total)
{
    const uint64_t grain =
        job->window < FIELD_VEC_PRODUCT_GRAIN ? FIELD_VEC_PRODUCT_GRAIN : job->window;
    field_vec_run(job, total, grain, job->window);
}

void field_vec_fma_interleave(FieldVector out, const FieldVector a, const FieldVector b,
                              const FieldVector c_even, const uint64_t *s_even,
                              const FieldVector c_odd, const uint64_t *s_odd)
{
    const uint64_t padded = a->allocated_n;
    if (padded == 0)
        return;
    FieldVecJob job = {.range = field_vec_fma_interleave_range,
                       .out = out,
                       .a = a,
                       .b = b,
                       .c = c_even,
                       .c2 = c_odd,
                       .s = s_even,
                       .s2 = s_odd,
                       .window = field_vec_scratch_window(2, a->d, padded)};
    field_vec_run_windows(&job, padded);
}

// Product with a table of prime-field scalars, one per position: `t` holds a
// single plane (d == 1) over the same modulus, and out[i] = a[i] * t[i] is
// each of a's d planes scaled position-wise by it -- d eltwise passes, no
// cross-plane terms and no reduction by w, against the 2d^2 + d - 1 passes of
// the full product. `out` may alias `a`.
static void field_vec_mul_plane_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t j = 0; j < job->a->d; j++)
        mod_eltwise_mul(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin,
                        job->c->coeffs[0] + begin, end - begin, job->a->mod);
}

void field_vec_mul_plane(FieldVector out, const FieldVector a, const FieldVector t)
{
    FieldVecJob job = {.range = field_vec_mul_plane_range, .out = out, .a = a, .c = t};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_PRODUCT_GRAIN, MOD_MIN_VECTOR_LEN);
}

// out = a + b * t for the same kind of table. `out` may alias any operand:
// the product goes through one scratch plane.
static void field_vec_fma_plane_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    const uint64_t len = end - begin;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(len * sizeof(uint64_t));
    for (uint64_t j = 0; j < job->a->d; j++)
    {
        mod_eltwise_mul(scratch, job->b->coeffs[j] + begin, job->c->coeffs[0] + begin, len,
                        job->a->mod);
        mod_eltwise_add(job->out->coeffs[j] + begin, job->a->coeffs[j] + begin, scratch, len,
                        job->a->mod);
    }
    free(scratch);
}

void field_vec_fma_plane(FieldVector out, const FieldVector a, const FieldVector b,
                         const FieldVector t)
{
    FieldVecJob job = {.range = field_vec_fma_plane_range, .out = out, .a = a, .b = b, .c = t};
    field_vec_run(&job, a->allocated_n, FIELD_VEC_PRODUCT_GRAIN, MOD_MIN_VECTOR_LEN);
}

// A table is either d planes (an element per position) or one (a prime-field
// scalar per position); this picks the product for it.
static void mul_by_table(FieldVector out, const FieldVector a, const FieldVector t)
{
    if (t->d == 1 && a->d > 1)
        field_vec_mul_plane(out, a, t);
    else
        field_vec_mul(out, a, t);
}

// From `a`, `b` and a table `t`,
//
//     out[2i]     = a[i] + b[i] * t[i mod period]
//     out[2i + 1] = a[i] - b[i] * t[i mod period]
//
// The product is formed once per pair -- the two positions differ only by its
// sign -- and the pair is interleaved while still cached, so `out` is written
// once. A table shorter than `a` is read cyclically in place rather than tiled
// out to a's length. `t` is d planes or one.
//
// `period` is the table's length. Below a->n it must be a power of two and a
// whole number of eltwise vectors, so that the windows below never straddle a
// repetition; at or above a->n the table is read like `a` itself.
// The windows from `begin` up to `end` of field_vec_lift_twisted; job->c is
// the table.
static void field_vec_lift_twisted_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    const FieldVector out = job->out, a = job->a, b = job->b, t = job->c;
    const uint64_t d = a->d, n = a->n, padded = end, window = job->window, period = job->period;
    const int periodic = period < n;

    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(2 * d * window * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)safe_malloc(5 * d * sizeof(uint64_t *));
    struct _FieldVector prod = *a, odd = *a, av = *a, bv = *a, tv = *t;

    prod.coeffs = planes;
    odd.coeffs = planes + d;
    av.coeffs = planes + 2 * d;
    bv.coeffs = planes + 3 * d;
    tv.coeffs = planes + 4 * d;
    for (uint64_t j = 0; j < d; j++)
    {
        prod.coeffs[j] = scratch + j * window;
        odd.coeffs[j] = scratch + (d + j) * window;
    }

    for (uint64_t start = begin; start < padded; start += window)
    {
        const uint64_t len = (padded - start < window) ? padded - start : window;
        const uint64_t offset = periodic ? start % period : start;
        for (uint64_t j = 0; j < d; j++)
        {
            av.coeffs[j] = a->coeffs[j] + start;
            bv.coeffs[j] = b->coeffs[j] + start;
        }
        for (uint64_t j = 0; j < t->d; j++)
            tv.coeffs[j] = t->coeffs[j] + offset;
        prod.n = odd.n = av.n = bv.n = tv.n = len;
        prod.allocated_n = odd.allocated_n = av.allocated_n = len;
        bv.allocated_n = tv.allocated_n = len;

        mul_by_table(&prod, &bv, &tv);
        field_vec_sub(&odd, &av, &prod);
        field_vec_add(&prod, &av, &prod);

        if (start < n)
        {
            const uint64_t take = (n - start < len) ? n - start : len;
            for (uint64_t j = 0; j < d; j++)
                vec_interleave_u64(out->coeffs[j] + 2 * start, prod.coeffs[j], odd.coeffs[j], take);
        }
    }

    free(planes);
    free(scratch);
}

void field_vec_lift_twisted(FieldVector out, const FieldVector a, const FieldVector b,
                            const FieldVector t, uint64_t period)
{
    const uint64_t padded = a->allocated_n;
    if (padded == 0)
        return;

    uint64_t window = field_vec_scratch_window(2, a->d, padded);
    if (period < a->n)
    {
        // A power of two no larger than the period, so that every window
        // start is a multiple of the window and every window lies inside one
        // repetition of the table.
        uint64_t pow2 = MOD_MIN_VECTOR_LEN;
        while (pow2 * 2 <= window)
            pow2 *= 2;
        window = pow2 < period ? pow2 : period;
    }
    FieldVecJob job = {.range = field_vec_lift_twisted_range,
                       .out = out,
                       .a = a,
                       .b = b,
                       .c = t,
                       .window = window,
                       .period = period};
    field_vec_run_windows(&job, padded);
}

// `word` read as adjacent pairs, folded with the table and one element r:
//
//     out[i] = (word[2i] + word[2i + 1]) / 2
//              + (word[2i] - word[2i + 1]) * untwist[i] * r
//
// computed as word[2i + 1] + (word[2i] - word[2i + 1]) * c[i] with
// c[i] = untwist[i] * r + 1/2: two products per pair. The passes run a
// window at a time so that the split's two halves and `c` stay cached instead
// of being streamed back from memory between each one.
//
// The windows from `begin` up to `end`; job->a is the word, job->c the table.
static void field_vec_fold_twisted_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    const FieldVector out = job->out, word = job->a, untwist = job->c;
    const uint64_t *r = job->s;
    const uint64_t d = word->d, padded = end, window = job->window;
    const uint64_t half = (word->mod->q + 1) / 2;

    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(3 * d * window * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)safe_malloc(6 * d * sizeof(uint64_t *));
    struct _FieldVector lo = *out, hi = *out, c = *out, pair = *word, t2i = *untwist, dst = *out;

    lo.coeffs = planes;
    hi.coeffs = planes + d;
    c.coeffs = planes + 2 * d;
    pair.coeffs = planes + 3 * d;
    t2i.coeffs = planes + 4 * d;
    dst.coeffs = planes + 5 * d;
    for (uint64_t j = 0; j < d; j++)
    {
        lo.coeffs[j] = scratch + j * window;
        hi.coeffs[j] = scratch + (d + j) * window;
        c.coeffs[j] = scratch + (2 * d + j) * window;
    }

    for (uint64_t start = begin; start < padded; start += window)
    {
        const uint64_t len = (padded - start < window) ? padded - start : window;
        // The result is padded to n and `word` to 2n, and those round
        // up differently, so the last window can ask for a few pairs past the
        // end of `word`. Those positions are padding of the result: the split
        // takes what is there and the rest is zeroed, which the arithmetic
        // carries through to the reduced zeros the padding has to hold.
        const uint64_t have = word->allocated_n / 2 - start;
        const uint64_t take = have < len ? have : len;

        for (uint64_t j = 0; j < d; j++)
        {
            dst.coeffs[j] = out->coeffs[j] + start;
            pair.coeffs[j] = word->coeffs[j] + 2 * start;
        }
        // The table is d planes, or one when it holds prime-field scalars;
        // either way it is read at this window's offset.
        for (uint64_t j = 0; j < untwist->d; j++)
            t2i.coeffs[j] = untwist->coeffs[j] + start;
        lo.n = hi.n = c.n = t2i.n = dst.n = len;
        lo.allocated_n = hi.allocated_n = c.allocated_n = len;
        t2i.allocated_n = dst.allocated_n = len;
        pair.n = pair.allocated_n = 2 * take;

        field_vec_split_even_odd(&lo, &hi, &pair);
        for (uint64_t j = 0; take < len && j < d; j++)
        {
            memset(lo.coeffs[j] + take, 0, (len - take) * sizeof(uint64_t));
            memset(hi.coeffs[j] + take, 0, (len - take) * sizeof(uint64_t));
        }

        // c = untwist * r + 1/2. A prime-field table scales each plane of
        // r by its one plane.
        if (t2i.d == 1 && d > 1)
            for (uint64_t j = 0; j < d; j++)
                mod_eltwise_scale(c.coeffs[j], t2i.coeffs[0], r[j], len, word->mod);
        else
            field_vec_scale(&c, &t2i, r);
        mod_eltwise_add_scalar(c.coeffs[0], c.coeffs[0], half, len, word->mod);

        field_vec_sub(&lo, &lo, &hi);
        field_vec_fma(&dst, &hi, &lo, &c);
    }

    free(planes);
    free(scratch);
}

void field_vec_fold_twisted(FieldVector out, const FieldVector word, const FieldVector untwist,
                            const uint64_t *r)
{
    const uint64_t padded = out->allocated_n;
    if (padded == 0)
        return;
    FieldVecJob job = {.range = field_vec_fold_twisted_range,
                       .out = out,
                       .a = word,
                       .c = untwist,
                       .s = r,
                       .window = field_vec_scratch_window(3, word->d, padded)};
    if (field_vec_planes_overlap(out, word))
        field_vec_fold_twisted_range(&job, 0, padded);
    else
        field_vec_run_windows(&job, padded);
}

/* Each output element is computed as v + (u - v) * c with
   c = untwist[i] * r[b] + 1/2, which equals the header's
   (u + v) / 2 + (u - v) * untwist[i] * r[b] with two products per pair. The
   output is produced a window at a time: the window's pairs are split into
   u and v, c is built for its positions, and the result is written, all in a
   scratch buffer small enough to stay in cache. */
static void field_vec_fold_twisted_blocks_range(const FieldVecJob *job, uint64_t begin,
                                                uint64_t end)
{
    const FieldVector out = job->out, word = job->a, untwist = job->c, r = job->b;
    const uint64_t d = word->d, m = untwist->n, total = end, window = job->window;
    const uint64_t half = (word->mod->q + 1) / 2;

    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(3 * d * window * sizeof(uint64_t));
    // The vector kernels take 64-byte-aligned planes in whole groups of
    // MOD_MIN_VECTOR_LEN. When m is not a multiple of that, a run of the
    // table can start or end inside a group, and such a run is computed here
    // and copied into c.
    uint64_t *unaligned_run = NULL;
    if (m % MOD_MIN_VECTOR_LEN != 0)
        unaligned_run = (uint64_t *)safe_aligned_malloc(d * window * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)safe_malloc(7 * d * sizeof(uint64_t *));
    uint64_t *r_block = (uint64_t *)safe_malloc(d * sizeof(uint64_t));
    struct _FieldVector lo = *out, hi = *out, c = *out, pair = *word, dst = *out;
    struct _FieldVector run = *out, table = *untwist;

    lo.coeffs = planes;
    hi.coeffs = planes + d;
    c.coeffs = planes + 2 * d;
    pair.coeffs = planes + 3 * d;
    dst.coeffs = planes + 4 * d;
    run.coeffs = planes + 5 * d;
    table.coeffs = planes + 6 * d;
    for (uint64_t j = 0; j < d; j++)
    {
        lo.coeffs[j] = scratch + j * window;
        hi.coeffs[j] = scratch + (d + j) * window;
        c.coeffs[j] = scratch + (2 * d + j) * window;
    }

    for (uint64_t start = begin; start < total; start += window)
    {
        const uint64_t len = (total - start < window) ? total - start : window;
        const uint64_t padded_len = field_vec_padded_length(len);
        for (uint64_t j = 0; j < d; j++)
        {
            dst.coeffs[j] = out->coeffs[j] + start;
            pair.coeffs[j] = word->coeffs[j] + 2 * start;
        }
        lo.n = hi.n = c.n = dst.n = len;
        lo.allocated_n = hi.allocated_n = c.allocated_n = dst.allocated_n = padded_len;
        pair.n = pair.allocated_n = 2 * len;
        field_vec_split_even_odd(&lo, &hi, &pair);
        // The kernels read the padding of a short last window: zeroed, it
        // holds values below q.
        if (padded_len > len)
            for (uint64_t j = 0; j < 3 * d; j++)
                memset(scratch + j * window + len, 0, (padded_len - len) * sizeof(uint64_t));

        /* c[t] = untwist[t mod m] * r[t / m] + 1/2 for the output positions
           t of the window. The window may cover parts of several blocks;
           the part of block b is the run of the table it covers, scaled by
           r[b] -- plane by plane when the table holds prime-field scalars. */
        for (uint64_t pos = start; pos < start + len;)
        {
            const uint64_t block = pos / m, offset = pos % m;
            const uint64_t take = (m - offset < start + len - pos) ? m - offset : start + len - pos;
            const bool aligned = offset % MOD_MIN_VECTOR_LEN == 0 &&
                                 (pos - start) % MOD_MIN_VECTOR_LEN == 0 &&
                                 take % MOD_MIN_VECTOR_LEN == 0;
            const uint64_t padded_take = field_vec_padded_length(take);
            for (uint64_t j = 0; j < d; j++)
            {
                r_block[j] = r->coeffs[j][block];
                run.coeffs[j] = aligned ? c.coeffs[j] + (pos - start) : unaligned_run + j * window;
            }
            // Away from c, the run is scaled in place: a table over the prime
            // subfield sits in the last plane, which is the last one written.
            for (uint64_t j = 0; j < untwist->d; j++)
            {
                if (aligned)
                {
                    table.coeffs[j] = untwist->coeffs[j] + offset;
                    continue;
                }
                table.coeffs[j] = run.coeffs[untwist->d == d ? j : d - 1];
                memcpy(table.coeffs[j], untwist->coeffs[j] + offset, take * sizeof(uint64_t));
                memset(table.coeffs[j] + take, 0, (padded_take - take) * sizeof(uint64_t));
            }
            run.n = table.n = take;
            run.allocated_n = table.allocated_n = padded_take;
            if (untwist->d == 1 && d > 1)
                for (uint64_t j = 0; j < d; j++)
                    mod_eltwise_scale(run.coeffs[j], table.coeffs[0], r_block[j], take, word->mod);
            else
                field_vec_scale(&run, &table, r_block);
            if (!aligned)
                for (uint64_t j = 0; j < d; j++)
                    memcpy(c.coeffs[j] + (pos - start), run.coeffs[j], take * sizeof(uint64_t));
            pos += take;
        }
        mod_eltwise_add_scalar(c.coeffs[0], c.coeffs[0], half, len, word->mod);

        field_vec_sub(&lo, &lo, &hi);
        field_vec_fma(&dst, &hi, &lo, &c);
    }

    free(r_block);
    free(planes);
    free(unaligned_run);
    free(scratch);
}

void field_vec_fold_twisted_blocks(FieldVector out, const FieldVector word,
                                   const FieldVector untwist, const FieldVector r)
{
    const uint64_t d = word->d, total = untwist->n * r->n;
    assert(word->n == 2 * total && out->n == total && r->d == d);
    if (total == 0)
        return;

    uint64_t window = FMA_INTERLEAVE_SCRATCH_BYTES / (3 * d * sizeof(uint64_t));
    window -= window % MOD_MIN_VECTOR_LEN;
    if (window < MOD_MIN_VECTOR_LEN)
        window = MOD_MIN_VECTOR_LEN;
    if (window > total)
        window = field_vec_padded_length(total);
    FieldVecJob job = {.range = field_vec_fold_twisted_blocks_range,
                       .out = out,
                       .a = word,
                       .b = r,
                       .c = untwist,
                       .window = window};
    if (field_vec_planes_overlap(out, word))
        field_vec_fold_twisted_blocks_range(&job, 0, total);
    else
        field_vec_run_windows(&job, total);
}

void field_vec_sum(uint64_t *out, const FieldVector a)
{
    for (uint64_t j = 0; j < a->d; j++)
    {
        const uint64_t q = a->mod->q;
        uint64_t acc = 0;
        for (uint64_t i = 0; i < a->n; i++)
            acc = add_modq(acc, a->coeffs[j][i], q);
        out[j] = acc;
    }
}

void field_vec_get_element(uint64_t *out, const FieldVector a, uint64_t index)
{
    for (uint64_t j = 0; j < a->d; j++)
        out[j] = a->coeffs[j][index];
}

void field_vec_set_element(FieldVector out, uint64_t index, const uint64_t *value)
{
    for (uint64_t j = 0; j < out->d; j++)
        out->coeffs[j][index] = value[j];
}

void field_vec_set_range(FieldVector out, uint64_t start, const uint64_t *values, uint64_t count)
{
    const uint64_t d = out->d;
    for (uint64_t j = 0; j < d; j++)
    {
        uint64_t *plane = out->coeffs[j] + start;
        for (uint64_t i = 0; i < count; i++)
            plane[i] = values[i * d + j];
    }
}

void field_vec_get_range(uint64_t *out, const FieldVector a, uint64_t start, uint64_t count)
{
    const uint64_t d = a->d;
    for (uint64_t j = 0; j < d; j++)
    {
        const uint64_t *plane = a->coeffs[j] + start;
        for (uint64_t i = 0; i < count; i++)
            out[i * d + j] = plane[i];
    }
}

void field_vec_copy(FieldVector out, const FieldVector a)
{
    for (uint64_t j = 0; j < a->d; j++)
        memcpy(out->coeffs[j], a->coeffs[j], a->allocated_n * sizeof(uint64_t));
}

void field_vec_split_even_odd(FieldVector even, FieldVector odd, const FieldVector a)
{
    const uint64_t half = a->n / 2;
    for (uint64_t j = 0; j < a->d; j++)
    {
        const uint64_t *plane = a->coeffs[j];
        for (uint64_t i = 0; i < half; i++)
        {
            even->coeffs[j][i] = plane[2 * i];
            odd->coeffs[j][i] = plane[2 * i + 1];
        }
    }
}

void field_vec_interleave(FieldVector out, const FieldVector even, const FieldVector odd)
{
    for (uint64_t j = 0; j < out->d; j++)
        vec_interleave_u64(out->coeffs[j], even->coeffs[j], odd->coeffs[j], even->n);
}

void field_vec_concat(FieldVector out, const FieldVector *parts, uint64_t count)
{
    uint64_t at = 0;
    for (uint64_t p = 0; p < count; p++)
    {
        for (uint64_t j = 0; j < out->d; j++)
            memcpy(out->coeffs[j] + at, parts[p]->coeffs[j], parts[p]->n * sizeof(uint64_t));
        at += parts[p]->n;
    }
}

bool field_vec_gather(FieldVector out, const FieldVector a, const uint64_t *indices, uint64_t count)
{
    // Checked here rather than by the caller: a gather is one pass over the
    // indices and so is validating them, but a caller outside C pays a call
    // per index, which at a message-sized gather costs many times the gather.
    // A pass of its own, before anything is written, so a rejected gather
    // leaves `out` untouched and the bound is tested once per index and not
    // once per plane.
    for (uint64_t i = 0; i < count; i++)
        if (indices[i] >= a->n)
            return false;
    for (uint64_t j = 0; j < a->d; j++)
        for (uint64_t i = 0; i < count; i++)
            out->coeffs[j][i] = a->coeffs[j][indices[i]];
    return true;
}

// Deinterleave into two scratch vectors of out's shape, then three whole-plane
// operations: one allocation and no Python-visible intermediates. The scratch
// planes are `half` words each, a multiple of the eltwise vector width, so each
// stays 64-byte aligned inside the one block.
void field_vec_fold(FieldVector out, const FieldVector a, const uint64_t *r)
{
    const uint64_t d = a->d, half = out->allocated_n;
    // Pairs that exist in a's allocation; the rest of out's padding stays zero.
    const uint64_t pairs = a->allocated_n / 2 < half ? a->allocated_n / 2 : half;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(2 * d * half * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)malloc(2 * d * sizeof(uint64_t *));
    struct _FieldVector even = *out, odd = *out;

    memset(scratch, 0, 2 * d * half * sizeof(uint64_t));
    even.coeffs = planes;
    odd.coeffs = planes + d;
    for (uint64_t j = 0; j < d; j++)
    {
        even.coeffs[j] = scratch + (2 * j) * half;
        odd.coeffs[j] = scratch + (2 * j + 1) * half;
        for (uint64_t i = 0; i < pairs; i++)
        {
            even.coeffs[j][i] = a->coeffs[j][2 * i];
            odd.coeffs[j][i] = a->coeffs[j][2 * i + 1];
        }
    }

    field_vec_sub(&odd, &odd, &even);
    field_vec_scale(&odd, &odd, r);
    field_vec_add(out, &even, &odd);

    free(planes);
    free(scratch);
}

// The runs of `block` elements that alternate between the two outputs: `lo`
// collects runs 0, 2, 4, ... and `hi` runs 1, 3, 5, .... With block == 1 that
// is field_vec_split_even_odd, which is what runs for it.
void field_vec_split_blocks(FieldVector lo, FieldVector hi, const FieldVector a, uint64_t block)
{
    if (block == 1)
    {
        field_vec_split_even_odd(lo, hi, a);
        return;
    }
    const uint64_t bytes = block * sizeof(uint64_t);
    for (uint64_t j = 0; j < a->d; j++)
    {
        const uint64_t *plane = a->coeffs[j];
        for (uint64_t c = 0, at = 0; at + 2 * block <= a->n; c++, at += 2 * block)
        {
            memcpy(lo->coeffs[j] + c * block, plane + at, bytes);
            memcpy(hi->coeffs[j] + c * block, plane + at + block, bytes);
        }
    }
}

// field_vec_fold over pairs `block` apart rather than adjacent: out[i] is
// lo[i] + r * (hi[i] - lo[i]) on the split field_vec_split_blocks makes, which
// is the interpolation binding the variable that `block` indexes.
//
// Three whole-plane passes either way; what `block` decides is whether the two
// operands have to be gathered first. At block == a->n / 2 -- the last variable
// of a multilinear table -- the halves are already contiguous, so they are used
// where they lie and the only allocation is the one temporary the passes need.
void field_vec_fold_blocks(FieldVector out, const FieldVector a, uint64_t block, const uint64_t *r)
{
    if (block == 1)
    {
        field_vec_fold(out, a, r);
        return;
    }

    const uint64_t d = a->d, half = out->allocated_n;
    // The halves are contiguous runs of `half` words only when nothing was
    // rounded up to reach it; otherwise they are gathered like any other block.
    const int in_place = (block == a->n / 2) && (block == half);
    const uint64_t planes_held = in_place ? 1 : 3;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(planes_held * d * half * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)malloc(3 * d * sizeof(uint64_t *));
    struct _FieldVector lo = *out, hi = *out, acc = *out;

    memset(scratch, 0, planes_held * d * half * sizeof(uint64_t));
    lo.coeffs = planes;
    hi.coeffs = planes + d;
    acc.coeffs = planes + 2 * d;
    for (uint64_t j = 0; j < d; j++)
        acc.coeffs[j] = scratch + j * half;

    if (in_place)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            lo.coeffs[j] = a->coeffs[j];
            hi.coeffs[j] = a->coeffs[j] + block;
        }
    }
    else
    {
        for (uint64_t j = 0; j < d; j++)
        {
            lo.coeffs[j] = scratch + (d + j) * half;
            hi.coeffs[j] = scratch + (2 * d + j) * half;
        }
        field_vec_split_blocks(&lo, &hi, a, block);
    }

    field_vec_sub(&acc, &hi, &lo);
    field_vec_scale(&acc, &acc, r);
    field_vec_add(out, &lo, &acc);

    free(planes);
    free(scratch);
}

int field_vec_is_equal(const FieldVector a, const FieldVector b)
{
    if (a->n != b->n || a->d != b->d)
        return 0;
    for (uint64_t j = 0; j < a->d; j++)
    {
        if (memcmp(a->coeffs[j], b->coeffs[j], a->n * sizeof(uint64_t)) != 0)
            return 0;
    }
    return 1;
}

// A window of `length` elements from `start`, as a vector in its own right:
// the planes are the parent's, offset. Both bounds are multiples of
// MOD_MIN_VECTOR_LEN, so every plane stays 64-byte aligned and the eltwise
// kernels keep their no-tail contract.
static struct _FieldVector field_vec_window(const FieldVector v, uint64_t start, uint64_t length,
                                            uint64_t **planes)
{
    struct _FieldVector window = *v;
    for (uint64_t j = 0; j < v->d; j++)
        planes[j] = v->coeffs[j] + start;
    window.coeffs = planes;
    window.n = length;
    window.allocated_n = length;
    return window;
}

// Montgomery's trick an element at a time: prefix products, one inversion of
// the last, then a reverse sweep peeling each factor back off. 3(n-1)
// multiplications and one inversion, all on single elements.
static int field_vec_inv_serial(FieldVector out, const FieldVector a)
{
    const uint64_t d = a->d, n = a->n;
    if (n == 0)
        return 1;

    // Prefix products, one inversion of the last, then a reverse sweep peeling
    // each factor back off: 3(n-1) multiplications and one inversion.
    uint64_t *prefix = (uint64_t *)malloc(n * d * sizeof(uint64_t));
    uint64_t *element = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *running = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)malloc(d * sizeof(uint64_t));

    field_vec_get_element(prefix, a, 0);
    for (uint64_t i = 1; i < n; i++)
    {
        field_vec_get_element(element, a, i);
        field_ext_mul(prefix + i * d, prefix + (i - 1) * d, element, d, a->w, a->mod);
    }

    int status = field_ext_inv(running, prefix + (n - 1) * d, d, a->w, a->mod);
    if (status)
    {
        for (uint64_t i = n; i-- > 1;)
        {
            // Read a[i] before writing out[i]: the two may be the same vector.
            field_vec_get_element(element, a, i);
            field_ext_mul(tmp, running, prefix + (i - 1) * d, d, a->w, a->mod);
            field_vec_set_element(out, i, tmp);
            field_ext_mul(running, running, element, d, a->w, a->mod);
        }
        field_vec_set_element(out, 0, running);
    }

    free(prefix);
    free(element);
    free(running);
    free(tmp);
    return status;
}

// Rows the parallel sweep aims for. The trick's two sweeps are a serial chain
// -- each step needs the one before it -- so one chain advances an element at a
// time however wide the vector is. Cut the vector into rows and the chains
// become the columns: a step is one whole-vector multiply over a row, and the
// chain is only as long as there are rows. More rows means fewer columns and so
// fewer of the single inversions the backward sweep starts from; 256 keeps
// those to about a nanosecond an element while leaving rows wide enough to be
// worth handing to a kernel.
#define FIELD_VEC_INV_ROWS 256

int field_vec_inv(FieldVector out, const FieldVector a)
{
    const uint64_t d = a->d, n = a->n;
    if (n == 0)
        return 1;

    // Rows must be a whole number of SIMD vectors for the windows to be
    // aligned; below one row's worth there is nothing to parallelise anyway.
    const uint64_t width = (n / FIELD_VEC_INV_ROWS) & ~(uint64_t)(MOD_MIN_VECTOR_LEN - 1);
    if (width == 0)
        return field_vec_inv_serial(out, a);
    const uint64_t rows = n / width, body = rows * width;

    uint64_t *prefix_block = (uint64_t *)safe_aligned_malloc(d * body * sizeof(uint64_t));
    uint64_t *carry_block = (uint64_t *)safe_aligned_malloc(3 * d * width * sizeof(uint64_t));
    uint64_t **planes = (uint64_t **)safe_malloc(8 * d * sizeof(uint64_t *));

    struct _FieldVector prefix = *a, running = *a, folded = *a, next = *a;
    for (uint64_t j = 0; j < d; j++)
    {
        planes[j] = prefix_block + j * body;
        planes[d + j] = carry_block + j * width;
        planes[2 * d + j] = carry_block + (d + j) * width;
        planes[3 * d + j] = carry_block + (2 * d + j) * width;
    }
    prefix.coeffs = planes;
    prefix.n = prefix.allocated_n = body;
    running.coeffs = planes + d;
    folded.coeffs = planes + 2 * d;
    next.coeffs = planes + 3 * d;
    running.n = running.allocated_n = width;
    folded.n = folded.allocated_n = width;
    next.n = next.allocated_n = width;

    uint64_t **a_planes = planes + 4 * d, **p_planes = planes + 5 * d;
    uint64_t **q_planes = planes + 6 * d, **o_planes = planes + 7 * d;

    // Forward: prefix row t is the product of rows 0..t, column by column.
    struct _FieldVector head = field_vec_window(&prefix, 0, width, p_planes);
    struct _FieldVector head_in = field_vec_window(a, 0, width, a_planes);
    field_vec_copy(&head, &head_in);
    for (uint64_t t = 1; t < rows; t++)
    {
        struct _FieldVector to = field_vec_window(&prefix, t * width, width, p_planes);
        struct _FieldVector from = field_vec_window(&prefix, (t - 1) * width, width, q_planes);
        struct _FieldVector row = field_vec_window(a, t * width, width, a_planes);
        field_vec_mul(&to, &from, &row);
    }

    // One inversion per column, which is the trick again over a single row.
    struct _FieldVector last = field_vec_window(&prefix, (rows - 1) * width, width, p_planes);
    int status = field_vec_inv_serial(&running, &last);

    // Whatever did not fit into whole rows is its own chain. It is inverted
    // before any row of `out` is written, so that a zero there leaves `out`
    // untouched; the backward sweep below reads only the whole rows of `a`.
    if (status && body < n)
    {
        struct _FieldVector tail_in = field_vec_window(a, body, n - body, a_planes);
        struct _FieldVector tail_out = field_vec_window(out, body, n - body, o_planes);
        status = field_vec_inv_serial(&tail_out, &tail_in);
    }

    if (status)
    {
        // Backward: out row t is the running product times prefix row t-1, and
        // the running product then absorbs row t. Row t of `a` is read into
        // `next` before out row t is written, because `out` may be `a`.
        for (uint64_t t = rows; t-- > 1;)
        {
            struct _FieldVector below = field_vec_window(&prefix, (t - 1) * width, width, p_planes);
            struct _FieldVector row = field_vec_window(a, t * width, width, a_planes);
            struct _FieldVector dest = field_vec_window(out, t * width, width, o_planes);
            field_vec_mul(&folded, &running, &below);
            field_vec_mul(&next, &running, &row);
            field_vec_copy(&dest, &folded);
            struct _FieldVector spare = running;
            running = next;
            next = spare;
        }
        struct _FieldVector first = field_vec_window(out, 0, width, o_planes);
        field_vec_copy(&first, &running);
    }

    free(prefix_block);
    free(carry_block);
    free(planes);
    return status;
}

// The Frobenius applied to every element: the same permutation of coefficient
// positions and the same constant per position as field_ext_frobenius, which is
// where the map is derived -- so here it is d whole-plane scalings, independent
// of how many elements the vector holds, rather than one exponentiation each.
//
// Written through a scratch plane set because the map permutes the planes and
// `out` may be `a`.
void field_vec_frobenius(FieldVector out, const FieldVector a, uint64_t k)
{
    const uint64_t d = a->d, len = a->allocated_n;
    if (d <= 1 || k % d == 0)
    {
        field_vec_copy(out, a);
        return;
    }
    uint64_t *constants = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *to = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(d * len * sizeof(uint64_t));

    frobenius_map(constants, to, k, d, a->w, a->mod);
    for (uint64_t j = 0; j < d; j++)
        mod_eltwise_scale(scratch + to[j] * len, a->coeffs[j], constants[j], len, a->mod);
    for (uint64_t j = 0; j < d; j++)
        memcpy(out->coeffs[j], scratch + j * len, len * sizeof(uint64_t));

    free(constants);
    free(to);
    free(scratch);
}

void field_vec_sample_random(FieldVector out, const uint8_t *seed, uint64_t seed_len,
                             uint64_t start)
{
    // One draw stream over all the coefficients, scattered into the planes:
    // element i takes draws i*d .. i*d + d - 1, so `start` is a position in
    // that stream and not a second stream. Drawing per plane from the same seed
    // would give every plane the same values, and hence every element the same
    // coefficient repeated.
    const uint64_t d = out->d, n = out->n;
    if (n == 0)
        return;
    uint64_t *flat = (uint64_t *)malloc(n * d * sizeof(uint64_t));
    prng_sample_below_from(flat, n * d, start * d, out->mod->q, "field_vec_sample", seed, seed_len);
    field_vec_set_range(out, 0, flat, n);
    free(flat);
}

void field_vec_sample_random_at(FieldVector out, const uint8_t *seed, uint64_t seed_len,
                                const uint64_t *indices, uint64_t count)
{
    const uint64_t d = out->d;
    if (count == 0)
        return;
    uint64_t *flat = (uint64_t *)safe_malloc(count * d * sizeof(uint64_t));
    // Consecutive indices are one run of the same draw stream, so a sorted
    // window costs what that window of a whole fill costs; anything scattered
    // falls back to a draw apiece, which is what a scattered request is.
    for (uint64_t k = 0; k < count;)
    {
        uint64_t run = 1;
        while (k + run < count && indices[k + run] == indices[k] + run)
            run++;
        prng_sample_below_from(&flat[k * d], run * d, indices[k] * d, out->mod->q,
                               "field_vec_sample", seed, seed_len);
        k += run;
    }
    field_vec_set_range(out, 0, flat, count);
    free(flat);
}

void field_vec_sample_random_element(uint64_t *out, const uint8_t *seed, uint64_t seed_len,
                                     uint64_t index, uint64_t d, uint64_t mod)
{
    // The same stream, at the d draws element `index` takes -- so this and the
    // fill above agree by construction rather than by two definitions that have
    // to be kept in step.
    prng_sample_below_from(out, d, index * d, mod, "field_vec_sample", seed, seed_len);
}

// The elements in index order, d words each, as the bytes a digest covers.
static void hash_span(uint8_t *out, const FieldVector a, uint64_t start, uint64_t count)
{
    const uint64_t d = a->d;
    uint64_t *element = (uint64_t *)safe_malloc(d * sizeof(uint64_t));
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    for (uint64_t i = 0; i < count; i++)
    {
        field_vec_get_element(element, a, start + i);
        blake3_hasher_update(&hasher, (const uint8_t *)element, d * sizeof(uint64_t));
    }
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
    free(element);
}

void field_vec_hash(uint8_t *out, const FieldVector a) { hash_span(out, a, 0, a->n); }

// Windows staged per batch, as a byte budget rather than a count: a window is
// anywhere from one element to a hundred, and what has to stay in cache is the
// bytes. The gather writes them and the hasher reads them straight back, so
// this wants to be small; 32 KiB also keeps the batch at 32 windows or more
// however large a window is, which is what keeps the SIMD lanes full.
#define FIELD_VEC_HASH_STAGE_BYTES (32 * 1024)

// One digest per window, where window k covers the `group` elements at
// `k * window_step + i * element_step`, i < group. Both window shapes this
// file offers are that: a contiguous run is (stride, 1) and a fiber is
// (1, stride), which is the whole difference between them.
//
// The digests go through the lane-parallel `hash_batch`, which is an order of
// magnitude over a hasher per window. It needs the windows laid end to end,
// and an element's d words live one per plane -- so a window's bytes are never
// already contiguous and a batch has to be staged. The gather is a pass over
// the vector either way; staging only decides where it writes.
//
// Windows are independent, so the digests split by runs of windows, each with
// its own staging. This is windows [begin, end), into job->bytes.
static void hash_windows_range(const FieldVecJob *job, uint64_t begin, uint64_t end)
{
    const FieldVector a = job->a;
    const uint64_t group = job->group, window_step = job->window_step;
    const uint64_t element_step = job->element_step;
    const uint64_t d = a->d;
    const uint64_t leaf = group * d * sizeof(uint64_t);
    uint8_t *out = job->bytes;

    if (!hash_batch_fits(leaf))
    {
        // A window past a BLAKE3 chunk, which hash_batch cannot take: one
        // hasher per window, fed element by element.
        uint64_t *element = (uint64_t *)safe_malloc(d * sizeof(uint64_t));
        for (uint64_t k = begin; k < end; k++)
        {
            blake3_hasher hasher;
            blake3_hasher_init(&hasher);
            for (uint64_t i = 0; i < group; i++)
            {
                field_vec_get_element(element, a, k * window_step + i * element_step);
                blake3_hasher_update(&hasher, (const uint8_t *)element, d * sizeof(uint64_t));
            }
            blake3_hasher_finalize(&hasher, &out[k * BLAKE3_OUT_LEN], BLAKE3_OUT_LEN);
        }
        free(element);
        return;
    }

    // hash_batch_fits caps `leaf` at a BLAKE3 chunk, so this is at least 32.
    const uint64_t batch = FIELD_VEC_HASH_STAGE_BYTES / leaf;
    uint64_t *staged = (uint64_t *)safe_aligned_malloc(batch * leaf);
    for (uint64_t k = begin; k < end;)
    {
        uint64_t run = end - k;
        if (run > batch)
            run = batch;
        for (uint64_t j = 0; j < run; j++)
            for (uint64_t i = 0; i < group; i++)
                field_vec_get_element(&staged[(j * group + i) * d], a,
                                      (k + j) * window_step + i * element_step);
        hash_batch(&out[k * BLAKE3_OUT_LEN], (const uint8_t *)staged, run, leaf);
        k += run;
    }
    free(staged);
}

static void hash_windows(uint8_t *out, const FieldVector a, uint64_t count, uint64_t group,
                         uint64_t window_step, uint64_t element_step)
{
    if (count == 0)
        return;
    const uint64_t leaf = group * a->d * sizeof(uint64_t);
    FieldVecJob job = {.range = hash_windows_range,
                       .a = a,
                       .group = group,
                       .window_step = window_step,
                       .element_step = element_step,
                       .bytes = out};
    field_vec_run(&job, count, FIELD_VEC_HASH_GRAIN_BYTES / leaf, 1);
}

uint64_t field_vec_hash_fiber_count(const FieldVector a, uint64_t group, uint64_t stride)
{
    if (group == 0 || stride == 0 || group > a->n / stride)
        return 0;
    return stride;
}

void field_vec_hash_fibers(uint8_t *out, const FieldVector a, uint64_t group, uint64_t stride)
{
    hash_windows(out, a, field_vec_hash_fiber_count(a, group, stride), group, 1, stride);
}

uint64_t field_vec_hash_count(const FieldVector a, uint64_t group, uint64_t stride)
{
    if (group == 0 || stride == 0 || a->n < group)
        return 0;
    return (a->n - group) / stride + 1;
}

void field_vec_hash_elements(uint8_t *out, const FieldVector a, uint64_t group, uint64_t stride)
{
    hash_windows(out, a, field_vec_hash_count(a, group, stride), group, stride, 1);
}
