// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"
#include "arith_generic.h"
#include <alloc.h>
#include <arith_trace.h>
#include <parallel.h>

// -------------------------------------------------------------
// Dense vector operations over a ring the caller does not name
// -------------------------------------------------------------

// Every operation below computes each output entry on its own, from entries
// of its inputs. A large table is split into contiguous chunks of entries
// that run on the library's threads; each entry is computed the same way
// whichever chunk holds it, so the result does not depend on the number of
// threads.

// An operation writing fewer bytes than this runs on the calling thread:
// handing entries to other threads would cost more than computing them.
#define MLE_PARALLEL_BYTES (1u << 20)
// The least output, in bytes, one chunk covers, so that the temporaries a
// chunk allocates are amortised over enough entries.
#define MLE_CHUNK_BYTES (1u << 18)
// Chunks per thread: several, so that a thread starting late leaves less
// work behind, but few, so that each chunk stays large.
#define MLE_CHUNKS_PER_THREAD 4

// How one binding operation finds the pair (lo, hi) for output entry i.
typedef enum
{
    MLE_PAIRS,   // (2i, 2i + 1)
    MLE_HALVES,  // (i, i + size)
    MLE_STRIDED, // the entries differing only in bit eval_var_idx
} MlePairLayout;

typedef struct MleJob MleJob;

// Computes output entries [begin, end) of a job.
typedef void (*MleRange)(const MleJob *job, uint64_t begin, uint64_t end);

// One operation over a table and its operands; which operands it reads
// depends on `range`.
struct MleJob
{
    ArithRing ring;
    ArithElement *out;
    const ArithElement *in, *in2;
    const ArithElement *a; // a ring-element operand, or NULL for `scalar`
    uint64_t scalar;
    MlePairLayout layout;
    uint64_t size;         // output entries
    uint64_t eval_var_idx; // for MLE_STRIDED
    ArithElement *const *tables;
    const ArithElement *coeff;
    uint64_t terms;
    int fused; // the linear combination sums through the RNS kernel, not the generic interface
    MleRange range;
    uint64_t chunks;
};

// The bytes of one entry, as far as sizing the work goes: a word per
// coefficient and residue.
static uint64_t mle_entry_bytes(ArithRing ring)
{
    uint64_t components = 1;
    if (ring->impl == ARITH_IMPL_RNS)
        components = (uint64_t)__builtin_popcountll(arith_rns_ring_mask(ring));
    return components * ring->N * sizeof(uint64_t);
}

static void mle_chunk(void *ctx, uint64_t c)
{
    const MleJob *job = (const MleJob *)ctx;
    const uint64_t q = job->size / job->chunks, r = job->size % job->chunks;
    const uint64_t begin = c * q + (c < r ? c : r);
    job->range(job, begin, begin + q + (c < r ? 1 : 0));
}

static void bind_range(const MleJob *job, uint64_t begin, uint64_t end);

// Whether an output entry is also an operand that another output entry
// reads: a pairs or strided binding written over its own input (entry i
// reads entries at or after i, so one pass in entry order is correct), or a
// ring-element operand that is itself an entry of the output.
static bool mle_reads_its_output(const MleJob *job)
{
    if (job->range == bind_range && job->layout != MLE_HALVES && job->size > 0 &&
        job->out[0].handle == job->in[0].handle)
        return true;
    if (job->a != NULL)
        for (uint64_t i = 0; i < job->size; i++)
            if (job->out[i].handle == job->a->handle)
                return true;
    return false;
}

// Runs job->range over every output entry: on the calling thread when the
// output is small, when the operations go on the execution tape (which
// records the calling thread only, and so gets them in entry order), or when
// the output overlaps operands other entries read; otherwise in contiguous
// chunks on up to the library's thread limit.
static void mle_for(MleJob *job)
{
    uint64_t chunks = 1;
    if (!ARITH_GENERIC_PATH() && !mle_reads_its_output(job))
    {
        const uint64_t bytes = job->size * mle_entry_bytes(job->ring);
        if (bytes >= MLE_PARALLEL_BYTES)
        {
            chunks = bytes / MLE_CHUNK_BYTES;
            chunks = chunks < job->size ? chunks : job->size;
        }
    }
    const uint64_t threads = vfhe_threads_for(0, chunks);
    if (threads == 1)
    {
        job->range(job, 0, job->size);
        return;
    }
    const uint64_t wanted = threads * MLE_CHUNKS_PER_THREAD;
    job->chunks = chunks < wanted ? chunks : wanted;
    vfhe_parallel_for(job->chunks, threads, mle_chunk, job);
}

static void add_range(const MleJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t i = begin; i < end; i++)
        arith_add(job->ring, &job->out[i], &job->in[i], &job->in2[i]);
}

static void sub_range(const MleJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t i = begin; i < end; i++)
        arith_sub(job->ring, &job->out[i], &job->in[i], &job->in2[i]);
}

static void scale_range(const MleJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t i = begin; i < end; i++)
        arith_mul(job->ring, &job->out[i], &job->in[i], job->a);
}

static void scale_scalar_range(const MleJob *job, uint64_t begin, uint64_t end)
{
    for (uint64_t i = begin; i < end; i++)
        arith_scale_int(job->ring, &job->out[i], &job->in[i], job->scalar);
}

void mle_dense_poly_add(ArithRing ring, ArithElement *out, ArithElement *in1, ArithElement *in2,
                        uint64_t size)
{
    MleJob job = {
        .ring = ring, .out = out, .in = in1, .in2 = in2, .size = size, .range = add_range};
    mle_for(&job);
}

void mle_dense_poly_sub(ArithRing ring, ArithElement *out, ArithElement *in1, ArithElement *in2,
                        uint64_t size)
{
    MleJob job = {
        .ring = ring, .out = out, .in = in1, .in2 = in2, .size = size, .range = sub_range};
    mle_for(&job);
}

void mle_dense_poly_scale(ArithRing ring, ArithElement *out, ArithElement *in,
                          const ArithElement *scale, uint64_t size)
{
    MleJob job = {
        .ring = ring, .out = out, .in = in, .a = scale, .size = size, .range = scale_range};
    mle_for(&job);
}

void mle_dense_poly_scale_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                 uint64_t scale, uint64_t size)
{
    MleJob job = {.ring = ring,
                  .out = out,
                  .in = in,
                  .scalar = scale,
                  .size = size,
                  .range = scale_scalar_range};
    mle_for(&job);
}

// Binding one variable folds table pairs as out = lo + a * (hi - lo). The
// entry points below only differ in where the pair (lo, hi) sits: adjacent
// entries for the LSB variable (pairs), the two table halves for the MSB
// variable (halves), and stride-computed indices for any variable in between
// (the generic fallback). The Python layer picks by the variable's position.

static void mle_dense_poly_bind(ArithRing ring, ArithElement *out, const ArithElement *lo,
                                const ArithElement *hi, const ArithElement *a, ArithElement *tmp,
                                ArithElement *tmp2)
{
    arith_sub(ring, tmp, hi, lo);
    arith_mul(ring, tmp2, a, tmp);
    arith_add(ring, out, lo, tmp2);
}

static void mle_dense_poly_bind_scalar(ArithRing ring, ArithElement *out, const ArithElement *lo,
                                       const ArithElement *hi, uint64_t a, ArithElement *tmp,
                                       ArithElement *tmp2)
{
    arith_sub(ring, tmp, hi, lo);
    arith_scale_int(ring, tmp2, tmp, a);
    arith_add(ring, out, lo, tmp2);
}

// Each range binds with its own two temporaries.
static void bind_range(const MleJob *job, uint64_t begin, uint64_t end)
{
    const uint64_t stride = 1ULL << job->eval_var_idx;
    ArithElement tmp, tmp2;
    arith_new_like(job->ring, &job->in[0], &tmp);
    arith_new_like(job->ring, &job->in[0], &tmp2);
    for (uint64_t i = begin; i < end; i++)
    {
        uint64_t lo;
        if (job->layout == MLE_PAIRS)
            lo = 2 * i;
        else if (job->layout == MLE_HALVES)
            lo = i;
        else
            lo = (i & (stride - 1)) + ((i >> job->eval_var_idx) << (job->eval_var_idx + 1));
        const uint64_t hi = job->layout == MLE_PAIRS    ? lo + 1
                            : job->layout == MLE_HALVES ? lo + job->size
                                                        : lo + stride;
        if (job->a != NULL)
            mle_dense_poly_bind(job->ring, &job->out[i], &job->in[lo], &job->in[hi], job->a, &tmp,
                                &tmp2);
        else
            mle_dense_poly_bind_scalar(job->ring, &job->out[i], &job->in[lo], &job->in[hi],
                                       job->scalar, &tmp, &tmp2);
    }
    arith_free(job->ring, &tmp);
    arith_free(job->ring, &tmp2);
}

static void mle_bind(ArithRing ring, ArithElement *out, const ArithElement *in,
                     const ArithElement *a, uint64_t scalar, MlePairLayout layout, uint64_t size,
                     uint64_t eval_var_idx)
{
    MleJob job = {.ring = ring,
                  .out = out,
                  .in = in,
                  .a = a,
                  .scalar = scalar,
                  .layout = layout,
                  .size = size,
                  .eval_var_idx = eval_var_idx,
                  .range = bind_range};
    mle_for(&job);
}

void mle_dense_poly_evaluate_pairs(ArithRing ring, ArithElement *out, ArithElement *in,
                                   const ArithElement *a, uint64_t size)
{
    mle_bind(ring, out, in, a, 0, MLE_PAIRS, size, 0);
}

void mle_dense_poly_evaluate_pairs_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                          uint64_t a, uint64_t size)
{
    mle_bind(ring, out, in, NULL, a, MLE_PAIRS, size, 0);
}

void mle_dense_poly_evaluate_halves(ArithRing ring, ArithElement *out, ArithElement *in,
                                    const ArithElement *a, uint64_t size)
{
    mle_bind(ring, out, in, a, 0, MLE_HALVES, size, 0);
}

void mle_dense_poly_evaluate_halves_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                           uint64_t a, uint64_t size)
{
    mle_bind(ring, out, in, NULL, a, MLE_HALVES, size, 0);
}

void mle_dense_poly_evaluate(ArithRing ring, ArithElement *out, ArithElement *in,
                             const ArithElement *a, uint64_t num_vars, uint64_t eval_var_idx)
{
    mle_bind(ring, out, in, a, 0, MLE_STRIDED, 1ULL << (num_vars - 1), eval_var_idx);
}

void mle_dense_poly_evaluate_scalar(ArithRing ring, ArithElement *out, ArithElement *in, uint64_t a,
                                    uint64_t num_vars, uint64_t eval_var_idx)
{
    mle_bind(ring, out, in, NULL, a, MLE_STRIDED, 1ULL << (num_vars - 1), eval_var_idx);
}

// Over RNS, off the generic interface, each entry is one sum of products
// (polynomial_matrix_mul_addto_RNS_polynomial with one row and one column),
// reduced once per coefficient where the kernel family allows; elsewhere it
// is a product and k - 1 accumulated products. Both leave every residue
// reduced, so over operands on the same primes they give the same words.
static void linear_combination_range(const MleJob *job, uint64_t begin, uint64_t end)
{
    const ArithRing ring = job->ring;
    const uint64_t k = job->terms;
    if (k == 0)
    {
        for (uint64_t j = begin; j < end; j++)
            arith_zero_in(ring, &job->out[j], arith_mul_domain(ring));
        return;
    }
    if (!job->fused)
    {
        for (uint64_t j = begin; j < end; j++)
        {
            arith_mul(ring, &job->out[j], &job->coeff[0], &job->tables[0][j]);
            for (uint64_t i = 1; i < k; i++)
                arith_mul_addto(ring, &job->out[j], &job->coeff[i], &job->tables[i][j]);
        }
        return;
    }
    RNS_Polynomial *coeff = (RNS_Polynomial *)safe_malloc(2 * k * sizeof(RNS_Polynomial));
    RNS_Polynomial *terms = coeff + k;
    uint64_t coeff_mask = UINT64_MAX;
    for (uint64_t i = 0; i < k; i++)
    {
        coeff[i] = arith_rns_polynomial(&job->coeff[i]);
        coeff_mask &= coeff[i]->rns_mask;
    }
    for (uint64_t j = begin; j < end; j++)
    {
        RNS_Polynomial o = arith_rns_polynomial(&job->out[j]);
        uint64_t mask = o->rns_mask & coeff_mask;
        for (uint64_t i = 0; i < k; i++)
        {
            terms[i] = arith_rns_polynomial(&job->tables[i][j]);
            mask &= terms[i]->rns_mask;
        }
        o->rns_mask = mask;
        polynomial_RNS_zero(o);
        polynomial_matrix_mul_addto_RNS_polynomial(&o, coeff, terms, 1, k, 1);
        job->out[j].domain = ARITH_DOMAIN_MUL;
    }
    free(coeff);
}

ArithStatus mle_dense_poly_linear_combination(ArithRing ring, ArithElement *out,
                                              ArithElement *const *tables,
                                              const ArithElement *coeff, uint64_t k, uint64_t size)
{
    const ArithDomain mul = arith_mul_domain(ring);
    for (uint64_t i = 0; i < k; i++)
    {
        if (coeff[i].domain != mul)
            return ARITH_BAD_DOMAIN;
        for (uint64_t j = 0; j < size; j++)
            if (tables[i][j].domain != mul)
                return ARITH_BAD_DOMAIN;
    }
    MleJob job = {.ring = ring,
                  .out = out,
                  .size = size,
                  .tables = tables,
                  .coeff = coeff,
                  .terms = k,
                  .fused = ring->impl == ARITH_IMPL_RNS && !ARITH_GENERIC_PATH(),
                  .range = linear_combination_range};
    mle_for(&job);
    return ARITH_OK;
}
