// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// The Reed-Solomon code over an extension field F_p[x]/(x^d - w), on the
// coefficient planes of a FieldVector. The evaluation points lie in F_p, so the
// transform is F_p-linear and runs once per plane with one NTT_Plan over the
// field's modulus; a plane is exactly the uint64_t array arith's NTT takes.
#include "rscode.h"

#include <string.h>

#include "alloc.h"
#include "parallel.h"

NTT_Plan rs_field_new_plan(uint64_t size, Modulus mod) { return ntt_new_plan(size, mod); }

void rs_field_free_plan(NTT_Plan plan) { ntt_free_plan(plan); }

uint64_t rs_field_plan_root(NTT_Plan plan) { return plan->root_of_unity; }

// Codeword words a chunk of planes transforms at least when a transform runs
// on several threads: a few hundred microseconds of transform per chunk, well
// above what handing a chunk to a thread costs.
#define RS_FIELD_PARALLEL_WORDS (1u << 16)

// Plane j of the codeword of `in`; `scratch` holds out->n words.
static inline void encode_plane(FieldVector out, const FieldVector in, NTT_Plan plan, uint64_t j,
                                uint64_t *scratch)
{
    const uint64_t size = out->n;
    const uint64_t degree = in->n;
    memcpy(scratch, in->coeffs[j], sizeof(uint64_t) * degree);
    memset(&scratch[degree], 0, sizeof(uint64_t) * (size - degree));
    ntt_forward(scratch, scratch, plan);
    memcpy(out->coeffs[j], scratch, sizeof(uint64_t) * size);
}

// Plane j of the decoding of `in`, written to `out` when non-NULL; returns
// whether its coefficients above `degree` vanish. `scratch` holds in->n
// words.
static inline int decode_plane(FieldVector out, const FieldVector in, NTT_Plan plan, uint64_t j,
                               uint64_t degree, uint64_t *scratch)
{
    const uint64_t size = in->n;
    memcpy(scratch, in->coeffs[j], sizeof(uint64_t) * size);
    ntt_reverse(scratch, scratch, plan);
    if (out != NULL)
    {
        memcpy(out->coeffs[j], scratch, sizeof(uint64_t) * degree);
    }
    // The degree check: a codeword of this code inverts to a message
    // that was zero-padded above `degree`.
    for (uint64_t k = degree; k < size; k++)
        if (scratch[k] != 0)
            return 0;
    return 1;
}

// The planes of a transform on several threads; a chunk is a run of
// consecutive planes.
typedef struct
{
    FieldVector out;
    FieldVector in;
    NTT_Plan plan;
    uint64_t degree, per_chunk;
    int *is_codeword; // decode only: one flag per chunk
} PlanesJob;

// How many chunks a transform of `planes` planes of `size` words is split
// into, with `per_chunk` planes each; 1 when it runs on the calling thread.
static uint64_t plane_chunks(uint64_t planes, uint64_t size, uint64_t *per_chunk)
{
    if (planes < 2)
        return 1;
    uint64_t per = (RS_FIELD_PARALLEL_WORDS + size - 1) / size;
    if (per > planes)
        per = planes;
    const uint64_t chunks = (planes + per - 1) / per;
    if (vfhe_threads_for(0, chunks) <= 1)
        return 1;
    *per_chunk = per;
    return chunks;
}

static void encode_chunk(void *ctx, uint64_t c)
{
    const PlanesJob *job = (const PlanesJob *)ctx;
    const uint64_t first = c * job->per_chunk;
    const uint64_t last = first + job->per_chunk < job->in->d ? first + job->per_chunk : job->in->d;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * job->out->n);
    for (uint64_t j = first; j < last; j++)
        encode_plane(job->out, job->in, job->plan, j, scratch);
    free(scratch);
}

static void decode_chunk(void *ctx, uint64_t c)
{
    const PlanesJob *job = (const PlanesJob *)ctx;
    const uint64_t first = c * job->per_chunk;
    const uint64_t last = first + job->per_chunk < job->in->d ? first + job->per_chunk : job->in->d;
    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * job->in->n);
    int ok = 1;
    for (uint64_t j = first; j < last && ok; j++)
        ok = decode_plane(job->out, job->in, job->plan, j, job->degree, scratch);
    job->is_codeword[c] = ok;
    free(scratch);
}

void rs_field_encode(FieldVector out, const FieldVector in, NTT_Plan plan)
{
    // The planes are independent codewords.
    uint64_t per_chunk = 0;
    const uint64_t chunks = plane_chunks(in->d, out->n, &per_chunk);
    if (chunks > 1)
    {
        PlanesJob job = {out, in, plan, in->n, per_chunk, NULL};
        vfhe_parallel_for(chunks, 0, encode_chunk, &job);
        return;
    }

    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * out->n);
    for (uint64_t j = 0; j < in->d; j++)
        encode_plane(out, in, plan, j, scratch);
    free(scratch);
}

int rs_field_decode(FieldVector out, const FieldVector in, NTT_Plan plan)
{
    const uint64_t degree = out == NULL ? 0 : out->n;
    uint64_t per_chunk = 0;
    const uint64_t chunks = plane_chunks(in->d, in->n, &per_chunk);
    if (chunks > 1)
    {
        // A chunk stops at its first plane that fails the degree check; the
        // answer is whether every chunk passed.
        int *is_codeword = (int *)safe_malloc(sizeof(int) * chunks);
        PlanesJob job = {out, in, plan, degree, per_chunk, is_codeword};
        vfhe_parallel_for(chunks, 0, decode_chunk, &job);
        int all = 1;
        for (uint64_t c = 0; c < chunks; c++)
            all &= is_codeword[c];
        free(is_codeword);
        return all;
    }

    uint64_t *scratch = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * in->n);
    int is_codeword = 1;
    for (uint64_t j = 0; j < in->d && is_codeword; j++)
        is_codeword = decode_plane(out, in, plan, j, degree, scratch);
    free(scratch);
    return is_codeword;
}
