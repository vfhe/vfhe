// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>

#include <alloc.h>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vec_chunks.h"

uint64_t double2int(double x) { return ((uint64_t)((int64_t)x)); }

static unsigned char char_rev(unsigned char b)
{
    b = (unsigned char)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4));
    b = (unsigned char)(((b & 0xCC) >> 2) | ((b & 0x33) << 2));
    b = (unsigned char)(((b & 0xAA) >> 1) | ((b & 0x55) << 1));
    return b;
}

uint32_t int_rev(uint32_t b)
{
#if defined(__GNUC__) || defined(__clang__)
    uint32_t a = __builtin_bswap32(b);
#else
    uint32_t a = ((b >> 24) & 0xffu) | ((b >> 8) & 0xff00u) | ((b << 8) & 0xff0000u) | (b << 24);
#endif
    unsigned char *a_vec = (unsigned char *)&a;
    a_vec[0] = char_rev(a_vec[0]);
    a_vec[1] = char_rev(a_vec[1]);
    a_vec[2] = char_rev(a_vec[2]);
    a_vec[3] = char_rev(a_vec[3]);
    return a;
}

void bit_rev(uint64_t *out, uint64_t *in, uint64_t n, uint64_t log_n)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = in[int_rev((uint32_t)i) >> (32 - log_n)];
    }
}

// --- occurrence counts ------------------------------------------------------
//
// On one thread, one pass: prior[i] is the running count of values[i] before it
// is incremented. On several, two passes over chunks of the values: each chunk
// counts its values, the counts become per-chunk starting offsets (the counts
// of every earlier chunk, added in chunk order), and each chunk then runs the
// one-pass loop from its offsets. Every count is an exact integer, so the
// result is the one-pass one.

// The fewest values a chunk covers. A chunk also covers at least
// OCCURRENCE_BOUND_FACTOR * bound values, so that the per-chunk tables of
// `bound` counts, and adding them up, stay small beside the values.
#define OCCURRENCE_GRAIN (1u << 16)
#define OCCURRENCE_BOUND_FACTOR 4

typedef struct
{
    VecChunks chunks;
    uint64_t *prior;
    const uint64_t *values;
    uint64_t bound;
    uint64_t *counts; // `bound` per chunk
    uint64_t *valid;  // one flag per chunk
} OccurrenceJob;

static void occurrence_count_chunk(void *ctx, uint64_t chunk)
{
    const OccurrenceJob *job = (const OccurrenceJob *)ctx;
    const uint64_t begin = vec_chunk_begin(&job->chunks, chunk);
    const uint64_t end = vec_chunk_end(&job->chunks, chunk);
    const uint64_t bound = job->bound;
    uint64_t *counts = job->counts + chunk * bound;
    uint64_t valid = 1;
    memset(counts, 0, bound * sizeof(uint64_t));
    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t v = job->values[i];
        if (v >= bound)
        {
            valid = 0;
            break;
        }
        counts[v]++;
    }
    job->valid[chunk] = valid;
}

static void occurrence_prior_chunk(void *ctx, uint64_t chunk)
{
    const OccurrenceJob *job = (const OccurrenceJob *)ctx;
    const uint64_t begin = vec_chunk_begin(&job->chunks, chunk);
    const uint64_t end = vec_chunk_end(&job->chunks, chunk);
    uint64_t *running = job->counts + chunk * job->bound;
    for (uint64_t i = begin; i < end; i++)
        job->prior[i] = running[job->values[i]]++;
}

bool vec_occurrence_counts(uint64_t *prior, uint64_t *totals, const uint64_t *values,
                           uint64_t count, uint64_t bound)
{
    OccurrenceJob job = {.prior = prior, .values = values, .bound = bound};
    uint64_t grain = OCCURRENCE_BOUND_FACTOR * bound;
    if (grain < OCCURRENCE_GRAIN)
        grain = OCCURRENCE_GRAIN;
    if (!vec_chunks_plan(&job.chunks, count, grain, 8))
    {
        memset(totals, 0, bound * sizeof(uint64_t));
        for (uint64_t i = 0; i < count; i++)
        {
            const uint64_t v = values[i];
            if (v >= bound)
                return false;
            prior[i] = totals[v]++;
        }
        return true;
    }

    const uint64_t chunks = job.chunks.count;
    job.counts = (uint64_t *)safe_malloc(chunks * bound * sizeof(uint64_t));
    job.valid = (uint64_t *)safe_malloc(chunks * sizeof(uint64_t));
    vfhe_parallel_for(chunks, 0, occurrence_count_chunk, &job);
    bool valid = true;
    for (uint64_t c = 0; c < chunks; c++)
        valid = valid && job.valid[c];
    if (valid)
    {
        for (uint64_t x = 0; x < bound; x++)
        {
            uint64_t running = 0;
            for (uint64_t c = 0; c < chunks; c++)
            {
                uint64_t *slot = job.counts + c * bound + x;
                const uint64_t here = *slot;
                *slot = running;
                running += here;
            }
            totals[x] = running;
        }
        vfhe_parallel_for(chunks, 0, occurrence_prior_chunk, &job);
    }
    free(job.valid);
    free(job.counts);
    return valid;
}
