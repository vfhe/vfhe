// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// The Reed-Solomon code over an RNS ring. The transform runs per prime and the
// codeword is gathered coefficient by coefficient, so the whole file is tied to
// the representation; only the signatures are generic, so a field version can
// sit beside it under the same entry points.
#include "rscode.h"
// RNS row width and accessors: this file is the RNS backend of the code.
#include "arith_internal.h"
#include <arith_generic.h>

#include <string.h>

#include "alloc.h"
#include "crypto.h"
#include "parallel.h"

// -------------------------------------------------------------
// Reed-Solomon (foldable) code over R_q, per RNS prime
// -------------------------------------------------------------
// The transform itself is arith's negacyclic NTT of the codeword length:
// gather one (prime, coefficient slot) column of the message vector into a
// scratch buffer, zero-pad it to the codeword length, transform, scatter the
// result back into the same column of the output vector. See rscode.h for the
// evaluation-point layout the fold depends on.

NTT_Plan *rs_new_plans(RNS_Base base, uint64_t rns_mask, uint64_t size)
{
    const uint64_t rows = (uint64_t)(rns_mask_get_last_active_index(rns_mask) + 1);
    NTT_Plan *plans = (NTT_Plan *)safe_malloc(sizeof(NTT_Plan) * rows);
    for (uint64_t i = 0; i < rows; i++)
    {
        // Borrow the RNS_Base's modulus: one set of Barrett constants per prime,
        // shared by this code's plan at every level.
        plans[i] = (rns_mask & (1ULL << i)) ? ntt_new_plan(size, base->mods[i]) : NULL;
    }
    return plans;
}

void rs_free_plans(NTT_Plan *plans, uint64_t count)
{
    for (uint64_t i = 0; i < count; i++)
    {
        if (plans[i] != NULL)
            ntt_free_plan(plans[i]);
    }
    free(plans);
}

uint64_t rs_plans_root(NTT_Plan *plans, uint64_t index)
{
    return plans[index] == NULL ? 0 : plans[index]->root_of_unity;
}

// Codeword words a chunk of columns transforms at least when a transform runs
// on several threads, so that a chunk's gather, transform and scatter outweigh
// handing it to a thread.
#define RS_PARALLEL_WORDS (1u << 16)
// Chunks per thread, so that threads finishing early take another one.
#define RS_CHUNKS_PER_THREAD 4
// A chunk's columns are a multiple of this many coefficient slots, so two
// chunks never write into one 64-byte line of an output row.
#define RS_COLUMN_ALIGN 16

// The column of prime `i` (base index), coefficient slot `j`: in[0..degree)
// zero-padded to `size`, transformed, written to out[0..size). `codeword`
// holds `size` words.
static inline void encode_column(ArithElement *out, ArithElement *in, uint64_t size,
                                 uint64_t degree, NTT_Plan plan, uint64_t i, uint64_t j,
                                 bool narrow, uint64_t *codeword)
{
    memset(&codeword[degree], 0, sizeof(uint64_t) * (size - degree));
    if (narrow)
        for (uint64_t k = 0; k < degree; k++)
            codeword[k] = arith_rns_polynomial(&in[k])->rows32[i][j];
    else
        for (uint64_t k = 0; k < degree; k++)
            codeword[k] = arith_rns_polynomial(&in[k])->rows64[i][j];
    ntt_forward(codeword, codeword, plan);
    if (narrow)
        for (uint64_t k = 0; k < size; k++)
            arith_rns_polynomial(&out[k])->rows32[i][j] = (uint32_t)codeword[k];
    else
        for (uint64_t k = 0; k < size; k++)
            arith_rns_polynomial(&out[k])->rows64[i][j] = codeword[k];
}

// The inverse of encode_column, writing the message to out[0..degree) when
// `out` is non-NULL; returns whether the coefficients above `degree` vanish.
static inline int decode_column(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree,
                                NTT_Plan plan, uint64_t i, uint64_t j, bool narrow,
                                uint64_t *codeword)
{
    if (narrow)
        for (uint64_t k = 0; k < size; k++)
            codeword[k] = arith_rns_polynomial(&in[k])->rows32[i][j];
    else
        for (uint64_t k = 0; k < size; k++)
            codeword[k] = arith_rns_polynomial(&in[k])->rows64[i][j];
    ntt_reverse(codeword, codeword, plan);
    if (out != NULL)
    {
        if (narrow)
            for (uint64_t k = 0; k < degree; k++)
                arith_rns_polynomial(&out[k])->rows32[i][j] = (uint32_t)codeword[k];
        else
            for (uint64_t k = 0; k < degree; k++)
                arith_rns_polynomial(&out[k])->rows64[i][j] = codeword[k];
    }
    // The degree check: a codeword of this code inverts to a message
    // that was zero-padded above `degree`.
    for (uint64_t k = degree; k < size; k++)
        if (codeword[k] != 0)
            return 0;
    return 1;
}

// The columns of a transform on several threads. Column c is slot c % N of
// the (c / N)-th active prime; a chunk is a run of consecutive columns.
typedef struct
{
    ArithElement *out, *in;
    uint64_t size, degree;
    NTT_Plan *plans;
    RNS_Base base;
    const uint64_t *rows; // the active base indices, ascending
    uint64_t columns, per_chunk;
    int *is_codeword; // decode only: one flag per chunk
} ColumnsJob;

// How many chunks a transform of `columns` columns of `size` words is split
// into, with `per_chunk` columns each; 1 when it runs on the calling thread.
static uint64_t column_chunks(uint64_t columns, uint64_t size, uint64_t *per_chunk)
{
    const uint64_t most_chunks = columns * size / RS_PARALLEL_WORDS;
    const uint64_t threads = vfhe_threads_for(0, most_chunks);
    if (threads <= 1)
        return 1;
    uint64_t chunks = threads * RS_CHUNKS_PER_THREAD;
    if (chunks > most_chunks)
        chunks = most_chunks;
    uint64_t per = (columns + chunks - 1) / chunks;
    per = (per + RS_COLUMN_ALIGN - 1) / RS_COLUMN_ALIGN * RS_COLUMN_ALIGN;
    *per_chunk = per;
    return (columns + per - 1) / per;
}

static void encode_chunk(void *ctx, uint64_t c)
{
    const ColumnsJob *job = (const ColumnsJob *)ctx;
    const uint64_t N = job->base->N;
    const uint64_t first = c * job->per_chunk;
    const uint64_t last =
        first + job->per_chunk < job->columns ? first + job->per_chunk : job->columns;
    uint64_t *codeword = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * job->size);
    for (uint64_t col = first; col < last; col++)
    {
        const uint64_t i = job->rows[col / N];
        encode_column(job->out, job->in, job->size, job->degree, job->plans[i], i, col % N,
                      rns_row_is_narrow(job->base, i), codeword);
    }
    free(codeword);
}

static void decode_chunk(void *ctx, uint64_t c)
{
    const ColumnsJob *job = (const ColumnsJob *)ctx;
    const uint64_t N = job->base->N;
    const uint64_t first = c * job->per_chunk;
    const uint64_t last =
        first + job->per_chunk < job->columns ? first + job->per_chunk : job->columns;
    uint64_t *codeword = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * job->size);
    int ok = 1;
    for (uint64_t col = first; col < last && ok; col++)
    {
        const uint64_t i = job->rows[col / N];
        ok = decode_column(job->out, job->in, job->size, job->degree, job->plans[i], i, col % N,
                           rns_row_is_narrow(job->base, i), codeword);
    }
    job->is_codeword[c] = ok;
    free(codeword);
}

// The active base indices of `rns_mask` below `l`, ascending; returns how many.
static uint64_t active_rows(uint64_t *rows, uint64_t rns_mask, uint64_t l)
{
    uint64_t count = 0;
    for (uint64_t i = 0; i < l; i++)
        if (rns_mask & (1ULL << i))
            rows[count++] = i;
    return count;
}

void rs_encode(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree, NTT_Plan *plans)
{
    RNS_Base base = arith_rns_polynomial(&in[0])->base;
    const uint64_t N = base->N;
    const uint64_t rns_mask = arith_rns_polynomial(&in[0])->rns_mask;

    for (uint64_t k = 0; k < size; k++)
    {
        arith_rns_polynomial(&out[k])->rns_mask = rns_mask;
    }

    // Every (prime, slot) column is an independent codeword.
    uint64_t rows[64], per_chunk = 0;
    const uint64_t columns = active_rows(rows, rns_mask, base->l) * N;
    const uint64_t chunks = column_chunks(columns, size, &per_chunk);
    if (chunks > 1)
    {
        ColumnsJob job = {out, in, size, degree, plans, base, rows, columns, per_chunk, NULL};
        vfhe_parallel_for(chunks, 0, encode_chunk, &job);
        return;
    }

    uint64_t *codeword = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * size);
    for (uint64_t i = 0; i < base->l; i++)
    {
        if (!(rns_mask & (1ULL << i)))
            continue;
        // one width test per prime; the gather and scatter are typed
        const bool narrow = rns_row_is_narrow(base, i);
        for (uint64_t j = 0; j < N; j++)
            encode_column(out, in, size, degree, plans[i], i, j, narrow, codeword);
    }
    free(codeword);
}

int rs_decode(ArithElement *out, ArithElement *in, uint64_t size, uint64_t degree, NTT_Plan *plans)
{
    RNS_Base base = arith_rns_polynomial(&in[0])->base;
    const uint64_t N = base->N;
    const uint64_t rns_mask = arith_rns_polynomial(&in[0])->rns_mask;

    if (out != NULL)
    {
        for (uint64_t k = 0; k < degree; k++)
        {
            arith_rns_polynomial(&out[k])->rns_mask = rns_mask;
        }
    }

    uint64_t rows[64], per_chunk = 0;
    const uint64_t columns = active_rows(rows, rns_mask, base->l) * N;
    const uint64_t chunks = column_chunks(columns, size, &per_chunk);
    if (chunks > 1)
    {
        // A chunk stops at its first column that fails the degree check; the
        // answer is whether every chunk passed.
        int *is_codeword = (int *)safe_malloc(sizeof(int) * chunks);
        ColumnsJob job = {out,  in,   size,    degree,    plans,
                          base, rows, columns, per_chunk, is_codeword};
        vfhe_parallel_for(chunks, 0, decode_chunk, &job);
        int all = 1;
        for (uint64_t c = 0; c < chunks; c++)
            all &= is_codeword[c];
        free(is_codeword);
        return all;
    }

    uint64_t *codeword = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * size);
    int is_codeword = 1;
    for (uint64_t i = 0; i < base->l && is_codeword; i++)
    {
        if (!(rns_mask & (1ULL << i)))
            continue;
        const bool narrow = rns_row_is_narrow(base, i);
        for (uint64_t j = 0; j < N && is_codeword; j++)
            is_codeword = decode_column(out, in, size, degree, plans[i], i, j, narrow, codeword);
    }
    free(codeword);
    return is_codeword;
}

// -------------------------------------------------------------
// Merkle leaf digests of a codeword over R_q
// -------------------------------------------------------------
// A leaf is `size` adjacent symbols, and its digest is the BLAKE3 digest of
// the symbols' own digests concatenated in order, a symbol's digest being
// polynomial_RNS_get_hash's. The symbols' digests are staged a block at a
// time and the block's leaves hashed from them with one hash_batch call.

// Bytes of a digest, as hash_stream_digest and hash_batch write it.
#define DIGEST_LEN 32
// Symbols whose digests a block stages: 128 KiB of digests.
#define LEAF_BLOCK_SYMBOLS 4096
// Symbol bytes a chunk of leaves hashes at least when the digests run on
// several threads, so that a chunk's hashing outweighs handing it to a thread.
#define LEAF_PARALLEL_BYTES (256u << 10)
// Chunks per thread, so that threads finishing early take another one.
#define LEAF_CHUNKS_PER_THREAD 4

// The digests of leaves [first, last).
static void leaf_digests_range(uint8_t *out, ArithElement *word, uint64_t leaf_size, uint64_t first,
                               uint64_t last)
{
    const uint64_t block = leaf_size < LEAF_BLOCK_SYMBOLS ? LEAF_BLOCK_SYMBOLS / leaf_size : 1;
    uint8_t *staged = (uint8_t *)safe_malloc(block * leaf_size * DIGEST_LEN);
    void *state = safe_malloc(hash_stream_size());
    for (uint64_t leaf = first; leaf < last; leaf += block)
    {
        const uint64_t leaves = last - leaf < block ? last - leaf : block;
        for (uint64_t s = 0; s < leaves * leaf_size; s++)
        {
            hash_stream_init(state);
            polynomial_RNS_hash_update(state, arith_rns_polynomial(&word[leaf * leaf_size + s]));
            hash_stream_digest(state, &staged[s * DIGEST_LEN]);
        }
        hash_batch(&out[leaf * DIGEST_LEN], staged, leaves, leaf_size * DIGEST_LEN);
    }
    free(state);
    free(staged);
}

typedef struct
{
    uint8_t *out;
    ArithElement *word;
    uint64_t leaf_size, leaves, per_chunk;
} LeavesJob;

static void leaf_digests_chunk(void *ctx, uint64_t c)
{
    const LeavesJob *job = (const LeavesJob *)ctx;
    const uint64_t first = c * job->per_chunk;
    const uint64_t last =
        first + job->per_chunk < job->leaves ? first + job->per_chunk : job->leaves;
    leaf_digests_range(job->out, job->word, job->leaf_size, first, last);
}

void rs_leaf_digests(uint8_t *out, ArithElement *word, uint64_t length, uint64_t leaf_size)
{
    const uint64_t leaves = length / leaf_size;
    if (leaves == 0)
        return;
    // The symbols of a codeword share a ring, so the first one's rows give the
    // bytes each symbol hashes.
    RNS_Polynomial first = arith_rns_polynomial(&word[0]);
    uint64_t symbol_bytes = 0;
    for (uint64_t i = 0; i < first->base->l; i++)
        if (first->rns_mask & (1ULL << i))
            symbol_bytes += first->base->N * (rns_row_is_narrow(first->base, i) ? sizeof(uint32_t)
                                                                                : sizeof(uint64_t));
    const uint64_t most_chunks = leaves * leaf_size * symbol_bytes / LEAF_PARALLEL_BYTES;
    const uint64_t threads = vfhe_threads_for(0, most_chunks);
    if (threads <= 1)
    {
        leaf_digests_range(out, word, leaf_size, 0, leaves);
        return;
    }
    uint64_t chunks = threads * LEAF_CHUNKS_PER_THREAD;
    if (chunks > most_chunks)
        chunks = most_chunks;
    // An even number of leaves per chunk: two chunks never write into one
    // 64-byte line of `out`.
    uint64_t per_chunk = (leaves + chunks - 1) / chunks;
    per_chunk += per_chunk & 1;
    LeavesJob job = {out, word, leaf_size, leaves, per_chunk};
    vfhe_parallel_for((leaves + per_chunk - 1) / per_chunk, 0, leaf_digests_chunk, &job);
}
