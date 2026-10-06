// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// An RNS polynomial as bytes, and an RNS polynomial from a seed.
#include "arith.h"
#include <alloc.h>
#include <parallel.h>
#include <crypto.h>
#include <string.h>

#include "arith_internal.h"

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the row encoding copies words as they are in memory, which assumes a little-endian host"
#endif

// The derive-key context of the seed expansion. Stored seeds depend on it:
// never change it, define a new one.
static const char SEED_CONTEXT[] = "vfhe 2026-10-04 RNS polynomial mul-domain row from a seed";

// Calls below these sizes run on one thread, where starting threads would
// cost more than the work. Expansion is compute-bound, copying memory-bound.
#define EXPAND_PARALLEL_BYTES (1u << 20)
#define COPY_PARALLEL_BYTES (16u << 20)

static void row_key(uint8_t key[16], const uint8_t *seed, uint64_t seed_len, uint64_t stream,
                    uint64_t q)
{
    const uint64_t label[2] = {stream, q};
    prng_expand_key(key, SEED_CONTEXT, seed, seed_len, label, 2);
}

typedef struct
{
    RNS_Polynomial out;
    const uint64_t *rows; // base indices to fill
    const uint8_t *seed;
    uint64_t seed_len, stream;
} ExpandJob;

static void expand_row(void *ctx, uint64_t r)
{
    const ExpandJob *job = (const ExpandJob *)ctx;
    const RNS_Polynomial out = job->out;
    const uint64_t i = job->rows[r];
    const uint64_t q = out->base->mods[i]->q;
    uint8_t key[16];
    row_key(key, job->seed, job->seed_len, job->stream, q);
    if (rns_row_is_narrow(out->base, i))
        prng_expand_below32(rns_row32(out, i), out->base->N, 0, q, key);
    else
        prng_expand_below(rns_row64(out, i), out->base->N, 0, q, key);
}

// The active base indices of `p`, ascending; returns how many.
static uint64_t active_rows(uint64_t *rows, RNS_Polynomial p)
{
    uint64_t count = 0;
    for (uint64_t i = 0; i < p->base->l; i++)
        if (p->rns_mask & (1ULL << i))
            rows[count++] = i;
    return count;
}

void polynomial_RNS_expand_seeded(RNS_Polynomial out, const uint8_t *seed, uint64_t seed_len,
                                  uint64_t stream)
{
    uint64_t rows[64];
    const uint64_t count = active_rows(rows, out);
    ExpandJob job = {out, rows, seed, seed_len, stream};
    const bool wide = count * out->base->N * sizeof(uint64_t) >= EXPAND_PARALLEL_BYTES;
    vfhe_parallel_for(count, wide ? 0 : 1, expand_row, &job);
}

bool polynomial_RNS_matches_seeded(RNS_Polynomial p, bool canonical, uint64_t index,
                                   const uint8_t *seed, uint64_t seed_len, uint64_t stream)
{
    const RNS_Base base = p->base;
    const uint64_t N = base->N;
    const uint64_t q = base->mods[index]->q;
    const uint64_t poly_size = N / base->split_degree;
    uint8_t key[16];
    row_key(key, seed, seed_len, stream, q);
    bool same;
    if (rns_row_is_narrow(base, index))
    {
        uint32_t *row = (uint32_t *)safe_aligned_malloc(N * sizeof(uint32_t));
        prng_expand_below32(row, N, 0, q, key);
        if (canonical)
            for (uint64_t k = 0; k < base->split_degree; k++)
                ntt_reverse_w32(&row[k * poly_size], &row[k * poly_size], base->plans[index]);
        same = memcmp(row, rns_row32(p, index), N * sizeof(uint32_t)) == 0;
        free(row);
    }
    else
    {
        uint64_t *row = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
        prng_expand_below(row, N, 0, q, key);
        if (canonical)
            for (uint64_t k = 0; k < base->split_degree; k++)
                ntt_reverse(&row[k * poly_size], &row[k * poly_size], base->plans[index]);
        same = memcmp(row, rns_row64(p, index), N * sizeof(uint64_t)) == 0;
        free(row);
    }
    return same;
}

// --- Rows as bytes -------------------------------------------------------

// Bits a residue modulo q needs: those of q - 1.
static unsigned residue_bits(uint64_t q)
{
    return q <= 1 ? 1 : 64 - (unsigned)__builtin_clzll(q - 1);
}

static size_t word_bytes(uint64_t q) { return q <= (1ULL << 32) ? 4 : 8; }

uint64_t rns_row_bytes(uint64_t q, uint64_t N, bool tight)
{
    if (!tight)
        return N * word_bytes(q);
    return (N * residue_bits(q) + 63) / 64 * 8;
}

typedef struct
{
    RNS_Polynomial p;
    uint8_t *bytes;
    const uint64_t *rows;    // base indices, in the order of the encoding
    const uint64_t *offsets; // where each row starts in `bytes`
    bool tight, validate;
    int64_t *bad; // read: the first row holding a value >= q, or -1
} RowsJob;

static inline void store64(uint8_t *at, uint64_t v) { memcpy(at, &v, sizeof v); }

static inline uint64_t load64(const uint8_t *at)
{
    uint64_t v;
    memcpy(&v, at, sizeof v);
    return v;
}

// Value k of row i, at either width. A per-value branch is cheap next to
// the bit packing that calls this.
static inline uint64_t row_get(RNS_Polynomial p, uint64_t i, uint64_t k)
{
    return rns_row_is_narrow(p->base, i) ? p->rows32[i][k] : p->rows64[i][k];
}

static void write_row(void *ctx, uint64_t r)
{
    const RowsJob *job = (const RowsJob *)ctx;
    const RNS_Polynomial p = job->p;
    const uint64_t i = job->rows[r];
    const uint64_t N = p->base->N;
    const uint64_t q = p->base->mods[i]->q;
    uint8_t *out = job->bytes + job->offsets[r];
    const bool narrow = rns_row_is_narrow(p->base, i);
    if (!job->tight)
    {
        if (word_bytes(q) == 8)
            memcpy(out, rns_row64(p, i), N * 8);
        else if (narrow)
            memcpy(out, rns_row32(p, i), N * 4);
        else
        {
            const uint64_t *row = rns_row64(p, i);
            for (uint64_t k = 0; k < N; k++)
            {
                const uint32_t v = (uint32_t)row[k];
                memcpy(&out[4 * k], &v, 4);
            }
        }
        return;
    }
    const unsigned b = residue_bits(q);
    uint64_t acc = 0;
    unsigned have = 0;
    for (uint64_t k = 0; k < N; k++)
    {
        const uint64_t v = row_get(p, i, k);
        acc |= v << have;
        have += b;
        if (have >= 64)
        {
            store64(out, acc);
            out += 8;
            have -= 64;
            acc = have ? v >> (b - have) : 0;
        }
    }
    if (have)
        store64(out, acc);
}

static void read_row(void *ctx, uint64_t r)
{
    const RowsJob *job = (const RowsJob *)ctx;
    const RNS_Polynomial p = job->p;
    const uint64_t i = job->rows[r];
    const uint64_t N = p->base->N;
    const uint64_t q = p->base->mods[i]->q;
    const uint8_t *in = job->bytes + job->offsets[r];
    const bool narrow = rns_row_is_narrow(p->base, i);
    if (!job->tight && word_bytes(q) == 8)
        memcpy(rns_row64(p, i), in, N * 8);
    else if (!job->tight && narrow)
        memcpy(rns_row32(p, i), in, N * 4);
    else if (!job->tight)
    {
        uint64_t *row = rns_row64(p, i);
        for (uint64_t k = 0; k < N; k++)
        {
            uint32_t v;
            memcpy(&v, &in[4 * k], 4);
            row[k] = v;
        }
    }
    else
    {
        const unsigned b = residue_bits(q);
        const uint64_t mask = b == 64 ? ~0ULL : (1ULL << b) - 1;
        for (uint64_t k = 0; k < N; k++)
        {
            const uint64_t bit = k * b;
            const uint64_t word = bit / 64;
            const unsigned off = (unsigned)(bit % 64);
            uint64_t v = load64(&in[8 * word]) >> off;
            if (off + b > 64)
                v |= load64(&in[8 * (word + 1)]) << (64 - off);
            v &= mask;
            if (narrow)
                p->rows32[i][k] = (uint32_t)v;
            else
                p->rows64[i][k] = v;
        }
    }
    if (!job->validate)
        return;
    for (uint64_t k = 0; k < N; k++)
    {
        if (row_get(p, i, k) >= q)
        {
            // Rows run concurrently: keep the smallest offending row.
            int64_t seen = __atomic_load_n(job->bad, __ATOMIC_RELAXED);
            while ((seen < 0 || (int64_t)r < seen) &&
                   !__atomic_compare_exchange_n(job->bad, &seen, (int64_t)r, false,
                                                __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            {
            }
            return;
        }
    }
}

// Fills each row's offset and returns the threads to copy them on.
static uint64_t row_offsets(uint64_t *offsets, RNS_Polynomial p, const uint64_t *rows,
                            uint64_t count, bool tight)
{
    uint64_t at = 0;
    for (uint64_t r = 0; r < count; r++)
    {
        offsets[r] = at;
        at += rns_row_bytes(p->base->mods[rows[r]]->q, p->base->N, tight);
    }
    return at >= COPY_PARALLEL_BYTES ? 0 : 1;
}

void polynomial_RNS_write_rows(uint8_t *out, RNS_Polynomial p, const uint64_t *rows, uint64_t count,
                               bool tight)
{
    uint64_t offsets[64];
    const uint64_t threads = row_offsets(offsets, p, rows, count, tight);
    RowsJob job = {p, out, rows, offsets, tight, false, NULL};
    vfhe_parallel_for(count, threads, write_row, &job);
}

int64_t polynomial_RNS_read_rows(RNS_Polynomial p, const uint8_t *in, const uint64_t *rows,
                                 uint64_t count, bool tight, bool validate)
{
    uint64_t offsets[64];
    const uint64_t threads = row_offsets(offsets, p, rows, count, tight);
    int64_t bad = -1;
    RowsJob job = {p, (uint8_t *)in, rows, offsets, tight, validate, &bad};
    vfhe_parallel_for(count, threads, read_row, &job);
    return bad;
}
