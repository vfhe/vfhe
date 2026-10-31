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

// The derive-key context of the seed expansion, part of the seeded-row
// format: a stored seed expands to the same rows only under this context.
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

// Garner's mixed radix: c = a_0 + a_1 Q_1 + ... + a_(L-1) Q_(L-1), with
// Q_r = q_0 ... q_(r-1) and 0 <= a_r < q_r, is the c in [0, q) with the given
// residues, and a_r = (c_r - (a_0 Q_0 + ... + a_(r-1) Q_(r-1))) Q_r^-1 mod q_r.
// Each a_r is one row computed from the rows before it: the sum is a lifted
// combination of those rows with the constants Q_k mod q_r, then a subtract
// and a scale by Q_r^-1, all modulo q_r through the element-wise kernels. The
// sum of the a_r Q_r is then a combination of the rows modulo 2^128, where it
// wraps.
typedef struct
{
    RNS_Polynomial *in;
    uint64_t *low, *high;
    const uint64_t *rows;                     // the base indices of the active primes
    const uint64_t *q;                        // their values
    const uint64_t *prefix;                   // prefix[r * L + k] = Q_k mod q_r, k < r
    const uint64_t *inverse;                  // inverse[r] = Q_r^-1 mod q_r
    const uint64_t *weight_low, *weight_high; // Q_r mod 2^128, as two words
    uint64_t L;
} LiftJob;

// Row `i` of `p` into `out` in natural coefficient order: coefficient u + s t
// sits at u * block + t, s the splitting degree and block = N / s.
static void natural_row(uint64_t *out, RNS_Polynomial p, uint64_t i)
{
    const uint64_t N = p->base->N, s = p->base->split_degree, block = N / s;
    if (rns_row_is_narrow(p->base, i))
    {
        const uint32_t *row = rns_row32(p, i);
        for (uint64_t u = 0; u < s; u++)
            for (uint64_t t = 0; t < block; t++)
                out[u + s * t] = row[u * block + t];
    }
    else if (s == 1)
        memcpy(out, rns_row64(p, i), N * sizeof(uint64_t));
    else
    {
        const uint64_t *row = rns_row64(p, i);
        for (uint64_t u = 0; u < s; u++)
            for (uint64_t t = 0; t < block; t++)
                out[u + s * t] = row[u * block + t];
    }
}

static void lift_polynomial(void *ctx, uint64_t e)
{
    const LiftJob *job = (const LiftJob *)ctx;
    const RNS_Polynomial p = job->in[e];
    const uint64_t N = p->base->N, L = job->L;
    // Aligned: the element-wise kernels load whole 64-byte vectors.
    uint64_t *a = (uint64_t *)safe_aligned_malloc(L * N * sizeof(uint64_t));
    uint64_t *residues = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));
    const uint64_t *terms[MOD_COMBINATION_MAX_TERMS];
    natural_row(a, p, job->rows[0]);
    for (uint64_t r = 1; r < L; r++)
    {
        const Modulus mod = p->base->mods[job->rows[r]];
        uint64_t *a_r = a + r * N;
        for (uint64_t k = 0; k < r; k++)
            terms[k] = a + k * N;
        const uint64_t *scale = job->prefix + r * L;
        if (!mod_eltwise_lifted_combination(a_r, terms, job->q, scale, r, false, N, mod))
            mod_eltwise_lifted_combination_gen(a_r, terms, job->q, scale, r, false, N, mod);
        natural_row(residues, p, job->rows[r]);
        mod_eltwise_sub(a_r, residues, a_r, N, mod);
        mod_eltwise_scale(a_r, a_r, job->inverse[r], N, mod);
    }
    for (uint64_t r = 0; r < L; r++)
        terms[r] = a + r * N;
    vec_combination_mod_2_128(job->low + e * N, job->high + e * N, terms, job->weight_low,
                              job->weight_high, L, N);
    free(residues);
    free(a);
}

void polynomial_RNS_lift_mod_2_128(uint64_t *low, uint64_t *high, RNS_Polynomial *in,
                                   uint64_t count)
{
    if (count == 0)
        return;
    const RNS_Base base = in[0]->base;
    for (uint64_t e = 0; e < count; e++)
        assert(in[e]->base == base && in[e]->rns_mask == in[0]->rns_mask);
    uint64_t rows[64], q[64];
    const uint64_t L = active_rows(rows, in[0]);
    assert(L <= MOD_COMBINATION_MAX_TERMS);
    for (uint64_t r = 0; r < L; r++)
        q[r] = base->mods[rows[r]]->q;
    uint64_t prefix[64 * 64], inverse[64], weight_low[64], weight_high[64];
    // Q_r mod 2^128 as two words: one 64 x 64 -> 128 product per prime.
    weight_low[0] = 1;
    weight_high[0] = 0;
    for (uint64_t k = 1; k < L; k++)
    {
        const unsigned __int128 low_product = (unsigned __int128)weight_low[k - 1] * q[k - 1];
        weight_low[k] = (uint64_t)low_product;
        weight_high[k] = (uint64_t)(low_product >> 64) + weight_high[k - 1] * q[k - 1];
    }
    for (uint64_t r = 0; r < L; r++)
    {
        const Modulus mod = base->mods[rows[r]];
        uint64_t product = modq(1, mod);
        for (uint64_t k = 0; k < r; k++)
        {
            prefix[r * L + k] = product;
            product = mul_modq(product, modq(q[k], mod), mod);
        }
        inverse[r] = r ? inverse_mod(product, q[r]) : 1;
    }
    LiftJob job = {in, low, high, rows, q, prefix, inverse, weight_low, weight_high, L};
    const bool wide = count * L * base->N * sizeof(uint64_t) >= EXPAND_PARALLEL_BYTES;
    vfhe_parallel_for(count, wide ? 0 : 1, lift_polynomial, &job);
}

// --- rows in natural order ---------------------------------------------------

typedef struct
{
    RNS_Polynomial p;
    uint64_t *words;      // N words per active row
    const uint64_t *rows; // the active base indices
} NaturalJob;

static void get_natural_row(void *ctx, uint64_t r)
{
    const NaturalJob *job = (const NaturalJob *)ctx;
    natural_row(job->words + r * job->p->base->N, job->p, job->rows[r]);
}

// The inverse of natural_row: word u + s t of the input goes to u * block + t.
static void set_natural_row(void *ctx, uint64_t r)
{
    const NaturalJob *job = (const NaturalJob *)ctx;
    const RNS_Polynomial p = job->p;
    const uint64_t i = job->rows[r];
    const uint64_t N = p->base->N, s = p->base->split_degree, block = N / s;
    const uint64_t *in = job->words + r * N;
    if (rns_row_is_narrow(p->base, i))
    {
        uint32_t *row = rns_row32(p, i);
        for (uint64_t u = 0; u < s; u++)
            for (uint64_t t = 0; t < block; t++)
                row[u * block + t] = (uint32_t)in[u + s * t];
    }
    else if (s == 1)
        memcpy(rns_row64(p, i), in, N * sizeof(uint64_t));
    else
    {
        uint64_t *row = rns_row64(p, i);
        for (uint64_t u = 0; u < s; u++)
            for (uint64_t t = 0; t < block; t++)
                row[u * block + t] = in[u + s * t];
    }
}

void polynomial_RNS_get_natural(uint64_t *out, RNS_Polynomial p)
{
    uint64_t rows[64];
    const uint64_t count = active_rows(rows, p);
    NaturalJob job = {p, out, rows};
    const bool wide = count * p->base->N * sizeof(uint64_t) >= COPY_PARALLEL_BYTES;
    vfhe_parallel_for(count, wide ? 0 : 1, get_natural_row, &job);
}

int64_t polynomial_RNS_set_natural(RNS_Polynomial p, const uint64_t *in)
{
    uint64_t rows[64];
    const uint64_t count = active_rows(rows, p), N = p->base->N;
    // Every row is checked before any is written, so a rejected input leaves
    // `p` as it was.
    for (uint64_t r = 0; r < count; r++)
    {
        const uint64_t q = p->base->mods[rows[r]]->q;
        const uint64_t *row = in + r * N;
        uint64_t above = 0;
        for (uint64_t k = 0; k < N; k++)
            above |= row[k] >= q;
        if (above)
            return (int64_t)r;
    }
    NaturalJob job = {p, (uint64_t *)in, rows};
    const bool wide = count * N * sizeof(uint64_t) >= COPY_PARALLEL_BYTES;
    vfhe_parallel_for(count, wide ? 0 : 1, set_natural_row, &job);
    return -1;
}
