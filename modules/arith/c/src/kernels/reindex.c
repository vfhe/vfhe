// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// Reindexing a plane of words by the bits of the positions (arith_internal.h).
// It only moves words, so the field and pseudo-Mersenne vectors both use it,
// one plane at a time.
//
// Two ways to copy. When bits 0 .. r-1 of a position map to bits 0 .. r-1 of
// the source, a run of 2^r positions reads 2^r consecutive words: one memcpy.
// Otherwise every position is looked up. An aligned group of 8 positions
// differs only in its low three bits, so its source indices are the first
// one ORed with eight fixed offsets: one gather under AVX-512.
#include <arith.h>
#include <string.h>
#include "arith_internal.h"

#if VFHE_HAVE_AVX512F
#include <immintrin.h>
#endif

// Below this many words a run is not worth a memcpy call.
#define REINDEX_MIN_RUN 8

bool vec_reindex_valid(const int64_t *bits, uint64_t count, uint64_t offset, uint64_t n)
{
    if (count > 63)
        return false;
    uint64_t mapped = 0;
    for (uint64_t j = 0; j < count; j++)
    {
        if (bits[j] < 0)
            continue;
        if (bits[j] > 63 || (mapped >> bits[j]) & 1)
            return false;
        mapped |= 1ULL << bits[j];
    }
    // The largest source index is offset | mapped, below n.
    return !(offset & mapped) && (offset | mapped) < n;
}

void vec_reindex_plan(VecReindex *plan, const int64_t *bits, uint64_t count, uint64_t offset)
{
    memset(plan->table, 0, sizeof plan->table);
    plan->offset = offset;
    plan->bytes = (count + 7) / 8;
    for (uint64_t j = 0; j < count; j++)
    {
        if (bits[j] < 0)
            continue;
        const uint64_t source = 1ULL << bits[j];
        for (uint64_t v = 0; v < 256; v++)
            if ((v >> (j % 8)) & 1)
                plan->table[j / 8][v] |= source;
    }
    uint64_t r = 0;
    while (r < count && bits[r] == (int64_t)r)
        r++;
    plan->run = 1ULL << r;
}

static inline uint64_t source_index(const VecReindex *plan, uint64_t i)
{
    uint64_t source = plan->offset;
    for (uint64_t b = 0; b < plan->bytes; b++)
        source |= plan->table[b][(i >> (8 * b)) & 255];
    return source;
}

static void reindex_runs(uint64_t *out, const uint64_t *in, const VecReindex *plan, uint64_t begin,
                         uint64_t end)
{
    for (uint64_t i = begin; i < end;)
    {
        uint64_t length = plan->run - (i & (plan->run - 1));
        if (length > end - i)
            length = end - i;
        memcpy(out + i, in + source_index(plan, i), length * sizeof(uint64_t));
        i += length;
    }
}

static void reindex_scalar(uint64_t *out, const uint64_t *in, const VecReindex *plan,
                           uint64_t begin, uint64_t end)
{
    for (uint64_t i = begin; i < end; i++)
        out[i] = in[source_index(plan, i)];
}

#if VFHE_HAVE_AVX512F

static void reindex_lookup(uint64_t *out, const uint64_t *in, const VecReindex *plan,
                           uint64_t begin, uint64_t end)
{
    // Up to the first multiple of 8, one position at a time.
    uint64_t i = begin;
    const uint64_t aligned = (begin + 7) & ~(uint64_t)7;
    if (aligned < end)
    {
        reindex_scalar(out, in, plan, i, aligned);
        i = aligned;
    }
    // Lane t of an aligned group reads its first source index ORed with
    // table[0][t], the source bits of t (t is in byte 0 of the position).
    const __m512i lanes = _mm512_loadu_si512((const void *)plan->table[0]);
    for (; i + 8 <= end; i += 8)
    {
        const __m512i at =
            _mm512_or_si512(_mm512_set1_epi64((long long)source_index(plan, i)), lanes);
        _mm512_storeu_si512((void *)(out + i), _mm512_i64gather_epi64(at, (const void *)in, 8));
    }
    reindex_scalar(out, in, plan, i, end);
}

#else

static void reindex_lookup(uint64_t *out, const uint64_t *in, const VecReindex *plan,
                           uint64_t begin, uint64_t end)
{
    reindex_scalar(out, in, plan, begin, end);
}

#endif

void vec_reindex_u64(uint64_t *out, const uint64_t *in, const VecReindex *plan, uint64_t begin,
                     uint64_t end)
{
    if (plan->run >= REINDEX_MIN_RUN)
        reindex_runs(out, in, plan, begin, end);
    else
        reindex_lookup(out, in, plan, begin, end);
}
