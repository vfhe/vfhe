// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
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
#include <string.h>

#include "arith_internal.h"
#include "pmf_vec_group.h"

uint64_t pmf_vec_padded_length(uint64_t n)
{
    const uint64_t unit = PMF_LANES;
    return ((n + unit - 1) / unit) * unit;
}

// The three entry-point shapes below differ only in where the right-hand operand
// comes from -- the other vector, or one element broadcast -- so each is a walk
// over the groups around one of the cores in pmf_vec_group.h.

#define PMF_VEC_BINARY(name, core)                                                                 \
    void name(PMFVector out, const PMFVector a, const PMFVector b)                                 \
    {                                                                                              \
        const uint64_t L = a->params->limbs;                                                       \
        pv_t av[PMF_MAX_LIMBS], bv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];                           \
        for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)                               \
        {                                                                                          \
            pmf_vec_load_group(av, a, i, L);                                                       \
            pmf_vec_load_group(bv, b, i, L);                                                       \
            core(T, av, bv, a->params);                                                            \
            pmf_vec_store_group(out, i, T, L);                                                     \
        }                                                                                          \
    }

#define PMF_VEC_BROADCAST(name, core)                                                              \
    void name(PMFVector out, const PMFVector a, const uint64_t *s)                                 \
    {                                                                                              \
        const uint64_t L = a->params->limbs;                                                       \
        pv_t av[PMF_MAX_LIMBS], sv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];                           \
        pmf_vec_load_scalar(sv, s, L);                                                             \
        for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)                               \
        {                                                                                          \
            pmf_vec_load_group(av, a, i, L);                                                       \
            core(T, av, sv, a->params);                                                            \
            pmf_vec_store_group(out, i, T, L);                                                     \
        }                                                                                          \
    }

PMF_VEC_BINARY(pmf_vec_add, pmf_vec_add_group)
PMF_VEC_BINARY(pmf_vec_sub, pmf_vec_sub_group)
PMF_VEC_BINARY(pmf_vec_mul, pmf_vec_mul_group)

PMF_VEC_BROADCAST(pmf_vec_add_scalar, pmf_vec_add_group)
PMF_VEC_BROADCAST(pmf_vec_sub_scalar, pmf_vec_sub_group)
PMF_VEC_BROADCAST(pmf_vec_scale, pmf_vec_mul_group)

void pmf_vec_scalar_sub(PMFVector out, const uint64_t *s, const PMFVector a)
{
    const uint64_t L = a->params->limbs;
    pv_t av[PMF_MAX_LIMBS], sv[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(sv, s, L);
    for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_group(av, a, i, L);
        pmf_vec_sub_group(T, sv, av, a->params);
        pmf_vec_store_group(out, i, T, L);
    }
}

void pmf_vec_neg(PMFVector out, const PMFVector a)
{
    const uint64_t L = a->params->limbs;
    const pv_t mask = pv_mask52();
    pv_t av[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS];

    for (uint64_t i = 0; i < a->allocated_n; i += PMF_VEC_GROUP)
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
            const pv_t difference = pv_sub(pv_sub(pv_bcast(a->params->p[k]), av[k]), borrow);
            borrow = pv_borrow(difference);
            T[k] = pv_and(pv_and(difference, mask), nonzero);
        }
        assert(pv_is_zero(borrow)); // a < p, so p - a never borrows out

        pmf_vec_store_group(out, i, T, L);
    }
}

void pmf_vec_sum(uint64_t *out, const PMFVector a)
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

    // Whole groups of live elements only: the padding is canonical but carries
    // no value, so it must not reach the total.
    const uint64_t whole = (a->n / PMF_VEC_GROUP) * PMF_VEC_GROUP;
    uint64_t since_reduce = 0;
    for (uint64_t i = 0; i < whole; i += PMF_VEC_GROUP)
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
    for (uint64_t lane = 0; lane < PMF_VEC_GROUP; lane++)
    {
        for (uint64_t k = 0; k < PMF_LANES; k++)
            element[k] = k < L ? partial[k * PMF_LANES + lane] : 0;
        pmf_add(out, out, element, a->params);
    }
    for (uint64_t i = whole; i < a->n; i++)
    {
        pmf_vec_get_element(element, a, i);
        pmf_add(out, out, element, a->params);
    }
}

// --- adjacent pairs -------------------------------------------------------

// The even- and odd-indexed elements among the 2 * PMF_VEC_GROUP input elements
// from 2 * i0 on, as one group each. Where the input's allocation ends both
// read as zero, which keeps whatever is computed from them canonical. The one
// place in this file lanes are rearranged: a pair sits in adjacent lanes and
// its two halves have to end up in the same lane of two registers.
static inline void pmf_vec_load_pairs(pv_t *E, pv_t *O, const PMFVector a, uint64_t i0, uint64_t L)
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

void pmf_vec_fold(PMFVector out, const PMFVector a, const uint64_t *r)
{
    const uint64_t L = a->params->limbs;
    pv_t E[PMF_MAX_LIMBS], O[PMF_MAX_LIMBS], R[PMF_MAX_LIMBS];
    pv_t D[PMF_MAX_LIMBS + 1], T[PMF_MAX_LIMBS + 1];

    pmf_vec_load_scalar(R, r, L);
    for (uint64_t i = 0; i < out->allocated_n; i += PMF_VEC_GROUP)
    {
        pmf_vec_load_pairs(E, O, a, i, L);
        pmf_vec_sub_group(D, O, E, a->params);
        pmf_vec_mul_group(T, D, R, a->params);
        pmf_vec_add_group(D, E, T, a->params);
        pmf_vec_store_group(out, i, D, L);
    }
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

void pmf_vec_copy(PMFVector out, const PMFVector a)
{
    for (uint64_t k = 0; k < a->params->limbs; k++)
        memcpy(out->limbs[k], a->limbs[k], a->allocated_n * sizeof(uint64_t));
}

void pmf_vec_split_even_odd(PMFVector even, PMFVector odd, const PMFVector a)
{
    const uint64_t half = a->n / 2;
    for (uint64_t k = 0; k < a->params->limbs; k++)
    {
        const uint64_t *plane = a->limbs[k];
        for (uint64_t i = 0; i < half; i++)
        {
            even->limbs[k][i] = plane[2 * i];
            odd->limbs[k][i] = plane[2 * i + 1];
        }
    }
}

void pmf_vec_interleave(PMFVector out, const PMFVector even, const PMFVector odd)
{
    for (uint64_t k = 0; k < out->params->limbs; k++)
    {
        uint64_t *plane = out->limbs[k];
        for (uint64_t i = 0; i < even->n; i++)
        {
            plane[2 * i] = even->limbs[k][i];
            plane[2 * i + 1] = odd->limbs[k][i];
        }
    }
}

void pmf_vec_concat(PMFVector out, const PMFVector *parts, uint64_t count)
{
    uint64_t at = 0;
    for (uint64_t p = 0; p < count; p++)
    {
        for (uint64_t k = 0; k < out->params->limbs; k++)
            memcpy(out->limbs[k] + at, parts[p]->limbs[k], parts[p]->n * sizeof(uint64_t));
        at += parts[p]->n;
    }
}

void pmf_vec_gather(PMFVector out, const PMFVector a, const uint64_t *indices, uint64_t count)
{
    for (uint64_t k = 0; k < a->params->limbs; k++)
        for (uint64_t i = 0; i < count; i++)
            out->limbs[k][i] = a->limbs[k][indices[i]];
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
// Both go through the canonical encoding of one element at a time. They are
// BLAKE3-bound rather than arithmetic-bound, so there is nothing for the limb
// planes to do here beyond handing over each element.

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

static void pmf_vec_hash_span(uint8_t *out, const PMFVector a, uint64_t start, uint64_t count)
{
    blake3_hasher hasher;
    uint64_t element[PMF_LANES];
    uint8_t encoded[PMF_MAX_LIMBS * 8];

    blake3_hasher_init(&hasher);
    for (uint64_t i = 0; i < count; i++)
    {
        pmf_vec_get_element(element, a, start + i);
        pmf_to_bytes(encoded, element, a->params);
        blake3_hasher_update(&hasher, encoded, a->params->nbytes);
    }
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

void pmf_vec_hash(uint8_t *out, const PMFVector a) { pmf_vec_hash_span(out, a, 0, a->n); }

uint64_t pmf_vec_hash_count(const PMFVector a, uint64_t group, uint64_t stride)
{
    if (group == 0 || stride == 0 || a->n < group)
        return 0;
    return (a->n - group) / stride + 1;
}

void pmf_vec_hash_elements(uint8_t *out, const PMFVector a, uint64_t group, uint64_t stride)
{
    const uint64_t count = pmf_vec_hash_count(a, group, stride);
    for (uint64_t k = 0; k < count; k++)
        pmf_vec_hash_span(out + k * BLAKE3_OUT_LEN, a, k * stride, group);
}
