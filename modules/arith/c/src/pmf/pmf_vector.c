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
#include "pmf_vec_ops.h"

uint64_t pmf_vec_padded_length(uint64_t n)
{
    const uint64_t unit = PMF_LANES;
    return ((n + unit - 1) / unit) * unit;
}

// --- the reduction tail, in registers -------------------------------------

// Carry every limb down below 2^52, folding what falls off the top back in on
// 2^(52L) == e. Mirrors step 2 of pmf_ref_reduce_wide, with the scalar loop's
// data-dependent repeat replaced by its proven worst case: each pass replaces
// carry * 2^(52L) with carry * e, so the value strictly decreases; the second
// pass leaves a carry of at most one and the third clears it. Lanes are
// independent, so per lane this is exactly the scalar argument.
static inline void pmf_vec_carry_down(pv_t *T, uint64_t L, pv_t fold, pv_t mask)
{
    for (int pass = 0; pass < 3; pass++)
    {
        pv_t carry = pv_zero();
        for (uint64_t k = 0; k < L; k++)
        {
            const pv_t acc = pv_add(T[k], carry);
            T[k] = pv_and(acc, mask);
            carry = pv_srl(acc, PMF_LIMB_BITS);
        }
        if (pass == 2)
        {
            // The third pass must find nothing left to fold.
            assert(pv_is_zero(carry));
            break;
        }
        // carry is small here, so fold * carry needs two limbs at most.
        T[0] = pv_madd52lo(T[0], fold, carry);
        T[1] = pv_madd52hi(T[1], fold, carry);
    }
}

// Subtract the modulus once where that leaves a non-negative result. Input must
// satisfy V < 2^n = p + c, so one conditional subtract lands in [0, p).
static inline void pmf_vec_cond_sub_p(pv_t *T, uint64_t L, const uint64_t *p, pv_t mask)
{
    pv_t t[PMF_MAX_LIMBS];
    pv_t borrow = pv_zero();

    for (uint64_t k = 0; k < L; k++)
    {
        const pv_t difference = pv_sub(pv_sub(T[k], pv_bcast(p[k])), borrow);
        borrow = pv_borrow(difference);
        t[k] = pv_and(difference, mask);
    }

    // A borrow out means T < p, so T was already canonical.
    const pv_t keep = pv_mask_from_flag(borrow);
    for (uint64_t k = 0; k < L; k++)
        T[k] = pv_select(keep, T[k], t[k]);
}

// The shared tail: T is L+1 limbs, each below 2^60, and is reduced in place to
// L canonical limbs. Mirrors pmf_ref_reduce_wide.
static inline void pmf_vec_reduce_wide(pv_t *T, PMFParams params)
{
    const uint64_t L = params->limbs;
    const pv_t mask = pv_mask52();
    const pv_t fold = pv_bcast(params->fold);

    // 1. Fold the overflow limb. T[L] may hold up to 60 bits, and madd52 reads
    //    only 52 of each operand, so the multiplicand is split in two halves and
    //    the product -- up to 112 bits -- lands across three limbs.
    {
        const pv_t low = pv_and(T[L], mask);
        const pv_t high = pv_srl(T[L], PMF_LIMB_BITS);
        T[0] = pv_madd52lo(T[0], fold, low);
        T[1] = pv_madd52hi(T[1], fold, low);
        T[1] = pv_madd52lo(T[1], fold, high);
        T[2] = pv_madd52hi(T[2], fold, high);
    }

    // 2. Carry down, folding whatever leaves the top.
    pmf_vec_carry_down(T, L, fold, mask);

    // 3. Reduce the bits at or above 2^n: they are the top s bits of the last
    //    limb, worth 2^n == c there. Only a shifted modulus has any. One round
    //    can carry a bit back up into them, so two clear it -- u * c < e < 2^52
    //    throughout, which is what keeps this a single madd52 into limb 0.
    if (params->shift)
    {
        const pv_t top_mask = pv_bcast(params->top_mask);
        const pv_t c = pv_bcast(params->c);
        for (int round = 0; round < 2; round++)
        {
            const pv_t above = pv_srl(T[L - 1], params->top_bits);
            T[L - 1] = pv_and(T[L - 1], top_mask);
            T[0] = pv_madd52lo(T[0], c, above);
            pv_t carry = pv_zero();
            for (uint64_t k = 0; k < L; k++)
            {
                const pv_t acc = pv_add(T[k], carry);
                T[k] = pv_and(acc, mask);
                carry = pv_srl(acc, PMF_LIMB_BITS);
            }
            assert(pv_is_zero(carry));
        }
        assert(pv_is_zero(pv_srl(T[L - 1], params->top_bits)));
    }

    // 4. V < 2^n == p + c now, so one conditional subtract finishes it.
    pmf_vec_cond_sub_p(T, L, params->p, mask);
}

// --- group load and store -------------------------------------------------

static inline void pmf_vec_load_group(pv_t *T, const PMFVector v, uint64_t i, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        T[k] = pv_load(v->limbs[k] + i);
}

static inline void pmf_vec_store_group(PMFVector v, uint64_t i, const pv_t *T, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        pv_store(v->limbs[k] + i, T[k]);
}

// One element broadcast into every lane of each limb plane.
static inline void pmf_vec_load_scalar(pv_t *S, const uint64_t *s, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        S[k] = pv_bcast(s[k]);
}

// --- arithmetic -----------------------------------------------------------

// a + b for one group. Canonical limbs sum below 2^53, so the tail's input
// contract holds with nothing above the top limb.
static inline void pmf_vec_add_group(pv_t *T, const pv_t *a, const pv_t *b, PMFParams params)
{
    const uint64_t L = params->limbs;
    for (uint64_t k = 0; k < L; k++)
        T[k] = pv_add(a[k], b[k]);
    T[L] = pv_zero();
    pmf_vec_reduce_wide(T, params);
}

// a - b for one group. A borrow out of the top means the true difference is
// negative; adding p back corrects it, and the carry that addition produces is
// the 2^(52L) the borrow took away, so it is dropped. Mirrors pmf_ref_sub, and
// like it needs no reduction tail.
static inline void pmf_vec_sub_group(pv_t *T, const pv_t *a, const pv_t *b, PMFParams params)
{
    const uint64_t L = params->limbs;
    const pv_t mask = pv_mask52();
    pv_t borrow = pv_zero();

    for (uint64_t k = 0; k < L; k++)
    {
        const pv_t difference = pv_sub(pv_sub(a[k], b[k]), borrow);
        borrow = pv_borrow(difference);
        T[k] = pv_and(difference, mask);
    }

    const pv_t correct = pv_mask_from_flag(borrow);
    pv_t carry = pv_zero();
    for (uint64_t k = 0; k < L; k++)
    {
        const pv_t acc = pv_add(pv_add(T[k], pv_and(correct, pv_bcast(params->p[k]))), carry);
        T[k] = pv_and(acc, mask);
        carry = pv_srl(acc, PMF_LIMB_BITS);
    }
}

// a * b for one group: schoolbook over the limb planes, then the Crandall fold.
//
// Each column accumulates the low halves of the products landing on it and the
// high halves of those one limb below. At most 2L addends, each below 2^52, so a
// column stays under 2^56 and the whole double loop runs with no intermediate
// normalization -- which is the payoff of doing this a group at a time.
static inline void pmf_vec_mul_group(pv_t *T, const pv_t *a, const pv_t *b, PMFParams params)
{
    const uint64_t L = params->limbs;
    const pv_t mask = pv_mask52();
    pv_t col[2 * PMF_MAX_LIMBS];

    for (uint64_t k = 0; k < 2 * L; k++)
        col[k] = pv_zero();

    for (uint64_t i = 0; i < L; i++)
    {
        for (uint64_t j = 0; j < L; j++)
        {
            col[i + j] = pv_madd52lo(col[i + j], a[i], b[j]);
            col[i + j + 1] = pv_madd52hi(col[i + j + 1], a[i], b[j]);
        }
    }

    // Carry the 2L columns down to proper limbs.
    pv_t carry = pv_zero();
    for (uint64_t k = 0; k < 2 * L; k++)
    {
        const pv_t acc = pv_add(col[k], carry);
        col[k] = pv_and(acc, mask);
        carry = pv_srl(acc, PMF_LIMB_BITS);
    }
    assert(pv_is_zero(carry));

    // Fold the top half in on 2^(52L) == e. Every operand is a proper limb here,
    // so one madd52 pair per column suffices.
    carry = pv_zero();
    for (uint64_t k = 0; k < L; k++)
    {
        pv_t acc = pv_add(col[k], carry);
        acc = pv_madd52lo(acc, pv_bcast(params->fold), col[k + L]);
        T[k] = pv_and(acc, mask);
        carry = pv_srl(acc, PMF_LIMB_BITS);
        carry = pv_madd52hi(carry, pv_bcast(params->fold), col[k + L]);
    }
    T[L] = carry;

    pmf_vec_reduce_wide(T, params);
}

// The three entry-point shapes below differ only in where the right-hand operand
// comes from -- the other vector, or one element broadcast -- so each is a walk
// over the groups around one of the cores above.

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
