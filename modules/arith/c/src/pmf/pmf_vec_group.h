// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The pseudo-Mersenne group kernels: one group of PMF_VEC_GROUP elements, held
// limb by limb in `pv_t` words, through add, sub, mul and the reduction tail.
//
// Every vector kernel -- the element-wise operations in pmf_vector.c and the
// transform in pmf_ntt.c -- is a walk over the groups of a vector around one of
// these cores. They mirror the algorithms in pmf.c (the same fold, the same
// passes, the same bounds) without sharing its code: those are the oracle these
// are tested against.
//
// A core takes its operands as L words each and writes T[0..L-1] canonical; T
// must have room for L+1 words, the extra one being the overflow limb the
// reduction tail consumes. Operands may alias T.
#ifndef VFHE_PMF_VEC_GROUP_H
#define VFHE_PMF_VEC_GROUP_H

#include <arith.h>
#include <assert.h>

#include "pmf_vec_ops.h"

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

#endif // VFHE_PMF_VEC_GROUP_H
