// SPDX-FileCopyrightText: 2026 Daniele Cozzo <daniele.cozzo@imdea.org>
// SPDX-License-Identifier: Apache-2.0
/**
 * @file pmf.c
 * @brief Pseudo-Mersenne field F_p, p = 2^n - c: parameters, the shared
 *        reduction tail, and the scalar arithmetic kernels.
 *
 * One file, and no ISA guard yet, for the same reason field.c has none: there is
 * no vector code here to guard. When the AVX-512 IFMA kernels arrive they go
 * behind a
 *
 *     #if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)
 *
 * inside the public pmf_mul / pmf_add wrappers at the bottom, with the scalar
 * pmf_ref_* bodies left OUTSIDE it -- they must compile into every engine so the
 * tuned build can differential-test against them in-process. Splitting this file
 * before that code exists only buys empty scaffolding.
 *
 * The reference multiply is deliberately plain schoolbook plus a Crandall fold,
 * NOT a scalar transcription of the planned vector kernel: it is the oracle, and
 * a shared structure could hide a shared mistake in both.
 */
#include <arith.h>
#include <inttypes.h>

#include "arith_internal.h"

// --- parameters -----------------------------------------------------------

PMFParams pmf_new_params(uint64_t n, uint64_t c)
{
    const uint64_t limbs = (n + PMF_LIMB_BITS - 1) / PMF_LIMB_BITS;

    if (limbs < 5 || limbs > PMF_MAX_LIMBS)
    {
        fprintf(stderr,
                "pmf_new_params: n=%" PRIu64 " needs %" PRIu64
                " limbs; only 5 (n in 209..260) and 6 (n in 261..312) have kernels\n",
                n, limbs);
        return NULL;
    }
    if (c == 0 || (c & 1) == 0)
    {
        fprintf(stderr,
                "pmf_new_params: c=%" PRIu64
                " must be odd and nonzero for 2^n - c to be an odd modulus\n",
                c);
        return NULL;
    }

    const uint64_t shift = PMF_LIMB_BITS * limbs - n;
    // e = c << shift must stay inside one limb, which is what bounds c -- and
    // hence which n are usable at all.
    const uint64_t c_bits = n - PMF_LIMB_BITS * (limbs - 1);
    if (c >> c_bits != 0)
    {
        fprintf(stderr,
                "pmf_new_params: c=%" PRIu64 " exceeds %" PRIu64 " bits, so the fold constant "
                "c << %" PRIu64 " would not fit a 52-bit limb\n",
                c, c_bits, shift);
        return NULL;
    }

    PMFParams params = (PMFParams)calloc(1, sizeof(*params));
    if (params == NULL)
        return NULL;

    params->n = n;
    params->c = c;
    params->limbs = limbs;
    params->shift = shift;
    params->fold = c << shift;
    params->top_bits = PMF_LIMB_BITS - shift;
    params->top_mask = (1ULL << params->top_bits) - 1;
    params->nbytes = (n + 7) / 8;

    // p = 2^n - c, as limbs: start from 2^n - 1 (every bit below n set) and
    // subtract c - 1, so nothing ever needs an extra limb.
    for (uint64_t k = 0; k < limbs; k++)
    {
        const uint64_t below = PMF_LIMB_BITS * k;
        if (n >= below + PMF_LIMB_BITS)
            params->p[k] = PMF_LIMB_MASK;
        else
            params->p[k] = (n > below) ? ((1ULL << (n - below)) - 1) : 0;
    }
    uint64_t sub = c - 1;
    for (uint64_t k = 0; k < limbs && sub != 0; k++)
    {
        if (params->p[k] >= sub)
        {
            params->p[k] -= sub;
            sub = 0;
        }
        else
        {
            params->p[k] = (params->p[k] + PMF_LIMB_MASK + 1) - sub;
            sub = 1;
        }
    }

    return params;
}

void pmf_free_params(PMFParams params) { free(params); }

uint64_t pmf_limbs(PMFParams params) { return params->limbs; }

uint64_t pmf_byte_length(PMFParams params) { return params->nbytes; }

// --- shared helpers -------------------------------------------------------

// Write the live limbs and zero the padding lanes. Every public output goes
// through here, so the "lanes L..7 are always zero" contract holds by
// construction rather than by each caller remembering.
static void pmf_store(uint64_t *out, const uint64_t *t, uint64_t limbs)
{
    for (uint64_t k = 0; k < limbs; k++)
        out[k] = t[k];
    for (uint64_t k = limbs; k < PMF_LANES; k++)
        out[k] = 0;
}

// Subtract the modulus once if that leaves a non-negative result. Input must
// already satisfy V < 2^n = p + c, so one conditional subtract lands in [0, p).
static void pmf_cond_sub_p(uint64_t *out, const uint64_t *T, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t t[PMF_MAX_LIMBS];
    uint64_t borrow = 0;

    for (uint64_t k = 0; k < L; k++)
    {
        uint64_t d = T[k] - params->p[k] - borrow;
        borrow = (T[k] < params->p[k] + borrow) ? 1 : 0;
        t[k] = d & PMF_LIMB_MASK;
    }

    // borrow == 1 means T < p, so T was already canonical.
    pmf_store(out, borrow ? T : t, L);
}

// --- the reduction tail ---------------------------------------------------

void pmf_ref_reduce_wide(uint64_t *out, uint64_t *T, PMFParams params)
{
    const uint64_t L = params->limbs;
    assert(L >= 3); // step 1 spills into T[2]

    // 1. Fold the overflow column: 2^(52L) == e (mod p). With e < 2^52 and
    //    T[L] < 2^60 the product reaches 2^112, so it occupies THREE limbs --
    //    not one, which is why the multiply leaves this column to us.
    unsigned __int128 w = (unsigned __int128)params->fold * T[L];
    T[0] += (uint64_t)(w & PMF_LIMB_MASK);
    T[1] += (uint64_t)((w >> PMF_LIMB_BITS) & PMF_LIMB_MASK);
    T[2] += (uint64_t)(w >> (2 * PMF_LIMB_BITS));

    // 2. Carry the limbs down below 2^52, folding whatever falls off the top
    //    back in. Each pass replaces carry * 2^(52L) by carry * e, so the value
    //    strictly decreases; the second pass leaves carry in {0,1} and the third
    //    clears it.
    uint64_t carry;
    int passes = 0;
    do
    {
        unsigned __int128 acc = 0;
        for (uint64_t k = 0; k < L; k++)
        {
            acc += T[k];
            T[k] = (uint64_t)acc & PMF_LIMB_MASK;
            acc >>= PMF_LIMB_BITS;
        }
        carry = (uint64_t)acc;
        if (carry)
        {
            w = (unsigned __int128)params->fold * carry;
            T[0] += (uint64_t)(w & PMF_LIMB_MASK);
            T[1] += (uint64_t)(w >> PMF_LIMB_BITS);
        }
        passes++;
        assert(passes <= 4);
    } while (carry);
    // T is now "normal": every limb below 2^52, value below 2^(52L).

    // 3. Reduce the bits at or above 2^n. They are exactly the top s bits of
    //    limb L-1, and each is worth c: V = u * 2^n + v == v + u * c (mod p),
    //    with u < 2^s so u * c < 2^s * c == e < 2^52. When s == 0 the shift
    //    yields u == 0 and this whole step is skipped.
    for (uint64_t u = T[L - 1] >> params->top_bits; u != 0; u = T[L - 1] >> params->top_bits)
    {
        T[L - 1] &= params->top_mask;
        unsigned __int128 acc = (unsigned __int128)T[0] + (unsigned __int128)u * params->c;
        T[0] = (uint64_t)acc & PMF_LIMB_MASK;
        acc >>= PMF_LIMB_BITS;
        for (uint64_t k = 1; acc != 0 && k < L; k++)
        {
            acc += T[k];
            T[k] = (uint64_t)acc & PMF_LIMB_MASK;
            acc >>= PMF_LIMB_BITS;
        }
    }

    // V < 2^(52L - s) == 2^n == p + c, so one conditional subtract finishes.
    pmf_cond_sub_p(out, T, params);
}

// --- scalar reference kernels --------------------------------------------

void pmf_ref_add(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t T[PMF_MAX_LIMBS + 1];

    // Each limb stays below 2^53, well inside the tail's 2^60 contract. Routing
    // through the tail rather than open-coding add-then-conditional-subtract is
    // deliberate: when s == 0 the sum can carry out of 52L bits, and the tail is
    // the one place already proven to handle that.
    for (uint64_t k = 0; k < L; k++)
        T[k] = a[k] + b[k];
    T[L] = 0;

    pmf_ref_reduce_wide(out, T, params);
}

void pmf_ref_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t t[PMF_MAX_LIMBS];
    uint64_t borrow = 0;

    // a, b < p, so a - b lies in (-p, p) and a single corrective +p is enough.
    // No trip through the reduction tail: the result is canonical already.
    for (uint64_t k = 0; k < L; k++)
    {
        uint64_t d = a[k] - b[k] - borrow;
        borrow = (a[k] < b[k] + borrow) ? 1 : 0;
        t[k] = d & PMF_LIMB_MASK;
    }

    if (borrow)
    {
        // The limbs currently hold a - b + 2^(52L). Adding p and DISCARDING the
        // carry out of the top limb subtracts that 2^(52L) again, leaving
        // a - b + p, which is in (0, p) exactly because a - b > -p.
        uint64_t carry = 0;
        for (uint64_t k = 0; k < L; k++)
        {
            const uint64_t s = t[k] + params->p[k] + carry;
            t[k] = s & PMF_LIMB_MASK;
            carry = s >> PMF_LIMB_BITS;
        }
    }

    pmf_store(out, t, L);
}

void pmf_ref_neg(uint64_t *out, const uint64_t *a, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t t[PMF_MAX_LIMBS];
    uint64_t borrow = 0;
    int is_zero = 1;

    for (uint64_t k = 0; k < L; k++)
        if (a[k] != 0)
            is_zero = 0;

    // -0 is 0, not p. For a != 0, p - a lies in (0, p) since 0 < a < p, so the
    // subtraction cannot borrow out and needs no correction.
    if (is_zero)
    {
        pmf_store(out, a, L);
        return;
    }

    for (uint64_t k = 0; k < L; k++)
    {
        const uint64_t d = params->p[k] - a[k] - borrow;
        borrow = (params->p[k] < a[k] + borrow) ? 1 : 0;
        t[k] = d & PMF_LIMB_MASK;
    }

    pmf_store(out, t, L);
}

void pmf_ref_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params)
{
    const uint64_t L = params->limbs;
    // Zeroed for the whole array rather than the 2L entries actually used: the
    // compiler cannot see that L is bounded, so a partial init reads as
    // maybe-uninitialized. Twelve __int128 stores is not worth arguing about.
    unsigned __int128 col[2 * PMF_MAX_LIMBS] = {0};
    uint64_t t[2 * PMF_MAX_LIMBS];
    uint64_t T[PMF_MAX_LIMBS + 1];

    // Schoolbook. A column holds at most L products below 2^104, so it peaks
    // under 2^107 and stays inside an unsigned __int128.
    for (uint64_t i = 0; i < L; i++)
        for (uint64_t j = 0; j < L; j++)
            col[i + j] += (unsigned __int128)a[i] * b[j];

    unsigned __int128 acc = 0;
    for (uint64_t k = 0; k < 2 * L - 1; k++)
    {
        acc += col[k];
        t[k] = (uint64_t)acc & PMF_LIMB_MASK;
        acc >>= PMF_LIMB_BITS;
    }
    // a, b < p < 2^n <= 2^(52L), so the product fits 2L limbs exactly.
    t[2 * L - 1] = (uint64_t)acc;

    // Crandall fold: the upper half comes back weighted by e.
    acc = 0;
    for (uint64_t k = 0; k < L; k++)
    {
        acc += (unsigned __int128)t[k] + (unsigned __int128)params->fold * t[k + L];
        T[k] = (uint64_t)acc & PMF_LIMB_MASK;
        acc >>= PMF_LIMB_BITS;
    }
    T[L] = (uint64_t)acc; // below 2^53

    pmf_ref_reduce_wide(out, T, params);
}

// --- public arithmetic ----------------------------------------------------
//
// Thin wrappers today. This is where the ISA guard goes when the vector kernels
// land: #if IFMA -> the tuned kernel, #else -> the pmf_ref_* call below.

void pmf_add(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params)
{
    pmf_ref_add(out, a, b, params);
}

void pmf_sub(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params)
{
    pmf_ref_sub(out, a, b, params);
}

void pmf_neg(uint64_t *out, const uint64_t *a, PMFParams params) { pmf_ref_neg(out, a, params); }

void pmf_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, PMFParams params)
{
    pmf_ref_mul(out, a, b, params);
}

int pmf_is_equal(const uint64_t *a, const uint64_t *b, PMFParams params)
{
    // A plain limb comparison is valid only because every output is canonical, so
    // one value has exactly one representation. Compares the live limbs only:
    // whatever sits in the padding lanes must not affect the answer.
    for (uint64_t k = 0; k < params->limbs; k++)
        if (a[k] != b[k])
            return 0;
    return 1;
}

void pmf_canonicalize(uint64_t *out, const uint64_t *in, PMFParams params)
{
    uint64_t T[PMF_MAX_LIMBS + 1];

    for (uint64_t k = 0; k < params->limbs; k++)
        T[k] = in[k];
    T[params->limbs] = 0;

    pmf_ref_reduce_wide(out, T, params);
}
