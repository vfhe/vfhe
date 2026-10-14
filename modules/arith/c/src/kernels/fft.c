// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
/* The negacyclic FFT over R[X]/(X^n + 1): plan construction, the AVX-512
 * transforms, and the dispatch to the scalar ones in fft_scalar.c.
 *
 * 1. There is no separate "twist" pass.
 *
 * The usual way to write this transform is three steps: pack the n real
 * coefficients into m = n/2 complex values c_j = a_j + i*a_{j+m}, multiply
 * each by zeta^j (the "twist", zeta = exp(i*pi/n)), then run an ordinary
 * m-point cyclic FFT. None of it is needed:
 *
 *     f(zeta^{4k+1}) = sum_{j<m} a_j     zeta^{(4k+1)j}
 *                    + sum_{j<m} a_{j+m} zeta^{(4k+1)(j+m)}
 *                    = sum_{j<m} (a_j + a_{j+m} zeta^{(4k+1)m}) zeta^{(4k+1)j}
 *
 * and zeta^{(4k+1)m} = zeta^{4km} * zeta^m = 1 * i, since zeta^{4m} = 1 and
 * zeta^m = exp(i*pi/2). So
 *
 *     f(zeta^{4k+1}) = sum_{j<m} c_j zeta^{(4k+1)j},   c_j = a_j + i*a_{j+m}
 *
 * which is a complex transform whose evaluation points are the m roots of
 * X^m = i. Two consequences:
 *
 *   - The pack is free. Re(c_j) is p[j] and Im(c_j) is p[m+j], which is
 *     already the split real/imaginary layout the core works in.
 *
 *   - The twist is a change of roots, not a pass. Cooley-Tukey does not care
 *     what the modulus is: X^{2t} - w^2 factors as (X^t - w)(X^t + w) for any
 *     w, and the butterfly u +- w*v is unchanged. Working modulo X^m - i
 *     instead of X^m - 1 is therefore the same code with a different twiddle
 *     table. Group i at level h (h groups at that level) carries the modulus
 *     X^{m/h} - zeta^{(m/h)(4 brv(i) + 1)}, so its butterfly twiddle is that
 *     root's square root:
 *
 *         ws[h + i] = zeta^{(m/2h)(4 brv(i) + 1)},  angle pi(4 brv(i)+1)/(4h)
 *
 *     Replacing the "+1" with "+0" would give the ordinary cyclic table. That
 *     single term is the entire negacyclic half-shift.
 *
 * 2. The output is in bit-reversed order, deliberately.
 *
 * The forward transform is decimation-in-time (natural input, bit-reversed
 * output) and the inverse is decimation-in-frequency (bit-reversed input,
 * natural output), so the two compose without any permutation pass.
 *
 * 3. Structure of a transform.
 *
 * Data is held as separate real and imaginary arrays (p[0..m-1] and
 * p[m..n-1]) so that butterflies are pure vertical FMA with no shuffles; the
 * only lane crossing happens in the small final stages, which are done inside
 * registers.
 *
 * The forward transform runs, from the top:
 *
 *   - recursion: one radix-4 level (two butterfly stages) over the current
 *     block, then the same on each of its four quarters, depth first, so that
 *     once a sub-problem is small enough to sit in L1 all of its remaining
 *     stages complete without leaving L1;
 *   - leaf (FFT_LEAF_COMPLEX elements or fewer): radix-4 stages down to a
 *     distance of 16, then the last four stages (distances 8, 4, 2, 1) fused
 *     into a single register-resident pass over each block of 16 complex
 *     values, using permutexvar and per-lane twiddle vectors with the
 *     butterfly's sign folded in.
 *
 * The inverse mirrors it exactly: the fused pass first (ascending), then the
 * radix-4 stages, then the recursion's merge levels, with conjugate twiddles
 * throughout and the 1/m normalisation applied on the way in.
 *
 * Stage-count parity is handled by a single radix-2 stage at the top of the
 * array (forward) or the bottom (inverse) when the number of stages is odd.
 *
 * 4. The twiddles are correctly rounded.
 *
 * They are computed in long double and rounded once. Computing the angle in
 * double first (pi times a ratio, two roundings) puts an error of a few ulp in
 * every twiddle, which is enough to make the transform's output error about
 * 1.7x larger. The tables are built once per plan, so the cost is irrelevant.
 * Where long double is no wider than double (Apple), the angle over pi, which
 * is exact in double, goes to __cospi / __sinpi instead; elsewhere such a
 * platform falls back to long double and gets the double-rounded twiddles.
 */
#include <arith.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>

#include "arith_internal.h"

#if VFHE_HAVE_AVX512F
#include <immintrin.h>
#endif

/* Sub-problem size, in complex elements, at which the recursion stops and the
 * flat core takes over. It should be the largest block whose data plus its
 * slice of the twiddle tables still fits in L1: 1024 complex is 16 KiB of
 * data and suits a 48 KiB L1. Rounded down to a power of four of at least 64:
 * an even log2 keeps the number of stages inside a leaf even, which is what
 * stops an extra twiddled radix-2 pass appearing in every leaf. */
#ifndef FFT_LEAF_COMPLEX
#define FFT_LEAF_COMPLEX 1024
#endif

// Shortest transform the AVX-512 code handles: its fused final pass works on
// blocks of 16 complex values, i.e. n = 32 real coefficients.
#define FFT_MIN_VECTOR_LEN 32

#define FFT_MAX_LEN (1ULL << 20)

static size_t lg2(size_t x)
{
    size_t r = 0;
    while (((size_t)1 << r) < x)
        r++;
    return r;
}

// Capped at m, in which case the whole transform is one leaf and the recursion
// never fires.
static size_t pick_leaf(size_t m)
{
    size_t L = lg2((size_t)FFT_LEAF_COMPLEX);
    if (((size_t)1 << L) > (size_t)FFT_LEAF_COMPLEX)
        L--;
    if (L & 1)
        L--;
    if (L < 6)
        L = 6;
    size_t leaf = (size_t)1 << L;
    return leaf > m ? m : leaf;
}

#if VFHE_HAVE_AVX512F

// Vector `b` of one of the fused tail's per-lane twiddle tables.
#define TAIL(table, b) _mm512_load_pd(c->table + 8 * (b))

/* lane index vectors for the in-register stages */
static const __m512i IU4 = {0, 1, 2, 3, 0, 1, 2, 3};
static const __m512i IV4 = {4, 5, 6, 7, 4, 5, 6, 7};
static const __m512i IU2 = {0, 1, 0, 1, 4, 5, 4, 5};
static const __m512i IV2 = {2, 3, 2, 3, 6, 7, 6, 7};
static const __m512i IU1 = {0, 0, 2, 2, 4, 4, 6, 6};
static const __m512i IV1 = {1, 1, 3, 3, 5, 5, 7, 7};

/* ---------------------------------------------------------------------------
 * Recursive (cache-blocked) CT_NR / GS_RN.
 *
 * The stages above the leaf run depth first rather than as full sweeps of
 * the array. A sweep per stage is the right shape while the array fits in L1
 * and increasingly wasteful once it does not: bytes touched per forward call
 * (data, twiddles and the expanded tables of the last stages) come to roughly
 * 65*n, so
 *
 *      n =   256     16 KiB     L1
 *      n =  1024     65 KiB     just past L1
 *      n =  4096    260 KiB     L2
 *      n = 32768    2.1 MiB     L2 boundary or L3
 *
 * and from about n = 8192 up every stage would stream the array out of L2.
 * Depth-first traversal keeps a sub-problem resident once it is small enough,
 * so all of its remaining stages complete without leaving L1. Every butterfly
 * is the same; only the order changes.
 *
 * Index algebra, which is what makes this need no extra tables. Stage
 * distance t has ng_t = m/(2t) groups; group gg spans [2*gg*t, 2*gg*t + 2t)
 * and uses ws[ng_t + gg]. A block at complex offset `off` of length `len`
 * therefore owns, at distance t, the len/(2t) groups starting at
 * gg = off/(2t) -- a contiguous slice of the same flat tables the
 * non-recursive path uses. The recursion carries (ngt, g0) for its own top
 * distance and derives the rest by shifting, so there is no integer division
 * on the hot path.
 * ------------------------------------------------------------------------- */

/* Leaf of the forward recursion: radix-4 stages down to distance 16, then the
 * fused distance-8/4/2/1 pass, restricted to one block.
 * (ngt, g0) are the absolute level and first group index at t = len/2. */
static void leaf_ct(const struct _FFT_Plan *c, double *re, double *im, double *out_re,
                    double *out_im, size_t len, size_t off, size_t ngt, size_t g0)
{
    size_t t = len >> 1;
    size_t cnt = 1; /* groups of this block at distance t */

    /* An odd number of stages leaves one radix-2 pass over; it goes here, at
     * the widest distance in the block. Note the twiddle is never 1: a cyclic
     * transform would have ws[1] = 1 and could do this stage with adds and
     * subtracts alone, but the negacyclic half-shift makes ws[1] = exp(i*pi/4),
     * so it is an ordinary complex butterfly at every level. Keeping the leaf
     * size a power of four means the branch is dead below the top level. */
    if ((lg2(len) - 4) & 1)
    {
        __m512d wr = _mm512_set1_pd(c->ws_re[ngt + g0]);
        __m512d wi = _mm512_set1_pd(c->ws_im[ngt + g0]);
        for (size_t j = 0; j < t; j += 8)
        {
            __m512d ur = _mm512_load_pd(re + j), ui = _mm512_load_pd(im + j);
            __m512d vr = _mm512_load_pd(re + j + t), vi = _mm512_load_pd(im + j + t);
            __m512d Vr = _mm512_fmsub_pd(wr, vr, _mm512_mul_pd(wi, vi));
            __m512d Vi = _mm512_fmadd_pd(wr, vi, _mm512_mul_pd(wi, vr));
            _mm512_store_pd(re + j, _mm512_add_pd(ur, Vr));
            _mm512_store_pd(im + j, _mm512_add_pd(ui, Vi));
            _mm512_store_pd(re + j + t, _mm512_sub_pd(ur, Vr));
            _mm512_store_pd(im + j + t, _mm512_sub_pd(ui, Vi));
        }
        t >>= 1;
        ngt <<= 1;
        g0 <<= 1;
        cnt = 2;
    }

    /* Big stages, radix-4 pairs: quad (j, j+tB, j+t, j+t+tB) is closed under
     * both stages, so two stages cost one pass over the block. */
    for (; t >= 32; t >>= 2, ngt <<= 2, g0 <<= 2, cnt <<= 2)
    {
        size_t tB = t >> 1;
        for (size_t k = 0; k < cnt; k++)
        {
            size_t gg = g0 + k;
            __m512d war = _mm512_set1_pd(c->ws_re[ngt + gg]);
            __m512d wai = _mm512_set1_pd(c->ws_im[ngt + gg]);
            __m512d wb0r = _mm512_set1_pd(c->ws_re[2 * ngt + 2 * gg]);
            __m512d wb0i = _mm512_set1_pd(c->ws_im[2 * ngt + 2 * gg]);
            __m512d wb1r = _mm512_set1_pd(c->ws_re[2 * ngt + 2 * gg + 1]);
            __m512d wb1i = _mm512_set1_pd(c->ws_im[2 * ngt + 2 * gg + 1]);
            double *pr = re + 2 * k * t, *pi = im + 2 * k * t;
            for (size_t j = 0; j < tB; j += 8)
            {
                __m512d x0r = _mm512_load_pd(pr + j);
                __m512d x0i = _mm512_load_pd(pi + j);
                __m512d x1r = _mm512_load_pd(pr + j + tB);
                __m512d x1i = _mm512_load_pd(pi + j + tB);
                __m512d x2r = _mm512_load_pd(pr + j + t);
                __m512d x2i = _mm512_load_pd(pi + j + t);
                __m512d x3r = _mm512_load_pd(pr + j + t + tB);
                __m512d x3i = _mm512_load_pd(pi + j + t + tB);

                /* stage A (dist t): (x0,x2) and (x1,x3) with wa */
                __m512d Vr = _mm512_fmsub_pd(war, x2r, _mm512_mul_pd(wai, x2i));
                __m512d Vi = _mm512_fmadd_pd(war, x2i, _mm512_mul_pd(wai, x2r));
                x2r = _mm512_sub_pd(x0r, Vr);
                x2i = _mm512_sub_pd(x0i, Vi);
                x0r = _mm512_add_pd(x0r, Vr);
                x0i = _mm512_add_pd(x0i, Vi);
                Vr = _mm512_fmsub_pd(war, x3r, _mm512_mul_pd(wai, x3i));
                Vi = _mm512_fmadd_pd(war, x3i, _mm512_mul_pd(wai, x3r));
                x3r = _mm512_sub_pd(x1r, Vr);
                x3i = _mm512_sub_pd(x1i, Vi);
                x1r = _mm512_add_pd(x1r, Vr);
                x1i = _mm512_add_pd(x1i, Vi);

                /* stage B (dist tB): (x0,x1) with wb0, (x2,x3) with wb1 */
                Vr = _mm512_fmsub_pd(wb0r, x1r, _mm512_mul_pd(wb0i, x1i));
                Vi = _mm512_fmadd_pd(wb0r, x1i, _mm512_mul_pd(wb0i, x1r));
                _mm512_store_pd(pr + j + tB, _mm512_sub_pd(x0r, Vr));
                _mm512_store_pd(pi + j + tB, _mm512_sub_pd(x0i, Vi));
                _mm512_store_pd(pr + j, _mm512_add_pd(x0r, Vr));
                _mm512_store_pd(pi + j, _mm512_add_pd(x0i, Vi));
                Vr = _mm512_fmsub_pd(wb1r, x3r, _mm512_mul_pd(wb1i, x3i));
                Vi = _mm512_fmadd_pd(wb1r, x3i, _mm512_mul_pd(wb1i, x3r));
                _mm512_store_pd(pr + j + t + tB, _mm512_sub_pd(x2r, Vr));
                _mm512_store_pd(pi + j + t + tB, _mm512_sub_pd(x2i, Vi));
                _mm512_store_pd(pr + j + t, _mm512_add_pd(x2r, Vr));
                _mm512_store_pd(pi + j + t, _mm512_add_pd(x2i, Vi));
            }
        }
    }

    /* Fused tail: t = 8, 4, 2, 1 in one register-resident pass per block of 16
     * complex, reading the scratch block and writing the output block. The
     * tables are indexed by the ABSOLUTE 16-complex block number. */
    {
        const size_t m = c->m;
        const size_t b0 = off >> 4;
        for (size_t bl = 0; bl < len / 16; bl++)
        {
            const size_t b = b0 + bl;
            __m512d r0 = _mm512_load_pd(re + 16 * bl);
            __m512d r1 = _mm512_load_pd(re + 16 * bl + 8);
            __m512d i0 = _mm512_load_pd(im + 16 * bl);
            __m512d i1 = _mm512_load_pd(im + 16 * bl + 8);

            /* t=8: u = vec 0, v = vec 1, one twiddle per block (w=1 for b=0). */
            {
                __m512d wr = _mm512_set1_pd(c->ws_re[m / 16 + b]);
                __m512d wi = _mm512_set1_pd(c->ws_im[m / 16 + b]);
                __m512d Vr = _mm512_fmsub_pd(wr, r1, _mm512_mul_pd(wi, i1));
                __m512d Vi = _mm512_fmadd_pd(wr, i1, _mm512_mul_pd(wi, r1));
                r1 = _mm512_sub_pd(r0, Vr);
                i1 = _mm512_sub_pd(i0, Vi);
                r0 = _mm512_add_pd(r0, Vr);
                i0 = _mm512_add_pd(i0, Vi);
            }

            /* t=4, t=2, t=1 on each vector, per-lane sign-absorbed tables. */
#define CT_SMALL(rr, ii, IU, IV, WR, WI)                                                           \
    {                                                                                              \
        __m512d ur = _mm512_permutexvar_pd(IU, rr);                                                \
        __m512d ui = _mm512_permutexvar_pd(IU, ii);                                                \
        __m512d vr = _mm512_permutexvar_pd(IV, rr);                                                \
        __m512d vi = _mm512_permutexvar_pd(IV, ii);                                                \
        rr = _mm512_fnmadd_pd(WI, vi, _mm512_fmadd_pd(WR, vr, ur));                                \
        ii = _mm512_fmadd_pd(WI, vr, _mm512_fmadd_pd(WR, vi, ui));                                 \
    }
            CT_SMALL(r0, i0, IU4, IV4, TAIL(ct4_re, 2 * b), TAIL(ct4_im, 2 * b))
            CT_SMALL(r1, i1, IU4, IV4, TAIL(ct4_re, 2 * b + 1), TAIL(ct4_im, 2 * b + 1))
            CT_SMALL(r0, i0, IU2, IV2, TAIL(ct2_re, 2 * b), TAIL(ct2_im, 2 * b))
            CT_SMALL(r1, i1, IU2, IV2, TAIL(ct2_re, 2 * b + 1), TAIL(ct2_im, 2 * b + 1))
            CT_SMALL(r0, i0, IU1, IV1, TAIL(ct1_re, 2 * b), TAIL(ct1_im, 2 * b))
            CT_SMALL(r1, i1, IU1, IV1, TAIL(ct1_re, 2 * b + 1), TAIL(ct1_im, 2 * b + 1))
#undef CT_SMALL

            _mm512_store_pd(out_re + 16 * bl, r0);
            _mm512_store_pd(out_re + 16 * bl + 8, r1);
            _mm512_store_pd(out_im + 16 * bl, i0);
            _mm512_store_pd(out_im + 16 * bl + 8, i1);
        }
    }
}

/* One radix-4 level (stages t = len/2 then len/4) over this block, then the
 * four quarters depth-first. */
static void rec_ct(const struct _FFT_Plan *c, double *re, double *im, double *out_re,
                   double *out_im, size_t len, size_t off, size_t ngt, size_t g0)
{
    if (len <= c->leaf)
    {
        leaf_ct(c, re, im, out_re, out_im, len, off, ngt, g0);
        return;
    }
    const size_t t = len >> 1, tB = len >> 2;
    __m512d war = _mm512_set1_pd(c->ws_re[ngt + g0]);
    __m512d wai = _mm512_set1_pd(c->ws_im[ngt + g0]);
    __m512d wb0r = _mm512_set1_pd(c->ws_re[2 * ngt + 2 * g0]);
    __m512d wb0i = _mm512_set1_pd(c->ws_im[2 * ngt + 2 * g0]);
    __m512d wb1r = _mm512_set1_pd(c->ws_re[2 * ngt + 2 * g0 + 1]);
    __m512d wb1i = _mm512_set1_pd(c->ws_im[2 * ngt + 2 * g0 + 1]);
    for (size_t j = 0; j < tB; j += 8)
    {
        __m512d x0r = _mm512_load_pd(re + j);
        __m512d x0i = _mm512_load_pd(im + j);
        __m512d x1r = _mm512_load_pd(re + j + tB);
        __m512d x1i = _mm512_load_pd(im + j + tB);
        __m512d x2r = _mm512_load_pd(re + j + t);
        __m512d x2i = _mm512_load_pd(im + j + t);
        __m512d x3r = _mm512_load_pd(re + j + t + tB);
        __m512d x3i = _mm512_load_pd(im + j + t + tB);

        __m512d Vr = _mm512_fmsub_pd(war, x2r, _mm512_mul_pd(wai, x2i));
        __m512d Vi = _mm512_fmadd_pd(war, x2i, _mm512_mul_pd(wai, x2r));
        x2r = _mm512_sub_pd(x0r, Vr);
        x2i = _mm512_sub_pd(x0i, Vi);
        x0r = _mm512_add_pd(x0r, Vr);
        x0i = _mm512_add_pd(x0i, Vi);
        Vr = _mm512_fmsub_pd(war, x3r, _mm512_mul_pd(wai, x3i));
        Vi = _mm512_fmadd_pd(war, x3i, _mm512_mul_pd(wai, x3r));
        x3r = _mm512_sub_pd(x1r, Vr);
        x3i = _mm512_sub_pd(x1i, Vi);
        x1r = _mm512_add_pd(x1r, Vr);
        x1i = _mm512_add_pd(x1i, Vi);

        Vr = _mm512_fmsub_pd(wb0r, x1r, _mm512_mul_pd(wb0i, x1i));
        Vi = _mm512_fmadd_pd(wb0r, x1i, _mm512_mul_pd(wb0i, x1r));
        _mm512_store_pd(re + j + tB, _mm512_sub_pd(x0r, Vr));
        _mm512_store_pd(im + j + tB, _mm512_sub_pd(x0i, Vi));
        _mm512_store_pd(re + j, _mm512_add_pd(x0r, Vr));
        _mm512_store_pd(im + j, _mm512_add_pd(x0i, Vi));
        Vr = _mm512_fmsub_pd(wb1r, x3r, _mm512_mul_pd(wb1i, x3i));
        Vi = _mm512_fmadd_pd(wb1r, x3i, _mm512_mul_pd(wb1i, x3r));
        _mm512_store_pd(re + j + t + tB, _mm512_sub_pd(x2r, Vr));
        _mm512_store_pd(im + j + t + tB, _mm512_sub_pd(x2i, Vi));
        _mm512_store_pd(re + j + t, _mm512_add_pd(x2r, Vr));
        _mm512_store_pd(im + j + t, _mm512_add_pd(x2i, Vi));
    }

    const size_t q = len >> 2;
    for (size_t k = 0; k < 4; k++)
    {
        rec_ct(c, re + k * q, im + k * q, out_re + k * q, out_im + k * q, q, off + k * q, ngt << 2,
               (g0 << 2) + k);
    }
}

/* CT_NR: u_k = sum_j c_j zeta^{(4k+1)j} at position brv(k) of out (split
 * re/im), from the natural-order c in s (split re/im). */
static void fft_ct(const struct _FFT_Plan *c, double *s, double *out)
{
    const size_t m = c->m;
    double *re = s, *im = s + m;

    if (m <= c->leaf)
    {
        leaf_ct(c, re, im, out, out + m, m, 0, 1, 0);
        return;
    }
    /* An odd number of stages above the leaf: spend the leftover radix-2 at
     * t = m/2. */
    if ((lg2(m) - lg2(c->leaf)) & 1)
    {
        const size_t t = m >> 1;
        /* ws[1] = exp(i*pi/4), not 1 — see the note in leaf_ct. */
        __m512d wr = _mm512_set1_pd(c->ws_re[1]);
        __m512d wi = _mm512_set1_pd(c->ws_im[1]);
        for (size_t j = 0; j < t; j += 8)
        {
            __m512d ur = _mm512_load_pd(re + j), ui = _mm512_load_pd(im + j);
            __m512d vr = _mm512_load_pd(re + j + t), vi = _mm512_load_pd(im + j + t);
            __m512d Vr = _mm512_fmsub_pd(wr, vr, _mm512_mul_pd(wi, vi));
            __m512d Vi = _mm512_fmadd_pd(wr, vi, _mm512_mul_pd(wi, vr));
            _mm512_store_pd(re + j, _mm512_add_pd(ur, Vr));
            _mm512_store_pd(im + j, _mm512_add_pd(ui, Vi));
            _mm512_store_pd(re + j + t, _mm512_sub_pd(ur, Vr));
            _mm512_store_pd(im + j + t, _mm512_sub_pd(ui, Vi));
        }
        rec_ct(c, re, im, out, out + m, t, 0, 2, 0);
        rec_ct(c, re + t, im + t, out + t, out + m + t, t, t, 2, 1);
        return;
    }
    rec_ct(c, re, im, out, out + m, m, 0, 1, 0);
}

/* ---------------------------------------------------------------------------
 * GS_RN with conjugate twiddles: exact inverse of fft_ct scaled by m.
 * Mirror of the above -- children first, then the merging radix-4 level.
 * ------------------------------------------------------------------------- */

static void leaf_gs(const struct _FFT_Plan *c, const double *in_re, const double *in_im, double *re,
                    double *im, size_t len, size_t off, size_t ngt, size_t g0)
{
    const size_t m = c->m;

    const __m512d SG4 = _mm512_setr_pd(1, 1, 1, 1, -1, -1, -1, -1);
    const __m512d SG2 = _mm512_setr_pd(1, 1, -1, -1, 1, 1, -1, -1);
    const __m512d SG1 = _mm512_setr_pd(1, -1, 1, -1, 1, -1, 1, -1);

    /* Fused head: t = 1, 2, 4, 8 in one register-resident pass per block of 16
     * complex, reading the input block and writing the scratch block. */
    {
        const size_t b0 = off >> 4;
        const __m512d SC = _mm512_set1_pd(c->inv_m);
        for (size_t bl = 0; bl < len / 16; bl++)
        {
            const size_t b = b0 + bl;
            /* The 1/m normalisation is applied here rather than in a pass of its
             * own: four multiplies per 16 complex values, in a pass that already
             * touches every element. */
            __m512d r0 = _mm512_mul_pd(_mm512_load_pd(in_re + 16 * bl), SC);
            __m512d r1 = _mm512_mul_pd(_mm512_load_pd(in_re + 16 * bl + 8), SC);
            __m512d i0 = _mm512_mul_pd(_mm512_load_pd(in_im + 16 * bl), SC);
            __m512d i1 = _mm512_mul_pd(_mm512_load_pd(in_im + 16 * bl + 8), SC);

#define GS_SMALL(rr, ii, IU, IV, SG, WR, WI)                                                       \
    {                                                                                              \
        __m512d sr =                                                                               \
            _mm512_fmadd_pd(SG, _mm512_permutexvar_pd(IV, rr), _mm512_permutexvar_pd(IU, rr));     \
        __m512d si =                                                                               \
            _mm512_fmadd_pd(SG, _mm512_permutexvar_pd(IV, ii), _mm512_permutexvar_pd(IU, ii));     \
        rr = _mm512_fmsub_pd(WR, sr, _mm512_mul_pd(WI, si));                                       \
        ii = _mm512_fmadd_pd(WR, si, _mm512_mul_pd(WI, sr));                                       \
    }
            GS_SMALL(r0, i0, IU1, IV1, SG1, TAIL(gs1_re, 2 * b), TAIL(gs1_im, 2 * b))
            GS_SMALL(r1, i1, IU1, IV1, SG1, TAIL(gs1_re, 2 * b + 1), TAIL(gs1_im, 2 * b + 1))
            GS_SMALL(r0, i0, IU2, IV2, SG2, TAIL(gs2_re, 2 * b), TAIL(gs2_im, 2 * b))
            GS_SMALL(r1, i1, IU2, IV2, SG2, TAIL(gs2_re, 2 * b + 1), TAIL(gs2_im, 2 * b + 1))
            GS_SMALL(r0, i0, IU4, IV4, SG4, TAIL(gs4_re, 2 * b), TAIL(gs4_im, 2 * b))
            GS_SMALL(r1, i1, IU4, IV4, SG4, TAIL(gs4_re, 2 * b + 1), TAIL(gs4_im, 2 * b + 1))
#undef GS_SMALL

            /* t=8: u' = u + v; v' = conj(w) * (u - v), one twiddle per block. */
            {
                __m512d wr = _mm512_set1_pd(c->ws_re[m / 16 + b]);
                __m512d wi = _mm512_set1_pd(c->wsn_im[m / 16 + b]);
                __m512d sr = _mm512_sub_pd(r0, r1);
                __m512d si = _mm512_sub_pd(i0, i1);
                r0 = _mm512_add_pd(r0, r1);
                i0 = _mm512_add_pd(i0, i1);
                r1 = _mm512_fmsub_pd(wr, sr, _mm512_mul_pd(wi, si));
                i1 = _mm512_fmadd_pd(wr, si, _mm512_mul_pd(wi, sr));
            }

            _mm512_store_pd(re + 16 * bl, r0);
            _mm512_store_pd(re + 16 * bl + 8, r1);
            _mm512_store_pd(im + 16 * bl, i0);
            _mm512_store_pd(im + 16 * bl + 8, i1);
        }
    }

    /* Big stages t = 16 .. len/2 ascending, radix-4 pairs (tB = t, tA = 2t).
     * At distance tA the absolute level is ngt << d and this block's first
     * group is g0 << d, with d = lg2(len) - 1 - lg2(tA); d starts at
     * lg2(len) - 6 for tA = 32 and drops by 2 per iteration. */
    size_t t = 16;
    if (2 * t <= len / 2)
    {
        const size_t d0 = lg2(len) - 6;
        size_t ngA = ngt << d0, gA = g0 << d0, cnt = (size_t)1 << d0;
        for (; 2 * t <= len / 2; t <<= 2, ngA >>= 2, gA >>= 2, cnt >>= 2)
        {
            const size_t tA = 2 * t;
            for (size_t k = 0; k < cnt; k++)
            {
                const size_t gg = gA + k;
                __m512d war = _mm512_set1_pd(c->ws_re[ngA + gg]);
                __m512d wai = _mm512_set1_pd(c->wsn_im[ngA + gg]);
                __m512d wb0r = _mm512_set1_pd(c->ws_re[2 * ngA + 2 * gg]);
                __m512d wb0i = _mm512_set1_pd(c->wsn_im[2 * ngA + 2 * gg]);
                __m512d wb1r = _mm512_set1_pd(c->ws_re[2 * ngA + 2 * gg + 1]);
                __m512d wb1i = _mm512_set1_pd(c->wsn_im[2 * ngA + 2 * gg + 1]);
                double *pr = re + 2 * k * tA, *pi = im + 2 * k * tA;
                for (size_t j = 0; j < t; j += 8)
                {
                    __m512d x0r = _mm512_load_pd(pr + j);
                    __m512d x0i = _mm512_load_pd(pi + j);
                    __m512d x1r = _mm512_load_pd(pr + j + t);
                    __m512d x1i = _mm512_load_pd(pi + j + t);
                    __m512d x2r = _mm512_load_pd(pr + j + tA);
                    __m512d x2i = _mm512_load_pd(pi + j + tA);
                    __m512d x3r = _mm512_load_pd(pr + j + tA + t);
                    __m512d x3i = _mm512_load_pd(pi + j + tA + t);

                    /* stage tB: (x0,x1) with conj wb0, (x2,x3) with conj wb1 */
                    __m512d sr = _mm512_sub_pd(x0r, x1r);
                    __m512d si = _mm512_sub_pd(x0i, x1i);
                    x0r = _mm512_add_pd(x0r, x1r);
                    x0i = _mm512_add_pd(x0i, x1i);
                    x1r = _mm512_fmsub_pd(wb0r, sr, _mm512_mul_pd(wb0i, si));
                    x1i = _mm512_fmadd_pd(wb0r, si, _mm512_mul_pd(wb0i, sr));
                    sr = _mm512_sub_pd(x2r, x3r);
                    si = _mm512_sub_pd(x2i, x3i);
                    x2r = _mm512_add_pd(x2r, x3r);
                    x2i = _mm512_add_pd(x2i, x3i);
                    x3r = _mm512_fmsub_pd(wb1r, sr, _mm512_mul_pd(wb1i, si));
                    x3i = _mm512_fmadd_pd(wb1r, si, _mm512_mul_pd(wb1i, sr));

                    /* stage tA: (x0,x2) and (x1,x3) with conj wa */
                    sr = _mm512_sub_pd(x0r, x2r);
                    si = _mm512_sub_pd(x0i, x2i);
                    _mm512_store_pd(pr + j, _mm512_add_pd(x0r, x2r));
                    _mm512_store_pd(pi + j, _mm512_add_pd(x0i, x2i));
                    _mm512_store_pd(pr + j + tA, _mm512_fmsub_pd(war, sr, _mm512_mul_pd(wai, si)));
                    _mm512_store_pd(pi + j + tA, _mm512_fmadd_pd(war, si, _mm512_mul_pd(wai, sr)));
                    sr = _mm512_sub_pd(x1r, x3r);
                    si = _mm512_sub_pd(x1i, x3i);
                    _mm512_store_pd(pr + j + t, _mm512_add_pd(x1r, x3r));
                    _mm512_store_pd(pi + j + t, _mm512_add_pd(x1i, x3i));
                    _mm512_store_pd(pr + j + tA + t,
                                    _mm512_fmsub_pd(war, sr, _mm512_mul_pd(wai, si)));
                    _mm512_store_pd(pi + j + tA + t,
                                    _mm512_fmadd_pd(war, si, _mm512_mul_pd(wai, sr)));
                }
            }
        }
    }

    /* Odd big-stage count -> trailing radix-2 at t = len/2. */
    if (t == len / 2)
    {
        /* GS mirror of the CT case above: always twiddled. */
        __m512d wr = _mm512_set1_pd(c->ws_re[ngt + g0]);
        __m512d wi = _mm512_set1_pd(c->wsn_im[ngt + g0]);
        for (size_t j = 0; j < t; j += 8)
        {
            __m512d ur = _mm512_load_pd(re + j), ui = _mm512_load_pd(im + j);
            __m512d vr = _mm512_load_pd(re + j + t), vi = _mm512_load_pd(im + j + t);
            __m512d sr = _mm512_sub_pd(ur, vr);
            __m512d si = _mm512_sub_pd(ui, vi);
            _mm512_store_pd(re + j, _mm512_add_pd(ur, vr));
            _mm512_store_pd(im + j, _mm512_add_pd(ui, vi));
            _mm512_store_pd(re + j + t, _mm512_fmsub_pd(wr, sr, _mm512_mul_pd(wi, si)));
            _mm512_store_pd(im + j + t, _mm512_fmadd_pd(wr, si, _mm512_mul_pd(wi, sr)));
        }
    }
}

static void rec_gs(const struct _FFT_Plan *c, const double *in_re, const double *in_im, double *re,
                   double *im, size_t len, size_t off, size_t ngt, size_t g0)
{
    if (len <= c->leaf)
    {
        leaf_gs(c, in_re, in_im, re, im, len, off, ngt, g0);
        return;
    }
    const size_t q = len >> 2;
    for (size_t k = 0; k < 4; k++)
    {
        rec_gs(c, in_re + k * q, in_im + k * q, re + k * q, im + k * q, q, off + k * q, ngt << 2,
               (g0 << 2) + k);
    }

    const size_t tA = len >> 1, t = len >> 2;
    __m512d war = _mm512_set1_pd(c->ws_re[ngt + g0]);
    __m512d wai = _mm512_set1_pd(c->wsn_im[ngt + g0]);
    __m512d wb0r = _mm512_set1_pd(c->ws_re[2 * ngt + 2 * g0]);
    __m512d wb0i = _mm512_set1_pd(c->wsn_im[2 * ngt + 2 * g0]);
    __m512d wb1r = _mm512_set1_pd(c->ws_re[2 * ngt + 2 * g0 + 1]);
    __m512d wb1i = _mm512_set1_pd(c->wsn_im[2 * ngt + 2 * g0 + 1]);
    for (size_t j = 0; j < t; j += 8)
    {
        __m512d x0r = _mm512_load_pd(re + j);
        __m512d x0i = _mm512_load_pd(im + j);
        __m512d x1r = _mm512_load_pd(re + j + t);
        __m512d x1i = _mm512_load_pd(im + j + t);
        __m512d x2r = _mm512_load_pd(re + j + tA);
        __m512d x2i = _mm512_load_pd(im + j + tA);
        __m512d x3r = _mm512_load_pd(re + j + tA + t);
        __m512d x3i = _mm512_load_pd(im + j + tA + t);

        __m512d sr = _mm512_sub_pd(x0r, x1r);
        __m512d si = _mm512_sub_pd(x0i, x1i);
        x0r = _mm512_add_pd(x0r, x1r);
        x0i = _mm512_add_pd(x0i, x1i);
        x1r = _mm512_fmsub_pd(wb0r, sr, _mm512_mul_pd(wb0i, si));
        x1i = _mm512_fmadd_pd(wb0r, si, _mm512_mul_pd(wb0i, sr));
        sr = _mm512_sub_pd(x2r, x3r);
        si = _mm512_sub_pd(x2i, x3i);
        x2r = _mm512_add_pd(x2r, x3r);
        x2i = _mm512_add_pd(x2i, x3i);
        x3r = _mm512_fmsub_pd(wb1r, sr, _mm512_mul_pd(wb1i, si));
        x3i = _mm512_fmadd_pd(wb1r, si, _mm512_mul_pd(wb1i, sr));

        sr = _mm512_sub_pd(x0r, x2r);
        si = _mm512_sub_pd(x0i, x2i);
        _mm512_store_pd(re + j, _mm512_add_pd(x0r, x2r));
        _mm512_store_pd(im + j, _mm512_add_pd(x0i, x2i));
        _mm512_store_pd(re + j + tA, _mm512_fmsub_pd(war, sr, _mm512_mul_pd(wai, si)));
        _mm512_store_pd(im + j + tA, _mm512_fmadd_pd(war, si, _mm512_mul_pd(wai, sr)));
        sr = _mm512_sub_pd(x1r, x3r);
        si = _mm512_sub_pd(x1i, x3i);
        _mm512_store_pd(re + j + t, _mm512_add_pd(x1r, x3r));
        _mm512_store_pd(im + j + t, _mm512_add_pd(x1i, x3i));
        _mm512_store_pd(re + j + tA + t, _mm512_fmsub_pd(war, sr, _mm512_mul_pd(wai, si)));
        _mm512_store_pd(im + j + tA + t, _mm512_fmadd_pd(war, si, _mm512_mul_pd(wai, sr)));
    }
}

static void fft_gs(const struct _FFT_Plan *c, const double *in, double *s)
{
    const size_t m = c->m;
    double *re = s, *im = s + m;

    if (m <= c->leaf)
    {
        leaf_gs(c, in, in + m, re, im, m, 0, 1, 0);
        return;
    }
    if ((lg2(m) - lg2(c->leaf)) & 1)
    {
        const size_t t = m >> 1;
        rec_gs(c, in, in + m, re, im, t, 0, 2, 0);
        rec_gs(c, in + t, in + m + t, re + t, im + t, t, t, 2, 1);
        __m512d wr = _mm512_set1_pd(c->ws_re[1]);
        __m512d wi = _mm512_set1_pd(c->wsn_im[1]);
        for (size_t j = 0; j < t; j += 8)
        {
            __m512d ur = _mm512_load_pd(re + j), ui = _mm512_load_pd(im + j);
            __m512d vr = _mm512_load_pd(re + j + t), vi = _mm512_load_pd(im + j + t);
            __m512d sr = _mm512_sub_pd(ur, vr);
            __m512d si = _mm512_sub_pd(ui, vi);
            _mm512_store_pd(re + j, _mm512_add_pd(ur, vr));
            _mm512_store_pd(im + j, _mm512_add_pd(ui, vi));
            _mm512_store_pd(re + j + t, _mm512_fmsub_pd(wr, sr, _mm512_mul_pd(wi, si)));
            _mm512_store_pd(im + j + t, _mm512_fmadd_pd(wr, si, _mm512_mul_pd(wi, sr)));
        }
        return;
    }
    rec_gs(c, in, in + m, re, im, m, 0, 1, 0);
}

#endif // VFHE_HAVE_AVX512F

static bool fft_vectorized(const struct _FFT_Plan *c)
{
#if VFHE_HAVE_AVX512F
    return c->n >= FFT_MIN_VECTOR_LEN;
#else
    (void)c;
    return false;
#endif
}

/* The fused tail's per-lane tables: one vector per vector of data, the
 * forward's with the butterfly's sign folded in (+w on the u lanes, -w on the
 * v lanes), the inverse's with the identity (1, 0) on the u lanes and conj(w)
 * on the v lanes. Every value is already in ws; storing them expanded trades
 * memory for a shorter inner loop, the better trade while the table fits in
 * cache. */
static void gen_tail_tables(struct _FFT_Plan *c)
{
    const size_t m = c->m;
    for (size_t k = 0; k < m / 8; k++)
    {
        double *ct4r = c->ct4_re + 8 * k, *ct4i = c->ct4_im + 8 * k;
        double *gs4r = c->gs4_re + 8 * k, *gs4i = c->gs4_im + 8 * k;
        double *ct2r = c->ct2_re + 8 * k, *ct2i = c->ct2_im + 8 * k;
        double *gs2r = c->gs2_re + 8 * k, *gs2i = c->gs2_im + 8 * k;
        double *ct1r = c->ct1_re + 8 * k, *ct1i = c->ct1_im + 8 * k;
        double *gs1r = c->gs1_re + 8 * k, *gs1i = c->gs1_im + 8 * k;

        // t = 4: group k, w = ws[m/8 + k], lanes 0-3 u and 4-7 v
        const double wr = c->ws_re[m / 8 + k], wi = c->ws_im[m / 8 + k];
        for (int l = 0; l < 4; l++)
        {
            ct4r[l] = wr, ct4i[l] = wi;
            ct4r[l + 4] = -wr, ct4i[l + 4] = -wi;
            gs4r[l] = 1.0, gs4i[l] = 0.0;
            gs4r[l + 4] = wr, gs4i[l + 4] = -wi;
        }
        // t = 2: groups 2k (lanes 0-3) and 2k+1 (lanes 4-7)
        for (int g = 0; g < 2; g++)
        {
            const double ar = c->ws_re[m / 4 + 2 * k + g], ai = c->ws_im[m / 4 + 2 * k + g];
            for (int l = 0; l < 2; l++)
            {
                ct2r[4 * g + l] = ar, ct2i[4 * g + l] = ai;
                ct2r[4 * g + 2 + l] = -ar, ct2i[4 * g + 2 + l] = -ai;
                gs2r[4 * g + l] = 1.0, gs2i[4 * g + l] = 0.0;
                gs2r[4 * g + 2 + l] = ar, gs2i[4 * g + 2 + l] = -ai;
            }
        }
        // t = 1: groups 4k+g at lanes (2g, 2g+1)
        for (int g = 0; g < 4; g++)
        {
            const double ar = c->ws_re[m / 2 + 4 * k + g], ai = c->ws_im[m / 2 + 4 * k + g];
            ct1r[2 * g] = ar, ct1i[2 * g] = ai;
            ct1r[2 * g + 1] = -ar, ct1i[2 * g + 1] = -ai;
            gs1r[2 * g] = 1.0, gs1i[2 * g] = 0.0;
            gs1r[2 * g + 1] = ar, gs1i[2 * g + 1] = -ai;
        }
    }
}

FFT_Plan fft_new_plan(uint64_t n)
{
    if (n < 2 || n > FFT_MAX_LEN || (n & (n - 1)) != 0)
    {
        fprintf(stderr, "fft_new_plan: n = %llu is not a power of two in [2, 2^20]\n",
                (unsigned long long)n);
        return NULL;
    }
    FFT_Plan c = (FFT_Plan)safe_malloc(sizeof(struct _FFT_Plan));
    c->n = n;
    c->m = n / 2;
    c->leaf = pick_leaf(c->m);
    c->inv_m = 1.0 / (double)c->m;
    const size_t m = c->m;
    c->ws_re = (double *)safe_aligned_malloc(m * sizeof(double));
    c->ws_im = (double *)safe_aligned_malloc(m * sizeof(double));
    c->wsn_im = (double *)safe_aligned_malloc(m * sizeof(double));

    // Group i at level h: angle pi (4 brv(i) + 1) / (4h). Index 0 is unused.
#if !(LDBL_MANT_DIG <= DBL_MANT_DIG && defined(__APPLE__))
    const long double pi = 3.141592653589793238462643383279502884L;
#endif
    c->ws_re[0] = 1.0;
    c->ws_im[0] = 0.0;
    unsigned bits = 0;
    for (size_t h = 1; h < m; h <<= 1, bits++)
    {
        for (size_t i = 0; i < h; i++)
        {
#if LDBL_MANT_DIG <= DBL_MANT_DIG && defined(__APPLE__)
            // An odd integer over a power of two below 2^22: exact in double.
            const double x = (4.0 * (double)arith_bit_reverse(i, bits) + 1.0) / (4.0 * (double)h);
            c->ws_re[h + i] = __cospi(x);
            c->ws_im[h + i] = __sinpi(x);
#else
            const long double x = pi * (4.0L * (long double)arith_bit_reverse(i, bits) + 1.0L) /
                                  (4.0L * (long double)h);
            c->ws_re[h + i] = (double)cosl(x);
            c->ws_im[h + i] = (double)sinl(x);
#endif
        }
    }
    for (size_t i = 0; i < m; i++)
        c->wsn_im[i] = -c->ws_im[i];

    double **tail[] = {&c->ct4_re, &c->ct4_im, &c->ct2_re, &c->ct2_im, &c->ct1_re, &c->ct1_im,
                       &c->gs4_re, &c->gs4_im, &c->gs2_re, &c->gs2_im, &c->gs1_re, &c->gs1_im};
    const bool vectorized = fft_vectorized(c);
    for (size_t i = 0; i < sizeof(tail) / sizeof(tail[0]); i++)
        *tail[i] = vectorized ? (double *)safe_aligned_malloc(m * sizeof(double)) : NULL;
    if (vectorized)
        gen_tail_tables(c);
    return c;
}

void fft_free_plan(FFT_Plan plan)
{
    if (!plan)
        return;
    double *tables[] = {plan->ws_re,  plan->ws_im,  plan->wsn_im, plan->ct4_re, plan->ct4_im,
                        plan->ct2_re, plan->ct2_im, plan->ct1_re, plan->ct1_im, plan->gs4_re,
                        plan->gs4_im, plan->gs2_re, plan->gs2_im, plan->gs1_re, plan->gs1_im};
    for (size_t i = 0; i < sizeof(tables) / sizeof(tables[0]); i++)
        free(tables[i]);
    free(plan);
}

void fft_forward(double *x, FFT_Plan plan)
{
#if VFHE_HAVE_AVX512F
    if (fft_vectorized(plan))
    {
        fft_ct(plan, x, x);
        return;
    }
#endif
    fft_forward_scalar(x, plan);
}

void fft_inverse(double *x, FFT_Plan plan)
{
#if VFHE_HAVE_AVX512F
    if (fft_vectorized(plan))
    {
        fft_gs(plan, x, x);
        return;
    }
#endif
    fft_inverse_scalar(x, plan);
}

void fft_pointwise_mul(double *out, const double *a, const double *b, FFT_Plan plan)
{
    const size_t m = plan->m;
    size_t i = 0;
#if VFHE_HAVE_AVX512F
    if (fft_vectorized(plan))
    {
        for (; i < m; i += 8)
        {
            const __m512d ar = _mm512_load_pd(a + i), ai = _mm512_load_pd(a + m + i);
            const __m512d br = _mm512_load_pd(b + i), bi = _mm512_load_pd(b + m + i);
            _mm512_store_pd(out + i, _mm512_fmsub_pd(ar, br, _mm512_mul_pd(ai, bi)));
            _mm512_store_pd(out + m + i, _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br)));
        }
    }
#endif
    for (; i < m; i++)
    {
        const double ar = a[i], ai = a[m + i], br = b[i], bi = b[m + i];
        out[i] = ar * br - ai * bi;
        out[m + i] = ar * bi + ai * br;
    }
}
