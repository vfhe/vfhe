// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
//
// The negacyclic NTT over a pseudo-Mersenne field, on the limb planes of a
// PMFVector.
//
// Same transform basis as arith's RNS kernels: Cooley-Tukey natural-to-
// bit-reversed forward, Gentleman-Sande bit-reversed-to-natural inverse, and
// one flat table of the n powers of the root in bit-reversed order, indexed
// ws[m + i] by stage and block. The table is stored plane-wise -- limb k of the
// twiddle at index e is ws[k][e] -- so a stage can broadcast or gather one limb
// of it straight into the pv_t word the group cores compute on.
//
// A butterfly pairs a[j] with a[j + t]. While t is at least the group width the
// two live in different groups, and a stage is the group cores of
// pmf_vec_group.h over pairs of groups with the block's twiddle broadcast; on
// the portable engine, whose group is one element, every stage is of that kind.
// The last log2(PMF_VEC_GROUP) stages of the tuned engine pair lanes inside one
// group. Those are the one place in the pmf kernels where lanes interact, and
// pmf_ntt_stage_*_in_group is where: it fetches each lane's partner with a
// permute, forms the low and high halves of every pair in both lanes, and runs
// the same cores over the whole group -- twice the arithmetic of a wide stage,
// but no per-element extraction. Padding lanes of a short vector pair only with
// each other and see a zero twiddle, so they stay canonical and stay apart from
// the live ones.
//
// Every element is canonical between operations: the cores reduce fully, and
// IFMA's 52-bit operand truncation leaves no room for a lazy variant.
//
// pmf_ref_ntt_* at the bottom are the scalar oracle: the same loops over the
// element kernels of pmf.c, compiled into every engine.
#include <arith.h>
#include <inttypes.h>
#include <string.h>

#include "arith_internal.h"
#include "pmf_vec_group.h"

// --- scalar helpers -------------------------------------------------------

static void pmf_ref_set_one(uint64_t *out)
{
    memset(out, 0, PMF_LANES * sizeof(*out));
    out[0] = 1;
}

void pmf_ref_pow(uint64_t *out, const uint64_t *a, const uint64_t *exp, uint64_t exp_limbs,
                 PMFParams params)
{
    uint64_t base[PMF_LANES], result[PMF_LANES];

    memcpy(base, a, sizeof(base));
    pmf_ref_set_one(result);
    // Right to left over the exponent bits, least significant limb first.
    for (uint64_t k = 0; k < exp_limbs; k++)
    {
        for (uint64_t bit = 0; bit < PMF_LIMB_BITS; bit++)
        {
            if ((exp[k] >> bit) & 1)
                pmf_ref_mul(result, result, base, params);
            pmf_ref_mul(base, base, base, params);
        }
    }
    memcpy(out, result, sizeof(result));
}

// a^e for a word-sized exponent.
static void pmf_ref_pow_u64(uint64_t *out, const uint64_t *a, uint64_t e, PMFParams params)
{
    const uint64_t exp[2] = {e & PMF_LIMB_MASK, e >> PMF_LIMB_BITS};
    pmf_ref_pow(out, a, exp, 2, params);
}

// a^(p-2), Fermat. Only zero has no inverse, and no caller passes it.
static void pmf_ref_inverse(uint64_t *out, const uint64_t *a, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t exp[PMF_MAX_LIMBS];
    uint64_t borrow = 2;

    memcpy(exp, params->p, L * sizeof(*exp));
    for (uint64_t k = 0; k < L && borrow; k++)
    {
        if (exp[k] >= borrow)
        {
            exp[k] -= borrow;
            borrow = 0;
        }
        else
        {
            exp[k] = exp[k] + (PMF_LIMB_MASK + 1) - borrow;
            borrow = 1;
        }
    }
    pmf_ref_pow(out, a, exp, L, params);
}

static uint64_t reverse_bits_pmf(uint64_t x, uint64_t bits)
{
    uint64_t res = 0;
    for (uint64_t i = 0; i < bits; i++)
    {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

// --- the plan -------------------------------------------------------------

// L planes of n words: plane k holds limb k of root^i at index brv(i).
static uint64_t **pmf_ntt_table(uint64_t n, uint64_t logn, const uint64_t *root, PMFParams params)
{
    const uint64_t L = params->limbs;
    uint64_t **planes = (uint64_t **)malloc(L * sizeof(*planes));
    uint64_t power[PMF_LANES];

    for (uint64_t k = 0; k < L; k++)
        planes[k] = (uint64_t *)malloc(n * sizeof(uint64_t));

    pmf_ref_set_one(power);
    for (uint64_t i = 0; i < n; i++)
    {
        const uint64_t idx = reverse_bits_pmf(i, logn);
        for (uint64_t k = 0; k < L; k++)
            planes[k][idx] = power[k];
        pmf_ref_mul(power, power, root, params);
    }
    return planes;
}

static void pmf_ntt_free_table(uint64_t **planes, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        free(planes[k]);
    free(planes);
}

PMFNTTPlan pmf_ntt_new_plan(uint64_t n, const uint64_t *root_of_unity, PMFParams params)
{
    uint64_t logn = 0;
    uint64_t minus_one[PMF_LANES], check[PMF_LANES], count[PMF_LANES];

    if (n == 0 || (n & (n - 1)) != 0)
    {
        fprintf(stderr, "pmf_ntt_new_plan: n=%" PRIu64 " is not a power of two\n", n);
        return NULL;
    }
    while ((1ULL << logn) < n)
        logn++;

    // A primitive 2n-th root has psi^n == -1: that pins the order exactly, since
    // psi^(2n) == 1 then follows and no smaller power of two reaches 1 first.
    pmf_ref_set_one(minus_one);
    pmf_ref_neg(minus_one, minus_one, params);
    pmf_ref_pow_u64(check, root_of_unity, n, params);
    if (!pmf_is_equal(check, minus_one, params))
    {
        fprintf(stderr,
                "pmf_ntt_new_plan: the root given is not a primitive %" PRIu64
                "-th root of unity (psi^n != -1)\n",
                2 * n);
        return NULL;
    }

    PMFNTTPlan plan = (PMFNTTPlan)calloc(1, sizeof(*plan));
    if (plan == NULL)
        return NULL;
    plan->params = params;
    plan->n = n;
    plan->logn = logn;
    memcpy(plan->root_of_unity, root_of_unity, sizeof(plan->root_of_unity));
    // psi^(2n) == 1, so psi^(2n-1) is its inverse.
    pmf_ref_pow_u64(plan->inv_root_of_unity, root_of_unity, 2 * n - 1, params);
    // n < p, so as a field element it is n itself, and it is nonzero.
    pmf_ref_set_one(count);
    count[0] = n & PMF_LIMB_MASK;
    count[1] = n >> PMF_LIMB_BITS;
    pmf_ref_inverse(plan->inv_n, count, params);

    plan->ws_fwd = pmf_ntt_table(n, logn, plan->root_of_unity, params);
    plan->ws_inv = pmf_ntt_table(n, logn, plan->inv_root_of_unity, params);
    return plan;
}

void pmf_ntt_free_plan(PMFNTTPlan plan)
{
    if (plan == NULL)
        return;
    pmf_ntt_free_table(plan->ws_fwd, plan->params->limbs);
    pmf_ntt_free_table(plan->ws_inv, plan->params->limbs);
    free(plan);
}

// --- vector stages, pairs across groups -----------------------------------

// One twiddle, broadcast limb by limb.
static inline void pmf_ntt_load_twiddle(pv_t *W, uint64_t **ws, uint64_t e, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        W[k] = pv_bcast(ws[k][e]);
}

// Cooley-Tukey stage: m blocks of 2t elements, block i twiddled by ws[m + i].
// a[j] <- a[j] + w a[j+t], a[j+t] <- a[j] - w a[j+t].
static void pmf_ntt_stage_fwd_wide(PMFVector a, uint64_t **ws, uint64_t m, uint64_t t,
                                   PMFParams params)
{
    const uint64_t L = params->limbs;
    pv_t u[PMF_MAX_LIMBS], v[PMF_MAX_LIMBS], w[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1];

    for (uint64_t i = 0; i < m; i++)
    {
        const uint64_t j1 = 2 * i * t;
        pmf_ntt_load_twiddle(w, ws, m + i, L);
        for (uint64_t j = j1; j < j1 + t; j += PMF_VEC_GROUP)
        {
            pmf_vec_load_group(u, a, j, L);
            pmf_vec_load_group(v, a, j + t, L);
            pmf_vec_mul_group(T, v, w, params);
            for (uint64_t k = 0; k < L; k++)
                v[k] = T[k];
            pmf_vec_sub_group(T, u, v, params);
            pmf_vec_store_group(a, j + t, T, L);
            pmf_vec_add_group(T, u, v, params);
            pmf_vec_store_group(a, j, T, L);
        }
    }
}

// Gentleman-Sande stage: h blocks of 2t elements, block i twiddled by ws[h + i].
// a[j] <- a[j] + a[j+t], a[j+t] <- (a[j] - a[j+t]) w.
static void pmf_ntt_stage_inv_wide(PMFVector a, uint64_t **ws, uint64_t h, uint64_t t,
                                   PMFParams params)
{
    const uint64_t L = params->limbs;
    pv_t u[PMF_MAX_LIMBS], v[PMF_MAX_LIMBS], w[PMF_MAX_LIMBS], T[PMF_MAX_LIMBS + 1],
        D[PMF_MAX_LIMBS + 1];

    for (uint64_t i = 0; i < h; i++)
    {
        const uint64_t j1 = 2 * i * t;
        pmf_ntt_load_twiddle(w, ws, h + i, L);
        for (uint64_t j = j1; j < j1 + t; j += PMF_VEC_GROUP)
        {
            pmf_vec_load_group(u, a, j, L);
            pmf_vec_load_group(v, a, j + t, L);
            pmf_vec_add_group(T, u, v, params);
            pmf_vec_store_group(a, j, T, L);
            pmf_vec_sub_group(D, u, v, params);
            pmf_vec_mul_group(T, D, w, params);
            pmf_vec_store_group(a, j + t, T, L);
        }
    }
}

// --- vector stages, pairs inside a group (tuned engine only) ---------------

#if VFHE_HAVE_AVX512IFMA

// Lane l's partner at pair distance t is lane l ^ t.
static inline pv_t pmf_ntt_swap_lanes(pv_t a, uint64_t t)
{
    const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i partner = _mm512_xor_si512(lanes, _mm512_set1_epi64((long long)t));
    return _mm512_permutexvar_epi64(partner, a);
}

// All-ones in the lanes whose index has bit t set: the high half of each pair.
static inline pv_t pmf_ntt_high_lanes(uint64_t t)
{
    __mmask8 bits = 0;
    for (unsigned l = 0; l < PMF_VEC_GROUP; l++)
        if (l & t)
            bits |= (__mmask8)(1u << l);
    return _mm512_maskz_set1_epi64(bits, -1LL);
}

// The lanes of the group at j0 that hold live elements; only a vector shorter
// than one group has any other kind.
static inline __mmask8 pmf_ntt_live_lanes(uint64_t j0, uint64_t n)
{
    return (j0 + PMF_VEC_GROUP <= n) ? (__mmask8)0xFF : (__mmask8)((1u << (n - j0)) - 1);
}

// Per-lane twiddles for the group at j0: lane l belongs to block
// (j0 + l) / (2t), so its twiddle sits at ws[base + (j0 + l) >> log2(2t)].
// Padding lanes read zero rather than past the table.
static inline void pmf_ntt_gather_twiddles(pv_t *W, uint64_t **ws, uint64_t base, uint64_t j0,
                                           uint64_t t, __mmask8 live, uint64_t L)
{
    const pv_t lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    uint64_t shift = 1;
    while ((1ULL << shift) < 2 * t)
        shift++;
    const pv_t idx = pv_add(pv_srl(pv_add(lanes, pv_bcast(j0)), shift), pv_bcast(base));
    for (uint64_t k = 0; k < L; k++)
        W[k] = _mm512_mask_i64gather_epi64(pv_zero(), live, idx, (const void *)ws[k], 8);
}

// Both halves of every pair, in both lanes: U holds a[j] and V holds a[j+t]
// whether the lane is the low or the high one.
static inline void pmf_ntt_pair_halves(pv_t *U, pv_t *V, const pv_t *A, pv_t hi, uint64_t t,
                                       uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
    {
        const pv_t partner = pmf_ntt_swap_lanes(A[k], t);
        U[k] = pv_select(hi, partner, A[k]);
        V[k] = pv_select(hi, A[k], partner);
    }
}

static void pmf_ntt_stage_fwd_in_group(PMFVector a, uint64_t **ws, uint64_t m, uint64_t t,
                                       PMFParams params)
{
    const uint64_t L = params->limbs;
    const pv_t hi = pmf_ntt_high_lanes(t);
    pv_t A[PMF_MAX_LIMBS], U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS];
    pv_t M[PMF_MAX_LIMBS + 1], S[PMF_MAX_LIMBS + 1], D[PMF_MAX_LIMBS + 1];

    for (uint64_t j0 = 0; j0 < a->allocated_n; j0 += PMF_VEC_GROUP)
    {
        const __mmask8 live = pmf_ntt_live_lanes(j0, a->n);
        pmf_vec_load_group(A, a, j0, L);
        pmf_ntt_pair_halves(U, V, A, hi, t, L);
        pmf_ntt_gather_twiddles(W, ws, m, j0, t, live, L);
        pmf_vec_mul_group(M, W, V, params);
        pmf_vec_add_group(S, U, M, params);
        pmf_vec_sub_group(D, U, M, params);
        for (uint64_t k = 0; k < L; k++)
            S[k] = pv_select(hi, D[k], S[k]);
        pmf_vec_store_group(a, j0, S, L);
    }
}

static void pmf_ntt_stage_inv_in_group(PMFVector a, uint64_t **ws, uint64_t h, uint64_t t,
                                       PMFParams params)
{
    const uint64_t L = params->limbs;
    const pv_t hi = pmf_ntt_high_lanes(t);
    pv_t A[PMF_MAX_LIMBS], U[PMF_MAX_LIMBS], V[PMF_MAX_LIMBS], W[PMF_MAX_LIMBS];
    pv_t M[PMF_MAX_LIMBS + 1], S[PMF_MAX_LIMBS + 1], D[PMF_MAX_LIMBS + 1];

    for (uint64_t j0 = 0; j0 < a->allocated_n; j0 += PMF_VEC_GROUP)
    {
        const __mmask8 live = pmf_ntt_live_lanes(j0, a->n);
        pmf_vec_load_group(A, a, j0, L);
        pmf_ntt_pair_halves(U, V, A, hi, t, L);
        pmf_ntt_gather_twiddles(W, ws, h, j0, t, live, L);
        pmf_vec_add_group(S, U, V, params);
        pmf_vec_sub_group(D, U, V, params);
        pmf_vec_mul_group(M, D, W, params);
        for (uint64_t k = 0; k < L; k++)
            S[k] = pv_select(hi, M[k], S[k]);
        pmf_vec_store_group(a, j0, S, L);
    }
}

#endif // VFHE_HAVE_AVX512IFMA

// --- the transforms -------------------------------------------------------

void pmf_vec_ntt_forward(PMFVector a, PMFNTTPlan plan)
{
    const uint64_t n = plan->n;
    uint64_t t = n;

    assert(a->n == n);
    for (uint64_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
#if VFHE_HAVE_AVX512IFMA
        if (t < PMF_VEC_GROUP)
        {
            pmf_ntt_stage_fwd_in_group(a, plan->ws_fwd, m, t, plan->params);
            continue;
        }
#endif
        pmf_ntt_stage_fwd_wide(a, plan->ws_fwd, m, t, plan->params);
    }
}

void pmf_vec_ntt_inverse(PMFVector a, PMFNTTPlan plan)
{
    const uint64_t n = plan->n;
    uint64_t t = 1;

    assert(a->n == n);
    for (uint64_t m = n; m > 1; m >>= 1)
    {
        const uint64_t h = m >> 1;
#if VFHE_HAVE_AVX512IFMA
        if (t < PMF_VEC_GROUP)
        {
            pmf_ntt_stage_inv_in_group(a, plan->ws_inv, h, t, plan->params);
            t <<= 1;
            continue;
        }
#endif
        pmf_ntt_stage_inv_wide(a, plan->ws_inv, h, t, plan->params);
        t <<= 1;
    }
    pmf_vec_scale(a, a, plan->inv_n);
}

// --- the scalar oracle ----------------------------------------------------

// Twiddle e as one element.
static void pmf_ref_twiddle(uint64_t *w, uint64_t **ws, uint64_t e, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        w[k] = ws[k][e];
    for (uint64_t k = L; k < PMF_LANES; k++)
        w[k] = 0;
}

void pmf_ref_ntt_forward(uint64_t *a, PMFNTTPlan plan)
{
    const uint64_t n = plan->n, L = plan->params->limbs;
    uint64_t w[PMF_LANES], v[PMF_LANES];
    uint64_t t = n;

    for (uint64_t m = 1; m < n; m <<= 1)
    {
        t >>= 1;
        for (uint64_t i = 0; i < m; i++)
        {
            const uint64_t j1 = 2 * i * t;
            pmf_ref_twiddle(w, plan->ws_fwd, m + i, L);
            for (uint64_t j = j1; j < j1 + t; j++)
            {
                uint64_t *lo = a + j * PMF_LANES, *hi = a + (j + t) * PMF_LANES;
                pmf_ref_mul(v, hi, w, plan->params);
                pmf_ref_sub(hi, lo, v, plan->params);
                pmf_ref_add(lo, lo, v, plan->params);
            }
        }
    }
}

void pmf_ref_ntt_inverse(uint64_t *a, PMFNTTPlan plan)
{
    const uint64_t n = plan->n, L = plan->params->limbs;
    uint64_t w[PMF_LANES], v[PMF_LANES];
    uint64_t t = 1;

    for (uint64_t m = n; m > 1; m >>= 1)
    {
        const uint64_t h = m >> 1;
        for (uint64_t i = 0; i < h; i++)
        {
            const uint64_t j1 = 2 * i * t;
            pmf_ref_twiddle(w, plan->ws_inv, h + i, L);
            for (uint64_t j = j1; j < j1 + t; j++)
            {
                uint64_t *lo = a + j * PMF_LANES, *hi = a + (j + t) * PMF_LANES;
                pmf_ref_sub(v, lo, hi, plan->params);
                pmf_ref_add(lo, lo, hi, plan->params);
                pmf_ref_mul(hi, v, w, plan->params);
            }
        }
        t <<= 1;
    }
    for (uint64_t i = 0; i < n; i++)
        pmf_ref_mul(a + i * PMF_LANES, a + i * PMF_LANES, plan->inv_n, plan->params);
}
