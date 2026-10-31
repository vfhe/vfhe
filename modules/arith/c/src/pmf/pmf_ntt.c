// SPDX-FileCopyrightText: 2026 The vFHE Authors
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
// From 64 elements on, the tuned engine runs those three stages instead on
// 64-element chunks transposed 8 x 8 per limb plane ("the transposed tail"),
// where every lane of a butterfly is a different pair and the twiddles load as
// plain vectors from per-plan tables.
//
// Every transform dispatches once on the limb count (PMF_WITH_LIMBS), so the
// stages see it as a literal.
//
// Every element is canonical between operations: the cores reduce fully, and
// IFMA's 52-bit operand truncation leaves no room for a lazy variant.
//
// pmf_ref_ntt_* at the bottom are the scalar oracle: the same loops over the
// element kernels of pmf.c, compiled into every engine.
#include <arith.h>
#include <inttypes.h>
#include <string.h>

#include "alloc.h"
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
void pmf_ref_inverse(uint64_t *out, const uint64_t *a, PMFParams params)
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

#if VFHE_HAVE_AVX512IFMA
// Twiddles per 64-element chunk in the transposed tail: 7 vectors of 8.
#define PMF_NTT_TAIL_WORDS 56

// The twiddles of the three stages with t < 8 in the order pmf_ntt_tail_*
// read them. In a transposed chunk c, lane g of register r holds element
// 64c + 8g + r, whose block at distance t is (64c + 8g + r) / 2t, so the
// twiddle varies along g:
//   [0, 8)    t = 4: ws[n/8 + 8c + g]
//   [8, 24)   t = 2: ws[n/4 + 16c + 2g + h],  h = r / 4   (two vectors)
//   [24, 56)  t = 1: ws[n/2 + 32c + 4g + q],  q = r / 2   (four vectors)
// The inverse's first three stages have the same blocks, so this serves both
// tables.
static uint64_t **pmf_ntt_tail_table(uint64_t n, uint64_t **ws, uint64_t L)
{
    const uint64_t chunks = n / 64;
    uint64_t **planes = (uint64_t **)malloc(L * sizeof(*planes));
    for (uint64_t k = 0; k < L; k++)
    {
        uint64_t *table =
            (uint64_t *)safe_aligned_malloc(chunks * PMF_NTT_TAIL_WORDS * sizeof(uint64_t));
        for (uint64_t c = 0; c < chunks; c++)
        {
            uint64_t *row = table + c * PMF_NTT_TAIL_WORDS;
            for (uint64_t g = 0; g < 8; g++)
            {
                row[g] = ws[k][n / 8 + 8 * c + g];
                for (uint64_t h = 0; h < 2; h++)
                    row[8 + 8 * h + g] = ws[k][n / 4 + 16 * c + 2 * g + h];
                for (uint64_t q = 0; q < 4; q++)
                    row[24 + 8 * q + g] = ws[k][n / 2 + 32 * c + 4 * g + q];
            }
        }
        planes[k] = table;
    }
    return planes;
}
#endif

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
    plan->limbs = params->limbs;
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
#if VFHE_HAVE_AVX512IFMA
    if (n >= 64)
    {
        plan->tail_fwd = pmf_ntt_tail_table(n, plan->ws_fwd, plan->limbs);
        plan->tail_inv = pmf_ntt_tail_table(n, plan->ws_inv, plan->limbs);
    }
#endif
    return plan;
}

void pmf_ntt_free_plan(PMFNTTPlan plan)
{
    if (plan == NULL)
        return;
    // plan->limbs, not plan->params->limbs: the params are borrowed and may
    // already be gone, since nothing orders one release before the other.
    pmf_ntt_free_table(plan->ws_fwd, plan->limbs);
    pmf_ntt_free_table(plan->ws_inv, plan->limbs);
    if (plan->tail_fwd)
    {
        pmf_ntt_free_table(plan->tail_fwd, plan->limbs);
        pmf_ntt_free_table(plan->tail_inv, plan->limbs);
    }
    free(plan);
}

// --- vector stages, pairs across groups -----------------------------------

// One twiddle, broadcast limb by limb.
PMF_INLINE void pmf_ntt_load_twiddle(pv_t *W, uint64_t **ws, uint64_t e, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        W[k] = pv_bcast(ws[k][e]);
}

// Cooley-Tukey stage: m blocks of 2t elements, block i twiddled by ws[m + i].
// a[j] <- a[j] + w a[j+t], a[j+t] <- a[j] - w a[j+t].
PMF_INLINE void pmf_ntt_stage_fwd_wide(PMFVector a, uint64_t **ws, uint64_t m, uint64_t t,
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
PMF_INLINE void pmf_ntt_stage_inv_wide(PMFVector a, uint64_t **ws, uint64_t h, uint64_t t,
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
PMF_INLINE pv_t pmf_ntt_swap_lanes(pv_t a, uint64_t t)
{
    const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i partner = _mm512_xor_si512(lanes, _mm512_set1_epi64((long long)t));
    return _mm512_permutexvar_epi64(partner, a);
}

// All-ones in the lanes whose index has bit t set: the high half of each pair.
PMF_INLINE pv_t pmf_ntt_high_lanes(uint64_t t)
{
    __mmask8 bits = 0;
    for (unsigned l = 0; l < PMF_VEC_GROUP; l++)
        if (l & t)
            bits |= (__mmask8)(1u << l);
    return _mm512_maskz_set1_epi64(bits, -1LL);
}

// The lanes of the group at j0 that hold live elements; only a vector shorter
// than one group has any other kind.
PMF_INLINE __mmask8 pmf_ntt_live_lanes(uint64_t j0, uint64_t n)
{
    return (j0 + PMF_VEC_GROUP <= n) ? (__mmask8)0xFF : (__mmask8)((1u << (n - j0)) - 1);
}

// Per-lane twiddles for the group at j0: lane l belongs to block
// (j0 + l) / (2t), so its twiddle sits at ws[base + (j0 + l) >> log2(2t)].
// Padding lanes read zero rather than past the table.
PMF_INLINE void pmf_ntt_gather_twiddles(pv_t *W, uint64_t **ws, uint64_t base, uint64_t j0,
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
PMF_INLINE void pmf_ntt_pair_halves(pv_t *U, pv_t *V, const pv_t *A, pv_t hi, uint64_t t,
                                    uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
    {
        const pv_t partner = pmf_ntt_swap_lanes(A[k], t);
        U[k] = pv_select(hi, partner, A[k]);
        V[k] = pv_select(hi, A[k], partner);
    }
}

PMF_INLINE void pmf_ntt_stage_fwd_in_group(PMFVector a, uint64_t **ws, uint64_t m, uint64_t t,
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

PMF_INLINE void pmf_ntt_stage_inv_in_group(PMFVector a, uint64_t **ws, uint64_t h, uint64_t t,
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

// --- the transposed tail (tuned engine only) -------------------------------

// 8 x 8 transpose: lane l of register r goes to lane r of register l. Three
// levels, each swapping bit d of the register index with bit d of the lane.
PMF_INLINE void pmf_ntt_transpose8(pv_t *x)
{
    for (unsigned d = 1; d < 8; d <<= 1)
    {
        long long lo_index[8], hi_index[8];
        for (unsigned l = 0; l < 8; l++)
        {
            lo_index[l] = (l & d) ? 8 + (long long)(l - d) : (long long)l;
            hi_index[l] = (l & d) ? 8 + (long long)l : (long long)(l + d);
        }
        const __m512i lo = _mm512_loadu_si512((const void *)lo_index);
        const __m512i hi = _mm512_loadu_si512((const void *)hi_index);
        for (unsigned r = 0; r < 8; r++)
        {
            if (r & d)
                continue;
            const pv_t a = x[r], b = x[r + d];
            x[r] = _mm512_permutex2var_epi64(a, lo, b);
            x[r + d] = _mm512_permutex2var_epi64(a, hi, b);
        }
    }
}

// The 64 elements from j0, transposed per limb: buf[r] holds elements j0 + 8g + r.
PMF_INLINE void pmf_ntt_load_chunk(pv_t (*buf)[PMF_MAX_LIMBS + 1], const PMFVector a, uint64_t j0,
                                   uint64_t L)
{
    pv_t x[8];
    for (uint64_t k = 0; k < L; k++)
    {
        for (unsigned r = 0; r < 8; r++)
            x[r] = pv_load(a->limbs[k] + j0 + 8 * r);
        pmf_ntt_transpose8(x);
        for (unsigned r = 0; r < 8; r++)
            buf[r][k] = x[r];
    }
}

PMF_INLINE void pmf_ntt_store_chunk(PMFVector a, pv_t (*buf)[PMF_MAX_LIMBS + 1], uint64_t j0,
                                    uint64_t L)
{
    pv_t x[8];
    for (uint64_t k = 0; k < L; k++)
    {
        for (unsigned r = 0; r < 8; r++)
            x[r] = buf[r][k];
        pmf_ntt_transpose8(x);
        for (unsigned r = 0; r < 8; r++)
            pv_store(a->limbs[k] + j0 + 8 * r, x[r]);
    }
}

PMF_INLINE void pmf_ntt_load_tail_twiddle(pv_t *W, uint64_t **table, uint64_t at, uint64_t L)
{
    for (uint64_t k = 0; k < L; k++)
        W[k] = pv_load(table[k] + at);
}

// u <- u + w v, v <- u - w v.
PMF_INLINE void pmf_ntt_butterfly_ct(pv_t *u, pv_t *v, const pv_t *w, PMFParams P)
{
    pv_t M[PMF_MAX_LIMBS + 1], S[PMF_MAX_LIMBS + 1];
    pmf_vec_mul_group(M, v, w, P);
    pmf_vec_add_group(S, u, M, P);
    pmf_vec_sub_group(v, u, M, P);
    for (uint64_t k = 0; k < P->limbs; k++)
        u[k] = S[k];
}

// u <- u + v, v <- (u - v) w.
PMF_INLINE void pmf_ntt_butterfly_gs(pv_t *u, pv_t *v, const pv_t *w, PMFParams P)
{
    pv_t S[PMF_MAX_LIMBS + 1], D[PMF_MAX_LIMBS + 1];
    pmf_vec_add_group(S, u, v, P);
    pmf_vec_sub_group(D, u, v, P);
    pmf_vec_mul_group(v, D, w, P);
    for (uint64_t k = 0; k < P->limbs; k++)
        u[k] = S[k];
}

// The stages t = 4, 2, 1 of the forward transform.
PMF_INLINE void pmf_ntt_tail_fwd(PMFVector a, PMFNTTPlan plan, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t buf[8][PMF_MAX_LIMBS + 1], W[PMF_MAX_LIMBS];
    for (uint64_t c = 0; c < plan->n / 64; c++)
    {
        const uint64_t at = c * PMF_NTT_TAIL_WORDS;
        pmf_ntt_load_chunk(buf, a, 64 * c, L);
        pmf_ntt_load_tail_twiddle(W, plan->tail_fwd, at, L);
        for (unsigned r = 0; r < 4; r++)
            pmf_ntt_butterfly_ct(buf[r], buf[r + 4], W, P);
        for (unsigned h = 0; h < 2; h++)
        {
            pmf_ntt_load_tail_twiddle(W, plan->tail_fwd, at + 8 + 8 * h, L);
            pmf_ntt_butterfly_ct(buf[4 * h], buf[4 * h + 2], W, P);
            pmf_ntt_butterfly_ct(buf[4 * h + 1], buf[4 * h + 3], W, P);
        }
        for (unsigned q = 0; q < 4; q++)
        {
            pmf_ntt_load_tail_twiddle(W, plan->tail_fwd, at + 24 + 8 * q, L);
            pmf_ntt_butterfly_ct(buf[2 * q], buf[2 * q + 1], W, P);
        }
        pmf_ntt_store_chunk(a, buf, 64 * c, L);
    }
}

// The stages t = 1, 2, 4 of the inverse transform.
PMF_INLINE void pmf_ntt_tail_inv(PMFVector a, PMFNTTPlan plan, PMFParams P)
{
    const uint64_t L = P->limbs;
    pv_t buf[8][PMF_MAX_LIMBS + 1], W[PMF_MAX_LIMBS];
    for (uint64_t c = 0; c < plan->n / 64; c++)
    {
        const uint64_t at = c * PMF_NTT_TAIL_WORDS;
        pmf_ntt_load_chunk(buf, a, 64 * c, L);
        for (unsigned q = 0; q < 4; q++)
        {
            pmf_ntt_load_tail_twiddle(W, plan->tail_inv, at + 24 + 8 * q, L);
            pmf_ntt_butterfly_gs(buf[2 * q], buf[2 * q + 1], W, P);
        }
        for (unsigned h = 0; h < 2; h++)
        {
            pmf_ntt_load_tail_twiddle(W, plan->tail_inv, at + 8 + 8 * h, L);
            pmf_ntt_butterfly_gs(buf[4 * h], buf[4 * h + 2], W, P);
            pmf_ntt_butterfly_gs(buf[4 * h + 1], buf[4 * h + 3], W, P);
        }
        pmf_ntt_load_tail_twiddle(W, plan->tail_inv, at, L);
        for (unsigned r = 0; r < 4; r++)
            pmf_ntt_butterfly_gs(buf[r], buf[r + 4], W, P);
        pmf_ntt_store_chunk(a, buf, 64 * c, L);
    }
}

#endif // VFHE_HAVE_AVX512IFMA

// --- the transforms -------------------------------------------------------

// The forward transform from stage m0 on (m0 = 1 is the whole transform): the
// stages before it are taken as done.
PMF_INLINE void pmf_vec_ntt_forward_from(PMFVector a, PMFNTTPlan plan, uint64_t m0, PMFParams P)
{
    const uint64_t n = plan->n;
    uint64_t t = n / m0;

    assert(a->n == n);
    for (uint64_t m = m0; m < n; m <<= 1)
    {
        t >>= 1;
#if VFHE_HAVE_AVX512IFMA
        if (t < PMF_VEC_GROUP)
        {
            if (plan->tail_fwd)
            {
                pmf_ntt_tail_fwd(a, plan, P);
                break;
            }
            pmf_ntt_stage_fwd_in_group(a, plan->ws_fwd, m, t, P);
            continue;
        }
#endif
        pmf_ntt_stage_fwd_wide(a, plan->ws_fwd, m, t, P);
    }
}

PMF_INLINE void pmf_vec_ntt_inverse_body(PMFVector a, PMFNTTPlan plan, PMFParams P)
{
    const uint64_t n = plan->n;
    uint64_t t = 1, m = n;

    assert(a->n == n);
#if VFHE_HAVE_AVX512IFMA
    if (plan->tail_inv)
    {
        pmf_ntt_tail_inv(a, plan, P);
        t = 8;
        m = n >> 3;
    }
#endif
    for (; m > 1; m >>= 1)
    {
        const uint64_t h = m >> 1;
#if VFHE_HAVE_AVX512IFMA
        if (t < PMF_VEC_GROUP)
        {
            pmf_ntt_stage_inv_in_group(a, plan->ws_inv, h, t, P);
            t <<= 1;
            continue;
        }
#endif
        pmf_ntt_stage_inv_wide(a, plan->ws_inv, h, t, P);
        t <<= 1;
    }
}

void pmf_vec_ntt_forward(PMFVector a, PMFNTTPlan plan)
{
    PMF_WITH_LIMBS(plan->params, pmf_vec_ntt_forward_from(a, plan, 1, &P_));
}

void pmf_vec_ntt_inverse(PMFVector a, PMFNTTPlan plan)
{
    PMF_WITH_LIMBS(plan->params, pmf_vec_ntt_inverse_body(a, plan, &P_));
    pmf_vec_scale(a, a, plan->inv_n);
}

// --- batches of transforms ------------------------------------------------

// Every block of `a` transformed as a vector of its own. A block that is a whole
// number of groups is one directly, over the same planes. A shorter block shares
// its group with its neighbours, and the in-group stages compute on every lane
// of a group, so it is transformed in a staging group whose other lanes are zero.
static void pmf_ntt_blocks(PMFVector a, uint64_t blocks, PMFNTTPlan plan, int inverse)
{
    const uint64_t n = plan->n, L = plan->limbs;
    uint64_t *planes[PMF_MAX_LIMBS];
    struct _PMFVector block = {planes, n, n, a->params};

    assert(a->n == n * blocks);
    if (n % PMF_VEC_GROUP == 0)
    {
        for (uint64_t b = 0; b < blocks; b++)
        {
            for (uint64_t k = 0; k < L; k++)
                planes[k] = a->limbs[k] + b * n;
            if (inverse)
                pmf_vec_ntt_inverse(&block, plan);
            else
                pmf_vec_ntt_forward(&block, plan);
        }
        return;
    }

    _Alignas(64) uint64_t staging[PMF_MAX_LIMBS][PMF_VEC_GROUP];
    memset(staging, 0, sizeof(staging));
    block.allocated_n = PMF_VEC_GROUP;
    for (uint64_t k = 0; k < L; k++)
        planes[k] = staging[k];
    for (uint64_t b = 0; b < blocks; b++)
    {
        for (uint64_t k = 0; k < L; k++)
            memcpy(staging[k], a->limbs[k] + b * n, n * sizeof(uint64_t));
        if (inverse)
            pmf_vec_ntt_inverse(&block, plan);
        else
            pmf_vec_ntt_forward(&block, plan);
        for (uint64_t k = 0; k < L; k++)
            memcpy(a->limbs[k] + b * n, staging[k], n * sizeof(uint64_t));
    }
}

void pmf_vec_ntt_forward_blocks(PMFVector a, uint64_t blocks, PMFNTTPlan plan)
{
    pmf_ntt_blocks(a, blocks, plan, 0);
}

void pmf_vec_ntt_inverse_blocks(PMFVector a, uint64_t blocks, PMFNTTPlan plan)
{
    pmf_ntt_blocks(a, blocks, plan, 1);
}

// Both directions are the k-way (de)interleave of every plane, with block b's
// run of n words at b * n as the k-th sequence.
void pmf_ntt_to_interleaved(PMFVector out, const PMFVector in, uint64_t blocks)
{
    const uint64_t n = in->n / blocks;
    const uint64_t **runs = (const uint64_t **)malloc(blocks * sizeof(*runs));

    for (uint64_t k = 0; k < in->params->limbs; k++)
    {
        for (uint64_t b = 0; b < blocks; b++)
            runs[b] = in->limbs[k] + b * n;
        vec_interleave_k_u64(out->limbs[k], runs, blocks, n);
    }
    free(runs);
}

void pmf_ntt_to_blocks(PMFVector out, const PMFVector in, uint64_t blocks)
{
    const uint64_t n = in->n / blocks;
    uint64_t **runs = (uint64_t **)malloc(blocks * sizeof(*runs));

    for (uint64_t k = 0; k < in->params->limbs; k++)
    {
        for (uint64_t b = 0; b < blocks; b++)
            runs[b] = out->limbs[k] + b * n;
        vec_deinterleave_u64(runs, in->limbs[k], blocks, n);
    }
    free(runs);
}

// --- the zero-padded batch encoder -----------------------------------------
//
// A block of n whose top n - k coefficients are zero leaves the first
// log2(c) forward stages (c = n / k) nothing to do but copy: every butterfly's
// partner is zero, so both outputs are its other input. After them the block
// is c copies of its message. So each message is written c times into its
// block, straight from the interleaved batch, and the transform starts at
// stage c. On the tuned engine that start must leave the transposed tail its
// three stages, which it always runs, and a block shorter than a group takes
// the staged whole transform: in both cases the message is written once,
// zero-padded, and transformed whole instead.

// Messages b0..b0+7 by 8 x 8 tiles: rows i0..i0+7 of the batch are one group
// per limb, and transposed, register r is message b0 + r at coefficients
// i0..i0+7.
#if VFHE_HAVE_AVX512IFMA
PMF_INLINE void pmf_ntt_place_tiles(PMFVector out, const PMFVector msg, uint64_t blocks,
                                    uint64_t b0, uint64_t n, uint64_t k, uint64_t L)
{
    pv_t x[8];
    for (uint64_t i0 = 0; i0 < k; i0 += 8)
        for (uint64_t l = 0; l < L; l++)
        {
            for (unsigned r = 0; r < 8; r++)
                x[r] = _mm512_loadu_si512((const void *)(msg->limbs[l] + (i0 + r) * blocks + b0));
            pmf_ntt_transpose8(x);
            for (unsigned r = 0; r < 8; r++)
                for (uint64_t q = 0; q < n / k; q++)
                    pv_store(out->limbs[l] + (b0 + r) * n + q * k + i0, x[r]);
        }
}
#endif

PMF_INLINE void pmf_vec_ntt_encode_body(PMFVector out, const PMFVector msg, uint64_t blocks,
                                        PMFNTTPlan plan, PMFParams P)
{
    const uint64_t n = plan->n, L = P->limbs, k = msg->n / blocks;
    uint64_t copies = n / k, start = n / k;
    uint64_t width = 1;
    int tiled = 0;
    if (n % PMF_VEC_GROUP)
        copies = start = 1;
#if VFHE_HAVE_AVX512IFMA
    if (plan->tail_fwd && k < 8)
        copies = start = 1;
    tiled = blocks % 8 == 0 && k % 8 == 0 && n % PMF_VEC_GROUP == 0;
    if (tiled)
        width = 8;
#endif

    for (uint64_t b0 = 0; b0 < blocks; b0 += width)
    {
#if VFHE_HAVE_AVX512IFMA
        if (tiled)
            pmf_ntt_place_tiles(out, msg, blocks, b0, n, k, L);
#endif
        if (!tiled)
            for (uint64_t l = 0; l < L; l++)
            {
                uint64_t *block = out->limbs[l] + b0 * n;
                if (copies == 1)
                    memset(block + k, 0, (n - k) * sizeof(uint64_t));
                for (uint64_t i = 0; i < k; i++)
                    for (uint64_t q = 0; q < copies; q++)
                        block[q * k + i] = msg->limbs[l][i * blocks + b0];
            }
        for (uint64_t r = 0; r < width; r++)
        {
            uint64_t *planes[PMF_MAX_LIMBS];
            for (uint64_t l = 0; l < L; l++)
                planes[l] = out->limbs[l] + (b0 + r) * n;
            struct _PMFVector block = {planes, n, n, plan->params};
            if (n % PMF_VEC_GROUP == 0)
            {
                pmf_vec_ntt_forward_from(&block, plan, start, P);
                continue;
            }
            // A block shorter than a group: the staged path of the batch above.
            pmf_ntt_blocks(&block, 1, plan, 0);
        }
    }
}

void pmf_vec_ntt_encode_blocks(PMFVector out, const PMFVector msg, uint64_t blocks, PMFNTTPlan plan)
{
    assert(out->n == blocks * plan->n && msg->n % blocks == 0);
    PMF_WITH_LIMBS(plan->params, pmf_vec_ntt_encode_body(out, msg, blocks, plan, &P_));
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
