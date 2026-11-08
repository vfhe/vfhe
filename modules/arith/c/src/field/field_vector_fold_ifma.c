// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// The IFMA body of the extension-field fold kernels, on the avx512ifma engine
// for q < 2^50 and d <= FIELD_MAX_FUSED_D. field_vector_fold.c describes the
// kernels.
#include <arith.h>
#include <assert.h>
#include <string.h>
#include "arith_internal.h"
#include "alloc.h"
#include "field_vector_fold.h"
#include "field_fused.h"

#if VFHE_HAVE_AVX512IFMA

// --- IFMA body: eight pairs at a time ----------------------------------------
//
// A group is eight consecutive pairs, lane l holding pair i + l, each element as
// d registers, one per coefficient. Coefficients are below q < 2^50, so the
// 52-bit multiply-add instructions (madd52lo / madd52hi) take them directly.
//
// An unreduced product's coefficient is accumulated per lane in three 64-bit
// words, value = lo + 2^52 hi + 2^104 top (FieldFoldLaneSum). madd52lo adds the
// low 52 bits of a product to lo and madd52hi the high bits to hi; a low half
// is below 2^52 and a high half below q^2 / 2^52 < 2^48. Normalizing carries
// lo's and hi's bits above 52 up into top, after which lo takes at least
// FIELD_FOLD_IFMA_LO_TERMS more terms without overflowing 64 bits; the bodies
// normalize often enough (field_fold_ifma_groups_per_normalization).

// The terms below 2^52 a normalized lo takes before it can overflow 64 bits.
#define FIELD_FOLD_IFMA_LO_TERMS 4095
// The most coefficients an unreduced product has at the degrees the IFMA body
// takes.
#define FIELD_FOLD_IFMA_UNREDUCED (2 * FIELD_MAX_FUSED_D - 1)

// One sum's accumulator at one node: for each coefficient k of the unreduced
// polynomial and each lane, the value lo[k] + 2^52 hi[k] + 2^104 top[k].

typedef struct
{
    __m512i lo[FIELD_FOLD_IFMA_UNREDUCED];
    __m512i hi[FIELD_FOLD_IFMA_UNREDUCED];
    __m512i top[FIELD_FOLD_IFMA_UNREDUCED];
} FieldFoldLaneSum;

// One chunk's accumulators, for each sum s and node t.
typedef struct
{
    FieldFoldLaneSum unreduced[FIELD_FOLD_MAX_SUMS][FIELD_FOLD_MAX_NODES];
} FieldFoldIfmaSums;

// The field's constants the IFMA body uses, broadcast to every lane where they
// are vectors.
typedef struct
{
    struct _Modulus mod; // a copy, so its fields stay in registers across stores
    uint64_t w_mod_q;    // w mod q
    __m512i q, neg_q;
    __m512i w, w_pre; // w and its Shoup precomputation, for field_shoup
    // The "fold and" variant's element r, and w r_k mod q, for
    // field_fold_ifma_mul_with_wb.
    __m512i r[FIELD_MAX_FUSED_D], wr[FIELD_MAX_FUSED_D];
} FieldFoldIfmaConstants;

FIELD_FOLD_INLINE void field_fold_ifma_constants(FieldFoldIfmaConstants *C, const FieldVector v,
                                                 const uint64_t *r)
{
    C->mod = *v->mod;
    C->w_mod_q = modq(v->w, v->mod);
    C->q = _mm512_set1_epi64((int64_t)v->mod->q);
    C->neg_q = _mm512_set1_epi64(-(int64_t)v->mod->q);
    C->w = _mm512_set1_epi64((int64_t)C->w_mod_q);
    C->w_pre = _mm512_set1_epi64((int64_t)(((unsigned __int128)C->w_mod_q << 52) / v->mod->q));
    for (uint64_t k = 0; r && k < v->d; k++)
    {
        const uint64_t rk = modq(r[k], v->mod);
        C->r[k] = _mm512_set1_epi64((int64_t)rk);
        C->wr[k] = _mm512_set1_epi64((int64_t)mul_modq(C->w_mod_q, rk, v->mod));
    }
}

FIELD_FOLD_INLINE __m512i field_fold_ifma_add(__m512i a, __m512i b, __m512i q)
{
    const __m512i sum = _mm512_add_epi64(a, b);
    return _mm512_min_epu64(sum, _mm512_sub_epi64(sum, q));
}

// a - b mod q, lane by lane. Both are below q < 2^50, so the difference is
// negative exactly when its sign bit is set.
FIELD_FOLD_INLINE __m512i field_fold_ifma_sub(__m512i a, __m512i b, __m512i q)
{
    const __m512i diff = _mm512_sub_epi64(a, b);
    return _mm512_mask_add_epi64(diff, _mm512_movepi64_mask(diff), diff, q);
}

// c = a * b in F_p[x]/(x^d - w), fully reduced, lane by lane, as
// field_fused_mul computes it. wb[j] = w b[j] mod q for j >= 1 (wb[0] is not
// read). c may alias a or b.
FIELD_FOLD_INLINE void field_fold_ifma_mul_with_wb(__m512i *c, const __m512i *a, const __m512i *b,
                                                   const __m512i *wb, uint64_t d, Modulus mod)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    __m512i out[FIELD_MAX_FUSED_D];
    for (uint64_t k = 0; k < d; k++)
    {
        __m512i lo = zero, hi = zero;
        for (uint64_t i = 0; i < d; i++)
        {
            const uint64_t j = FIELD_FUSED_PARTNER(i, k, d);
            const __m512i p = FIELD_FUSED_WRAPS(i, k, d) ? wb[j] : b[j];
            lo = _mm512_madd52lo_epu64(lo, a[i], p);
            hi = _mm512_madd52hi_epu64(hi, a[i], p);
        }
        hi = _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52));
        lo = _mm512_and_epi64(lo, mask52);
        out[k] = field_reduce128(hi, lo, mod);
    }
    for (uint64_t k = 0; k < d; k++)
        c[k] = out[k];
}

FIELD_FOLD_INLINE void field_fold_ifma_mul(__m512i *c, const __m512i *a, const __m512i *b,
                                           uint64_t d, const FieldFoldIfmaConstants *C)
{
    __m512i wb[FIELD_MAX_FUSED_D];
    for (uint64_t j = 1; j < d; j++)
        wb[j] = field_shoup(b[j], C->w, C->w_pre, C->q, C->neg_q);
    field_fold_ifma_mul_with_wb(c, a, b, wb, d, (Modulus)&C->mod);
}

// Add the coefficients of a * b, with no reduction at all, into the
// accumulator `target`, lane by lane.
FIELD_FOLD_INLINE void field_fold_ifma_add_unreduced_product(FieldFoldLaneSum *A, const __m512i *a,
                                                             const __m512i *b, uint64_t d)
{
    for (uint64_t k = 0; k < 2 * d - 1; k++)
    {
        const uint64_t low = k < d ? 0 : k - d + 1, high = k < d ? k : d - 1;
        __m512i lo = A->lo[k], hi = A->hi[k];
        for (uint64_t i = low; i <= high; i++)
        {
            lo = _mm512_madd52lo_epu64(lo, a[i], b[k - i]);
            hi = _mm512_madd52hi_epu64(hi, a[i], b[k - i]);
        }
        A->lo[k] = lo;
        A->hi[k] = hi;
    }
}

FIELD_FOLD_INLINE void field_fold_ifma_normalize(FieldFoldLaneSum *A, uint64_t d)
{
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    for (uint64_t k = 0; k < 2 * d - 1; k++)
    {
        const __m512i hi = _mm512_add_epi64(A->hi[k], _mm512_srli_epi64(A->lo[k], 52));
        A->lo[k] = _mm512_and_epi64(A->lo[k], mask52);
        A->hi[k] = _mm512_and_epi64(hi, mask52);
        A->top[k] = _mm512_add_epi64(A->top[k], _mm512_srli_epi64(hi, 52));
    }
}

// Load the pairs i .. i + live - 1 of a table as one group: their first
// members into U and their second members into V, d registers each, with the
// lanes past `live` zero. `i` is a multiple of 8, so both loads are aligned; a
// load whose lanes all lie past `live` is skipped, and every other one lies
// inside the planes.
FIELD_FOLD_INLINE void field_fold_ifma_load_pairs(__m512i *U, __m512i *V, uint64_t *const *planes,
                                                  uint64_t i, uint64_t live, uint64_t d)
{
    const __m512i evens = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i odds = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    const __mmask8 m0 = live >= 4 ? 0xFF : (__mmask8)((1u << (2 * live)) - 1);
    const __mmask8 m1 = live <= 4 ? 0 : (__mmask8)((1u << (2 * (live - 4))) - 1);
    for (uint64_t k = 0; k < d; k++)
    {
        const __m512i x0 = _mm512_maskz_load_epi64(m0, planes[k] + 2 * i);
        const __m512i x1 =
            m1 ? _mm512_maskz_load_epi64(m1, planes[k] + 2 * i + 8) : _mm512_setzero_si512();
        U[k] = _mm512_permutex2var_epi64(x0, evens, x1);
        V[k] = _mm512_permutex2var_epi64(x0, odds, x1);
    }
}

// The pairs of the 16 consecutive elements held in registers F0 and F1, split
// into first members (U) and second members (V), as field_fold_ifma_load_pairs
// gives them from memory.
FIELD_FOLD_INLINE void field_fold_ifma_pairs_in_registers(__m512i *U, __m512i *V, const __m512i *F0,
                                                          const __m512i *F1, uint64_t d)
{
    const __m512i evens = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i odds = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    for (uint64_t k = 0; k < d; k++)
    {
        U[k] = _mm512_permutex2var_epi64(F0[k], evens, F1[k]);
        V[k] = _mm512_permutex2var_epi64(F0[k], odds, F1[k]);
    }
}

FIELD_FOLD_INLINE void field_fold_ifma_nodes(__m512i (*F)[FIELD_MAX_FUSED_D], const __m512i *U,
                                             const __m512i *V, uint64_t nodes, uint64_t d,
                                             __m512i q)
{
    __m512i diff[FIELD_MAX_FUSED_D];
    for (uint64_t k = 0; k < d; k++)
    {
        F[0][k] = U[k];
        F[1][k] = V[k];
        diff[k] = field_fold_ifma_sub(V[k], U[k], q);
    }
    for (uint64_t t = 2; t < nodes; t++)
        for (uint64_t k = 0; k < d; k++)
            F[t][k] = field_fold_ifma_add(F[t - 1][k], diff[k], q);
}

// Add one group's contribution to every sum at every node, as
// field_fold_scalar_add_pair does for one pair. F[j][t] is the folds at node t
// of the group of table j; `weight` is the group's weights, or NULL for
// weights of one.
FIELD_FOLD_INLINE void
field_fold_ifma_add_group(FieldFoldIfmaSums *acc,
                          __m512i (*F)[FIELD_FOLD_MAX_NODES][FIELD_MAX_FUSED_D],
                          const __m512i *weight, const FieldProductShape *shape, uint64_t d,
                          const FieldFoldIfmaConstants *C)
{
    const __m512i zero = _mm512_setzero_si512();
    __m512i running_product[FIELD_MAX_FUSED_D], signed_sum[FIELD_MAX_FUSED_D];

    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            FieldFoldLaneSum *target = &acc->unreduced[s][t];
            for (uint64_t p = shape->first_product[s]; p < shape->first_product[s + 1]; p++)
            {
                const uint64_t *factors = shape->factors[p];
                const uint64_t last = shape->factor_count[p] - (weight ? 0 : 1);
                for (uint64_t k = 0; k < d; k++)
                    running_product[k] = F[factors[0]][t][k];
                for (uint64_t f = 1; f < last; f++)
                    field_fold_ifma_mul(running_product, running_product, F[factors[f]][t], d, C);
                if (weight)
                {
                    const bool first = p == shape->first_product[s];
                    for (uint64_t k = 0; k < d; k++)
                    {
                        const __m512i base = first ? zero : signed_sum[k];
                        signed_sum[k] = shape->negated[p]
                                            ? field_fold_ifma_sub(base, running_product[k], C->q)
                                            : field_fold_ifma_add(base, running_product[k], C->q);
                    }
                    continue;
                }
                if (shape->negated[p])
                    for (uint64_t k = 0; k < d; k++)
                        running_product[k] = field_fold_ifma_sub(zero, running_product[k], C->q);
                if (shape->factor_count[p] == 1)
                    for (uint64_t k = 0; k < d; k++)
                        target->lo[k] = _mm512_add_epi64(target->lo[k], running_product[k]);
                else
                    field_fold_ifma_add_unreduced_product(target, running_product,
                                                          F[factors[last]][t], d);
            }
            if (weight)
                field_fold_ifma_add_unreduced_product(target, signed_sum, weight, d);
        }
}

// How many groups the accumulators take between normalizations. A group adds
// to one coefficient's lo at most d terms per unreduced product, and one term
// per product of a single factor.
static uint64_t field_fold_ifma_groups_per_normalization(const FieldProductShape *shape,
                                                         bool weighted, uint64_t d)
{
    uint64_t most = d;
    for (uint64_t s = 0; !weighted && s < shape->sums; s++)
    {
        uint64_t terms = 0;
        for (uint64_t p = shape->first_product[s]; p < shape->first_product[s + 1]; p++)
            terms += shape->factor_count[p] == 1 ? 1 : d;
        most = terms > most ? terms : most;
    }
    return FIELD_FOLD_IFMA_LO_TERMS / most;
}

static void field_fold_ifma_sums_init(FieldFoldIfmaSums *acc, const FieldProductShape *shape,
                                      uint64_t d)
{
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
            for (uint64_t k = 0; k < 2 * d - 1; k++)
                acc->unreduced[s][t].lo[k] = acc->unreduced[s][t].hi[k] =
                    acc->unreduced[s][t].top[k] = _mm512_setzero_si512();
}

static void field_fold_ifma_normalize_all(FieldFoldIfmaSums *acc, const FieldProductShape *shape,
                                          uint64_t d)
{
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
            field_fold_ifma_normalize(&acc->unreduced[s][t], d);
}

// Reduce each accumulator into the element sums[s * nodes + t]: every lane's
// coefficients modulo q, added over the lanes, then the polynomial modulo
// x^d - w.
static void field_fold_ifma_sums_finish(uint64_t *const *sums, FieldFoldIfmaSums *acc,
                                        const FieldProductShape *shape, uint64_t d,
                                        const FieldFoldIfmaConstants *C)
{
    const Modulus mod = (Modulus)&C->mod;
    const uint64_t q = mod->q;
    _Alignas(64) uint64_t lo[FIELD_FUSED_LANES], hi[FIELD_FUSED_LANES], top[FIELD_FUSED_LANES];
    uint64_t poly[FIELD_FOLD_IFMA_UNREDUCED];

    field_fold_ifma_normalize_all(acc, shape, d);
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            const FieldFoldLaneSum *target = &acc->unreduced[s][t];
            for (uint64_t k = 0; k < 2 * d - 1; k++)
            {
                _mm512_store_si512((__m512i *)lo, target->lo[k]);
                _mm512_store_si512((__m512i *)hi, target->hi[k]);
                _mm512_store_si512((__m512i *)top, target->top[k]);
                poly[k] = 0;
                for (uint64_t lane = 0; lane < FIELD_FUSED_LANES; lane++)
                {
                    // lo and hi are below 2^52, so lo + 2^52 hi fits 104 bits.
                    uint64_t value = modq_wide(hi[lane] >> 12, (hi[lane] << 52) | lo[lane], mod);
                    if (top[lane])
                        value = add_modq(value, mul_modq(modq(top[lane], mod), mod->mp_w2, mod), q);
                    poly[k] = add_modq(poly[k], value, q);
                }
            }
            uint64_t *out = sums[s * shape->nodes + t];
            for (uint64_t k = 0; k < d; k++)
                out[k] = k + 1 < d ? add_modq(poly[k], mul_modq(C->w_mod_q, poly[k + d], mod), q)
                                   : poly[k];
        }
}

// The sums of the pairs [begin, end) of the tables, into sums[0 .. slots - 1].
// `begin` is a multiple of 8.
static inline void field_fold_ifma_sums(uint64_t *const *sums, const FieldFoldJob *job,
                                        uint64_t begin, uint64_t end, const uint64_t d)
{
    const FieldProductShape *shape = job->shape;
    const FieldVector weights = job->weights;
    FieldFoldIfmaConstants C;
    field_fold_ifma_constants(&C, job->tables[0], NULL);
    FieldFoldIfmaSums acc;
    __m512i F[FIELD_FOLD_MAX_TABLES][FIELD_FOLD_MAX_NODES][FIELD_MAX_FUSED_D];
    __m512i U[FIELD_MAX_FUSED_D], V[FIELD_MAX_FUSED_D], weight[FIELD_MAX_FUSED_D];
    const uint64_t budget = field_fold_ifma_groups_per_normalization(shape, weights != NULL, d);
    uint64_t since = 0;

    field_fold_ifma_sums_init(&acc, shape, d);
    for (uint64_t i = begin; i < end; i += FIELD_FOLD_GROUP)
    {
        const uint64_t live = end - i < FIELD_FOLD_GROUP ? end - i : FIELD_FOLD_GROUP;
        for (uint64_t x = 0; x < shape->used_count; x++)
        {
            const uint64_t j = shape->used[x];
            field_fold_ifma_load_pairs(U, V, job->tables[j]->coeffs, i, live, d);
            field_fold_ifma_nodes(F[j], U, V, shape->nodes, d, C.q);
        }
        // Lanes past `live` read the weights' padding, whatever it holds; their
        // folds are zero, so they add nothing.
        if (weights)
            for (uint64_t k = 0; k < d; k++)
                weight[k] = _mm512_load_si512((const __m512i *)(weights->coeffs[k] + i));
        field_fold_ifma_add_group(&acc, F, weights ? weight : NULL, shape, d, &C);
        if (++since == budget)
        {
            field_fold_ifma_normalize_all(&acc, shape, d);
            since = 0;
        }
    }
    field_fold_ifma_sums_finish(sums, &acc, shape, d, &C);
}

// The fold at r of one group of pairs, U + r (V - U).
FIELD_FOLD_INLINE void field_fold_ifma_fold(__m512i *out, const __m512i *U, const __m512i *V,
                                            uint64_t d, const FieldFoldIfmaConstants *C)
{
    __m512i diff[FIELD_MAX_FUSED_D];
    for (uint64_t k = 0; k < d; k++)
        diff[k] = field_fold_ifma_sub(V[k], U[k], C->q);
    field_fold_ifma_mul_with_wb(diff, diff, C->r, C->wr, d, (Modulus)&C->mod);
    for (uint64_t k = 0; k < d; k++)
        out[k] = field_fold_ifma_add(U[k], diff[k], C->q);
}

// The "fold and" variant over the pairs [begin, end) of the folded tables, as
// field_fold_scalar_fold_sums; `begin` is a multiple of 8. One step handles
// folded pairs j .. j + 7: for every table it folds the input pairs
// 2j .. 2j + 15 into the output elements 2j .. 2j + 15 (two groups), stores
// them, and splits the same registers into the folded pairs. Lanes past the
// input's pairs fold zeros, so the last step writes zeros into the outputs'
// padding: the steps cover the outputs up to 16 ceil(total / 8) elements, at
// least their padded length.
static inline void field_fold_ifma_fold_sums(uint64_t *const *sums, const FieldFoldJob *job,
                                             uint64_t begin, uint64_t end, const uint64_t d)
{
    const FieldProductShape *shape = job->shape;
    const FieldVector weights = job->weights;
    const uint64_t folded = 2 * job->total, padded = job->outs[0]->allocated_n;
    FieldFoldIfmaConstants C;
    field_fold_ifma_constants(&C, job->tables[0], job->r);
    FieldFoldIfmaSums acc;
    __m512i F[FIELD_FOLD_MAX_TABLES][FIELD_FOLD_MAX_NODES][FIELD_MAX_FUSED_D];
    __m512i U[FIELD_MAX_FUSED_D], V[FIELD_MAX_FUSED_D], weight[FIELD_MAX_FUSED_D];
    __m512i A0[FIELD_MAX_FUSED_D], A1[FIELD_MAX_FUSED_D];
    const uint64_t budget = field_fold_ifma_groups_per_normalization(shape, weights != NULL, d);
    uint64_t since = 0;

    field_fold_ifma_sums_init(&acc, shape, d);
    for (uint64_t j = begin; j < end; j += FIELD_FOLD_GROUP)
    {
        const uint64_t o = 2 * j, rest = folded - o;
        const uint64_t live0 = rest < 8 ? rest : 8;
        const uint64_t live1 = rest <= 8 ? 0 : rest - 8 < 8 ? rest - 8 : 8;
        for (uint64_t x = 0; x < job->count; x++)
        {
            uint64_t *const *out = job->outs[x]->coeffs;
            field_fold_ifma_load_pairs(U, V, job->tables[x]->coeffs, o, live0, d);
            field_fold_ifma_fold(A0, U, V, d, &C);
            if (live1)
            {
                field_fold_ifma_load_pairs(U, V, job->tables[x]->coeffs, o + 8, live1, d);
                field_fold_ifma_fold(A1, U, V, d, &C);
            }
            else
                for (uint64_t k = 0; k < d; k++)
                    A1[k] = _mm512_setzero_si512();
            for (uint64_t k = 0; k < d; k++)
            {
                _mm512_store_si512((__m512i *)(out[k] + o), A0[k]);
                if (o + 8 < padded)
                    _mm512_store_si512((__m512i *)(out[k] + o + 8), A1[k]);
            }
            if (shape->is_used[x])
            {
                field_fold_ifma_pairs_in_registers(U, V, A0, A1, d);
                field_fold_ifma_nodes(F[x], U, V, shape->nodes, d, C.q);
            }
        }
        if (weights)
            for (uint64_t k = 0; k < d; k++)
                weight[k] = _mm512_load_si512((const __m512i *)(weights->coeffs[k] + j));
        field_fold_ifma_add_group(&acc, F, weights ? weight : NULL, shape, d, &C);
        if (++since == budget)
        {
            field_fold_ifma_normalize_all(&acc, shape, d);
            since = 0;
        }
    }
    field_fold_ifma_sums_finish(sums, &acc, shape, d, &C);
}

// The gate of the fused extension product (q < 2^50, d <= FIELD_MAX_FUSED_D),
// asked for one group's length because the body handles any number of pairs.
bool field_fold_ifma_applies(const FieldVector v)
{
    return field_fused_applies(v->mod, v->d, FIELD_FUSED_LANES);
}

// --- this body's ranges, for field_vector_fold.c ---------------------------

FIELD_FOLD_BODIES(field_fold_ifma_sums)
FIELD_FOLD_BODIES(field_fold_ifma_fold_sums)

void field_fold_ifma_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                uint64_t *const *sums)
{
    FIELD_FOLD_CALL(field_fold_ifma_sums, job->tables[0]->d, sums, job, begin, end);
}

void field_fold_ifma_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                     uint64_t *const *sums)
{
    FIELD_FOLD_CALL(field_fold_ifma_fold_sums, job->tables[0]->d, sums, job, begin, end);
}

#endif // VFHE_HAVE_AVX512IFMA
