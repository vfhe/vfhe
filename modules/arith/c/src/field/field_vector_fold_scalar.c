// SPDX-FileCopyrightText: 2026 The vFHE Authors
// SPDX-License-Identifier: Apache-2.0
//
// The scalar body of the extension-field fold kernels, for the engines without
// AVX-512 IFMA. field_vector_fold.c describes the kernels.
#include <arith.h>
#include <assert.h>
#include <string.h>
#include "arith_internal.h"
#include "alloc.h"
#include "field_vector_fold.h"

#if !VFHE_HAVE_AVX512IFMA

// --- scalar body: one pair at a time -----------------------------------------
//
// Coefficients are below q < 2^62, so the product of two is below 2^124, and a
// sum of up to d <= 16 such products is below 2^128. An accumulator
// (FieldFoldU192) adds those sums into a 128-bit word and counts the carries
// out of it in a third word, so no number of pairs overflows it.

// A 192-bit unsigned accumulator, low + 2^128 high.
typedef struct
{
    unsigned __int128 low;
    uint64_t high;
} FieldFoldU192;

FIELD_FOLD_INLINE void field_fold_u192_add(FieldFoldU192 *acc, unsigned __int128 value)
{
    acc->low += value;
    acc->high += acc->low < value;
}

// One chunk's accumulators: for each sum s and node t, the 2d - 1 coefficients
// of the sum's unreduced polynomial.
typedef struct
{
    FieldFoldU192 unreduced[FIELD_FOLD_MAX_SUMS][FIELD_FOLD_MAX_NODES][FIELD_FOLD_MAX_UNREDUCED];
} FieldFoldScalarSums;

// c = a * b in F_p[x]/(x^d - w), fully reduced, with one modular reduction
// per coefficient. wb[j] = w b[j] mod q for j >= 1 (wb[0] is not read), which
// a caller multiplying by the same b again computes once. c may alias a or b.
FIELD_FOLD_INLINE void field_fold_scalar_mul_with_wb(uint64_t *c, const uint64_t *a,
                                                     const uint64_t *b, const uint64_t *wb,
                                                     uint64_t d, Modulus mod)
{
    uint64_t out[FIELD_FOLD_MAX_DEGREE];
    for (uint64_t k = 0; k < d; k++)
    {
        unsigned __int128 s = 0;
        for (uint64_t i = 0; i < d; i++)
            s += (unsigned __int128)a[i] * (i > k ? wb[k + d - i] : b[k - i]);
        out[k] = modq_wide((uint64_t)(s >> 64), (uint64_t)s, mod);
    }
    memcpy(c, out, d * sizeof(uint64_t));
}

// c = a * b in F_p[x]/(x^d - w), fully reduced. c may alias a or b.
FIELD_FOLD_INLINE void field_fold_scalar_mul(uint64_t *c, const uint64_t *a, const uint64_t *b,
                                             uint64_t d, uint64_t w_mod_q, Modulus mod)
{
    uint64_t wb[FIELD_FOLD_MAX_DEGREE];
    for (uint64_t j = 1; j < d; j++)
        wb[j] = mul_modq(w_mod_q, b[j], mod);
    field_fold_scalar_mul_with_wb(c, a, b, wb, d, mod);
}

// Add the coefficients of a * b, with no reduction at all, into the
// accumulators acc[0 .. 2d - 2].
FIELD_FOLD_INLINE void field_fold_scalar_add_unreduced_product(FieldFoldU192 *acc,
                                                               const uint64_t *a, const uint64_t *b,
                                                               uint64_t d)
{
    for (uint64_t k = 0; k < 2 * d - 1; k++)
    {
        const uint64_t low = k < d ? 0 : k - d + 1, high = k < d ? k : d - 1;
        unsigned __int128 s = 0;
        for (uint64_t i = low; i <= high; i++)
            s += (unsigned __int128)a[i] * b[k - i];
        field_fold_u192_add(&acc[k], s);
    }
}

// The element the accumulators acc[0 .. 2d - 2] add up to, fully reduced, into
// out (d words): each coefficient modulo q, then the polynomial modulo x^d - w,
// which moves the coefficient of x^(d + k) to x^k times w. two128_mod_q is
// 2^128 mod q and w_mod_q is w mod q.
static void field_fold_scalar_reduce_sum(uint64_t *out, const FieldFoldU192 *acc, uint64_t d,
                                         uint64_t w_mod_q, uint64_t two128_mod_q, Modulus mod)
{
    const uint64_t q = mod->q;
    uint64_t poly[FIELD_FOLD_MAX_UNREDUCED];
    for (uint64_t k = 0; k < 2 * d - 1; k++)
    {
        uint64_t value = modq_wide((uint64_t)(acc[k].low >> 64), (uint64_t)acc[k].low, mod);
        if (acc[k].high)
            value = add_modq(value, mul_modq(modq(acc[k].high, mod), two128_mod_q, mod), q);
        poly[k] = value;
    }
    for (uint64_t k = 0; k < d; k++)
        out[k] = k + 1 < d ? add_modq(poly[k], mul_modq(w_mod_q, poly[k + d], mod), q) : poly[k];
}

// The folds of one pair (u, v) at the nodes 0 .. nodes - 1, F[t] = u + t (v - u),
// each from the one before by adding v - u.
FIELD_FOLD_INLINE void field_fold_scalar_nodes(uint64_t (*F)[FIELD_FOLD_MAX_DEGREE],
                                               const uint64_t *u, const uint64_t *v, uint64_t nodes,
                                               uint64_t d, uint64_t q)
{
    uint64_t diff[FIELD_FOLD_MAX_DEGREE];
    for (uint64_t k = 0; k < d; k++)
    {
        F[0][k] = u[k];
        F[1][k] = v[k];
        diff[k] = sub_modq(v[k], u[k], q);
    }
    for (uint64_t t = 2; t < nodes; t++)
        for (uint64_t k = 0; k < d; k++)
            F[t][k] = add_modq(F[t - 1][k], diff[k], q);
}

// Add one pair's contribution to every sum at every node. F[j][t] is the fold
// at node t of the pair of table j, for every table the shape reads; `weight`
// is the pair's weight, or NULL for a weight of one.
//
// Without weights, each product's factors but the last are multiplied with
// reduction, and the last multiplication goes into the accumulators unreduced.
// With weights, the sum's signed products are reduced and added up first, and
// the multiplication by the weight is the unreduced one.
FIELD_FOLD_INLINE void
field_fold_scalar_add_pair(FieldFoldScalarSums *acc,
                           uint64_t (*F)[FIELD_FOLD_MAX_NODES][FIELD_FOLD_MAX_DEGREE],
                           const uint64_t *weight, const FieldProductShape *shape, uint64_t d,
                           uint64_t w_mod_q, Modulus mod)
{
    const uint64_t q = mod->q;
    uint64_t running_product[FIELD_FOLD_MAX_DEGREE], signed_sum[FIELD_FOLD_MAX_DEGREE];

    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
        {
            FieldFoldU192 *target = acc->unreduced[s][t];
            for (uint64_t p = shape->first_product[s]; p < shape->first_product[s + 1]; p++)
            {
                const uint64_t *factors = shape->factors[p];
                // The factors multiplied with reduction: all of them with
                // weights, all but the last without.
                const uint64_t last = shape->factor_count[p] - (weight ? 0 : 1);
                memcpy(running_product, F[factors[0]][t], d * sizeof(uint64_t));
                for (uint64_t f = 1; f < last; f++)
                    field_fold_scalar_mul(running_product, running_product, F[factors[f]][t], d,
                                          w_mod_q, mod);
                if (weight)
                {
                    const bool first = p == shape->first_product[s];
                    for (uint64_t k = 0; k < d; k++)
                    {
                        const uint64_t base = first ? 0 : signed_sum[k];
                        signed_sum[k] = shape->negated[p] ? sub_modq(base, running_product[k], q)
                                                          : add_modq(base, running_product[k], q);
                    }
                    continue;
                }
                if (shape->negated[p])
                    for (uint64_t k = 0; k < d; k++)
                        running_product[k] = sub_modq(0, running_product[k], q);
                if (shape->factor_count[p] == 1)
                    for (uint64_t k = 0; k < d; k++)
                        field_fold_u192_add(&target[k], running_product[k]);
                else
                    field_fold_scalar_add_unreduced_product(target, running_product,
                                                            F[factors[last]][t], d);
            }
            if (weight)
                field_fold_scalar_add_unreduced_product(target, signed_sum, weight, d);
        }
}

// The field's constants the scalar body uses.
typedef struct
{
    Modulus mod;
    uint64_t w_mod_q;      // w mod q
    uint64_t two128_mod_q; // 2^128 mod q
} FieldFoldScalarConstants;

static void field_fold_scalar_constants(FieldFoldScalarConstants *C, const FieldVector v)
{
    C->mod = v->mod;
    C->w_mod_q = modq(v->w, v->mod);
    const uint64_t c64 = modq_wide(1, 0, v->mod);
    C->two128_mod_q = mul_modq(c64, c64, v->mod);
}

static void field_fold_scalar_sums_init(FieldFoldScalarSums *acc, const FieldProductShape *shape,
                                        uint64_t d)
{
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
            memset(acc->unreduced[s][t], 0, (2 * d - 1) * sizeof(FieldFoldU192));
}

static void field_fold_scalar_sums_finish(uint64_t *const *sums, const FieldFoldScalarSums *acc,
                                          const FieldProductShape *shape, uint64_t d,
                                          const FieldFoldScalarConstants *C)
{
    for (uint64_t s = 0; s < shape->sums; s++)
        for (uint64_t t = 0; t < shape->nodes; t++)
            field_fold_scalar_reduce_sum(sums[s * shape->nodes + t], acc->unreduced[s][t], d,
                                         C->w_mod_q, C->two128_mod_q, C->mod);
}

// The sums of the pairs [begin, end) of the tables, into sums[0 .. slots - 1].
static inline void field_fold_scalar_sums(uint64_t *const *sums, const FieldFoldJob *job,
                                          uint64_t begin, uint64_t end, const uint64_t d)
{
    const FieldProductShape *shape = job->shape;
    const uint64_t nodes = shape->nodes;
    const FieldVector weights = job->weights;
    FieldFoldScalarConstants C;
    field_fold_scalar_constants(&C, job->tables[0]);
    const uint64_t q = C.mod->q;
    FieldFoldScalarSums acc;
    uint64_t F[FIELD_FOLD_MAX_TABLES][FIELD_FOLD_MAX_NODES][FIELD_FOLD_MAX_DEGREE];
    uint64_t u[FIELD_FOLD_MAX_DEGREE], v[FIELD_FOLD_MAX_DEGREE], weight[FIELD_FOLD_MAX_DEGREE];

    field_fold_scalar_sums_init(&acc, shape, d);
    for (uint64_t i = begin; i < end; i++)
    {
        for (uint64_t x = 0; x < shape->used_count; x++)
        {
            const uint64_t j = shape->used[x];
            uint64_t *const *planes = job->tables[j]->coeffs;
            for (uint64_t k = 0; k < d; k++)
            {
                u[k] = planes[k][2 * i];
                v[k] = planes[k][2 * i + 1];
            }
            field_fold_scalar_nodes(F[j], u, v, nodes, d, q);
        }
        if (weights)
            for (uint64_t k = 0; k < d; k++)
                weight[k] = weights->coeffs[k][i];
        field_fold_scalar_add_pair(&acc, F, weights ? weight : NULL, shape, d, C.w_mod_q, C.mod);
    }
    field_fold_scalar_sums_finish(sums, &acc, shape, d, &C);
}

// The "fold and" variant over the pairs [begin, end) of the folded tables. A
// table's folded pair i is made of the folds at r of its input pairs 2i and
// 2i + 1; it is written to the table's output and added into the sums. The
// chunk that ends the range also zeroes the outputs' padding.
static inline void field_fold_scalar_fold_sums(uint64_t *const *sums, const FieldFoldJob *job,
                                               uint64_t begin, uint64_t end, const uint64_t d)
{
    const FieldProductShape *shape = job->shape;
    const uint64_t nodes = shape->nodes;
    const FieldVector weights = job->weights;
    FieldFoldScalarConstants C;
    field_fold_scalar_constants(&C, job->tables[0]);
    const uint64_t q = C.mod->q;
    FieldFoldScalarSums acc;
    uint64_t F[FIELD_FOLD_MAX_TABLES][FIELD_FOLD_MAX_NODES][FIELD_FOLD_MAX_DEGREE];
    uint64_t pair[2][FIELD_FOLD_MAX_DEGREE], diff[FIELD_FOLD_MAX_DEGREE];
    uint64_t r[FIELD_FOLD_MAX_DEGREE], wr[FIELD_FOLD_MAX_DEGREE], weight[FIELD_FOLD_MAX_DEGREE];

    for (uint64_t k = 0; k < d; k++)
    {
        r[k] = modq(job->r[k], C.mod);
        wr[k] = mul_modq(C.w_mod_q, r[k], C.mod);
    }
    field_fold_scalar_sums_init(&acc, shape, d);
    for (uint64_t i = begin; i < end; i++)
    {
        for (uint64_t j = 0; j < job->count; j++)
        {
            uint64_t *const *in = job->tables[j]->coeffs;
            uint64_t *const *out = job->outs[j]->coeffs;
            for (uint64_t h = 0; h < 2; h++)
            {
                const uint64_t o = 2 * i + h;
                for (uint64_t k = 0; k < d; k++)
                {
                    pair[h][k] = in[k][2 * o];
                    diff[k] = sub_modq(in[k][2 * o + 1], pair[h][k], q);
                }
                field_fold_scalar_mul_with_wb(diff, diff, r, wr, d, C.mod);
                for (uint64_t k = 0; k < d; k++)
                {
                    pair[h][k] = add_modq(pair[h][k], diff[k], q);
                    out[k][o] = pair[h][k];
                }
            }
            if (shape->is_used[j])
                field_fold_scalar_nodes(F[j], pair[0], pair[1], nodes, d, q);
        }
        if (weights)
            for (uint64_t k = 0; k < d; k++)
                weight[k] = weights->coeffs[k][i];
        field_fold_scalar_add_pair(&acc, F, weights ? weight : NULL, shape, d, C.w_mod_q, C.mod);
    }
    if (end == job->total)
        for (uint64_t j = 0; j < job->count; j++)
        {
            const FieldVector out = job->outs[j];
            for (uint64_t k = 0; k < d; k++)
                memset(out->coeffs[k] + 2 * end, 0,
                       (out->allocated_n - 2 * end) * sizeof(uint64_t));
        }
    field_fold_scalar_sums_finish(sums, &acc, shape, d, &C);
}

// --- this body's ranges, for field_vector_fold.c ---------------------------

FIELD_FOLD_BODIES(field_fold_scalar_sums)
FIELD_FOLD_BODIES(field_fold_scalar_fold_sums)

void field_fold_scalar_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                  uint64_t *const *sums)
{
    FIELD_FOLD_CALL(field_fold_scalar_sums, job->tables[0]->d, sums, job, begin, end);
}

void field_fold_scalar_fold_sums_range(const FieldFoldJob *job, uint64_t begin, uint64_t end,
                                       uint64_t *const *sums)
{
    FIELD_FOLD_CALL(field_fold_scalar_fold_sums, job->tables[0]->d, sums, job, begin, end);
}

#endif // !VFHE_HAVE_AVX512IFMA
