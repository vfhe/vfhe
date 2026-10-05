// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "zyl17.h"

#include <stdlib.h>
#include <util.h>

uint64_t zyl17_group_size(const UnfoldedRotation *R, uint64_t g)
{
    const uint64_t left = R->n - g * R->unfolding;
    return left < R->unfolding ? left : R->unfolding;
}

int zyl17_group_is_combined(const UnfoldedRotation *R, uint64_t g)
{
    return R->all_patterns || zyl17_group_size(R, g) > 1;
}

int zyl17_needs_combination(const UnfoldedRotation *R)
{
    return R->all_patterns || (R->unfolding > 1 && R->n > 1);
}

static uint64_t rows_of(const UnfoldedRotation *R) { return (R->r + 1) * R->ell; }

static ArithElement *component(RNS_MLWE c, uint64_t i, uint64_t r)
{
    return i < r ? &c->a[i] : &c->b;
}

void zyl17_combined_key_init(CombinedKey *ck, const UnfoldedRotation *R)
{
    const uint64_t rows = rows_of(R);
    ck->rows = (RNS_MLWE *)safe_malloc(rows * sizeof(RNS_MLWE));
    for (uint64_t k = 0; k < rows; k++)
        ck->rows[k] = mlwe_alloc_sample(R->key_ring, R->r);
    ck->zero = 1;
}

void zyl17_combined_key_free(CombinedKey *ck, const UnfoldedRotation *R)
{
    const uint64_t rows = rows_of(R);
    for (uint64_t k = 0; k < rows; k++)
        free_mlwe_RNS_sample(ck->rows[k]);
    free(ck->rows);
}

// `one` is the constant 1, canonical; `factor` holds a transformed factor
// (evaluation) or a rotated key component (coefficient).
void zyl17_scratch_init(CombineScratch *s, const UnfoldedRotation *R)
{
    const uint64_t value = 1;
    arith_new(R->key_ring, &s->one);
    arith_from_int_array(R->key_ring, &s->one, &value, 1);
    arith_to_canonical(R->key_ring, &s->one);
    arith_new(R->key_ring, &s->factor);
}

void zyl17_scratch_free(CombineScratch *s, const UnfoldedRotation *R)
{
    arith_free(R->key_ring, &s->one);
    arith_free(R->key_ring, &s->factor);
}

/* ------------------------------------------------------------------------------------------------
 * Monomial table: x[m] = X^m in the mul domain for 0 < m < N (x[0], the constant 1, is never
 * read: a unit factor is an addition).
 * ------------------------------------------------------------------------------------------------
 */
struct _ZYL17_MonomialTable
{
    ArithRing ring;
    uint64_t N;
    ArithElement *x;
};

typedef struct
{
    ZYL17_MonomialTable table;
    const ArithElement *one;
} TableBuild;

static void table_entry(void *ctx, uint64_t m)
{
    const TableBuild *B = (const TableBuild *)ctx;
    ArithRing ring = B->table->ring;
    arith_new(ring, &B->table->x[m]);
    arith_mul_by_monomial(ring, &B->table->x[m], B->one, m, 0);
    arith_to_mul(ring, &B->table->x[m]);
}

ZYL17_MonomialTable zyl17_monomial_table_new(ArithRing ring, uint64_t n_threads)
{
    ZYL17_MonomialTable T = (ZYL17_MonomialTable)safe_malloc(sizeof(*T));
    T->ring = ring;
    T->N = ring->N;
    T->x = (ArithElement *)safe_malloc(T->N * sizeof(ArithElement));
    const uint64_t value = 1;
    ArithElement one;
    arith_new(ring, &one);
    arith_from_int_array(ring, &one, &value, 1);
    arith_to_canonical(ring, &one);
    TableBuild B = {T, &one};
    vfhe_parallel_for(T->N, n_threads, table_entry, &B);
    arith_free(ring, &one);
    return T;
}

const ArithElement *zyl17_monomial(ZYL17_MonomialTable table, uint64_t m) { return &table->x[m]; }

void zyl17_monomial_table_free(ZYL17_MonomialTable table)
{
    for (uint64_t m = 0; m < table->N; m++)
        arith_free(table->ring, &table->x[m]);
    free(table->x);
    free(table);
}

/* ------------------------------------------------------------------------------------------------
 * Combination. Every strategy zeroes the combined key and adds one term f_j BK_j per pattern
 * whose factor does not vanish ([BMMP18]'s X^0 - 1 = 0); a unit factor ([ZYL+17]'s X^0) is an
 * addition. In the mul domain a term is a multiply-accumulate per row component; in the
 * canonical one it is a rotation (fused with the -1) and an addition, and the sum is transformed
 * once at the end.
 * ------------------------------------------------------------------------------------------------
 */
uint64_t zyl17_pattern_exponent(const UnfoldedRotation *R, uint64_t g, uint64_t j)
{
    const uint64_t first = g * R->unfolding, size = zyl17_group_size(R, g);
    uint64_t e = 0;
    for (uint64_t t = 0; t < size; t++)
        if ((j >> t) & 1)
            e += R->a[first + t];
    return e % R->two_n;
}

// dst += sign * src * factor, or dst += sign * src when `factor` is NULL.
static void accumulate(ArithRing ring, ArithElement *dst, const ArithElement *src,
                       const ArithElement *factor, int negate)
{
    if (factor)
    {
        if (negate)
            arith_mul_subto(ring, dst, src, factor);
        else
            arith_mul_addto(ring, dst, src, factor);
    }
    else if (negate)
        arith_sub(ring, dst, dst, src);
    else
        arith_add(ring, dst, dst, src);
}

// Adds f BK to every component of `out` in the mul domain, f = X^e or X^e - 1.
static void add_term_mul_domain(const UnfoldedRotation *R, CombinedKey *out, RNS_MLWE *key,
                                uint64_t e, CombineScratch *s)
{
    ArithRing ring = R->key_ring;
    const uint64_t rows = rows_of(R), N = ring->N;
    const int minus_one = !R->all_patterns;
    // X^e = sign * X^m. The table holds X^m and the -1 becomes a subtraction
    // of the key; on the fly, the whole factor is transformed at once.
    const ArithElement *factor = NULL;
    int negate = 0, subtract_key = 0;
    if (R->combination == ZYL17_COMBINE_PRECOMPUTED && R->monomials)
    {
        const uint64_t m = e % N;
        negate = e >= N;
        subtract_key = minus_one;
        factor = m ? &R->monomials->x[m] : NULL;
    }
    else if (e != 0)
    {
        arith_mul_by_monomial(ring, &s->factor, &s->one, e, minus_one);
        arith_to_mul(ring, &s->factor);
        factor = &s->factor;
    }
    for (uint64_t k = 0; k < rows; k++)
        for (uint64_t i = 0; i <= R->r; i++)
        {
            ArithElement *dst = component(out->rows[k], i, R->r);
            const ArithElement *src = component(key[k], i, R->r);
            accumulate(ring, dst, src, factor, negate);
            if (subtract_key)
                arith_sub(ring, dst, dst, src);
        }
}

// Adds f BK to every component of `out` in the canonical domain.
static void add_term_canonical(const UnfoldedRotation *R, CombinedKey *out, RNS_MLWE *key,
                               uint64_t e, CombineScratch *s)
{
    ArithRing ring = R->key_ring;
    const uint64_t rows = rows_of(R);
    for (uint64_t k = 0; k < rows; k++)
        for (uint64_t i = 0; i <= R->r; i++)
        {
            ArithElement *dst = component(out->rows[k], i, R->r);
            const ArithElement *src = component(key[k], i, R->r);
            if (e == 0)
                arith_add(ring, dst, dst, src);
            else
            {
                arith_mul_by_monomial(ring, &s->factor, src, e, !R->all_patterns);
                arith_add(ring, dst, dst, &s->factor);
            }
        }
}

void zyl17_combine_group(const UnfoldedRotation *R, uint64_t g, CombinedKey *out, CombineScratch *s)
{
    if (!zyl17_group_is_combined(R, g))
        return;
    ArithRing ring = R->key_ring;
    const int canonical = R->combination == ZYL17_COMBINE_COEFFICIENT;
    const ArithDomain domain = canonical ? ARITH_DOMAIN_CANONICAL : arith_mul_domain(ring);
    const uint64_t size = zyl17_group_size(R, g);
    const uint64_t rows = rows_of(R), skip = R->all_patterns ? 0 : 1;
    RNS_MLWE *const *keys = R->bk + g * R->keys_per_group;

    for (uint64_t k = 0; k < rows; k++)
        for (uint64_t i = 0; i <= R->r; i++)
            arith_zero_in(ring, component(out->rows[k], i, R->r), domain);

    int written = 0;
    for (uint64_t j = skip; j < (1ULL << size); j++)
    {
        const uint64_t e = zyl17_pattern_exponent(R, g, j);
        if (e == 0 && !R->all_patterns)
            continue;
        if (canonical)
            add_term_canonical(R, out, keys[j - skip], e, s);
        else
            add_term_mul_domain(R, out, keys[j - skip], e, s);
        written = 1;
    }
    out->zero = !written;
    if (canonical && written)
        for (uint64_t k = 0; k < rows; k++)
            for (uint64_t i = 0; i <= R->r; i++)
                arith_to_mul(ring, component(out->rows[k], i, R->r));
}
