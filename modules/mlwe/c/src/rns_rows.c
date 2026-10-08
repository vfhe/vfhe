// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "rns_rows.h"

// Below this ring dimension a row is too little work to hand to another
// thread: the hand-off costs more than the row.
#define ROWS_MIN_N 4096

typedef struct
{
    uint64_t rows[64];
    uint64_t n_rows, parts;
    void (*body)(void *ctx, uint64_t row, uint64_t part);
    void *ctx;
} RowLoop;

static void row_loop_item(void *ctx, uint64_t k)
{
    RowLoop *loop = (RowLoop *)ctx;
    loop->body(loop->ctx, loop->rows[k / loop->parts], k % loop->parts);
}

void rns_rows_for(uint64_t N, uint64_t mask, uint64_t parts,
                  void (*body)(void *ctx, uint64_t row, uint64_t part), void *ctx)
{
    RowLoop loop = {{0}, 0, parts, body, ctx};
    for (uint64_t i = 0; i < 64; i++)
    {
        if (mask & (1ULL << i))
            loop.rows[loop.n_rows++] = i;
    }
    vfhe_parallel_for(loop.n_rows * parts, N >= ROWS_MIN_N ? 0 : 1, row_loop_item, &loop);
}

// The r + 1 elements of a sample: its mask components, then its body.
static ArithElement *element_of(MLWE c, uint64_t k) { return k < c->r ? &c->a[k] : &c->b; }

typedef struct
{
    MLWE out, in;
    ArithDomain to;
    const uint8_t *convert; // per element: whether it changes domain
} Conversion;

static void convert_row(void *ctx, uint64_t row, uint64_t k)
{
    Conversion *c = (Conversion *)ctx;
    if (!c->convert[k])
        return;
    struct _RNS_Polynomial out_storage, in_storage;
    RNS_Polynomial in = arith_rns_polynomial(element_of(c->in, k));
    if (!(in->rns_mask & (1ULL << row)))
        return;
    RNS_Polynomial in_row = polynomial_RNS_view(&in_storage, in, 1ULL << row);
    RNS_Polynomial out_row =
        polynomial_RNS_view(&out_storage, arith_rns_polynomial(element_of(c->out, k)), 1ULL << row);
    if (c->to == ARITH_DOMAIN_MUL)
        polynomial_RNSc_to_RNS(out_row, (RNSc_Polynomial)in_row);
    else
        polynomial_RNS_to_RNSc((RNSc_Polynomial)out_row, in_row);
}

void mlwe_rns_convert(MLWE out, MLWE in, ArithDomain to)
{
    const uint64_t parts = in->r + 1;
    uint8_t convert[parts];
    uint64_t rows = 0;
    for (uint64_t k = 0; k < parts; k++)
    {
        ArithElement *x = element_of(in, k);
        convert[k] = x->domain != to;
        if (convert[k])
            rows |= arith_rns_polynomial(x)->rns_mask;
        else if (out != in)
            arith_copy(out->ring, element_of(out, k), x);
    }
    Conversion c = {out, in, to, convert};
    rns_rows_for(in->ring->N, rows, parts, convert_row, &c);
    for (uint64_t k = 0; k < parts; k++)
    {
        if (!convert[k])
            continue;
        arith_rns_polynomial(element_of(out, k))->rns_mask =
            arith_rns_polynomial(element_of(in, k))->rns_mask;
        element_of(out, k)->domain = to;
    }
}

typedef struct
{
    MLWE out, in;
    uint64_t gen;
    const uint8_t *permute; // per element: whether it is permuted
} Permutation;

static void permute_row(void *ctx, uint64_t row, uint64_t k)
{
    Permutation *p = (Permutation *)ctx;
    RNSc_Polynomial in = (RNSc_Polynomial)arith_rns_polynomial(element_of(p->in, k));
    if (p->permute[k] && (in->rns_mask & (1ULL << row)))
        polynomial_RNSc_permute_rows((RNSc_Polynomial)arith_rns_polynomial(element_of(p->out, k)),
                                     in, p->gen, 1ULL << row);
}

void mlwe_rns_permute(MLWE out, MLWE in, uint64_t gen)
{
    const uint64_t parts = in->r + 1;
    uint8_t permute[parts];
    uint64_t rows = 0;
    for (uint64_t k = 0; k < parts; k++)
    {
        // As arith_permute: only a canonical element moves, and the rows it
        // does not hold come out zero.
        permute[k] = element_of(in, k)->domain == ARITH_DOMAIN_CANONICAL;
        if (!permute[k])
            continue;
        RNS_Polynomial x = arith_rns_polynomial(element_of(in, k));
        RNS_Polynomial y = arith_rns_polynomial(element_of(out, k));
        struct _RNS_Polynomial outside = *y;
        outside.rns_mask = ~x->rns_mask & ((y->base->l < 64 ? 1ULL << y->base->l : 0) - 1);
        polynomial_RNS_zero(&outside);
        y->rns_mask = x->rns_mask;
        rows |= x->rns_mask;
    }
    Permutation p = {out, in, gen, permute};
    rns_rows_for(in->ring->N, rows, parts, permute_row, &p);
    for (uint64_t k = 0; k < parts; k++)
    {
        if (permute[k])
            element_of(out, k)->domain = ARITH_DOMAIN_CANONICAL;
    }
}

typedef struct
{
    MLWE c;
    uint64_t divide_mask;
    const uint8_t *divide; // per element: whether it is divided
} RoundDivision;

static void round_division_row(void *ctx, uint64_t row, uint64_t k)
{
    RoundDivision *d = (RoundDivision *)ctx;
    if (d->divide[k])
        polynomial_round_division_RNSc_rows(
            (RNSc_Polynomial)arith_rns_polynomial(element_of(d->c, k)), d->divide_mask,
            1ULL << row);
}

void mlwe_rns_round_division(MLWE out, ArithRing to)
{
    const uint64_t parts = out->r + 1;
    // As arith_round_division: the destination's mask says what stays, and
    // only a canonical element moves.
    const uint64_t divide_mask = arith_rns_ring_mask(out->ring) & ~arith_rns_ring_mask(to);
    uint8_t divide[parts];
    uint64_t staying = 0;
    for (uint64_t k = 0; k < parts; k++)
    {
        divide[k] = element_of(out, k)->domain == ARITH_DOMAIN_CANONICAL;
        if (!divide[k])
            continue;
        RNSc_Polynomial p = (RNSc_Polynomial)arith_rns_polynomial(element_of(out, k));
        polynomial_round_division_RNSc_prepare(p, divide_mask);
        staying |= p->rns_mask & ~divide_mask;
    }
    RoundDivision d = {out, divide_mask, divide};
    rns_rows_for(out->ring->N, staying, parts, round_division_row, &d);
    for (uint64_t k = 0; k < parts; k++)
    {
        if (divide[k])
            polynomial_round_division_RNSc_finish(
                (RNSc_Polynomial)arith_rns_polynomial(element_of(out, k)), divide_mask);
    }
    out->ring = to;
}

typedef struct
{
    ArithElement *out;
    MLWE in1, in2;
} TensorProduct;

// The products of mlwe_tensor_product, on one row of every operand.
static void tensor_product_row(void *ctx, uint64_t row, uint64_t part)
{
    (void)part;
    TensorProduct *t = (TensorProduct *)ctx;
    const uint64_t r = t->in1->r, R = mlwe_extended_rank(r);
    const uint64_t mask = 1ULL << row;
    struct _RNS_Polynomial a1_storage[r + 1], a2_storage[r + 1], out_storage[R + 1];
    RNS_Polynomial a1[r + 1], a2[r + 1], o[R + 1];
    for (uint64_t k = 0; k <= r; k++)
    {
        a1[k] =
            polynomial_RNS_view(&a1_storage[k], arith_rns_polynomial(element_of(t->in1, k)), mask);
        a2[k] =
            polynomial_RNS_view(&a2_storage[k], arith_rns_polynomial(element_of(t->in2, k)), mask);
    }
    for (uint64_t k = 0; k <= R; k++)
        o[k] = polynomial_RNS_view(&out_storage[k], arith_rns_polynomial(&t->out[k]), mask);
    // a[r] is the body b.
    size_t k = 0;
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = i; j < r; j++)
        {
            polynomial_mul_RNS_polynomial(o[k], a1[i], a2[j]);
            if (i != j)
                polynomial_mul_addto_RNS_polynomial(o[k], a1[j], a2[i]);
            k++;
        }
    }
    for (size_t i = 0; i < r; i++)
    {
        polynomial_mul_RNS_polynomial(o[k], a1[i], a2[r]);
        polynomial_mul_addto_RNS_polynomial(o[k], a1[r], a2[i]);
        k++;
    }
    polynomial_mul_RNS_polynomial(o[R], a1[r], a2[r]);
}

int mlwe_rns_tensor_product(ArithElement *out, MLWE in1, MLWE in2)
{
    const uint64_t r = in1->r, R = mlwe_extended_rank(r);
    const ArithDomain mul = arith_mul_domain(in1->ring);
    const uint64_t mask = arith_rns_polynomial(&in1->b)->rns_mask;
    for (uint64_t k = 0; k <= r; k++)
    {
        const ArithElement *x = element_of(in1, k), *y = element_of(in2, k);
        if (x->domain != mul || y->domain != mul || arith_rns_polynomial(x)->rns_mask != mask ||
            arith_rns_polynomial(y)->rns_mask != mask)
            return -1;
    }
    TensorProduct t = {out, in1, in2};
    rns_rows_for(in1->ring->N, mask, 1, tensor_product_row, &t);
    for (uint64_t k = 0; k <= R; k++)
    {
        arith_rns_polynomial(&out[k])->rns_mask = mask;
        out[k].domain = mul;
    }
    return 0;
}
