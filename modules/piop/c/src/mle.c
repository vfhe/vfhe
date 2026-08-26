// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"
#include "arith_generic.h"

// -------------------------------------------------------------
// Dense vector operations over a ring the caller does not name
// -------------------------------------------------------------

void mle_dense_poly_add(ArithRing ring, ArithElement *out, ArithElement *in1, ArithElement *in2,
                        uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        arith_add(ring, &out[i], &in1[i], &in2[i]);
    }
}

void mle_dense_poly_sub(ArithRing ring, ArithElement *out, ArithElement *in1, ArithElement *in2,
                        uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        arith_sub(ring, &out[i], &in1[i], &in2[i]);
    }
}

void mle_dense_poly_scale(ArithRing ring, ArithElement *out, ArithElement *in,
                          const ArithElement *scale, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        arith_mul(ring, &out[i], &in[i], scale);
    }
}

void mle_dense_poly_scale_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                 uint64_t scale, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        arith_scale_int(ring, &out[i], &in[i], scale);
    }
}

// Binding one variable folds table pairs as out = lo + a * (hi - lo). The
// entry points below only differ in where the pair (lo, hi) sits: adjacent
// entries for the LSB variable (pairs), the two table halves for the MSB
// variable (halves), and stride-computed indices for any variable in between
// (the generic fallback). The Python layer picks by the variable's position.

static void mle_dense_poly_bind(ArithRing ring, ArithElement *out, const ArithElement *lo,
                                const ArithElement *hi, const ArithElement *a, ArithElement *tmp,
                                ArithElement *tmp2)
{
    arith_sub(ring, tmp, hi, lo);
    arith_mul(ring, tmp2, a, tmp);
    arith_add(ring, out, lo, tmp2);
}

static void mle_dense_poly_bind_scalar(ArithRing ring, ArithElement *out, const ArithElement *lo,
                                       const ArithElement *hi, uint64_t a, ArithElement *tmp,
                                       ArithElement *tmp2)
{
    arith_sub(ring, tmp, hi, lo);
    arith_scale_int(ring, tmp2, tmp, a);
    arith_add(ring, out, lo, tmp2);
}

void mle_dense_poly_evaluate_pairs(ArithRing ring, ArithElement *out, ArithElement *in,
                                   const ArithElement *a, uint64_t size)
{
    ArithElement tmp, tmp2;
    arith_new_like(ring, &in[0], &tmp);
    arith_new_like(ring, &in[0], &tmp2);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind(ring, &out[i], &in[2 * i], &in[2 * i + 1], a, &tmp, &tmp2);
    }
    arith_free(ring, &tmp);
    arith_free(ring, &tmp2);
}

void mle_dense_poly_evaluate_pairs_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                          uint64_t a, uint64_t size)
{
    ArithElement tmp, tmp2;
    arith_new_like(ring, &in[0], &tmp);
    arith_new_like(ring, &in[0], &tmp2);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind_scalar(ring, &out[i], &in[2 * i], &in[2 * i + 1], a, &tmp, &tmp2);
    }
    arith_free(ring, &tmp);
    arith_free(ring, &tmp2);
}

void mle_dense_poly_evaluate_halves(ArithRing ring, ArithElement *out, ArithElement *in,
                                    const ArithElement *a, uint64_t size)
{
    ArithElement tmp, tmp2;
    arith_new_like(ring, &in[0], &tmp);
    arith_new_like(ring, &in[0], &tmp2);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind(ring, &out[i], &in[i], &in[i + size], a, &tmp, &tmp2);
    }
    arith_free(ring, &tmp);
    arith_free(ring, &tmp2);
}

void mle_dense_poly_evaluate_halves_scalar(ArithRing ring, ArithElement *out, ArithElement *in,
                                           uint64_t a, uint64_t size)
{
    ArithElement tmp, tmp2;
    arith_new_like(ring, &in[0], &tmp);
    arith_new_like(ring, &in[0], &tmp2);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind_scalar(ring, &out[i], &in[i], &in[i + size], a, &tmp, &tmp2);
    }
    arith_free(ring, &tmp);
    arith_free(ring, &tmp2);
}

void mle_dense_poly_evaluate(ArithRing ring, ArithElement *out, ArithElement *in,
                             const ArithElement *a, uint64_t num_vars, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    uint64_t size = 1ULL << (num_vars - 1);

    ArithElement temp, temp2;
    arith_new_like(ring, &in[0], &temp);
    arith_new_like(ring, &in[0], &temp2);

    for (uint64_t i = 0; i < size; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        uint64_t idx1 = idx0 + stride;

        mle_dense_poly_bind(ring, &out[i], &in[idx0], &in[idx1], a, &temp, &temp2);
    }

    arith_free(ring, &temp);
    arith_free(ring, &temp2);
}

void mle_dense_poly_evaluate_scalar(ArithRing ring, ArithElement *out, ArithElement *in, uint64_t a,
                                    uint64_t num_vars, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    uint64_t size = 1ULL << (num_vars - 1);

    ArithElement temp, temp2;
    arith_new_like(ring, &in[0], &temp);
    arith_new_like(ring, &in[0], &temp2);

    for (uint64_t i = 0; i < size; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        uint64_t idx1 = idx0 + stride;

        mle_dense_poly_bind_scalar(ring, &out[i], &in[idx0], &in[idx1], a, &temp, &temp2);
    }

    arith_free(ring, &temp);
    arith_free(ring, &temp2);
}
