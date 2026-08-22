// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"

// -------------------------------------------------------------
// Polynomial (RNS_Polynomial) Dense Vector operations
// -------------------------------------------------------------

void mle_dense_poly_add(RNS_Polynomial *out, RNS_Polynomial *in1, RNS_Polynomial *in2,
                        uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        polynomial_add_RNS_polynomial(out[i], in1[i], in2[i]);
    }
}

void mle_dense_poly_sub(RNS_Polynomial *out, RNS_Polynomial *in1, RNS_Polynomial *in2,
                        uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        polynomial_sub_RNS_polynomial(out[i], in1[i], in2[i]);
    }
}

void mle_dense_poly_scale(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial scale,
                          uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        polynomial_mul_RNS_polynomial(out[i], in[i], scale);
    }
}

void mle_dense_poly_scale_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t scale,
                                 uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        polynomial_scale_RNS_polynomial(out[i], in[i], scale);
    }
}

// Binding one variable folds table pairs as out = lo + a * (hi - lo). The
// entry points below only differ in where the pair (lo, hi) sits: adjacent
// entries for the LSB variable (pairs), the two table halves for the MSB
// variable (halves), and stride-computed indices for any variable in between
// (the generic fallback). The Python layer picks by the variable's position.

static void mle_dense_poly_bind(RNS_Polynomial out, RNS_Polynomial lo, RNS_Polynomial hi,
                                RNS_Polynomial a, RNS_Polynomial tmp, RNS_Polynomial tmp2)
{
    polynomial_sub_RNS_polynomial(tmp, hi, lo);
    polynomial_mul_RNS_polynomial(tmp2, a, tmp);
    polynomial_add_RNS_polynomial(out, lo, tmp2);
}

static void mle_dense_poly_bind_scalar(RNS_Polynomial out, RNS_Polynomial lo,
                                       RNS_Polynomial hi, uint64_t a, RNS_Polynomial tmp,
                                       RNS_Polynomial tmp2)
{
    polynomial_sub_RNS_polynomial(tmp, hi, lo);
    polynomial_scale_RNS_polynomial(tmp2, tmp, a);
    polynomial_add_RNS_polynomial(out, lo, tmp2);
}

void mle_dense_poly_evaluate_pairs(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial a,
                                   uint64_t size)
{
    incNTT ntt = in[0]->ntt;
    RNS_Polynomial tmp = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind(out[i], in[2 * i], in[2 * i + 1], a, tmp, tmp2);
    }
    free_RNS_polynomial(tmp);
    free_RNS_polynomial(tmp2);
}

void mle_dense_poly_evaluate_pairs_scalar(RNS_Polynomial *out, RNS_Polynomial *in,
                                          uint64_t a, uint64_t size)
{
    incNTT ntt = in[0]->ntt;
    RNS_Polynomial tmp = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind_scalar(out[i], in[2 * i], in[2 * i + 1], a, tmp, tmp2);
    }
    free_RNS_polynomial(tmp);
    free_RNS_polynomial(tmp2);
}

void mle_dense_poly_evaluate_halves(RNS_Polynomial *out, RNS_Polynomial *in,
                                    RNS_Polynomial a, uint64_t size)
{
    incNTT ntt = in[0]->ntt;
    RNS_Polynomial tmp = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind(out[i], in[i], in[i + size], a, tmp, tmp2);
    }
    free_RNS_polynomial(tmp);
    free_RNS_polynomial(tmp2);
}

void mle_dense_poly_evaluate_halves_scalar(RNS_Polynomial *out, RNS_Polynomial *in,
                                           uint64_t a, uint64_t size)
{
    incNTT ntt = in[0]->ntt;
    RNS_Polynomial tmp = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, in[0]->rns_mask, ntt);
    for (uint64_t i = 0; i < size; i++)
    {
        mle_dense_poly_bind_scalar(out[i], in[i], in[i + size], a, tmp, tmp2);
    }
    free_RNS_polynomial(tmp);
    free_RNS_polynomial(tmp2);
}

void mle_dense_poly_evaluate(RNS_Polynomial *out, RNS_Polynomial *in, RNS_Polynomial a,
                             uint64_t num_vars, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    uint64_t size = 1ULL << (num_vars - 1);

    incNTT ntt = in[0]->ntt;
    uint64_t N = ntt->N;
    uint64_t rns_mask = in[0]->rns_mask;

    RNS_Polynomial temp = polynomial_new_RNS_polynomial(N, rns_mask, ntt);
    RNS_Polynomial temp2 = polynomial_new_RNS_polynomial(N, rns_mask, ntt);

    for (uint64_t i = 0; i < size; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        uint64_t idx1 = idx0 + stride;

        mle_dense_poly_bind(out[i], in[idx0], in[idx1], a, temp, temp2);
    }

    free_RNS_polynomial(temp);
    free_RNS_polynomial(temp2);
}

void mle_dense_poly_evaluate_scalar(RNS_Polynomial *out, RNS_Polynomial *in, uint64_t a,
                                    uint64_t num_vars, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    uint64_t size = 1ULL << (num_vars - 1);

    incNTT ntt = in[0]->ntt;
    uint64_t N = ntt->N;
    uint64_t rns_mask = in[0]->rns_mask;

    RNS_Polynomial temp = polynomial_new_RNS_polynomial(N, rns_mask, ntt);
    RNS_Polynomial temp2 = polynomial_new_RNS_polynomial(N, rns_mask, ntt);

    for (uint64_t i = 0; i < size; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        uint64_t idx1 = idx0 + stride;

        mle_dense_poly_bind_scalar(out[i], in[idx0], in[idx1], a, temp, temp2);
    }

    free_RNS_polynomial(temp);
    free_RNS_polynomial(temp2);
}
