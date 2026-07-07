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

        polynomial_sub_RNS_polynomial(temp, in[idx1], in[idx0]);
        polynomial_mul_RNS_polynomial(temp2, a, temp);
        polynomial_add_RNS_polynomial(out[i], in[idx0], temp2);
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

        polynomial_sub_RNS_polynomial(temp, in[idx1], in[idx0]);
        polynomial_scale_RNS_polynomial(temp2, temp, a);
        polynomial_add_RNS_polynomial(out[i], in[idx0], temp2);
    }

    free_RNS_polynomial(temp);
    free_RNS_polynomial(temp2);
}
