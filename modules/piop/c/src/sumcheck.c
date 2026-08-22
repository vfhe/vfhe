// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"

// -------------------------------------------------------------
// Sumcheck prover kernels over dense-MLE (RNS_Polynomial) tables
// -------------------------------------------------------------
// Libra-style evaluation-basis prover: per round, accumulate the round
// polynomial's evaluations from the table pairs (lo, hi) = (f|x=0, f|x=1)
// of the round variable. The round variable may sit at any position: the
// entry points only differ in where the pair sits — adjacent entries for
// the LSB variable (pairs), the two table halves for the MSB variable
// (halves), and stride-computed indices for any position (the generic
// fallback taking eval_var_idx). The Python layer picks by the variable's
// position, mirroring the mle.c binding kernels. `size` is the live table
// size (2^num_vars); entries must be in NTT (RNS) form. Folding with the
// challenge is binding the round variable — mle_dense_poly_evaluate* in
// mle.c, not duplicated here.

static void sumcheck_round_accumulate(RNS_Polynomial g0, RNS_Polynomial g1, RNS_Polynomial lo,
                                      RNS_Polynomial hi)
{
    polynomial_add_RNS_polynomial(g0, g0, lo);
    polynomial_add_RNS_polynomial(g1, g1, hi);
}

void sumcheck_round_pairs(RNS_Polynomial g0, RNS_Polynomial g1, RNS_Polynomial *table,
                          uint64_t size)
{
    polynomial_RNS_zero(g0);
    polynomial_RNS_zero(g1);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_round_accumulate(g0, g1, table[2 * i], table[2 * i + 1]);
    }
}

void sumcheck_round_halves(RNS_Polynomial g0, RNS_Polynomial g1, RNS_Polynomial *table,
                           uint64_t size)
{
    polynomial_RNS_zero(g0);
    polynomial_RNS_zero(g1);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_round_accumulate(g0, g1, table[i], table[i + size / 2]);
    }
}

void sumcheck_round(RNS_Polynomial g0, RNS_Polynomial g1, RNS_Polynomial *table, uint64_t size,
                    uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    polynomial_RNS_zero(g0);
    polynomial_RNS_zero(g1);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        sumcheck_round_accumulate(g0, g1, table[idx0], table[idx0 + stride]);
    }
}

// Degree-2 accumulation for one pair of each factor: g_out[0] += lo_f*lo_g,
// g_out[1] += hi_f*hi_g, g_out[2] += (2*hi_f - lo_f)*(2*hi_g - lo_g) (the
// evaluation at t = 2, extrapolated division-free).
static void sumcheck_prod2_accumulate(RNS_Polynomial *g_out, RNS_Polynomial f_lo,
                                      RNS_Polynomial f_hi, RNS_Polynomial g_lo, RNS_Polynomial g_hi,
                                      RNS_Polynomial tmp1, RNS_Polynomial tmp2)
{
    polynomial_mul_addto_RNS_polynomial(g_out[0], f_lo, g_lo);
    polynomial_mul_addto_RNS_polynomial(g_out[1], f_hi, g_hi);
    polynomial_scale_RNS_polynomial(tmp1, f_hi, 2);
    polynomial_sub_RNS_polynomial(tmp1, tmp1, f_lo);
    polynomial_scale_RNS_polynomial(tmp2, g_hi, 2);
    polynomial_sub_RNS_polynomial(tmp2, tmp2, g_lo);
    polynomial_mul_addto_RNS_polynomial(g_out[2], tmp1, tmp2);
}

void sumcheck_prod2_round_pairs(RNS_Polynomial *g_out, RNS_Polynomial *tf, RNS_Polynomial *tg,
                                uint64_t size)
{
    incNTT ntt = tf[0]->ntt;
    RNS_Polynomial tmp1 = polynomial_new_RNS_polynomial(ntt->N, tf[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, tg[0]->rns_mask, ntt);

    polynomial_RNS_zero(g_out[0]);
    polynomial_RNS_zero(g_out[1]);
    polynomial_RNS_zero(g_out[2]);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_prod2_accumulate(g_out, tf[2 * i], tf[2 * i + 1], tg[2 * i], tg[2 * i + 1], tmp1,
                                  tmp2);
    }

    free_RNS_polynomial(tmp1);
    free_RNS_polynomial(tmp2);
}

void sumcheck_prod2_round_halves(RNS_Polynomial *g_out, RNS_Polynomial *tf, RNS_Polynomial *tg,
                                 uint64_t size)
{
    incNTT ntt = tf[0]->ntt;
    RNS_Polynomial tmp1 = polynomial_new_RNS_polynomial(ntt->N, tf[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, tg[0]->rns_mask, ntt);

    polynomial_RNS_zero(g_out[0]);
    polynomial_RNS_zero(g_out[1]);
    polynomial_RNS_zero(g_out[2]);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_prod2_accumulate(g_out, tf[i], tf[i + size / 2], tg[i], tg[i + size / 2], tmp1,
                                  tmp2);
    }

    free_RNS_polynomial(tmp1);
    free_RNS_polynomial(tmp2);
}

void sumcheck_prod2_round(RNS_Polynomial *g_out, RNS_Polynomial *tf, RNS_Polynomial *tg,
                          uint64_t size, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    incNTT ntt = tf[0]->ntt;
    RNS_Polynomial tmp1 = polynomial_new_RNS_polynomial(ntt->N, tf[0]->rns_mask, ntt);
    RNS_Polynomial tmp2 = polynomial_new_RNS_polynomial(ntt->N, tg[0]->rns_mask, ntt);

    polynomial_RNS_zero(g_out[0]);
    polynomial_RNS_zero(g_out[1]);
    polynomial_RNS_zero(g_out[2]);
    for (uint64_t i = 0; i < size / 2; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        sumcheck_prod2_accumulate(g_out, tf[idx0], tf[idx0 + stride], tg[idx0], tg[idx0 + stride],
                                  tmp1, tmp2);
    }

    free_RNS_polynomial(tmp1);
    free_RNS_polynomial(tmp2);
}
