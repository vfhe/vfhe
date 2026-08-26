// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "arith.h"
#include "arith_generic.h"

// -------------------------------------------------------------
// Sumcheck prover kernels over dense-MLE tables of ring elements
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

static void sumcheck_round_accumulate(ArithRing ring, ArithElement *g0, ArithElement *g1,
                                      const ArithElement *lo, const ArithElement *hi)
{
    arith_add(ring, g0, g0, lo);
    arith_add(ring, g1, g1, hi);
}

void sumcheck_round_pairs(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                          uint64_t size)
{
    arith_zero_in(ring, g0, arith_mul_domain(ring));
    arith_zero_in(ring, g1, arith_mul_domain(ring));
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_round_accumulate(ring, g0, g1, &table[2 * i], &table[2 * i + 1]);
    }
}

void sumcheck_round_halves(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                           uint64_t size)
{
    arith_zero_in(ring, g0, arith_mul_domain(ring));
    arith_zero_in(ring, g1, arith_mul_domain(ring));
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_round_accumulate(ring, g0, g1, &table[i], &table[i + size / 2]);
    }
}

void sumcheck_round(ArithRing ring, ArithElement *g0, ArithElement *g1, ArithElement *table,
                    uint64_t size, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    arith_zero_in(ring, g0, arith_mul_domain(ring));
    arith_zero_in(ring, g1, arith_mul_domain(ring));
    for (uint64_t i = 0; i < size / 2; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        sumcheck_round_accumulate(ring, g0, g1, &table[idx0], &table[idx0 + stride]);
    }
}

// Degree-2 accumulation for one pair of each factor: g_out[0] += lo_f*lo_g,
// g_out[1] += hi_f*hi_g, g_out[2] += (2*hi_f - lo_f)*(2*hi_g - lo_g) (the
// evaluation at t = 2, extrapolated division-free).
static void sumcheck_prod2_accumulate(ArithRing ring, ArithElement *g_out, const ArithElement *f_lo,
                                      const ArithElement *f_hi, const ArithElement *g_lo,
                                      const ArithElement *g_hi, ArithElement *tmp1,
                                      ArithElement *tmp2)
{
    arith_mul_addto(ring, &g_out[0], f_lo, g_lo);
    arith_mul_addto(ring, &g_out[1], f_hi, g_hi);
    arith_scale_int(ring, tmp1, f_hi, 2);
    arith_sub(ring, tmp1, tmp1, f_lo);
    arith_scale_int(ring, tmp2, g_hi, 2);
    arith_sub(ring, tmp2, tmp2, g_lo);
    arith_mul_addto(ring, &g_out[2], tmp1, tmp2);
}

void sumcheck_prod2_round_pairs(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                ArithElement *tg, uint64_t size)
{
    ArithElement tmp1, tmp2;
    arith_new_like(ring, &tf[0], &tmp1);
    arith_new_like(ring, &tg[0], &tmp2);

    arith_zero_in(ring, &g_out[0], arith_mul_domain(ring));
    arith_zero_in(ring, &g_out[1], arith_mul_domain(ring));
    arith_zero_in(ring, &g_out[2], arith_mul_domain(ring));
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_prod2_accumulate(ring, g_out, &tf[2 * i], &tf[2 * i + 1], &tg[2 * i],
                                  &tg[2 * i + 1], &tmp1, &tmp2);
    }

    arith_free(ring, &tmp1);
    arith_free(ring, &tmp2);
}

void sumcheck_prod2_round_halves(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                 ArithElement *tg, uint64_t size)
{
    ArithElement tmp1, tmp2;
    arith_new_like(ring, &tf[0], &tmp1);
    arith_new_like(ring, &tg[0], &tmp2);

    arith_zero_in(ring, &g_out[0], arith_mul_domain(ring));
    arith_zero_in(ring, &g_out[1], arith_mul_domain(ring));
    arith_zero_in(ring, &g_out[2], arith_mul_domain(ring));
    for (uint64_t i = 0; i < size / 2; i++)
    {
        sumcheck_prod2_accumulate(ring, g_out, &tf[i], &tf[i + size / 2], &tg[i], &tg[i + size / 2],
                                  &tmp1, &tmp2);
    }

    arith_free(ring, &tmp1);
    arith_free(ring, &tmp2);
}

void sumcheck_prod2_round(ArithRing ring, ArithElement *g_out, ArithElement *tf, ArithElement *tg,
                          uint64_t size, uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    ArithElement tmp1, tmp2;
    arith_new_like(ring, &tf[0], &tmp1);
    arith_new_like(ring, &tg[0], &tmp2);

    arith_zero_in(ring, &g_out[0], arith_mul_domain(ring));
    arith_zero_in(ring, &g_out[1], arith_mul_domain(ring));
    arith_zero_in(ring, &g_out[2], arith_mul_domain(ring));
    for (uint64_t i = 0; i < size / 2; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        sumcheck_prod2_accumulate(ring, g_out, &tf[idx0], &tf[idx0 + stride], &tg[idx0],
                                  &tg[idx0 + stride], &tmp1, &tmp2);
    }

    arith_free(ring, &tmp1);
    arith_free(ring, &tmp2);
}
