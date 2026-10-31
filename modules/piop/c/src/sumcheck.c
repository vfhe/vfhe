// SPDX-FileCopyrightText: 2026 The vFHE Authors
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

// Degree-3 accumulation for one pair of each of three factors: g_out[t] +=
// f_t * g_t * h_t for t in {0, 1, 2, 3}, a factor's value at t = 2 and t = 3
// extrapolated division-free from its pair as hi + (hi - lo) and
// hi + 2 (hi - lo). `diff` and `ext` hold one temporary per factor.
static void sumcheck_prod3_accumulate(ArithRing ring, ArithElement *g_out,
                                      const ArithElement *const lo[3],
                                      const ArithElement *const hi[3], ArithElement *diff,
                                      ArithElement *ext, ArithElement *prod)
{
    arith_mul(ring, prod, lo[0], lo[1]);
    arith_mul_addto(ring, &g_out[0], prod, lo[2]);
    arith_mul(ring, prod, hi[0], hi[1]);
    arith_mul_addto(ring, &g_out[1], prod, hi[2]);
    for (int j = 0; j < 3; j++)
    {
        arith_sub(ring, &diff[j], hi[j], lo[j]);
        arith_add(ring, &ext[j], hi[j], &diff[j]);
    }
    arith_mul(ring, prod, &ext[0], &ext[1]);
    arith_mul_addto(ring, &g_out[2], prod, &ext[2]);
    for (int j = 0; j < 3; j++)
    {
        arith_add(ring, &ext[j], &ext[j], &diff[j]);
    }
    arith_mul(ring, prod, &ext[0], &ext[1]);
    arith_mul_addto(ring, &g_out[3], prod, &ext[2]);
}

// The degree-3 round over the pairs of the variable at `eval_var_idx`: for
// the LSB variable they are adjacent entries, for the MSB one the two table
// halves, both instances of the stride computation.
static void sumcheck_prod3_round_strided(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                         ArithElement *tg, ArithElement *th, uint64_t size,
                                         uint64_t eval_var_idx)
{
    uint64_t stride = 1ULL << eval_var_idx;
    ArithElement *tables[3] = {tf, tg, th};
    ArithElement diff[3], ext[3], prod;
    for (int j = 0; j < 3; j++)
    {
        arith_new_like(ring, &tables[j][0], &diff[j]);
        arith_new_like(ring, &tables[j][0], &ext[j]);
    }
    arith_new_like(ring, &tf[0], &prod);

    for (int t = 0; t < 4; t++)
    {
        arith_zero_in(ring, &g_out[t], arith_mul_domain(ring));
    }
    for (uint64_t i = 0; i < size / 2; i++)
    {
        uint64_t i_low = i & (stride - 1);
        uint64_t i_high = i >> eval_var_idx;
        uint64_t idx0 = i_low + (i_high << (eval_var_idx + 1));
        const ArithElement *lo[3] = {&tf[idx0], &tg[idx0], &th[idx0]};
        const ArithElement *hi[3] = {&tf[idx0 + stride], &tg[idx0 + stride], &th[idx0 + stride]};
        sumcheck_prod3_accumulate(ring, g_out, lo, hi, diff, ext, &prod);
    }

    for (int j = 0; j < 3; j++)
    {
        arith_free(ring, &diff[j]);
        arith_free(ring, &ext[j]);
    }
    arith_free(ring, &prod);
}

void sumcheck_prod3_round_pairs(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                ArithElement *tg, ArithElement *th, uint64_t size)
{
    sumcheck_prod3_round_strided(ring, g_out, tf, tg, th, size, 0);
}

void sumcheck_prod3_round_halves(ArithRing ring, ArithElement *g_out, ArithElement *tf,
                                 ArithElement *tg, ArithElement *th, uint64_t size)
{
    uint64_t eval_var_idx = 0;
    while ((2ULL << eval_var_idx) < size)
    {
        eval_var_idx++;
    }
    sumcheck_prod3_round_strided(ring, g_out, tf, tg, th, size, eval_var_idx);
}

void sumcheck_prod3_round(ArithRing ring, ArithElement *g_out, ArithElement *tf, ArithElement *tg,
                          ArithElement *th, uint64_t size, uint64_t eval_var_idx)
{
    sumcheck_prod3_round_strided(ring, g_out, tf, tg, th, size, eval_var_idx);
}
