// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "util.h"

void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                           uint64_t special_primes)
{
    const uint64_t r = in->r;
    (void)special_primes;

    // The products accumulate in the ring the MGSW key lives in, which is
    // wider than `out`'s: `out` is only allocated for its own ring. The
    // rescale afterwards brings the result back, and which primes leave
    // follows from the two rings rather than from counting special ones.
    RNS_MLWE acc = mlwe_alloc_sample(mgsw[0]->ring, out->r);
    mlwe_RNS_trivial_sample_of_zero(acc);

    for (size_t j = 0; j < r; j++)
    {
        gadget_mul_addto_polynomial(acc, &mgsw[j * ell], &in->a[j]);
    }
    gadget_mul_addto_polynomial(acc, &mgsw[r * ell], &in->b);

    mlwe_RNS_to_RNSc(acc, acc);
    mlwe_round_division(acc, out->ring);
    mlwe_RNSc_to_RNS(acc, acc);
    mlwe_copy_RNS_sample(out, acc);
    free_mlwe_RNS_sample(acc);
}

void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
               uint64_t special_primes)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;

    RNSc_MLWE diff = mlwe_alloc_sample(ring, r);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    mgsw_external_product(out, mgsw, diff, ell, special_primes);

    RNS_MLWE in1_NTT = mlwe_alloc_sample(ring, r);
    mlwe_copy_RNS_sample(in1_NTT, in1);
    mlwe_RNSc_to_RNS(in1_NTT, in1_NTT);
    mlwe_add_RNS_sample(out, out, in1_NTT);

    free_mlwe_RNS_sample(diff);
    free_mlwe_RNS_sample(in1_NTT);
}

void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE_KS_Key ksk,
                uint64_t ell, uint64_t special_primes)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;
    const uint64_t gen = 2 * ring->N - 1;

    RNSc_MLWE tmp = mlwe_alloc_sample(ring, r);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX(out, in1, tmp, mgsw, ell, special_primes);

    free_mlwe_RNS_sample(tmp);
}

/* ------------------------------------------------------------------------------------------------
 * Coefficient-domain output variants of CMUX / NCMUX, for callers (the GP25 monomial multiply)
 * that immediately convert the result to coefficient form for the next stage.
 *
 * Standard CMUX computes  out = ExtProduct(in2 - in1) + in1  entirely in the NTT domain, which
 * requires a forward NTT of `in1`; the caller then inverse-NTTs `out`. Since the (inverse) NTT is
 * linear, invNTT(ExtProduct + NTT(in1)) == invNTT(ExtProduct) + in1, so we instead inverse-NTT the
 * external product alone and add `in1` directly in coefficient form. This is bit-for-bit equivalent
 * but drops the forward NTT of `in1` (and folds the caller's inverse NTT in here).
 * ------------------------------------------------------------------------------------------------
 */
void mgsw_CMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
                        uint64_t special_primes)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;

    RNSc_MLWE diff = mlwe_alloc_sample(ring, r);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    mgsw_external_product(out, mgsw, diff, ell, special_primes); /* out in NTT  */
    mlwe_RNS_to_RNSc(out, out);                                  /* out -> coeff */
    mlwe_addto_RNSc_sample(out, in1); /* out += in1 (coeff; no fwd NTT of in1) */

    free_mlwe_RNS_sample(diff);
}

void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw,
                         RNS_MLWE_KS_Key ksk, uint64_t ell, uint64_t special_primes)
{
    const uint64_t r = in1->r;
    ArithRing ring = in1->ring;
    const uint64_t gen = 2 * ring->N - 1;

    RNSc_MLWE tmp = mlwe_alloc_sample(ring, r);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX_to_coeff(out, in1, tmp, mgsw, ell, special_primes);

    free_mlwe_RNS_sample(tmp);
}
