#include "mlwe.h"
#include "misc.h"

void mgsw_external_product(RNS_MLWE out, RNS_MLWE *mgsw, RNSc_MLWE in, uint64_t ell,
                           uint64_t special_primes)
{
    const uint64_t r = in->r;
    uint64_t extended_mask = mgsw[0]->b->rns_mask;
    for (size_t i = 0; i < out->r; i++)
    {
        out->a[i]->rns_mask = extended_mask;
    }
    out->b->rns_mask = extended_mask;

    mlwe_RNS_trivial_sample_of_zero(out);

    for (size_t j = 0; j < r; j++)
    {
        gadget_mul_addto_polynomial(out, &mgsw[j * ell], in->a[j]);
    }

    gadget_mul_addto_polynomial(out, &mgsw[r * ell], in->b);

    if (special_primes > 0)
    {
        mlwe_RNS_to_RNSc((RNSc_MLWE)out, out);
        uint64_t divide_mask = 0;
        uint64_t temp_mask = ((RNSc_Polynomial)out->b)->rns_mask;
        uint64_t count = 0;
        for (int idx = 63; idx >= 0 && count < special_primes; idx--)
        {
            if (temp_mask & (1ULL << idx))
            {
                divide_mask |= (1ULL << idx);
                count++;
            }
        }
        if (divide_mask > 0)
        {
            for (size_t j = 0; j < out->r; j++)
            {
                polynomial_round_division_RNSc_wo_free(((RNSc_MLWE)out)->a[j], divide_mask);
            }
            polynomial_round_division_RNSc_wo_free(((RNSc_MLWE)out)->b, divide_mask);
        }
        mlwe_RNSc_to_RNS(out, (RNSc_MLWE)out);
    }
}

void mgsw_CMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, uint64_t ell,
               uint64_t special_primes)
{
    uint64_t N = in1->b->ntt->N;
    uint64_t r = in1->r;
    uint64_t mask = in1->b->rns_mask;
    incNTT ntt = in1->b->ntt;

    RNSc_MLWE diff = mlwe_alloc_RNSc_sample(N, r, mask, ntt);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    mgsw_external_product(out, mgsw, diff, ell, special_primes);

    RNS_MLWE in1_NTT = mlwe_alloc_RNS_sample(N, r, mask, ntt);
    mlwe_copy_RNS_sample(in1_NTT, (RNS_MLWE)in1);
    mlwe_RNSc_to_RNS(in1_NTT, (RNSc_MLWE)in1_NTT);
    mlwe_add_RNS_sample(out, out, in1_NTT);

    free_mlwe_RNS_sample(diff);
    free_mlwe_RNS_sample(in1_NTT);
}

void mgsw_NCMUX(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE **ksk,
                uint64_t ell, uint64_t special_primes)
{
    uint64_t N = in1->b->ntt->N;
    uint64_t r = in1->r;
    uint64_t mask = in1->b->rns_mask;
    incNTT ntt = in1->b->ntt;
    uint64_t gen = 2 * N - 1;

    RNSc_MLWE tmp = mlwe_alloc_RNSc_sample(N, r, mask, ntt);

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
    uint64_t N = in1->b->ntt->N;
    uint64_t r = in1->r;
    uint64_t mask = in1->b->rns_mask;
    incNTT ntt = in1->b->ntt;

    RNSc_MLWE diff = mlwe_alloc_RNSc_sample(N, r, mask, ntt);
    mlwe_sub_RNSc_sample(diff, in2, in1);

    mgsw_external_product(out, mgsw, diff, ell, special_primes); /* out in NTT  */
    mlwe_RNS_to_RNSc((RNSc_MLWE)out, out);                       /* out -> coeff */
    mlwe_addto_RNSc_sample((RNSc_MLWE)out, in1); /* out += in1 (coeff; no fwd NTT of in1) */

    free_mlwe_RNS_sample(diff);
}

void mgsw_NCMUX_to_coeff(RNS_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2, RNS_MLWE *mgsw, RNS_MLWE **ksk,
                         uint64_t ell, uint64_t special_primes)
{
    uint64_t N = in1->b->ntt->N;
    uint64_t r = in1->r;
    uint64_t mask = in1->b->rns_mask;
    incNTT ntt = in1->b->ntt;
    uint64_t gen = 2 * N - 1;

    RNSc_MLWE tmp = mlwe_alloc_RNSc_sample(N, r, mask, ntt);

    mlwe_automorphism_RNSc_GHS(tmp, in2, gen, ksk, ell);

    mgsw_CMUX_to_coeff(out, in1, tmp, mgsw, ell, special_primes);

    free_mlwe_RNS_sample(tmp);
}
