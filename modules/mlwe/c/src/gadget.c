#include "mlwe.h"

void gadget_mul_addto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, RNSc_Polynomial poly)
{
    RNSc_Polynomial tmp = (RNSc_Polynomial)polynomial_new_RNS_polynomial(
        ksk[0]->b->ntt->N, ksk[0]->b->rns_mask, ksk[0]->b->ntt);
    uint64_t ksk_idx = 0;
    for (size_t j = 0; j < out->b->ntt->l; j++)
    {
        if (poly->rns_mask & (1ULL << j))
        {
            polynomial_RNSc_mod_reduce_lifted(tmp, poly, j);
            polynomial_RNSc_to_RNS((RNS_Polynomial)tmp, tmp);
            mlwe_RNS_mul_addto_by_poly(out, ksk[ksk_idx++], (RNS_Polynomial)tmp);
        }
    }
    free_RNS_polynomial(tmp);
}

void gadget_mul_subto_polynomial(RNS_MLWE out, RNS_MLWE *ksk, RNSc_Polynomial poly)
{
    RNSc_Polynomial tmp = (RNSc_Polynomial)polynomial_new_RNS_polynomial(
        ksk[0]->b->ntt->N, ksk[0]->b->rns_mask, ksk[0]->b->ntt);
    uint64_t ksk_idx = 0;
    for (size_t j = 0; j < out->b->ntt->l; j++)
    {
        if (poly->rns_mask & (1ULL << j))
        {
            polynomial_RNSc_mod_reduce_lifted(tmp, poly, j);
            polynomial_RNSc_to_RNS((RNS_Polynomial)tmp, tmp);
            mlwe_RNS_mul_subto_by_poly(out, ksk[ksk_idx++], (RNS_Polynomial)tmp);
        }
    }
    free_RNS_polynomial(tmp);
}
