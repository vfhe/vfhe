#include "mlwe.h"
#include "misc.h"

// MLWE RNS functions

RNS_MLWE_Key mlwe_alloc_RNS_key(uint64_t N, uint64_t r, uint64_t l, incNTT ntt, double sigma)
{
    RNS_MLWE_Key res;
    res = (RNS_MLWE_Key)safe_malloc(sizeof(*res));
    res->sigma = sigma;
    res->N = N;
    res->l = l;
    res->r = r;
    res->s = polynomial_new_int_polynomial_array(r, N);
    res->s_RNS = polynomial_new_RNS_polynomial_array(r, N, (1ULL << l) - 1, ntt);
    return res;
}

void free_mlwe_RNS_key(RNS_MLWE_Key key)
{
    free_polynomial_array(key->r, key->s);
    free_RNS_polynomial_array(key->r, key->s_RNS);
    free(key);
}

RNS_MLWE_Key mlwe_new_RNS_key_from_array(uint64_t *array, uint64_t N, uint64_t r, uint64_t l,
                                         incNTT ntt, double sigma)
{
    RNS_MLWE_Key res = mlwe_alloc_RNS_key(N, r, l, ntt, sigma);
    for (size_t i = 0; i < r; i++)
    {
        memcpy(res->s[i]->coeffs, &array[i * N], N * sizeof(uint64_t));
        polynomial_to_RNS(res->s_RNS[i], res->s[i]);
    }
    return res;
}

LWE mlwe_extract_LWE(RNSc_MLWE in, uint64_t idx)
{
    const uint64_t N = in->a[0]->ntt->N;
    const uint64_t l = rns_mask_to_l(in->a[0]->rns_mask);
    const uint64_t r = in->r;
    LWE res = lwe_alloc_sample(r * N, l, in->a[0]->ntt);

    for (size_t j = 0; j < l; j++)
    {
        int g_idx = rns_mask_get_active_index(in->a[0]->rns_mask, j);
        assert(g_idx >= 0);
        NTT_proc proc = in->a[0]->ntt->ntt[g_idx];

        for (size_t k = 0; k < r; k++)
        {
            // Reverse and negate for negacyclic
            for (size_t i = 0; i <= idx; i++)
            {
                res->a[j][k * N + i] = in->a[k]->coeffs[g_idx][idx - i];
            }
            for (size_t i = idx + 1; i < N; i++)
            {
                res->a[j][k * N + i] = negate_modq(in->a[k]->coeffs[g_idx][N + idx - i], proc->q);
            }
        }
        res->b[j] = in->b->coeffs[g_idx][idx];
    }
    return res;
}

RNS_MLWE_Key mlwe_new_RNS_gaussian_key(uint64_t N, uint64_t r, uint64_t l, double key_sigma,
                                       incNTT ntt, double sigma)
{
    RNS_MLWE_Key res = mlwe_alloc_RNS_key(N, r, l, ntt, sigma);
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            res->s[i]->coeffs[j] = (uint64_t)((int64_t)generate_normal_random(key_sigma));
        }
        polynomial_to_RNS(res->s_RNS[i], res->s[i]);
    }
    return res;
}

RNS_MLWE_Key mlwe_get_RNS_key_from_array(uint64_t N, uint64_t r, uint64_t l, uint64_t *array,
                                         incNTT ntt, double sigma)
{
    RNS_MLWE_Key res = mlwe_alloc_RNS_key(N, r, l, ntt, sigma);
    for (size_t j = 0; j < r; j++)
    {
        for (size_t i = 0; i < N; i++)
            res->s[j]->coeffs[i] = array[j * N + i];
        polynomial_to_RNS(res->s_RNS[j], res->s[j]);
    }
    return res;
}

RNS_MLWE mlwe_alloc_RNS_sample(uint64_t N, uint64_t r, uint64_t mask, incNTT ntt)
{
    RNS_MLWE res;
    res = (RNS_MLWE)safe_malloc(sizeof(*res));
    res->a = polynomial_new_RNS_polynomial_array(r, N, mask, ntt);
    res->b = polynomial_new_RNS_polynomial(N, mask, ntt);
    res->r = r;
    return res;
}

RNSc_MLWE mlwe_alloc_RNSc_sample(uint64_t N, uint64_t r, uint64_t mask, incNTT ntt)
{
    return (RNSc_MLWE)mlwe_alloc_RNS_sample(N, r, mask, ntt);
}

void mlwe_copy_array(RNS_MLWE *out, RNS_MLWE *in, uint64_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        mlwe_copy_RNS_sample(out[i], in[i]);
    }
}

RNS_MLWE *mlwe_create_copy_array(RNS_MLWE *in, uint64_t size)
{
    RNS_MLWE *res = mlwe_alloc_RNS_sample_array(size, in[0]->b->ntt->N, in[0]->r,
                                                in[0]->b->rns_mask, in[0]->b->ntt);
    mlwe_copy_array(res, in, size);
    return res;
}

RNS_MLWE *mlwe_alloc_RNS_sample_array(uint64_t size, uint64_t N, uint64_t r, uint64_t mask,
                                      incNTT ntt)
{
    RNS_MLWE *res;
    res = (RNS_MLWE *)safe_malloc(size * sizeof(*res));
    for (size_t i = 0; i < size; i++)
    {
        res[i] = mlwe_alloc_RNS_sample(N, r, mask, ntt);
    }
    return res;
}

RNS_MLWE *mlwe_alloc_RNS_sample_array2(uint64_t size, RNS_MLWE c)
{
    RNS_MLWE *res;
    res = (RNS_MLWE *)safe_malloc(size * sizeof(*res));
    for (size_t i = 0; i < size; i++)
    {
        res[i] = mlwe_alloc_RNS_sample(c->b->ntt->N, c->r, c->b->rns_mask, c->b->ntt);
    }
    return res;
}

void free_RNS_mlwe_array(uint64_t size, RNS_MLWE *v)
{
    for (size_t i = 0; i < size; i++)
    {
        free_mlwe_RNS_sample(v[i]);
    }
    free(v);
}

void free_RNS_mlwe_sample(RNS_MLWE c)
{
    free_RNS_polynomial_array(c->r, c->a);
    free_RNS_polynomial(c->b);
    free(c);
}

void mlwe_copy_RNS_sample(RNS_MLWE out, RNS_MLWE in)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_copy_RNS_polynomial(out->a[i], in->a[i]);
    }
    polynomial_copy_RNS_polynomial(out->b, in->b);
}

void mlwe_copy_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in)
{
    mlwe_copy_RNS_sample((RNS_MLWE)out, (RNS_MLWE)in);
}

void free_mlwe_RNS_sample(void *p)
{
    const RNS_MLWE pp = (RNS_MLWE)p;
    free_RNS_polynomial_array(pp->r, pp->a);
    free_RNS_polynomial(pp->b);
    free(pp);
}

void mlwe_RNS_sample_of_zero(RNS_MLWE out, RNS_MLWE_Key key)
{
    polynomial_gen_gaussian_RNSc_polynomial((RNSc_Polynomial)(out->b), key->sigma);
    polynomial_RNSc_to_RNS(out->b, (RNSc_Polynomial)out->b);
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_gen_random_RNSc_polynomial((RNSc_Polynomial)out->a[i]);
        polynomial_RNSc_to_RNS(out->a[i], (RNSc_Polynomial)out->a[i]);
        polynomial_mul_addto_RNS_polynomial(out->b, key->s_RNS[i], out->a[i]);
    }
}

void mlwe_RNSc_sample_of_zero(RNSc_MLWE out, RNS_MLWE_Key key)
{
    mlwe_RNS_sample_of_zero((RNS_MLWE)out, key);
    mlwe_RNS_to_RNSc(out, (RNS_MLWE)out);
}

void mlwe_scale_RNSc_mlwe(RNSc_MLWE c, uint64_t scale)
{
    for (size_t i = 0; i < c->r; i++)
    {
        polynomial_scale_RNSc_polynomial(c->a[i], c->a[i], scale);
    }
    polynomial_scale_RNSc_polynomial(c->b, c->b, scale);
}

void mlwe_scale_RNS_mlwe_RNS(RNS_MLWE c, uint64_t *scale)
{
    for (size_t i = 0; i < c->r; i++)
    {
        polynomial_scale_RNS_polynomial_RNS(c->a[i], c->a[i], scale);
    }
    polynomial_scale_RNS_polynomial_RNS(c->b, c->b, scale);
}

// out += in*scale
void mlwe_scale_RNS_mlwe_addto(RNS_MLWE out, RNS_MLWE in, uint64_t scale)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_scale_addto_RNS_polynomial(out->a[i], in->a[i], scale);
    }
    polynomial_scale_addto_RNS_polynomial(out->b, in->b, scale);
}

void mlwe_RNSc_sample(RNSc_MLWE out, RNS_MLWE_Key key, RNSc_Polynomial m)
{
    assert(m->rns_mask == out->b->rns_mask);
    mlwe_RNSc_sample_of_zero(out, key);
    polynomial_add_RNSc_polynomial(out->b, out->b, m);
}

void mlwe_RNS_phase(RNS_Polynomial out, RNS_MLWE in, RNS_MLWE_Key key)
{
    polynomial_mul_RNS_polynomial(out, in->a[0], key->s_RNS[0]);
    for (size_t i = 1; i < in->r; i++)
    {
        polynomial_mul_addto_RNS_polynomial(out, in->a[i], key->s_RNS[i]);
    }

    polynomial_sub_RNS_polynomial(out, in->b, out);
}

void mlwe_RNS_mul_by_poly(RNS_MLWE out, RNS_MLWE in, RNS_Polynomial poly)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_mul_RNS_polynomial(out->a[i], in->a[i], poly);
    }
    polynomial_mul_RNS_polynomial(out->b, in->b, poly);
}

void mlwe_RNS_mul_addto_by_poly(RNS_MLWE out, RNS_MLWE in, RNS_Polynomial poly)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_mul_addto_RNS_polynomial(out->a[i], in->a[i], poly);
    }
    polynomial_mul_addto_RNS_polynomial(out->b, in->b, poly);
}

void mlwe_RNS_mul_subto_by_poly(RNS_MLWE out, RNS_MLWE in, RNS_Polynomial poly)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_mul_subto_RNS_polynomial(out->a[i], in->a[i], poly);
    }
    polynomial_mul_subto_RNS_polynomial(out->b, in->b, poly);
}

RNSc_MLWE mlwe_new_RNSc_sample_of_zero(RNS_MLWE_Key key)
{
    RNSc_MLWE res = (RNSc_MLWE)mlwe_alloc_RNS_sample(key->N, key->r, key->l, key->s_RNS[0]->ntt);
    mlwe_RNSc_sample_of_zero(res, key);
    return res;
}

RNS_MLWE mlwe_new_RNS_sample_of_zero(RNS_MLWE_Key key)
{
    RNS_MLWE res = mlwe_alloc_RNS_sample(key->N, key->r, key->l, key->s_RNS[0]->ntt);
    mlwe_RNS_sample_of_zero(res, key);
    return res;
}

RNS_MLWE mlwe_new_RNS_trivial_sample_of_zero(uint64_t N, uint64_t r, uint64_t mask, incNTT ntt)
{
    RNS_MLWE res = mlwe_alloc_RNS_sample(N, r, mask, ntt);
    mlwe_RNS_trivial_sample_of_zero(res);
    return res;
}

void mlwe_RNS_trivial_sample_of_zero(RNS_MLWE out)
{
    for (size_t i = 0; i < out->a[0]->ntt->l; i++)
    {
        for (size_t j = 0; j < out->r; j++)
        {
            memset(out->a[j]->coeffs[i], 0, sizeof(uint64_t) * out->a[j]->ntt->N);
        }
        memset(out->b->coeffs[i], 0, sizeof(uint64_t) * out->b->ntt->N);
    }
}

void mlwe_automorphism_RNSc_GHS(RNSc_MLWE out, RNSc_MLWE in, uint64_t gen, RNS_MLWE **ksk,
                                uint64_t lvl)
{
    RNSc_MLWE tmp = (RNSc_MLWE)mlwe_alloc_RNS_sample(out->a[0]->ntt->N, out->r, out->a[0]->rns_mask,
                                                     out->a[0]->ntt);
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_RNSc_permute(tmp->a[i], in->a[i], gen);
    }
    polynomial_RNSc_permute(tmp->b, in->b, gen);
    mlwe_RNSc_GHS_hybrid_keyswitch(out, tmp, ksk, lvl);
    free_mlwe_RNS_sample(tmp);
}

void mlwe_partial_trace(RNSc_MLWE out, RNSc_MLWE in, uint64_t *gens, RNS_MLWE ***ksks,
                        uint64_t size, uint64_t lvl)
{
    RNSc_MLWE tmp = (RNSc_MLWE)mlwe_alloc_RNS_sample(out->a[0]->ntt->N, out->r, out->a[0]->rns_mask,
                                                     out->a[0]->ntt);
    mlwe_copy_RNSc_sample(tmp, in);
    for (size_t i = 0; i < size; i++)
    {
        mlwe_automorphism_RNSc_GHS(out, tmp, gens[i], ksks[i], lvl);
        mlwe_addto_RNSc_sample(tmp, out);
    }
    mlwe_copy_RNSc_sample(out, tmp);
    free_mlwe_RNS_sample(tmp);
}

void mlwe_full_packing_keyswitch(RNS_MLWE out, LWE *in, uint64_t size, RNS_MLWE **ksk, uint64_t lvl)
{
    (void)lvl;
    const uint64_t N = out->b->ntt->N;
    const uint64_t in_n = in[0]->n;
    const uint64_t lwe_l = in[0]->l;

    const uint64_t target_mask = out->b->rns_mask;
    const uint64_t extended_mask = ksk[0][0]->b->rns_mask;
    const uint64_t divide_mask = extended_mask & ~target_mask;

    for (size_t i = 0; i < out->r; i++)
    {
        out->a[i]->rns_mask = extended_mask;
    }
    out->b->rns_mask = extended_mask;

    mlwe_RNS_trivial_sample_of_zero(out);

    RNSc_Polynomial tmp_poly =
        (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, target_mask, out->b->ntt);
    RNSc_Polynomial tmp_poly_red =
        (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, extended_mask, out->b->ntt);
    RNS_Polynomial tmp_rns =
        (RNS_Polynomial)polynomial_new_RNS_polynomial(N, extended_mask, out->b->ntt);

    for (size_t i = 0; i < in_n; i++)
    {
        for (size_t j = 0; j < tmp_poly->ntt->l; j++)
        {
            if (tmp_poly->rns_mask & (1ULL << j))
            {
                memset(tmp_poly->coeffs[j], 0, N * sizeof(uint64_t));
            }
        }

        for (size_t limb = 0; limb < lwe_l; limb++)
        {
            int g_idx = rns_mask_get_active_index(target_mask, limb);
            assert(g_idx >= 0);
            for (size_t k = 0; k < size; k++)
            {
                tmp_poly->coeffs[g_idx][k] = in[k]->a[limb][i];
            }
        }

        uint64_t ksk_idx = 0;
        for (size_t j = 0; j < tmp_poly->ntt->l; j++)
        {
            if (tmp_poly->rns_mask & (1ULL << j))
            {
                polynomial_RNSc_mod_reduce_lifted(tmp_poly_red, tmp_poly, j);
                polynomial_RNSc_to_RNS(tmp_rns, tmp_poly_red);
                mlwe_RNS_mul_subto_by_poly(out, ksk[i][ksk_idx++], tmp_rns);
            }
        }
    }

    // body part: out->b += sum B_k X^k
    for (size_t j = 0; j < tmp_poly->ntt->l; j++)
    {
        if (tmp_poly->rns_mask & (1ULL << j))
        {
            memset(tmp_poly->coeffs[j], 0, N * sizeof(uint64_t));
        }
    }
    for (size_t limb = 0; limb < lwe_l; limb++)
    {
        int g_idx = rns_mask_get_active_index(target_mask, limb);
        assert(g_idx >= 0);
        for (size_t k = 0; k < size; k++)
        {
            tmp_poly->coeffs[g_idx][k] = in[k]->b[limb];
        }
    }

    mlwe_RNS_to_RNSc((RNSc_MLWE)out, out);
    if (divide_mask > 0)
    {
        for (size_t j = 0; j < out->r; j++)
        {
            polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)out->a[j], divide_mask);
        }
        polynomial_round_division_RNSc_wo_free((RNSc_Polynomial)out->b, divide_mask);
    }
    polynomial_add_RNSc_polynomial((RNSc_Polynomial)out->b, (RNSc_Polynomial)out->b, tmp_poly);
    mlwe_RNSc_to_RNS(out, (RNSc_MLWE)out);

    free_RNS_polynomial(tmp_poly);
    free_RNS_polynomial(tmp_poly_red);
    free_RNS_polynomial(tmp_rns);
}

void mlwe_trace(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE ***ksks, uint64_t lvl)
{
    const uint64_t log_N = (uint64_t)log2(in->a[0]->ntt->N);
    uint64_t *gens = (uint64_t *)malloc(log_N * sizeof(uint64_t));
    for (size_t i = 1; i <= log_N; i++)
        gens[i - 1] = (1ULL << (log_N - i + 1)) + 1;
    mlwe_partial_trace(out, in, gens, ksks, log_N, lvl);
    free(gens);
}

void mlwe_add_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_add_RNSc_polynomial(out->a[i], in1->a[i], in2->a[i]);
    }
    polynomial_add_RNSc_polynomial(out->b, in1->b, in2->b);
}

void mlwe_add_RNS_sample(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_add_RNS_polynomial(out->a[i], in1->a[i], in2->a[i]);
    }
    polynomial_add_RNS_polynomial(out->b, in1->b, in2->b);
}

void mlwe_add_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, RNSc_Polynomial in2)
{
    polynomial_add_RNSc_polynomial(out->b, in1->b, in2);
}

void mlwe_sub_RNSc_polynomial(RNSc_MLWE out, RNSc_MLWE in1, RNSc_Polynomial in2)
{
    polynomial_sub_RNSc_polynomial(out->b, in1->b, in2);
}

void mlwe_RNS_add_polynomial(RNS_MLWE out, RNS_MLWE in1, RNS_Polynomial in2)
{
    polynomial_add_RNS_polynomial(out->b, in1->b, in2);
}

void mlwe_RNS_sub_polynomial(RNS_MLWE out, RNS_MLWE in1, RNS_Polynomial in2)
{
    polynomial_sub_RNS_polynomial(out->b, in1->b, in2);
}

void mlwe_sub_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in1, RNSc_MLWE in2)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_sub_RNSc_polynomial(out->a[i], in1->a[i], in2->a[i]);
    }
    polynomial_sub_RNSc_polynomial(out->b, in1->b, in2->b);
}

void mlwe_RNSc_mul_by_xai(RNSc_MLWE out, RNSc_MLWE in, uint64_t a)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_RNSc_mul_by_xai(out->a[i], in->a[i], a);
    }
    polynomial_RNSc_mul_by_xai(out->b, in->b, a);
}

void mlwe_RNSc_mul_by_xai_minus1(RNSc_MLWE out, RNSc_MLWE in, uint64_t a)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_RNSc_mul_by_xai_minus1(out->a[i], in->a[i], a);
    }
    polynomial_RNSc_mul_by_xai_minus1(out->b, in->b, a);
}

void mlwe_addto_RNSc_sample(RNSc_MLWE out, RNSc_MLWE in) { mlwe_add_RNSc_sample(out, out, in); }

void mlwe_RNSc_to_RNS(RNS_MLWE out, RNSc_MLWE in)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_RNSc_to_RNS(out->a[i], in->a[i]);
    }
    polynomial_RNSc_to_RNS(out->b, in->b);
}

void mlwe_RNS_to_RNSc(RNSc_MLWE out, RNS_MLWE in)
{
    for (size_t i = 0; i < out->r; i++)
    {
        polynomial_RNS_to_RNSc(out->a[i], in->a[i]);
    }
    polynomial_RNS_to_RNSc(out->b, in->b);
}

RNS_MLWE_KS_Key mlwe_new_RNS_ks_key(RNS_MLWE_Key out_key, RNS_MLWE_Key in_key)
{
    (void)out_key;
    (void)in_key;
    assert(false); // not implemented
    return NULL;
}

void free_mlwe_RNS_ks_key(RNS_MLWE_KS_Key key)
{
    for (size_t i = 0; i < key->ell; i++)
    {
        free_mlwe_RNS_sample(key->s[i]);
    }
    free(key->s);
    free(key);
}

RNS_MLWE_KS_Key mlwe_new_RNS_automorphism_key(RNS_MLWE_Key key, uint64_t gen)
{
    (void)key;
    (void)gen;
    assert(false); // todo: reimplement to consider new ring
    return NULL;
}

void mlwe_RNSc_GHS_hybrid_keyswitch(RNSc_MLWE out, RNSc_MLWE in, RNS_MLWE **ksk, uint64_t lvl)
{
    (void)lvl;
    assert(in != out);
    const uint64_t extended_mask = ksk[0][0]->b->rns_mask;
    const uint64_t divide_mask = extended_mask & ~in->b->rns_mask;

    for (size_t i = 0; i < out->r; i++)
    {
        out->a[i]->rns_mask = extended_mask;
    }
    out->b->rns_mask = extended_mask;

    // compute -a_i^T * ksk_i
    mlwe_RNS_trivial_sample_of_zero((RNS_MLWE)out);
    for (size_t i = 0; i < in->r; i++)
    {
        gadget_mul_subto_polynomial((RNS_MLWE)out, ksk[i], in->a[i]);
    }
    // convert to RNSc
    mlwe_RNS_to_RNSc(out, (RNS_MLWE)out);
    // rescale to in's ring
    if (divide_mask > 0)
    {
        for (size_t j = 0; j < in->r; j++)
        {
            polynomial_round_division_RNSc_wo_free(out->a[j], divide_mask);
        }
        polynomial_round_division_RNSc_wo_free(out->b, divide_mask);
    }
    polynomial_add_RNSc_polynomial(out->b, out->b, in->b);
}

void mlwe_full_packing_keyswitch_scaled_rec(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE ***ksks,
                                            uint64_t lvl)
{
    if (ell == 0)
    {
        return;
    }
    const uint64_t half = 1ULL << (ell - 1);
    RNSc_MLWE *even = (RNSc_MLWE *)malloc(half * sizeof(RNSc_MLWE));
    RNSc_MLWE *odd = (RNSc_MLWE *)malloc(half * sizeof(RNSc_MLWE));
    for (size_t i = 0; i < half; i++)
    {
        even[i] = vec[2 * i];
        odd[i] = vec[2 * i + 1];
    }

    mlwe_full_packing_keyswitch_scaled_rec(even, ell - 1, ksks, lvl);
    mlwe_full_packing_keyswitch_scaled_rec(odd, ell - 1, ksks, lvl);

    RNSc_MLWE C_tilde = even[0];
    uint64_t N = vec[0]->b->ntt->N;
    uint64_t r = vec[0]->r;
    incNTT ntt = vec[0]->b->ntt;

    RNSc_MLWE tmp = mlwe_alloc_RNSc_sample(N, r, vec[0]->b->rns_mask, ntt);
    uint64_t extended_mask = ksks[ell - 1][0][0]->b->rns_mask;
    RNSc_MLWE tmp2 = mlwe_alloc_RNSc_sample(N, r, extended_mask, ntt);

    // tmp = odd[0] * X^(N>>ell)
    mlwe_RNSc_mul_by_xai(tmp, odd[0], N >> ell);

    // C_tilde = even[0] - tmp
    mlwe_sub_RNSc_sample(C_tilde, even[0], tmp);

    // tmp2 = autom(C_tilde, (1<<ell) + 1)
    uint64_t gen = (1ULL << ell) + 1;
    mlwe_automorphism_RNSc_GHS(tmp2, C_tilde, gen, ksks[ell - 1], lvl);

    // C_tilde = C_tilde + tmp2 + 2 * tmp
    mlwe_scale_RNSc_mlwe(tmp, 2);
    mlwe_addto_RNSc_sample(C_tilde, tmp2);
    mlwe_addto_RNSc_sample(C_tilde, tmp);

    free_mlwe_RNS_sample(tmp);
    free_mlwe_RNS_sample(tmp2);
    free(even);
    free(odd);
}

void mlwe_full_packing_keyswitch_scaled(RNSc_MLWE *vec, uint64_t ell, RNS_MLWE ***ksks,
                                        uint64_t lvl)
{
    if (ell == 0)
    {
        return;
    }
    mlwe_full_packing_keyswitch_scaled_rec(vec, ell, ksks, lvl);
}

void mlwe_discrete_convolution(RNS_Polynomial *out, RNS_MLWE in1, RNS_MLWE in2)
{
    uint64_t r = in1->r;
    for (size_t k = 0; k <= 2 * r; k++)
    {
        int first = 1;
        for (size_t i = 0; i <= k; i++)
        {
            if (i <= r && (k - i) <= r)
            {
                RNS_Polynomial A_i = (i < r) ? in1->a[i] : in1->b;
                RNS_Polynomial B_ki = ((k - i) < r) ? in2->a[k - i] : in2->b;
                if (first)
                {
                    polynomial_mul_RNS_polynomial(out[k], A_i, B_ki);
                    first = 0;
                }
                else
                {
                    polynomial_mul_addto_RNS_polynomial(out[k], A_i, B_ki);
                }
            }
        }
    }
}

void mlwe_multiply(RNS_MLWE out, RNS_MLWE in1, RNS_MLWE in2, RNS_MLWE **ksk)
{
    uint64_t N = in1->b->ntt->N;
    uint64_t r = in1->r;
    uint64_t mask = in1->b->rns_mask;
    incNTT ntt = in1->b->ntt;

    RNS_Polynomial *O = polynomial_new_RNS_polynomial_array(2 * r + 1, N, mask, ntt);
    mlwe_discrete_convolution(O, in1, in2);

    for (size_t j = 0; j < r; j++)
    {
        polynomial_copy_RNS_polynomial(out->a[j], O[j + r]);
    }
    polynomial_copy_RNS_polynomial(out->b, O[2 * r]);

    RNSc_Polynomial tmp_coeff = (RNSc_Polynomial)polynomial_new_RNS_polynomial(N, mask, ntt);
    for (size_t i = 0; i < r; i++)
    {
        polynomial_RNS_to_RNSc(tmp_coeff, O[i]);
        gadget_mul_subto_polynomial(out, ksk[i], tmp_coeff);
    }

    free_RNS_polynomial(tmp_coeff);
    free_RNS_polynomial_array(2 * r + 1, O);
}

void mlwe_round_division_RNSc(RNSc_MLWE out, uint64_t divide_mask)
{
    if (divide_mask > 0)
    {
        for (size_t j = 0; j < out->r; j++)
        {
            polynomial_round_division_RNSc_wo_free(out->a[j], divide_mask);
        }
        polynomial_round_division_RNSc_wo_free(out->b, divide_mask);
    }
}
