#include "arith.h"
#include "misc.h"
#include <blake3.h>

incNTT new_incomplete_ntt_list(uint64_t *primes, uint64_t split_degree, uint64_t N, uint64_t l)
{
    const uint64_t poly_size = N / split_degree;
    const uint64_t log_poly_size = (uint64_t)log2(poly_size);
    uint64_t *w_p = (uint64_t *)safe_malloc(2 * poly_size * sizeof(uint64_t));
    incNTT ntt = (incNTT)safe_malloc(sizeof(*ntt));
    ntt->N = N;
    ntt->l = l;
    ntt->ntt = new_ntt_list(primes, poly_size, l);
    ntt->split_degree = split_degree;
    ntt->w = (uint64_t **)safe_malloc(sizeof(uint64_t *) * l);
    for (size_t i = 0; i < l; i++)
    {
        ntt->w[i] = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
        uint64_t w1 = ntt->ntt[i]->root_of_unity;
        w_p[0] = w1;
        for (size_t j = 1; j < 2 * poly_size; j++)
        {
            w_p[j] = mul_modq(w_p[j - 1], w1, ntt->ntt[i]);
        }
        bit_rev(ntt->w[i], w_p, poly_size, log_poly_size + 1);
    }
    free(w_p);
    return ntt;
}

void incNTT_extend_with_primes(incNTT ntt, uint64_t *new_primes, uint64_t count)
{
    if (count == 0)
        return;
    uint64_t new_l = ntt->l + count;
    const uint64_t poly_size = ntt->N / ntt->split_degree;
    const uint64_t log_poly_size = (uint64_t)log2(poly_size);
    uint64_t *w_p = (uint64_t *)safe_malloc(2 * poly_size * sizeof(uint64_t));

    ntt->ntt = (NTT_proc *)realloc(ntt->ntt, sizeof(NTT_proc) * new_l);
    ntt->w = (uint64_t **)realloc(ntt->w, sizeof(uint64_t *) * new_l);

    for (size_t i = ntt->l; i < new_l; i++)
    {
        uint64_t prime = new_primes[i - ntt->l];
        ntt->ntt[i] = ntt_new_proc(poly_size, prime);
        ntt->w[i] = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
        uint64_t w1 = ntt->ntt[i]->root_of_unity;
        w_p[0] = w1;
        for (size_t j = 1; j < 2 * poly_size; j++)
        {
            w_p[j] = mul_modq(w_p[j - 1], w1, ntt->ntt[i]);
        }
        bit_rev(ntt->w[i], w_p, poly_size, log_poly_size + 1);
    }

    ntt->l = new_l;
    free(w_p);
}

uint64_t **incNTT_get_rou_matrix(incNTT ntt) { return ntt->w; }

RNS_Polynomial polynomial_new_RNS_polynomial(uint64_t N, uint64_t rns_mask, incNTT ntt)
{
    RNS_Polynomial res;
    res = (RNS_Polynomial)safe_malloc(sizeof(*res));
    res->coeffs = (uint64_t **)safe_malloc(sizeof(uint64_t *) * ntt->l);
    for (size_t i = 0; i < ntt->l; i++)
    {
        res->coeffs[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * ntt->N);
    }
    res->ntt = ntt;
    res->rns_mask = rns_mask;
    res->allocated_l = ntt->l;
    return res;
}

RNS_Polynomial *polynomial_new_RNS_polynomial_array(uint64_t size, uint64_t N, uint64_t rns_mask,
                                                    incNTT ntt)
{
    RNS_Polynomial *res;
    res = (RNS_Polynomial *)safe_malloc(sizeof(RNS_Polynomial) * size);
    for (size_t i = 0; i < size; i++)
    {
        res[i] = polynomial_new_RNS_polynomial(N, rns_mask, ntt);
    }
    return res;
}

bool polynomial_eq(RNS_Polynomial a, RNS_Polynomial b)
{
    const uint64_t max_l = a->ntt->l > b->ntt->l ? a->ntt->l : b->ntt->l;
    for (size_t i = 0; i < max_l; i++)
    {
        bool active_a = (i < a->ntt->l) && (a->rns_mask & (1ULL << i));
        bool active_b = (i < b->ntt->l) && (b->rns_mask & (1ULL << i));
        if (active_a && active_b)
        {
            if (memcmp(a->coeffs[i], b->coeffs[i], a->ntt->N * sizeof(uint64_t)) != 0)
            {
                return false;
            }
        }
        else if (active_a)
        {
            for (size_t j = 0; j < a->ntt->N; j++)
            {
                if (a->coeffs[i][j])
                    return false;
            }
        }
        else if (active_b)
        {
            for (size_t j = 0; j < b->ntt->N; j++)
            {
                if (b->coeffs[i][j])
                    return false;
            }
        }
    }
    return true;
}

void polynomial_copy_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            memcpy(out->coeffs[i], in->coeffs[i], sizeof(uint64_t) * out->ntt->N);
        }
    }
}

void polynomial_copy_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in)
{
    polynomial_copy_RNS_polynomial((RNS_Polynomial)out, (RNS_Polynomial)in);
}

void polynomial_RNS_zero(RNS_Polynomial p)
{
    for (size_t i = 0; i < p->ntt->l; i++)
    {
        if (p->rns_mask & (1ULL << i))
        {
            memset(p->coeffs[i], 0, sizeof(uint64_t) * p->ntt->N);
        }
    }
}

void free_RNS_polynomial(void *p)
{
    RNS_Polynomial pp = (RNS_Polynomial)p;
    for (size_t i = 0; i < pp->allocated_l; i++)
    {
        free(pp->coeffs[i]);
    }
    free(pp->coeffs);
    free(pp);
}

void free_RNS_polynomial_array(uint64_t size, RNS_Polynomial *p)
{
    for (size_t i = 0; i < size; i++)
    {
        free_RNS_polynomial(p[i]);
    }
    free(p);
}

RNS_Polynomial *polynomial_new_array_of_RNS_polynomials(uint64_t N, uint64_t rns_mask,
                                                        uint64_t size, incNTT ntt)
{
    RNS_Polynomial *res = (RNS_Polynomial *)safe_malloc(sizeof(RNS_Polynomial) * size);
    for (size_t i = 0; i < size; i++)
        res[i] = polynomial_new_RNS_polynomial(N, rns_mask, ntt);
    return res;
}

void polynomial_to_RNS(RNS_Polynomial out, IntPolynomial in)
{
    const uint64_t modMask = out->ntt->split_degree - 1,
                   poly_size = out->ntt->N / out->ntt->split_degree;
    uint64_t *temp = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            NTT_proc proc = out->ntt->ntt[i];
            mod_eltwise_reduce_signed(temp, (int64_t *)in->coeffs, out->ntt->N, proc);
            for (size_t j = 0; j < out->ntt->N; j++)
            {
                out->coeffs[i][(j & modMask) * poly_size + j / out->ntt->split_degree] = temp[j];
            }
        }
    }
    free(temp);
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void int_array_to_RNS(RNS_Polynomial out, uint64_t *in)
{
    const uint64_t modMask = out->ntt->split_degree - 1,
                   poly_size = out->ntt->N / out->ntt->split_degree;
    uint64_t *temp = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            NTT_proc proc = out->ntt->ntt[i];
            mod_eltwise_reduce_signed(temp, (int64_t *)in, out->ntt->N, proc);
            for (size_t j = 0; j < out->ntt->N; j++)
            {
                out->coeffs[i][(j & modMask) * poly_size + j / out->ntt->split_degree] = temp[j];
            }
        }
    }
    free(temp);
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void array_to_RNS(RNS_Polynomial out, uint64_t **in)
{
    const uint64_t modMask = out->ntt->split_degree - 1,
                   poly_size = out->ntt->N / out->ntt->split_degree;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t j = 0; j < out->ntt->N; j++)
            {
                out->coeffs[i][(j & modMask) * poly_size + j / out->ntt->split_degree] = in[i][j];
            }
        }
    }
    polynomial_RNSc_to_RNS(out, (RNSc_Polynomial)out);
}

void polynomial_gen_random_RNSc_polynomial(RNSc_Polynomial out)
{
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            const uint64_t p = out->ntt->ntt[i]->q;
            generate_random_bytes(sizeof(uint64_t) * out->ntt->N, (uint8_t *)out->coeffs[i]);
            array_mod_switch_from_2k(out->coeffs[i], out->coeffs[i], p, p, out->ntt->N);
        }
    }
}

void polynomial_gen_gaussian_RNSc_polynomial(RNSc_Polynomial out, double sigma)
{
    int64_t *noise_arr = (int64_t *)safe_aligned_malloc(out->ntt->N * sizeof(int64_t));
    for (size_t j = 0; j < out->ntt->N; j++)
    {
        noise_arr[j] = (int64_t)round(generate_normal_random(sigma));
    }
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            NTT_proc proc = out->ntt->ntt[i];
            mod_eltwise_reduce_signed(out->coeffs[i], noise_arr, out->ntt->N, proc);
        }
    }
    free(noise_arr);
}

void polynomial_multo_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in)
{
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = out->rns_mask & in->rns_mask;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
    uint64_t *tmp2 = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            memcpy(tmp2, out->coeffs[i], sizeof(uint64_t) * out->ntt->N);
            memset(out->coeffs[i], 0, sizeof(uint64_t) * out->ntt->N);
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                for (size_t k = 0; k < out->ntt->split_degree - j; k++)
                {
                    mod_eltwise_mul(tmp, &in->coeffs[i][j * poly_size], &tmp2[k * poly_size],
                                    poly_size, out->ntt->ntt[i]);
                    mod_eltwise_add(&out->coeffs[i][(j + k) * poly_size],
                                    &out->coeffs[i][(j + k) * poly_size], tmp, poly_size,
                                    out->ntt->ntt[i]);
                }
                for (size_t k = out->ntt->split_degree - j; k < out->ntt->split_degree; k++)
                {
                    mod_eltwise_mul(tmp, &in->coeffs[i][j * poly_size], &tmp2[k * poly_size],
                                    poly_size, out->ntt->ntt[i]);
                    mod_eltwise_mul(tmp, tmp, out->ntt->w[i], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_add(&out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    &out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    tmp, poly_size, out->ntt->ntt[i]);
                }
            }
        }
    }
    free(tmp);
    free(tmp2);
}

void polynomial_mul_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    assert(out != in1);
    assert(out != in2);
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t));
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            memset(out->coeffs[i], 0, out->ntt->N * sizeof(uint64_t));
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                for (size_t k = 0; k < out->ntt->split_degree - j; k++)
                {
                    if (j == 0)
                    {
                        mod_eltwise_mul(
                            &out->coeffs[i][(j + k) * poly_size], &in1->coeffs[i][j * poly_size],
                            &in2->coeffs[i][k * poly_size], poly_size, out->ntt->ntt[i]);
                    }
                    else
                    {
                        mod_eltwise_mul(tmp, &in1->coeffs[i][j * poly_size],
                                        &in2->coeffs[i][k * poly_size], poly_size,
                                        out->ntt->ntt[i]);
                        mod_eltwise_add(&out->coeffs[i][(j + k) * poly_size],
                                        &out->coeffs[i][(j + k) * poly_size], tmp, poly_size,
                                        out->ntt->ntt[i]);
                    }
                }
                for (size_t k = out->ntt->split_degree - j; k < out->ntt->split_degree; k++)
                {
                    mod_eltwise_mul(tmp, &in1->coeffs[i][j * poly_size],
                                    &in2->coeffs[i][k * poly_size], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_mul(tmp, tmp, out->ntt->w[i], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_add(&out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    &out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    tmp, poly_size, out->ntt->ntt[i]);
                }
            }
        }
    }
    free(tmp);
}

void polynomial_mul_addto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    assert(out != in1);
    assert(out != in2);
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    uint64_t mask = out->rns_mask;
    // tmp is only needed for the twiddle (cross-block) term; with split_degree==1 it is unused.
    uint64_t *tmp = (out->ntt->split_degree > 1)
                        ? (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t))
                        : NULL;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (mask & (1ULL << i))
        {
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                for (size_t k = 0; k < out->ntt->split_degree - j; k++)
                {
                    // fused: out += in1*in2 in one pass (no temp, no separate add)
                    mod_eltwise_mul_addto(
                        &out->coeffs[i][(j + k) * poly_size], &in1->coeffs[i][j * poly_size],
                        &in2->coeffs[i][k * poly_size], poly_size, out->ntt->ntt[i]);
                }
                for (size_t k = out->ntt->split_degree - j; k < out->ntt->split_degree; k++)
                {
                    mod_eltwise_mul(tmp, &in1->coeffs[i][j * poly_size],
                                    &in2->coeffs[i][k * poly_size], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_mul(tmp, tmp, out->ntt->w[i], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_add(&out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    &out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    tmp, poly_size, out->ntt->ntt[i]);
                }
            }
        }
    }
    if (tmp)
        free(tmp);
}

void polynomial_mul_subto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    assert(out != in1);
    assert(out != in2);
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    uint64_t mask = out->rns_mask;
    uint64_t *tmp = (out->ntt->split_degree > 1)
                        ? (uint64_t *)safe_aligned_malloc(poly_size * sizeof(uint64_t))
                        : NULL;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (mask & (1ULL << i))
        {
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                for (size_t k = 0; k < out->ntt->split_degree - j; k++)
                {
                    // fused: out -= in1*in2 in one pass
                    mod_eltwise_mul_subto(
                        &out->coeffs[i][(j + k) * poly_size], &in1->coeffs[i][j * poly_size],
                        &in2->coeffs[i][k * poly_size], poly_size, out->ntt->ntt[i]);
                }
                for (size_t k = out->ntt->split_degree - j; k < out->ntt->split_degree; k++)
                {
                    mod_eltwise_mul(tmp, &in1->coeffs[i][j * poly_size],
                                    &in2->coeffs[i][k * poly_size], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_mul(tmp, tmp, out->ntt->w[i], poly_size, out->ntt->ntt[i]);
                    mod_eltwise_sub(&out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    &out->coeffs[i][(j + k - out->ntt->split_degree) * poly_size],
                                    tmp, poly_size, out->ntt->ntt[i]);
                }
            }
        }
    }
    if (tmp)
        free(tmp);
}

void polynomial_sub_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            mod_eltwise_sub(out->coeffs[i], in1->coeffs[i], in2->coeffs[i], out->ntt->N,
                            out->ntt->ntt[i]);
        }
    }
}

void polynomial_sub_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, RNSc_Polynomial in2)
{
    polynomial_sub_RNS_polynomial((RNS_Polynomial)out, (RNS_Polynomial)in1, (RNS_Polynomial)in2);
}

void polynomial_add_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, RNSc_Polynomial in2)
{
    out->rns_mask = in1->rns_mask & in2->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            mod_eltwise_add(out->coeffs[i], in1->coeffs[i], in2->coeffs[i], out->ntt->N,
                            out->ntt->ntt[i]);
        }
    }
}

void polynomial_add_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, RNS_Polynomial in2)
{
    polynomial_add_RNSc_polynomial((RNSc_Polynomial)out, (RNSc_Polynomial)in1,
                                   (RNSc_Polynomial)in2);
}

void polynomial_RNSc_add_integer(RNSc_Polynomial out, RNSc_Polynomial in1, uint64_t in2)
{
    out->rns_mask = in1->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (out != in1)
                memcpy(out->coeffs[i], in1->coeffs[i], out->ntt->N * sizeof(uint64_t));
            const uint64_t q = out->ntt->ntt[i]->q;
            const uint64_t in_mod_q = in2 & (1ULL << 63) ? q - ((-in2) % q) : in2 % q;
            out->coeffs[i][0] = (out->coeffs[i][0] + in_mod_q) % q;
        }
    }
}

void polynomial_RNS_add_integer(RNS_Polynomial out, RNS_Polynomial in1, uint64_t in2)
{
    out->rns_mask = in1->rns_mask;
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            NTT_proc proc = out->ntt->ntt[i];
            if (out != in1)
                memcpy(out->coeffs[i], in1->coeffs[i], out->ntt->N * sizeof(uint64_t));
            const uint64_t q = proc->q;
            uint64_t in_mod_q;
            if (in2 & (1ULL << 63))
            {
                in_mod_q = negate_modq(modq(-in2, proc), q);
            }
            else
            {
                in_mod_q = modq(in2, proc);
            }
            mod_eltwise_add_scalar(out->coeffs[i], out->coeffs[i], in_mod_q, poly_size, proc);
        }
    }
}

void polynomial_scale_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1, uint64_t scale)
{
    out->rns_mask = in1->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            mod_eltwise_scale(out->coeffs[i], in1->coeffs[i], scale, out->ntt->N, out->ntt->ntt[i]);
        }
    }
}

void polynomial_scale_addto_RNSc_polynomial(RNSc_Polynomial out, RNSc_Polynomial in1,
                                            uint64_t scale)
{
    out->rns_mask = in1->rns_mask;
    uint64_t mask = out->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (mask & (1ULL << i))
        {
            mod_eltwise_fma(out->coeffs[i], in1->coeffs[i], scale, out->ntt->N, out->ntt->ntt[i]);
        }
    }
}

void polynomial_scale_addto_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, uint64_t scale)
{
    polynomial_scale_addto_RNSc_polynomial((RNSc_Polynomial)out, (RNSc_Polynomial)in1, scale);
}

void polynomial_scale_RNS_polynomial(RNS_Polynomial out, RNS_Polynomial in1, uint64_t scale)
{
    polynomial_scale_RNSc_polynomial((RNSc_Polynomial)out, (RNSc_Polynomial)in1, scale);
}

void polynomial_scale_RNS_polynomial_RNS(RNS_Polynomial out, RNS_Polynomial in1, uint64_t *scale)
{
    out->rns_mask = in1->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            mod_eltwise_scale(out->coeffs[i], in1->coeffs[i], scale[i], out->ntt->N,
                              out->ntt->ntt[i]);
        }
    }
}

void polynomial_RNSc_negate(RNSc_Polynomial out, RNSc_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            mod_eltwise_negate(out->coeffs[i], in->coeffs[i], out->ntt->N, out->ntt->ntt[i]);
        }
    }
}

void polynomial_RNS_negate(RNS_Polynomial out, RNS_Polynomial in)
{
    polynomial_RNSc_negate((RNSc_Polynomial)out, (RNSc_Polynomial)in);
}

void polynomial_RNSc_to_RNS(RNS_Polynomial out, RNSc_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t k = 0; k < out->ntt->split_degree; k++)
            {
                ntt_forward(&out->coeffs[i][k * poly_size], &in->coeffs[i][k * poly_size],
                            out->ntt->ntt[i]);
            }
        }
    }
}

void polynomial_RNS_to_RNSc(RNSc_Polynomial out, RNS_Polynomial in)
{
    out->rns_mask = in->rns_mask;
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t k = 0; k < out->ntt->split_degree; k++)
            {
                ntt_reverse(&out->coeffs[i][k * poly_size], &in->coeffs[i][k * poly_size],
                            out->ntt->ntt[i]);
            }
        }
    }
}

void polynomial_RNSc_add_noise(RNSc_Polynomial out, RNSc_Polynomial in, double sigma)
{
    int64_t *noise_arr = (int64_t *)safe_aligned_malloc(out->ntt->N * sizeof(int64_t));
    for (size_t j = 0; j < out->ntt->N; j++)
    {
        noise_arr[j] = (int64_t)round(generate_normal_random(sigma));
    }
    uint64_t *noise_reduced = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            NTT_proc proc = out->ntt->ntt[i];
            mod_eltwise_reduce_signed(noise_reduced, noise_arr, out->ntt->N, proc);
            mod_eltwise_add(out->coeffs[i], in->coeffs[i], noise_reduced, out->ntt->N, proc);
        }
    }
    free(noise_reduced);
    free(noise_arr);
}

RNS_BaseConversionParams init_base_conversion_params(incNTT ntt, uint64_t in_mask,
                                                     uint64_t out_mask)
{
    RNS_BaseConversionParams params = (RNS_BaseConversionParams)safe_malloc(sizeof(*params));
    params->in_mask = in_mask;
    params->out_mask = out_mask;

    params->D = (uint32_t *)safe_malloc(sizeof(uint32_t) * ntt->l);
    params->P = (uint32_t *)safe_malloc(sizeof(uint32_t) * ntt->l);
    params->w = 0;
    params->v = 0;

    for (size_t i = 0; i < ntt->l; i++)
    {
        if (in_mask & (1ULL << i))
        {
            params->D[params->w++] = (uint32_t)i;
        }
    }

    for (size_t i = 0; i < ntt->l; i++)
    {
        if ((out_mask & (1ULL << i)) && !(in_mask & (1ULL << i)))
        {
            params->P[params->v++] = (uint32_t)i;
        }
    }

    if (params->v == 0)
    {
        params->Dhat = NULL;
        params->D_mod_p = NULL;
        return params;
    }

    assert(params->w > 0);

    params->Dhat = (uint64_t *)safe_malloc(sizeof(uint64_t) * params->w);
    for (size_t j = 0; j < params->w; j++)
    {
        uint64_t idx_j = params->D[j];
        NTT_proc proc_j = ntt->ntt[idx_j];
        uint64_t q_j = proc_j->q;
        uint64_t prod = 1;
        for (size_t k = 0; k < params->w; k++)
        {
            if (k == j)
                continue;
            uint64_t idx_k = params->D[k];
            uint64_t q_k = ntt->ntt[idx_k]->q;
            prod = mul_modq(prod, modq(q_k, proc_j), proc_j);
        }
        params->Dhat[j] = inverse_mod(prod, q_j);
    }

    params->D_mod_p = (uint64_t **)safe_malloc(sizeof(uint64_t *) * params->v);
    for (size_t i = 0; i < params->v; i++)
    {
        params->D_mod_p[i] = (uint64_t *)safe_malloc(sizeof(uint64_t) * params->w);
        uint64_t idx_i = params->P[i];
        NTT_proc proc_i = ntt->ntt[idx_i];
        for (size_t j = 0; j < params->w; j++)
        {
            uint64_t prod = 1;
            for (size_t k = 0; k < params->w; k++)
            {
                if (k == j)
                    continue;
                uint64_t idx_k = params->D[k];
                uint64_t q_k = ntt->ntt[idx_k]->q;
                prod = mul_modq(prod, modq(q_k, proc_i), proc_i);
            }
            params->D_mod_p[i][j] = prod;
        }
    }

    return params;
}

void free_base_conversion_params(RNS_BaseConversionParams params)
{
    if (params == NULL)
        return;
    if (params->D_mod_p != NULL)
    {
        for (size_t i = 0; i < params->v; i++)
        {
            free(params->D_mod_p[i]);
        }
        free(params->D_mod_p);
    }
    if (params->Dhat != NULL)
    {
        free(params->Dhat);
    }
    free(params->D);
    free(params->P);
    free(params);
}

void polynomial_base_conversion_RNSc(RNSc_Polynomial out, RNSc_Polynomial in,
                                     RNS_BaseConversionParams params)
{
    assert(in->ntt->l == out->ntt->l);
    assert(in->ntt->N == out->ntt->N);

    uint64_t in_mask = in->rns_mask;
    uint64_t out_mask = out->rns_mask;

    // 1. Copy matching active RNS components from in to out (if not in-place)
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if ((in_mask & (1ULL << i)) && (out_mask & (1ULL << i)))
        {
            if (out != in)
            {
                memcpy(out->coeffs[i], in->coeffs[i], sizeof(uint64_t) * out->ntt->N);
            }
        }
    }

    RNS_BaseConversionParams local_params = params;
    if (local_params == NULL)
    {
        local_params = init_base_conversion_params(in->ntt, in_mask, out_mask);
    }
    else
    {
        assert(local_params->in_mask == in_mask);
        assert(local_params->out_mask == out_mask);
    }

    uint32_t w = local_params->w;
    uint32_t v = local_params->v;
    uint32_t *D = local_params->D;
    uint32_t *P = local_params->P;
    uint64_t *Dhat = local_params->Dhat;
    uint64_t **D_mod_p = local_params->D_mod_p;

    if (v == 0)
    {
        out->rns_mask = out_mask;
        if (params == NULL)
        {
            free_base_conversion_params(local_params);
        }
        return;
    }

    // 2. Initialize the output target polynomial coefficients to zero
    for (size_t i = 0; i < v; i++)
    {
        uint64_t idx_i = P[i];
        memset(out->coeffs[idx_i], 0, sizeof(uint64_t) * out->ntt->N);
    }

    // 3. Perform Fast Base Extension
    uint64_t *v_tmp = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));
    uint64_t *v_tmp2 = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));

    for (size_t j = 0; j < w; j++)
    {
        uint64_t idx_j = D[j];
        NTT_proc proc_j = in->ntt->ntt[idx_j];
        mod_eltwise_scale(v_tmp, in->coeffs[idx_j], Dhat[j], out->ntt->N, proc_j);

        for (size_t i = 0; i < v; i++)
        {
            uint64_t idx_i = P[i];
            NTT_proc proc_i = out->ntt->ntt[idx_i];
            mod_eltwise_reduce(v_tmp2, v_tmp, out->ntt->N, proc_i);
            mod_eltwise_fma(out->coeffs[idx_i], v_tmp2, D_mod_p[i][j], out->ntt->N, proc_i);
        }
    }

    free(v_tmp);
    free(v_tmp2);

    if (params == NULL)
    {
        free_base_conversion_params(local_params);
    }

    out->rns_mask = out_mask;
}

void polynomial_RNSc_mod_reduce_lifted(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t idx)
{
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            if (i == idx)
            {
                // in->coeffs[idx] is the residue mod p_idx (already < p_idx); reducing mod
                // p_i (== p_idx) is the identity, so just copy instead of running Barrett.
                if (out->coeffs[i] != in->coeffs[idx])
                    memcpy(out->coeffs[i], in->coeffs[idx], out->ntt->N * sizeof(uint64_t));
            }
            else
            {
                mod_eltwise_reduce(out->coeffs[i], in->coeffs[idx], out->ntt->N, out->ntt->ntt[i]);
            }
        }
    }
}

void polynomial_RNSc_mod_reduce(RNSc_Polynomial out, RNSc_Polynomial in)
{
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            memcpy(out->coeffs[i], in->coeffs[i], out->ntt->N * sizeof(uint64_t));
        }
    }
}

void polynomial_RNSc_decompose_small(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t log_base,
                                     uint64_t level)
{
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(out->ntt->N * sizeof(uint64_t));
    const uint64_t mask = (1ULL << log_base) - 1;
    const uint64_t shift = log_base * level;
    int last_active = rns_mask_get_last_active_index(in->rns_mask);
    assert(last_active >= 0);
    for (size_t i = 0; i < out->ntt->N; i++)
    {
        tmp[i] = (in->coeffs[last_active][i] >> shift) & mask;
    }
    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            memcpy(out->coeffs[i], tmp, sizeof(uint64_t) * out->ntt->N);
        }
    }
    free(tmp);
}

void polynomial_RNSc_to_multiprecision(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t log_base,
                                       uint64_t level)
{
    polynomial_RNSc_decompose_small(out, in, log_base, level);
}

void polynomial_RNS_get_hash(uint64_t *out, RNS_Polynomial p)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    for (size_t i = 0; i < p->ntt->l; i++)
    {
        if (p->rns_mask & (1ULL << i))
        {
            blake3_hasher_update(&hasher, p->coeffs[i], p->ntt->N * sizeof(uint64_t));
        }
    }
    blake3_hasher_finalize(&hasher, (uint8_t *)out, BLAKE3_OUT_LEN);
}

uint64_t *polynomial_RNS_get_hash_p(RNS_Polynomial p)
{
    uint64_t *out = (uint64_t *)safe_malloc(4 * sizeof(uint64_t));
    polynomial_RNS_get_hash(out, p);
    return out;
}

void polynomial_floor_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask)
{
    uint64_t mask = divide_mask & out->rns_mask;
    if (mask == 0)
        return;

    const uint64_t N = out->ntt->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    for (size_t idx = 0; idx < out->ntt->l; idx++)
    {
        if (mask & (1ULL << idx))
        {
            const uint64_t p = out->ntt->ntt[idx]->q;
            for (size_t i = 0; i < out->ntt->l; i++)
            {
                if (out->rns_mask & (1ULL << i))
                {
                    if (i == idx)
                        continue;
                    const uint64_t q = out->ntt->ntt[i]->q;
                    const uint64_t inv_p = inverse_mod(p, q);
                    mod_eltwise_reduce(tmp, out->coeffs[idx], N, out->ntt->ntt[i]);
                    mod_eltwise_sub(out->coeffs[i], out->coeffs[i], tmp, N, out->ntt->ntt[i]);
                    mod_eltwise_scale(out->coeffs[i], out->coeffs[i], inv_p, N, out->ntt->ntt[i]);
                }
            }
            memset(out->coeffs[idx], 0, sizeof(uint64_t) * N);
            out->rns_mask &= ~(1ULL << idx);
        }
    }
    free(tmp);
}

void polynomial_round_division_RNSc_wo_free(RNSc_Polynomial out, uint64_t divide_mask)
{
    uint64_t mask = divide_mask & out->rns_mask;
    if (mask == 0)
        return;

    const uint64_t N = out->ntt->N;
    uint64_t *tmp = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    for (size_t idx = 0; idx < out->ntt->l; idx++)
    {
        if (mask & (1ULL << idx))
        {
            const uint64_t p = out->ntt->ntt[idx]->q, half_p = p / 2;
            mod_eltwise_add_scalar(out->coeffs[idx], out->coeffs[idx], half_p, N,
                                   out->ntt->ntt[idx]);
            for (size_t i = 0; i < out->ntt->l; i++)
            {
                if (out->rns_mask & (1ULL << i))
                {
                    if (i == idx)
                        continue;
                    const uint64_t q = out->ntt->ntt[i]->q;
                    const uint64_t inv_p = inverse_mod(p, q);
                    const uint64_t half_p_mod_q = half_p % q;
                    mod_eltwise_reduce(tmp, out->coeffs[idx], N, out->ntt->ntt[i]);
                    mod_eltwise_add_scalar(out->coeffs[i], out->coeffs[i], half_p_mod_q, N,
                                           out->ntt->ntt[i]);
                    mod_eltwise_sub(out->coeffs[i], out->coeffs[i], tmp, N, out->ntt->ntt[i]);
                    mod_eltwise_scale(out->coeffs[i], out->coeffs[i], inv_p, N, out->ntt->ntt[i]);
                }
            }
            memset(out->coeffs[idx], 0, sizeof(uint64_t) * N);
            out->rns_mask &= ~(1ULL << idx);
        }
    }
    free(tmp);
}

void polynomial_floor_division_RNSc(RNSc_Polynomial out)
{
    int last_active = rns_mask_get_last_active_index(out->rns_mask);
    if (last_active >= 0)
    {
        polynomial_floor_division_RNSc_wo_free(out, 1ULL << last_active);
    }
}

void polynomial_round_division_RNSc(RNSc_Polynomial out)
{
    int last_active = rns_mask_get_last_active_index(out->rns_mask);
    if (last_active >= 0)
    {
        polynomial_round_division_RNSc_wo_free(out, 1ULL << last_active);
    }
}

void polynomial_RNSc_permute(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t gen)
{
    assert(out != in);
    const uint64_t N = out->ntt->N, mod_mask = N - 1, split_degree = out->ntt->split_degree,
                   split_degree_mod = split_degree - 1;
    int split_degree_log = 0;
    while ((1ULL << split_degree_log) < split_degree)
        split_degree_log++;
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    assert(gen < 2 * N);
    assert(gen > 0);
    polynomial_RNS_zero((RNS_Polynomial)out);

    int64_t *temp_signed = (int64_t *)safe_aligned_malloc(N * sizeof(int64_t));
    out->rns_mask = in->rns_mask;
    for (size_t j = 0; j < out->ntt->l; j++)
    {
        if (out->rns_mask & (1ULL << j))
        {
            NTT_proc proc = out->ntt->ntt[j];
            for (size_t i = 0; i < split_degree; i++)
            {
                for (size_t i2 = 0; i2 < poly_size; i2++)
                {
                    const uint64_t idx = ((i + (i2 << split_degree_log)) * gen);
                    const uint64_t dst = (idx & split_degree_mod) * poly_size +
                                         ((idx & mod_mask) >> split_degree_log);
                    int64_t val = (int64_t)in->coeffs[j][i * poly_size + i2];
                    temp_signed[dst] = (idx & N) ? -val : val;
                }
            }
            mod_eltwise_reduce_signed(out->coeffs[j], temp_signed, N, proc);
        }
        else
        {
            memset(out->coeffs[j], 0, sizeof(uint64_t) * N);
        }
    }
    free(temp_signed);
}

void polynomial_int_permute_mod_Q(IntPolynomial out, IntPolynomial in, uint64_t gen)
{
    const uint64_t N = in->N;
    uint64_t idx = 0;
    for (size_t i = 0; i < N; i++)
    {
        out->coeffs[idx] = in->coeffs[i];
        idx = (idx + gen) % N;
    }
}

void polynomial_RNSc_mul_by_xai(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t a)
{
    assert(in != out);
    assert(out->ntt->split_degree == 1);
    const uint64_t N = out->ntt->N;
    a &= ((N << 1) - 1);
    if (a == 0)
    {
        polynomial_copy_RNSc_polynomial(out, in);
        return;
    }
    out->rns_mask = in->rns_mask;
    for (size_t j = 0; j < out->ntt->l; j++)
    {
        if (out->rns_mask & (1ULL << j))
        {
            NTT_proc proc = in->ntt->ntt[j];
            uint64_t q = proc->q;
            if (a < N)
            {
                if (a % 8 == 0)
                {
                    mod_eltwise_negate(out->coeffs[j], in->coeffs[j] + N - a, a, proc);
                    memcpy(out->coeffs[j] + a, in->coeffs[j], (N - a) * sizeof(uint64_t));
                }
                else
                {
                    for (size_t i = 0; i < a; i++)
                    {
                        out->coeffs[j][i] = negate_modq(in->coeffs[j][i - a + N], q);
                    }
                    for (size_t i = a; i < N; i++)
                    {
                        out->coeffs[j][i] = in->coeffs[j][i - a];
                    }
                }
            }
            else
            {
                if (a % 8 == 0)
                {
                    memcpy(out->coeffs[j], in->coeffs[j] + 2 * N - a, (a - N) * sizeof(uint64_t));
                    mod_eltwise_negate(out->coeffs[j] + a - N, in->coeffs[j], 2 * N - a, proc);
                }
                else
                {
                    for (size_t i = 0; i < a - N; i++)
                    {
                        out->coeffs[j][i] = in->coeffs[j][i - a + 2 * N];
                    }
                    for (size_t i = a - N; i < N; i++)
                    {
                        out->coeffs[j][i] = negate_modq(in->coeffs[j][i - a + N], q);
                    }
                }
            }
        }
    }
}

void polynomial_RNSc_mul_by_xai_minus1(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t a)
{
    assert(in != out);
    assert(out->ntt->split_degree == 1);
    const uint64_t N = out->ntt->N;
    a &= ((N << 1) - 1);
    if (a == 0)
    {
        for (size_t j = 0; j < out->ntt->l; j++)
        {
            memset(out->coeffs[j], 0, sizeof(uint64_t) * N);
        }
        out->rns_mask = in->rns_mask;
        return;
    }
    out->rns_mask = in->rns_mask;
    for (size_t j = 0; j < out->ntt->l; j++)
    {
        if (out->rns_mask & (1ULL << j))
        {
            NTT_proc proc = in->ntt->ntt[j];
            uint64_t q = proc->q;
            if (a < N)
            {
                if (a % 8 == 0)
                {
                    mod_eltwise_negate(out->coeffs[j], in->coeffs[j] + N - a, a, proc);
                    mod_eltwise_sub(out->coeffs[j], out->coeffs[j], in->coeffs[j], a, proc);
                    mod_eltwise_sub(out->coeffs[j] + a, in->coeffs[j], in->coeffs[j] + a, N - a,
                                    proc);
                }
                else
                {
                    for (size_t i = 0; i < a; i++)
                    {
                        uint64_t term1 = negate_modq(in->coeffs[j][i - a + N], q);
                        out->coeffs[j][i] = sub_modq(term1, in->coeffs[j][i], q);
                    }
                    for (size_t i = a; i < N; i++)
                    {
                        out->coeffs[j][i] = sub_modq(in->coeffs[j][i - a], in->coeffs[j][i], q);
                    }
                }
            }
            else
            {
                if (a % 8 == 0)
                {
                    mod_eltwise_sub(out->coeffs[j], in->coeffs[j] + 2 * N - a, in->coeffs[j], a - N,
                                    proc);
                    mod_eltwise_negate(out->coeffs[j] + a - N, in->coeffs[j], 2 * N - a, proc);
                    mod_eltwise_sub(out->coeffs[j] + a - N, out->coeffs[j] + a - N,
                                    in->coeffs[j] + a - N, 2 * N - a, proc);
                }
                else
                {
                    for (size_t i = 0; i < a - N; i++)
                    {
                        out->coeffs[j][i] =
                            sub_modq(in->coeffs[j][i - a + 2 * N], in->coeffs[j][i], q);
                    }
                    for (size_t i = a - N; i < N; i++)
                    {
                        uint64_t term1 = negate_modq(in->coeffs[j][i - a + N], q);
                        out->coeffs[j][i] = sub_modq(term1, in->coeffs[j][i], q);
                    }
                }
            }
        }
    }
}

void polynomial_int_decompose_i(IntPolynomial out, IntPolynomial in, uint64_t Bg_bit, uint64_t l,
                                uint64_t q, uint64_t bit_size, uint64_t i)
{
    const uint64_t N = in->N;
    const uint64_t h_mask = (1UL << Bg_bit) - 1;
    const uint64_t h_bit = bit_size - (i + 1) * Bg_bit;
    uint64_t offset = 1ULL << (bit_size - l * Bg_bit - 1);
    for (size_t c = 0; c < N; c++)
    {
        const uint64_t coeff_off = in->coeffs[c] + offset;
        out->coeffs[c] = (coeff_off >> h_bit) & h_mask;
    }
}

IntPolynomial polynomial_new_int_polynomial(uint64_t N)
{
    IntPolynomial res;
    res = (IntPolynomial)safe_malloc(sizeof(*res));
    res->coeffs = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * N);
    res->N = N;
    return res;
}

IntPolynomial *polynomial_new_int_polynomial_array(uint64_t size, uint64_t N)
{
    IntPolynomial *res = (IntPolynomial *)safe_malloc(sizeof(IntPolynomial) * size);
    for (size_t i = 0; i < size; i++)
    {
        res[i] = polynomial_new_int_polynomial(N);
    }
    return res;
}

void free_polynomial(void *p)
{
    free(((IntPolynomial)p)->coeffs);
    free(p);
}

void free_polynomial_array(uint64_t size, IntPolynomial *p)
{
    for (size_t i = 0; i < size; i++)
    {
        free_polynomial(p[i]);
    }
    free(p);
}

void polynomial_RNS_broadcast_slot(RNS_Polynomial out, RNS_Polynomial in, uint64_t slot_idx)
{
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                for (size_t k = 0; k < poly_size; k++)
                {
                    out->coeffs[i][j * poly_size + k] = in->coeffs[i][j * poly_size + slot_idx];
                }
            }
        }
    }
}

void polynomial_RNS_broadcast_RNS_comp(RNS_Polynomial out, RNS_Polynomial in, uint64_t rns_comp)
{
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t k = 0; k < out->ntt->N; k++)
            {
                out->coeffs[i][k] = in->coeffs[rns_comp][k];
            }
        }
    }
}

void polynomial_RNS_rotate_slot(RNS_Polynomial out, RNS_Polynomial in, uint64_t rot)
{
    assert(out != in);
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                for (size_t k = 0; k < poly_size - rot; k++)
                {
                    out->coeffs[i][j * poly_size + k] = in->coeffs[i][j * poly_size + k + rot];
                }
                for (size_t k = poly_size - rot; k < poly_size; k++)
                {
                    out->coeffs[i][j * poly_size + k] =
                        in->coeffs[i][j * poly_size + k + rot - poly_size];
                }
            }
        }
    }
}

void polynomial_RNS_copy_slot(RNS_Polynomial out, uint64_t dst, RNS_Polynomial in, uint64_t src)
{
    const uint64_t poly_size = out->ntt->N / out->ntt->split_degree;
    out->rns_mask = in->rns_mask;
    for (size_t i = 0; i < in->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            for (size_t j = 0; j < out->ntt->split_degree; j++)
            {
                out->coeffs[i][j * poly_size + dst] = in->coeffs[i][j * poly_size + src];
            }
        }
    }
}

// Extended-Euclidean modular inverse (adapted from mersenneforum / uecm_modinv_64).
// Requires 0 < a < p and gcd(a, p) == 1. p must be odd.
static inline uint64_t modinv_ext_euclid_64(uint64_t a, uint64_t p)
{
    uint64_t ps1, ps2, parity, dividend, divisor, rem, q, t;

    q = 1;
    rem = a;
    dividend = p;
    divisor = a;
    ps1 = 1;
    ps2 = 0;
    parity = 0;

    while (divisor > 1)
    {
        rem = dividend - divisor;
        t = rem - divisor;
        if (rem >= divisor)
        {
            q += ps1;
            rem = t;
            t -= divisor;
            if (rem >= divisor)
            {
                q += ps1;
                rem = t;
                t -= divisor;
                if (rem >= divisor)
                {
                    q += ps1;
                    rem = t;
                    t -= divisor;
                    if (rem >= divisor)
                    {
                        q += ps1;
                        rem = t;
                        t -= divisor;
                        if (rem >= divisor)
                        {
                            q += ps1;
                            rem = t;
                            t -= divisor;
                            if (rem >= divisor)
                            {
                                q += ps1;
                                rem = t;
                                t -= divisor;
                                if (rem >= divisor)
                                {
                                    q += ps1;
                                    rem = t;
                                    t -= divisor;
                                    if (rem >= divisor)
                                    {
                                        q += ps1;
                                        rem = t;
                                        if (rem >= divisor)
                                        {
                                            q = dividend / divisor;
                                            rem = dividend % divisor;
                                            q *= ps1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        q += ps2;
        parity = ~parity;
        dividend = divisor;
        divisor = rem;
        ps2 = ps1;
        ps1 = q;
    }

    return (parity == 0) ? ps1 : p - ps1;
}

int polynomial_RNS_inverse(RNS_Polynomial out, RNS_Polynomial in)
{
    if (in->ntt->split_degree != 1)
        return -1;
    const uint64_t N = in->ntt->N;
    out->rns_mask = in->rns_mask;
    assert(out->ntt->N == N);

    uint64_t *prefix = (uint64_t *)safe_aligned_malloc(N * sizeof(uint64_t));

    for (size_t i = 0; i < in->ntt->l; i++)
    {
        if (in->rns_mask & (1ULL << i))
        {
            const uint64_t q = in->ntt->ntt[i]->q;
            NTT_proc proc = in->ntt->ntt[i];
            const uint64_t *a = in->coeffs[i];

            if (a[0] == 0)
            {
                free(prefix);
                return -2;
            }
            prefix[0] = a[0];
            for (size_t k = 1; k < N; k++)
            {
                if (a[k] == 0)
                {
                    free(prefix);
                    return -2;
                }
                prefix[k] = mul_modq(prefix[k - 1], a[k], proc);
            }

            uint64_t t = modinv_ext_euclid_64(prefix[N - 1], q);

            uint64_t *o = out->coeffs[i];
            for (size_t k = N - 1; k > 0; k--)
            {
                o[k] = mul_modq(t, prefix[k - 1], proc);
                t = mul_modq(t, a[k], proc);
            }
            o[0] = t;
        }
    }

    free(prefix);
    return 0;
}

int rns_mask_get_active_index(uint64_t mask, uint64_t i)
{
    uint64_t count = 0;
    for (int idx = 0; idx < 64; idx++)
    {
        if (mask & (1ULL << idx))
        {
            if (count == i)
            {
                return idx;
            }
            count++;
        }
    }
    return -1;
}

int rns_mask_get_last_active_index(uint64_t mask)
{
    for (int idx = 63; idx >= 0; idx--)
    {
        if (mask & (1ULL << idx))
        {
            return idx;
        }
    }
    return -1;
}

void rns_compute_scaling_factors(uint64_t *delta_out, incNTT ntt, uint64_t in_mask,
                                 uint64_t out_mask)
{
    uint64_t diff_mask = out_mask & ~in_mask;
    for (size_t i = 0; i < ntt->l; i++)
    {
        if (out_mask & (1ULL << i))
        {
            NTT_proc proc_i = ntt->ntt[i];
            if (in_mask & (1ULL << i))
            {
                uint64_t delta_i = 1;
                for (size_t j = 0; j < ntt->l; j++)
                {
                    if (diff_mask & (1ULL << j))
                    {
                        uint64_t p_j = ntt->ntt[j]->q;
                        delta_i = mul_modq(delta_i, modq(p_j, proc_i), proc_i);
                    }
                }
                delta_out[i] = delta_i;
            }
            else
            {
                delta_out[i] = 0;
            }
        }
        else
        {
            delta_out[i] = 0;
        }
    }
}

void polynomial_RNSc_scaled_lift(RNSc_Polynomial out, RNSc_Polynomial in, uint64_t *delta)
{
    assert(in->ntt->N == out->ntt->N);
    const uint64_t N = out->ntt->N;

    uint64_t local_delta[64];
    uint64_t *actual_delta = delta;
    if (actual_delta == NULL)
    {
        rns_compute_scaling_factors(local_delta, out->ntt, in->rns_mask, out->rns_mask);
        actual_delta = local_delta;
    }

    for (size_t i = 0; i < out->ntt->l; i++)
    {
        if (out->rns_mask & (1ULL << i))
        {
            NTT_proc proc_i = out->ntt->ntt[i];
            if (in->rns_mask & (1ULL << i))
            {
                mod_eltwise_scale(out->coeffs[i], in->coeffs[i], actual_delta[i], N, proc_i);
            }
            else
            {
                memset(out->coeffs[i], 0, N * sizeof(uint64_t));
            }
        }
    }
}
