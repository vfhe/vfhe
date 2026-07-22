#include "mlwe.h"
#include "misc.h"

LWE_Key lwe_alloc_key(uint64_t n, uint64_t l, incNTT ntt)
{
    LWE_Key key = (LWE_Key)safe_malloc(sizeof(*key));
    key->s = (uint64_t **)safe_malloc(sizeof(uint64_t *) * l);
    for (size_t i = 0; i < l; i++)
    {
        key->s[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    }
    key->n = n;
    key->l = l;
    key->ntt = ntt;
    return key;
}

LWE lwe_alloc_sample(uint64_t n, uint64_t l, incNTT ntt)
{
    LWE c = (LWE)safe_malloc(sizeof(*c));
    c->a = (uint64_t **)safe_malloc(sizeof(uint64_t *) * l);
    for (size_t i = 0; i < l; i++)
    {
        c->a[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    }
    c->b = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * l);
    c->n = n;
    c->l = l;
    c->ntt = ntt;
    return c;
}

void free_lwe_sample(LWE c)
{
    for (size_t i = 0; i < c->l; i++)
    {
        free(c->a[i]);
    }
    free(c->a);
    free(c->b);
    free(c);
}

void free_lwe_key(LWE_Key key)
{
    for (size_t i = 0; i < key->l; i++)
    {
        free(key->s[i]);
    }
    free(key->s);
    free(key);
}

LWE_Key lwe_new_key(uint64_t n, uint64_t l, incNTT ntt, double sec_sigma, double err_sigma)
{
    LWE_Key key = lwe_alloc_key(n, l, ntt);
    for (size_t i = 0; i < n; i++)
    {
        int64_t s_val = (int64_t)double2int(generate_normal_random(sec_sigma));
        for (size_t j = 0; j < l; j++)
        {
            uint64_t q = ntt->ntt[j]->q;
            key->s[j][i] = s_val < 0 ? negate_modq(-s_val, q) : modq(s_val, ntt->ntt[j]);
        }
    }
    key->sigma = err_sigma;
    return key;
}

LWE_Key lwe_new_sparse_ternary_key(uint64_t n, uint64_t l, incNTT ntt, uint64_t h, double err_sigma)
{
    LWE_Key key = lwe_alloc_key(n, l, ntt);
    uint64_t *tmp = (uint64_t *)safe_malloc(sizeof(uint64_t) * n);
    gen_sparse_ternary_array_modq(tmp, n, h, 3);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < l; j++)
        {
            uint64_t q = ntt->ntt[j]->q;
            if (tmp[i] == 1)
                key->s[j][i] = 1;
            else if (tmp[i] == 2)
                key->s[j][i] = q - 1;
            else
                key->s[j][i] = 0;
        }
    }
    free(tmp);
    key->sigma = err_sigma;
    return key;
}

void lwe_sample(LWE c, uint64_t *m, LWE_Key key)
{
    uint64_t *as = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * key->n);
    int64_t e_val = (int64_t)double2int(generate_normal_random(key->sigma));
    for (size_t i = 0; i < key->l; i++)
    {
        generate_random_bytes(key->n * sizeof(uint64_t), (uint8_t *)c->a[i]);
        array_reduce_mod_N(c->a[i], c->a[i], key->n,
                           key->ntt->ntt[i]->q); // Fallback, could use modq
        for (size_t j = 0; j < key->n; j++)
            c->a[i][j] = modq(c->a[i][j], key->ntt->ntt[i]);

        uint64_t q = key->ntt->ntt[i]->q;
        uint64_t e = e_val < 0 ? negate_modq(-e_val, q) : e_val;

        mod_eltwise_mul(as, c->a[i], key->s[i], key->n, key->ntt->ntt[i]);
        uint64_t b = e;
        for (size_t j = 0; j < key->n; j++)
        {
            b = add_modq(b, as[j], q);
        }
        if (m)
            b = add_modq(b, m[i], q);
        c->b[i] = b;
    }
    free(as);
}

LWE lwe_new_sample(uint64_t *m, LWE_Key key)
{
    LWE c = lwe_alloc_sample(key->n, key->l, key->ntt);
    lwe_sample(c, m, key);
    return c;
}

LWE lwe_new_trivial_sample(uint64_t *m, uint64_t n, uint64_t l, incNTT ntt)
{
    LWE c = lwe_alloc_sample(n, l, ntt);
    for (size_t i = 0; i < l; i++)
    {
        memset(c->a[i], 0, sizeof(uint64_t) * n);
        c->b[i] = m ? m[i] : 0;
    }
    return c;
}

void lwe_phase(uint64_t *out, LWE c, LWE_Key key)
{
    uint64_t *as = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * key->n);
    for (size_t i = 0; i < key->l; i++)
    {
        uint64_t q = key->ntt->ntt[i]->q;
        mod_eltwise_mul(as, c->a[i], key->s[i], key->n, key->ntt->ntt[i]);
        uint64_t sum = 0;
        for (size_t j = 0; j < key->n; j++)
        {
            sum = add_modq(sum, as[j], q);
        }
        out[i] = sub_modq(c->b[i], sum, q);
    }
    free(as);
}

void lwe_subto(LWE out, LWE in)
{
    assert(out->n == in->n);
    assert(out->l == in->l);
    for (size_t i = 0; i < out->l; i++)
    {
        mod_eltwise_sub(out->a[i], out->a[i], in->a[i], out->n, out->ntt->ntt[i]);
        out->b[i] = sub_modq(out->b[i], in->b[i], out->ntt->ntt[i]->q);
    }
}

// KS is disabled for LWE since we use mlwe_full_packing_keyswitch
LWE_KS_Key lwe_new_KS_key(LWE_Key out_key, LWE_Key in_key, uint64_t t, uint64_t base_bit)
{
    assert(false);
    return NULL;
}

void lwe_keyswitch(LWE out, LWE in, LWE_KS_Key ks_key) { assert(false); }
