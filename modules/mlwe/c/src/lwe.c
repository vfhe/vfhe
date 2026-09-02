// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include "mlwe.h"
#include "util.h"
#include <crypto.h>

// Generates a sparse ternary array with Hamming Weight h, balanced (h/2 ones and h/2 negative ones)
void gen_sparse_ternary_array_modq(uint64_t *out, uint64_t size, uint64_t h, uint64_t q)
{
    memset(out, 0, sizeof(uint64_t) * size);
    uint64_t hw = 0, val = 1, *rnd_buffer;
    const uint64_t buffer_size = h * 10;
    rnd_buffer = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * buffer_size);
    while (hw < h)
    {
        generate_random_bytes(sizeof(uint64_t) * buffer_size, (uint8_t *)rnd_buffer);
        array_mod_switch_from_2k(rnd_buffer, rnd_buffer, size, size, buffer_size);
        uint64_t i = 0;
        while (i < buffer_size && hw < h)
        {
            const uint64_t idx = rnd_buffer[i++];
            if (out[idx])
                continue;
            out[idx] = (uint64_t)((q + (int64_t)val) % q);
            val = -val;
            hw++;
        }
    }
    free(rnd_buffer);
#ifndef NDEBUG
    uint64_t hw_check = 0, sum_check = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum_check += out[i];
        hw_check += (out[i] != 0);
    }
    assert(hw_check == h);
    assert((sum_check % q) == 0);
#endif
}

LWE_Key lwe_alloc_key(uint64_t n, uint64_t l, RNS_Base base)
{
    LWE_Key key = (LWE_Key)safe_malloc(sizeof(*key));
    key->s = (uint64_t **)safe_malloc(sizeof(uint64_t *) * l);
    for (size_t i = 0; i < l; i++)
    {
        key->s[i] = (uint64_t *)safe_aligned_malloc(sizeof(uint64_t) * n);
    }
    key->n = n;
    key->l = l;
    key->base = base;
    return key;
}

LWE lwe_alloc_sample(uint64_t n, uint64_t l, RNS_Base base)
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
    c->base = base;
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

LWE_Key lwe_new_key(uint64_t n, uint64_t l, RNS_Base base, double sec_sigma, double err_sigma)
{
    LWE_Key key = lwe_alloc_key(n, l, base);
    for (size_t i = 0; i < n; i++)
    {
        int64_t s_val = (int64_t)double2int(generate_normal_random(sec_sigma));
        for (size_t j = 0; j < l; j++)
        {
            uint64_t q = base->mods[j]->q;
            key->s[j][i] = s_val < 0 ? negate_modq(-s_val, q) : modq(s_val, base->mods[j]);
        }
    }
    key->sigma = err_sigma;
    return key;
}

LWE_Key lwe_new_sparse_ternary_key(uint64_t n, uint64_t l, RNS_Base base, uint64_t h,
                                   double err_sigma)
{
    LWE_Key key = lwe_alloc_key(n, l, base);
    uint64_t *tmp = (uint64_t *)safe_malloc(sizeof(uint64_t) * n);
    gen_sparse_ternary_array_modq(tmp, n, h, 3);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < l; j++)
        {
            uint64_t q = base->mods[j]->q;
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
                           key->base->mods[i]->q); // Fallback, could use modq
        for (size_t j = 0; j < key->n; j++)
            c->a[i][j] = modq(c->a[i][j], key->base->mods[i]);

        uint64_t q = key->base->mods[i]->q;
        uint64_t e = e_val < 0 ? negate_modq((uint64_t)(-e_val), q) : (uint64_t)e_val;

        mod_eltwise_mul(as, c->a[i], key->s[i], key->n, key->base->mods[i]);
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
    LWE c = lwe_alloc_sample(key->n, key->l, key->base);
    lwe_sample(c, m, key);
    return c;
}

LWE lwe_new_trivial_sample(uint64_t *m, uint64_t n, uint64_t l, RNS_Base base)
{
    LWE c = lwe_alloc_sample(n, l, base);
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
        uint64_t q = key->base->mods[i]->q;
        mod_eltwise_mul(as, c->a[i], key->s[i], key->n, key->base->mods[i]);
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
        mod_eltwise_sub(out->a[i], out->a[i], in->a[i], out->n, out->base->mods[i]);
        out->b[i] = sub_modq(out->b[i], in->b[i], out->base->mods[i]->q);
    }
}

// KS is disabled for LWE since we use mlwe_full_packing_keyswitch
LWE_KS_Key lwe_new_KS_key(LWE_Key out_key, LWE_Key in_key, uint64_t t, uint64_t base_bit)
{
    assert(false);
    return NULL;
}

void lwe_keyswitch(LWE out, LWE in, LWE_KS_Key ks_key) { assert(false); }
