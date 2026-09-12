// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include <blake3.h>
#include "arith_internal.h"
#include "util.h"
#include <crypto.h>

// Helper functions for polynomial operations used in inversion
static int poly_deg(const uint64_t *p, int n)
{
    for (int i = n - 1; i >= 0; i--)
    {
        if (p[i] != 0)
            return i;
    }
    return -1;
}

static void poly_set_zero(uint64_t *p, uint64_t n) { memset(p, 0, n * sizeof(uint64_t)); }

static void poly_copy(uint64_t *dest, const uint64_t *src, uint64_t n)
{
    memcpy(dest, src, n * sizeof(uint64_t));
}

static int poly_divrem(uint64_t *q, uint64_t *r, const uint64_t *f, const uint64_t *g, uint64_t n,
                       Modulus mod)
{
    uint64_t p = mod->q;
    poly_set_zero(q, n);
    poly_copy(r, f, n);

    int deg_g = poly_deg(g, n);
    if (deg_g < 0)
    {
        return 0; // division by zero
    }

    uint64_t inv_lead_g = inverse_mod(g[deg_g], p);

    int deg_r = poly_deg(r, n);
    while (deg_r >= deg_g && deg_r >= 0)
    {
        uint64_t coeff = mul_modq(r[deg_r], inv_lead_g, mod);
        uint64_t shift = (uint64_t)(deg_r - deg_g);
        q[shift] = coeff;

        for (int j = 0; j <= deg_g; ++j)
        {
            uint64_t term = mul_modq(coeff, g[j], mod);
            r[shift + j] = sub_modq(r[shift + j], term, p);
        }

        deg_r = poly_deg(r, n);
    }
    return 1;
}

// Field operations implementation

void field_ext_add(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q)
{
    for (uint64_t i = 0; i < d; i++)
    {
        c[i] = add_modq(a[i], b[i], q);
    }
}

void field_ext_sub(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t q)
{
    for (uint64_t i = 0; i < d; i++)
    {
        c[i] = sub_modq(a[i], b[i], q);
    }
}

void field_ext_neg(uint64_t *c, const uint64_t *a, uint64_t d, uint64_t q)
{
    for (uint64_t i = 0; i < d; i++)
    {
        c[i] = negate_modq(a[i], q);
    }
}

void field_ext_mul(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t w,
                   Modulus mod)
{
    const uint64_t q = mod->q;
    uint64_t *tmp = (uint64_t *)malloc((2 * d - 1) * sizeof(uint64_t));
    memset(tmp, 0, (2 * d - 1) * sizeof(uint64_t));

    for (uint64_t i = 0; i < d; i++)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            uint64_t prod = mul_modq(a[i], b[j], mod);
            tmp[i + j] = add_modq(tmp[i + j], prod, q);
        }
    }

    for (uint64_t i = 2 * d - 2; i >= d; i--)
    {
        uint64_t folded = mul_modq(tmp[i], w, mod);
        tmp[i - d] = add_modq(tmp[i - d], folded, q);
    }

    for (uint64_t i = 0; i < d; i++)
    {
        c[i] = tmp[i];
    }
    free(tmp);
}

void field_ext_pow(uint64_t *res, const uint64_t *base, const uint64_t *exp, uint64_t exp_words,
                   uint64_t d, uint64_t w, Modulus mod)
{
    uint64_t *b = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *out = (uint64_t *)malloc(d * sizeof(uint64_t));

    memcpy(b, base, d * sizeof(uint64_t));
    out[0] = 1;
    for (uint64_t i = 1; i < d; i++)
        out[i] = 0;

    // Right to left over the exponent bits, least significant word first. The
    // caller passes the words it has, so the trailing squarings past the top
    // set bit cost at most one word's worth.
    for (uint64_t k = 0; k < exp_words; k++)
    {
        for (unsigned bit = 0; bit < 64; bit++)
        {
            if ((exp[k] >> bit) & 1)
            {
                field_ext_mul(tmp, out, b, d, w, mod);
                memcpy(out, tmp, d * sizeof(uint64_t));
            }
            field_ext_mul(tmp, b, b, d, w, mod);
            memcpy(b, tmp, d * sizeof(uint64_t));
        }
    }

    memcpy(res, out, d * sizeof(uint64_t)); // written only now, so res may be base
    free(b);
    free(tmp);
    free(out);
}

// The Frobenius x -> x^(p^k) is a coefficient map, not an exponentiation: the
// coefficients of an element lie in F_p, so Fermat leaves each of them fixed
// and all that moves is the monomial. Writing p = q*d + r, x^(j*p) is
// (x^d)^(j*q) * x^(j*r) = w^(j*q + (j*r)/d) * x^((j*r) mod d) -- one permutation
// of the positions with one constant per position.
//
// `to[j]` and `constants[j]` are that map for a single application. The
// exponent of w is reduced mod p - 1, its order, so it fits a 64-bit exponent
// whatever d is.
static void frobenius_step(uint64_t *constants, uint64_t *to, uint64_t d, uint64_t w, Modulus mod)
{
    const uint64_t p = mod->q, r = p % d;
    for (uint64_t j = 0; j < d; j++)
    {
        const __uint128_t e = (__uint128_t)(p / d) * j + (__uint128_t)(r * j) / d;
        constants[j] = power_mod(w, (uint64_t)(e % (p - 1)), p);
        to[j] = (j * r) % d;
    }
}

// The k-fold composition of that map, as one permutation and one constant. The
// Galois group is cyclic of order d, so `k` is taken modulo d first and the
// loop runs at most d - 1 times over d positions -- the cost does not depend on
// how many elements the map is then applied to.
void frobenius_map(uint64_t *constants, uint64_t *to, uint64_t k, uint64_t d, uint64_t w,
                   Modulus mod)
{
    uint64_t *step_c = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *step_to = (uint64_t *)malloc(d * sizeof(uint64_t));
    frobenius_step(step_c, step_to, d, w, mod);

    for (uint64_t j = 0; j < d; j++)
    {
        constants[j] = 1;
        to[j] = j;
    }
    for (uint64_t step = 0; step < k % d; step++)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            // The step constant belongs to the position the coefficient has
            // reached, not to where it started.
            constants[j] = mul_modq(constants[j], step_c[to[j]], mod);
            to[j] = step_to[to[j]];
        }
    }
    free(step_c);
    free(step_to);
}

void field_ext_frobenius(uint64_t *res, const uint64_t *a, uint64_t k, uint64_t d, uint64_t w,
                         Modulus mod)
{
    if (d <= 1 || k % d == 0)
    {
        memmove(res, a, d * sizeof(uint64_t));
        return;
    }
    uint64_t *constants = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *to = (uint64_t *)malloc(d * sizeof(uint64_t));
    uint64_t *out = (uint64_t *)malloc(d * sizeof(uint64_t));
    frobenius_map(constants, to, k, d, w, mod);
    for (uint64_t j = 0; j < d; j++)
        out[to[j]] = mul_modq(a[j], constants[j], mod);
    memcpy(res, out, d * sizeof(uint64_t)); // written only now, so res may be a
    free(constants);
    free(to);
    free(out);
}

static void poly_mul_mod_xd_w(uint64_t *res, const uint64_t *a, const uint64_t *b, uint64_t d,
                              uint64_t w, Modulus mod)
{
    uint64_t q = mod->q;
    uint64_t *tmp = (uint64_t *)malloc((2 * d + 1) * sizeof(uint64_t));
    memset(tmp, 0, (2 * d + 1) * sizeof(uint64_t));

    for (uint64_t i = 0; i <= d; i++)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            uint64_t prod = mul_modq(a[i], b[j], mod);
            tmp[i + j] = add_modq(tmp[i + j], prod, q);
        }
    }

    for (uint64_t i = 2 * d; i >= d; i--)
    {
        if (tmp[i] == 0)
            continue;
        uint64_t folded = mul_modq(tmp[i], w, mod);
        tmp[i - d] = add_modq(tmp[i - d], folded, q);
    }

    for (uint64_t i = 0; i < d; i++)
    {
        res[i] = tmp[i];
    }
    free(tmp);
}

int field_ext_inv(uint64_t *ainv, const uint64_t *a, uint64_t d, uint64_t w, Modulus mod)
{
    uint64_t p = mod->q;
    const uint64_t n = d + 1;

    uint64_t *r0 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *r1 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *r2 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *t0 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *t1 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *t2 = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *q = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)malloc(n * sizeof(uint64_t));

    poly_set_zero(r0, n);
    poly_set_zero(r1, n);
    poly_set_zero(r2, n);
    poly_set_zero(t0, n);
    poly_set_zero(t1, n);
    poly_set_zero(t2, n);
    poly_set_zero(q, n);
    poly_set_zero(tmp, n);

    /* r0 = X^d - w */
    if (w == 0)
    {
        r0[0] = 0;
    }
    else
    {
        r0[0] = p - w;
    }
    r0[d] = 1;

    /* r1 = a */
    for (uint64_t i = 0; i < d; i++)
    {
        r1[i] = a[i];
    }

    /* t0 = 0, t1 = 1 */
    t1[0] = 1;

    int status = 1;
    int iter = 0;
    while (poly_deg(r1, n) >= 0)
    {
        iter++;
        if (iter > 1000)
        {
            printf("ERROR: field_ext_inv infinite loop detected!\n");
            status = 0;
            break;
        }
        if (!poly_divrem(q, r2, r0, r1, n, mod))
        {
            status = 0;
            break;
        }

        // tmp = q * t1 mod (X^d - w)
        poly_mul_mod_xd_w(tmp, q, t1, d, w, mod);

        // t2 = t0 - tmp mod p
        for (uint64_t i = 0; i < d; i++)
        {
            t2[i] = sub_modq(t0[i], tmp[i], p);
        }

        poly_copy(r0, r1, n);
        poly_copy(r1, r2, n);
        poly_copy(t0, t1, n);
        poly_copy(t1, t2, n);
    }

    if (status)
    {
        int degr0 = poly_deg(r0, n);
        if (degr0 != 0 || r0[0] == 0)
        {
            status = 0;
        }
        else
        {
            uint64_t c_inv = inverse_mod(r0[0], p);
            for (uint64_t i = 0; i < d; ++i)
            {
                ainv[i] = mul_modq(t0[i], c_inv, mod);
            }
        }
    }

    free(r0);
    free(r1);
    free(r2);
    free(t0);
    free(t1);
    free(t2);
    free(q);
    free(tmp);

    return status;
}

/* Uniform in F_p^d from a seed. The generator lives in misc's prng.c; the
   domain-separation tag is the field's to choose. */
void field_sample_random_element(uint64_t *a, const uint8_t *seed, uint64_t seed_len, uint64_t d,
                                 uint64_t mod)
{
    prng_sample_below(a, d, mod, "field_sample_element", seed, seed_len);
}

void field_hash_element(uint8_t *out, const uint64_t *a, uint64_t d)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, (const uint8_t *)a, d * sizeof(uint64_t));
    blake3_hasher_finalize(&hasher, out, BLAKE3_OUT_LEN);
}

int field_ext_is_equal(const uint64_t *a, const uint64_t *b, uint64_t d)
{
    for (uint64_t i = 0; i < d; i++)
    {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

static uint64_t inverse_mod_eea_generic(uint64_t a, uint64_t m)
{
    int64_t t = 0;
    int64_t newt = 1;
    int64_t r = m;
    int64_t newr = a;

    while (newr != 0)
    {
        int64_t quotient = r / newr;
        int64_t temp_t = t - quotient * newt;
        t = newt;
        newt = temp_t;

        int64_t temp_r = r - quotient * newr;
        r = newr;
        newr = temp_r;
    }

    if (r > 1)
        return 0; // not invertible
    if (t < 0)
        t = t + m;
    return (uint64_t)t;
}

void field_base_conversion(uint64_t *out, const uint64_t *in, uint64_t source_component,
                           uint64_t target_component, uint64_t d, uint64_t poly_size,
                           const uint64_t *w_i, Modulus mod)
{
    uint64_t q = mod->q;
    if (source_component == target_component)
    {
        for (uint64_t j = 0; j < d; j++)
            out[j] = in[j];
        return;
    }

    uint64_t log_poly_size = (uint64_t)log2(poly_size);
    uint64_t log_n = log_poly_size + 1;

    uint64_t x_s1 = (int_rev((uint32_t)source_component) >> (32 - log_n)) + 1;
    uint64_t x_s2 = (int_rev((uint32_t)target_component) >> (32 - log_n)) + 1;

    uint64_t w_s2 = w_i[target_component];

    uint64_t M = 2 * poly_size * d;
    uint64_t inv_x_s2 = inverse_mod_eea_generic(x_s2, M);
    uint64_t e = (x_s1 * inv_x_s2) % M;

    for (uint64_t j = 0; j < d; j++)
        out[j] = 0;

    for (uint64_t j = 0; j < d; j++)
    {
        uint64_t coeff = in[j];
        if (coeff == 0)
            continue;
        uint64_t power = j * e;
        uint64_t dest_idx = power % d;
        uint64_t exp_term = power / d;
        uint64_t factor = power_mod(w_s2, exp_term, q);
        uint64_t prod = mul_modq(coeff, factor, mod);
        out[dest_idx] = add_modq(out[dest_idx], prod, q);
    }
}
