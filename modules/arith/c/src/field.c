#include <arith.h>
#include <blake3.h>
#include "misc.h"

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

static void poly_set_zero(uint64_t *p, uint64_t n)
{
    memset(p, 0, n * sizeof(uint64_t));
}

static void poly_copy(uint64_t *dest, const uint64_t *src, uint64_t n)
{
    memcpy(dest, src, n * sizeof(uint64_t));
}

static int poly_divrem(uint64_t *q, uint64_t *r, const uint64_t *f, const uint64_t *g, uint64_t n, NTT_proc proc)
{
    uint64_t p = proc->q;
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
        uint64_t coeff = mul_modq(r[deg_r], inv_lead_g, proc);
        uint64_t shift = (uint64_t)(deg_r - deg_g);
        q[shift] = coeff;

        for (int j = 0; j <= deg_g; ++j)
        {
            uint64_t term = mul_modq(coeff, g[j], proc);
            r[shift + j] = sub_modq(r[shift + j], term, p);
        }

        deg_r = poly_deg(r, n);
    }
    return 1;
}

// Field operations implementation

NTT_proc field_new_proc(uint64_t q)
{
    uint64_t k = 64;
    unsigned __int128 m_128 = ((unsigned __int128)1 << k) / q;
    while (m_128 < (1ULL << 63))
    {
        k++;
        m_128 = ((unsigned __int128)1 << k) / q;
    }
    uint64_t m = (uint64_t)m_128;
    NTT_proc res = (NTT_proc)malloc(sizeof(struct _NTT_proc));
    res->n = 1;
    res->q = q;
    res->root_of_unity = 0;
    res->inv_root_of_unity = 0;
    res->k = k;
    res->m = m;
    res->m52 = (k - 52 >= 64) ? 0 : (m >> (k - 52));
    res->mp_w1 = (uint64_t)(((unsigned __int128)1 << 52) % q);
    res->mp_w2 = (uint64_t)(((unsigned __int128)1 << 104) % q);
    res->ws_fwd = NULL;
    res->w_precon_fwd = NULL;
    res->ws_inv = NULL;
    res->w_precon_inv = NULL;
    return res;
}

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

void field_ext_mul(uint64_t *c, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t w, NTT_proc proc)
{
    uint64_t mod = proc->q;
    uint64_t *tmp = (uint64_t *)malloc((2 * d - 1) * sizeof(uint64_t));
    memset(tmp, 0, (2 * d - 1) * sizeof(uint64_t));

    for (uint64_t i = 0; i < d; i++)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            uint64_t prod = mul_modq(a[i], b[j], proc);
            tmp[i + j] = add_modq(tmp[i + j], prod, mod);
        }
    }

    for (uint64_t i = 2 * d - 2; i >= d; i--)
    {
        uint64_t folded = mul_modq(tmp[i], w, proc);
        tmp[i - d] = add_modq(tmp[i - d], folded, mod);
    }

    for (uint64_t i = 0; i < d; i++)
    {
        c[i] = tmp[i];
    }
    free(tmp);
}

void field_ext_pow(uint64_t *res, const uint64_t *base, uint64_t exp_lo, uint64_t exp_hi, uint64_t d, uint64_t w, NTT_proc proc)
{
    __uint128_t exp = ((__uint128_t)exp_hi << 64) | exp_lo;
    uint64_t *b = (uint64_t *)malloc(d * sizeof(uint64_t));
    memcpy(b, base, d * sizeof(uint64_t));

    res[0] = 1;
    for (uint64_t i = 1; i < d; i++)
    {
        res[i] = 0;
    }

    uint64_t *tmp = (uint64_t *)malloc(d * sizeof(uint64_t));
    while (exp > 0)
    {
        if (exp & 1)
        {
            field_ext_mul(tmp, res, b, d, w, proc);
            memcpy(res, tmp, d * sizeof(uint64_t));
        }
        field_ext_mul(tmp, b, b, d, w, proc);
        memcpy(b, tmp, d * sizeof(uint64_t));
        exp >>= 1;
    }
    free(b);
    free(tmp);
}

static void poly_mul_mod_xd_w(uint64_t *res, const uint64_t *a, const uint64_t *b, uint64_t d, uint64_t w, NTT_proc proc)
{
    uint64_t q = proc->q;
    uint64_t *tmp = (uint64_t *)malloc((2 * d + 1) * sizeof(uint64_t));
    memset(tmp, 0, (2 * d + 1) * sizeof(uint64_t));

    for (uint64_t i = 0; i <= d; i++)
    {
        for (uint64_t j = 0; j < d; j++)
        {
            uint64_t prod = mul_modq(a[i], b[j], proc);
            tmp[i + j] = add_modq(tmp[i + j], prod, q);
        }
    }

    for (uint64_t i = 2 * d; i >= d; i--)
    {
        if (tmp[i] == 0) continue;
        uint64_t folded = mul_modq(tmp[i], w, proc);
        tmp[i - d] = add_modq(tmp[i - d], folded, q);
    }

    for (uint64_t i = 0; i < d; i++)
    {
        res[i] = tmp[i];
    }
    free(tmp);
}

int field_ext_inv(uint64_t *ainv, const uint64_t *a, uint64_t d, uint64_t w, NTT_proc proc)
{
    uint64_t p = proc->q;
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
        if (!poly_divrem(q, r2, r0, r1, n, proc))
        {
            status = 0;
            break;
        }

        // tmp = q * t1 mod (X^d - w)
        poly_mul_mod_xd_w(tmp, q, t1, d, w, proc);

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
                ainv[i] = mul_modq(t0[i], c_inv, proc);
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

void field_sample_random_element(uint64_t *a, const uint8_t *seed, uint64_t seed_len, uint64_t d, uint64_t mod)
{
    blake3_hasher hasher;
    blake3_hasher_init_derive_key(&hasher, "field_sample_element");
    blake3_hasher_update(&hasher, seed, seed_len);
    uint8_t buffer[BLAKE3_OUT_LEN];
    uint64_t mask = mod;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    for (uint64_t i = 0; i < d; i++)
    {
        while (true)
        {
            blake3_hasher_finalize(&hasher, buffer, BLAKE3_OUT_LEN);
            uint64_t sampled = (*((uint64_t *)buffer)) & mask;
            if (sampled < mod)
            {
                a[i] = sampled;
                break;
            }
            blake3_hasher_update(&hasher, buffer, BLAKE3_OUT_LEN);
        }
    }
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
    int64_t t = 0;   int64_t newt = 1;
    int64_t r = m;   int64_t newr = a;

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

    if (r > 1) return 0; // not invertible
    if (t < 0) t = t + m;
    return (uint64_t)t;
}

void field_base_conversion(uint64_t *out, const uint64_t *in,
                           uint64_t source_component, uint64_t target_component,
                           uint64_t d, uint64_t poly_size, const uint64_t *w_i, NTT_proc proc)
{
    uint64_t q = proc->q;
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
        uint64_t prod = mul_modq(coeff, factor, proc);
        out[dest_idx] = add_modq(out[dest_idx], prod, q);
    }
}
