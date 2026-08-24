// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include "arith_internal.h"

#if defined(__AVX512IFMA__) && !defined(PORTABLE_BUILD) && !defined(PORTABLE)

/* The Barrett constants this engine's `modq` above expects. Sized so that the
   32-bit fast path's `x * m` cannot overflow 128 bits (m stays around 2^50 for
   a 50-bit prime), which the portable engine's larger m would not satisfy --
   that is why each engine derives its own. */
Modulus mod_new(uint64_t q)
{
    int q_bits = 0;
    uint64_t temp_q = q;
    while (temp_q > 0)
    {
        q_bits++;
        temp_q >>= 1;
    }
    uint64_t k = 50 + q_bits;
    // The multiprecision path shifts by `k - 64`, so k below 64 would shift by
    // a negative amount. Small primes are not used today; clamping keeps them
    // correct rather than undefined (m only shrinks, so the bound above holds).
    if (k < 64)
        k = 64;

    uint64_t m = (uint64_t)(((unsigned __int128)1 << k) / q);

    Modulus res = (Modulus)malloc(sizeof(struct _Modulus));
    res->q = q;
    res->k = k;
    res->m = m;
    res->m52 = (k - 52 >= 64) ? 0 : (m >> (k - 52));
    res->ifma_barr_lo = m & ((1ULL << 52) - 1);
    res->ifma_prod_right_shift = k - 52;
    res->mp_w1 = (uint64_t)(((unsigned __int128)1 << 52) % q);
    res->mp_w2 = (uint64_t)(((unsigned __int128)1 << 104) % q);
    return res;
}

void mod_free(Modulus mod) { free(mod); }

uint64_t modq(unsigned __int128 x, Modulus mod)
{
    if (mod->q < (1ULL << 32))
    {
        if (x <= 0xFFFFFFFFFFFFFFFFULL)
        {
            uint64_t x64 = (uint64_t)x;
            uint64_t q_hat = (uint64_t)(((unsigned __int128)x64 * mod->m) >> mod->k);
            uint64_t res = x64 - q_hat * mod->q;
            if (res >= mod->q)
                res -= mod->q;
            return res;
        }
    }

    if (x < (1ULL << 52))
    {
        uint64_t q_hat = (uint64_t)((x * mod->m52) >> 52);
        uint64_t res = (uint64_t)x - q_hat * mod->q;
        if (res >= mod->q)
            res -= mod->q;
        return res;
    }

    uint64_t x0 = (uint64_t)x & ((1ULL << 52) - 1);
    uint64_t x1 = (uint64_t)(x >> 52) & ((1ULL << 52) - 1);
    uint64_t x2 = (uint64_t)(x >> 104);

    unsigned __int128 reduced =
        x0 + (unsigned __int128)x1 * mod->mp_w1 + (unsigned __int128)x2 * mod->mp_w2;

    uint64_t r_lo = (uint64_t)reduced;
    uint64_t r_hi = (uint64_t)(reduced >> 64);
    unsigned __int128 r_lo_m = (unsigned __int128)r_lo * mod->m;
    unsigned __int128 r_hi_m = (unsigned __int128)r_hi * mod->m;
    unsigned __int128 prod_hi = r_hi_m + (r_lo_m >> 64);

    uint64_t q_hat = (uint64_t)(prod_hi >> (mod->k - 64));
    uint64_t res = (uint64_t)reduced - q_hat * mod->q;

    while (res >= mod->q)
        res -= mod->q;

    return res;
}

uint64_t add_modq(uint64_t a, uint64_t b, uint64_t q)
{
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

uint64_t sub_modq(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }

uint64_t negate_modq(uint64_t a, uint64_t q) { return a == 0 ? 0 : q - a; }

uint64_t mul_modq(uint64_t a, uint64_t b, Modulus mod)
{
    return modq((unsigned __int128)a * b, mod);
}

void mod_eltwise_mul(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_mul_gen(out, in1, in2, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_mul_32(out, in1, in2, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_mul_50(out, in1, in2, n, mod);
    }
    else
    {
        mod_eltwise_mul_64(out, in1, in2, n, mod);
    }
}

void mod_eltwise_mul_addto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_mul_addto_gen(out, in1, in2, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_mul_addto_32(out, in1, in2, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_mul_addto_50(out, in1, in2, n, mod);
    }
    else
    {
        mod_eltwise_mul_addto_64(out, in1, in2, n, mod);
    }
}

void mod_eltwise_mul_subto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_mul_subto_gen(out, in1, in2, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_mul_subto_32(out, in1, in2, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_mul_subto_50(out, in1, in2, n, mod);
    }
    else
    {
        mod_eltwise_mul_subto_64(out, in1, in2, n, mod);
    }
}

void mod_eltwise_scale(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_scale_gen(out, in, scale, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_scale_32(out, in, scale, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_scale_50(out, in, scale, n, mod);
    }
    else
    {
        mod_eltwise_scale_64(out, in, scale, n, mod);
    }
}

void mod_eltwise_fma(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_fma_gen(out, in, scale, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_fma_32(out, in, scale, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_fma_50(out, in, scale, n, mod);
    }
    else
    {
        mod_eltwise_fma_64(out, in, scale, n, mod);
    }
}

void mod_eltwise_add_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_add_scalar_gen(out, in, scalar, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_add_scalar_32(out, in, scalar, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_add_scalar_50(out, in, scalar, n, mod);
    }
    else
    {
        mod_eltwise_add_scalar_64(out, in, scalar, n, mod);
    }
}

void mod_eltwise_sub_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_sub_scalar_gen(out, in, scalar, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_sub_scalar_32(out, in, scalar, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_sub_scalar_50(out, in, scalar, n, mod);
    }
    else
    {
        mod_eltwise_sub_scalar_64(out, in, scalar, n, mod);
    }
}

void mod_eltwise_negate(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_negate_gen(out, in, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_negate_32(out, in, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_negate_50(out, in, n, mod);
    }
    else
    {
        mod_eltwise_negate_64(out, in, n, mod);
    }
}

void mod_eltwise_add(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_add_gen(out, in1, in2, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_add_32(out, in1, in2, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_add_50(out, in1, in2, n, mod);
    }
    else
    {
        mod_eltwise_add_64(out, in1, in2, n, mod);
    }
}

void mod_eltwise_sub(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_sub_gen(out, in1, in2, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_sub_32(out, in1, in2, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_sub_50(out, in1, in2, n, mod);
    }
    else
    {
        mod_eltwise_sub_64(out, in1, in2, n, mod);
    }
}

void mod_eltwise_reduce(uint64_t *out, uint64_t *in, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_reduce_gen(out, in, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_reduce_32(out, in, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_reduce_50(out, in, n, mod);
    }
    else
    {
        mod_eltwise_reduce_64(out, in, n, mod);
    }
}

void mod_eltwise_reduce_signed(uint64_t *out, int64_t *in, uint64_t n, Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_eltwise_reduce_signed_gen(out, in, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_eltwise_reduce_signed_32(out, in, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_eltwise_reduce_signed_50(out, in, n, mod);
    }
    else
    {
        mod_eltwise_reduce_signed_64(out, in, n, mod);
    }
}

void mod_reduce_array_mp(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                         Modulus mod)
{
    if (n < MOD_MIN_VECTOR_LEN)
    {
        mod_reduce_array_mp_gen(out, in_high, in_low, n, mod);
        return;
    }
    if (mod->q < (1ULL << 32))
    {
        mod_reduce_array_mp_32(out, in_high, in_low, n, mod);
    }
    else if (mod->q < (1ULL << 50))
    {
        mod_reduce_array_mp_50(out, in_high, in_low, n, mod);
    }
    else
    {
        mod_reduce_array_mp_64(out, in_high, in_low, n, mod);
    }
}

#endif
