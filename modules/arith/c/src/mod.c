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

// Barrett-reduce a value already split into 52-bit limbs, x = x0 + x1*2^52 +
// x2*2^104, folded onto q with the precomputed residues of 2^52 and 2^104.
static uint64_t fold_limbs_modq(uint64_t x0, uint64_t x1, uint64_t x2, Modulus mod)
{
    const uint64_t q = mod->q;
    const unsigned __int128 reduced =
        x0 + (unsigned __int128)x1 * mod->mp_w1 + (unsigned __int128)x2 * mod->mp_w2;

    const uint64_t r_lo = (uint64_t)reduced, r_hi = (uint64_t)(reduced >> 64);
    const unsigned __int128 prod_hi =
        (unsigned __int128)r_hi * mod->m + (((unsigned __int128)r_lo * mod->m) >> 64);

    const uint64_t q_hat = (uint64_t)(prod_hi >> (mod->k - 64));
    uint64_t res = (uint64_t)reduced - q_hat * q;
    while (res >= q)
        res -= q;
    return res;
}

uint64_t modq(uint64_t x, Modulus mod)
{
    const uint64_t q = mod->q;

    // A 32-bit modulus keeps m small enough that x * m stays inside 128 bits.
    if (q < (1ULL << 32))
    {
        const uint64_t q_hat = (uint64_t)(((unsigned __int128)x * mod->m) >> mod->k);
        uint64_t res = x - q_hat * q;
        if (res >= q)
            res -= q;
        return res;
    }

    if (x < (1ULL << 52))
    {
        const uint64_t q_hat = (uint64_t)(((unsigned __int128)x * mod->m52) >> 52);
        uint64_t res = x - q_hat * q;
        if (res >= q)
            res -= q;
        return res;
    }

    return fold_limbs_modq(x & ((1ULL << 52) - 1), x >> 52, 0, mod);
}

uint64_t modq_wide(uint64_t hi, uint64_t lo, Modulus mod)
{
    if (hi == 0)
        return modq(lo, mod);

    const uint64_t mask52 = (1ULL << 52) - 1;
    return fold_limbs_modq(lo & mask52, ((lo >> 52) | (hi << 12)) & mask52, hi >> 40, mod);
}

uint64_t add_modq(uint64_t a, uint64_t b, uint64_t q)
{
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

// Branchless: the subtraction wraps exactly when b > a, and `d > a` detects
// that, so q is added back under a mask. The ternary form costs a
// data-dependent branch per element once this is inlined into a kernel loop.
uint64_t sub_modq(uint64_t a, uint64_t b, uint64_t q)
{
    const uint64_t d = a - b;
    return d + (q & -(uint64_t)(d > a));
}

uint64_t negate_modq(uint64_t a, uint64_t q) { return a == 0 ? 0 : q - a; }

// The only place a 64x64 -> 128 product is formed; it goes straight into
// modq_wide's two words.
uint64_t mul_modq(uint64_t a, uint64_t b, Modulus mod)
{
    const unsigned __int128 prod = (unsigned __int128)a * b;
    return modq_wide((uint64_t)(prod >> 64), (uint64_t)prod, mod);
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
