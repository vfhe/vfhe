// SPDX-FileCopyrightText: 2026 Antonio Guimarães <antonio.guimaraes@imdea.org>
// SPDX-License-Identifier: Apache-2.0
#include <arith.h>
#include "arith_internal.h"

#if !defined(__AVX512IFMA__) || defined(PORTABLE_BUILD) || defined(PORTABLE)

uint64_t modq(unsigned __int128 x, NTT_proc proc)
{
    if (x < (1ULL << 52))
    {
        uint64_t q_hat = (uint64_t)((x * proc->m52) >> 52);
        uint64_t res = (uint64_t)x - q_hat * proc->q;
        if (res >= proc->q)
            res -= proc->q;
        return res;
    }

    uint64_t x0 = (uint64_t)x & ((1ULL << 52) - 1);
    uint64_t x1 = (uint64_t)(x >> 52) & ((1ULL << 52) - 1);
    uint64_t x2 = (uint64_t)(x >> 104);

    unsigned __int128 reduced =
        x0 + (unsigned __int128)x1 * proc->mp_w1 + (unsigned __int128)x2 * proc->mp_w2;

    uint64_t r_lo = (uint64_t)reduced;
    uint64_t r_hi = (uint64_t)(reduced >> 64);
    unsigned __int128 r_lo_m = (unsigned __int128)r_lo * proc->m;
    unsigned __int128 r_hi_m = (unsigned __int128)r_hi * proc->m;
    unsigned __int128 prod_hi = r_hi_m + (r_lo_m >> 64);

    uint64_t q_hat = (uint64_t)(prod_hi >> (proc->k - 64));
    uint64_t res = (uint64_t)reduced - q_hat * proc->q;

    while (res >= proc->q)
        res -= proc->q;

    return res;
}

uint64_t add_modq(uint64_t a, uint64_t b, uint64_t q)
{
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

uint64_t sub_modq(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }

uint64_t negate_modq(uint64_t a, uint64_t q) { return a == 0 ? 0 : q - a; }

uint64_t mul_modq(uint64_t a, uint64_t b, NTT_proc proc)
{
    return modq((unsigned __int128)a * b, proc);
}

// The element-wise kernels are the shared size-generic ones (mod_scalar.c);
// this engine has no vectorized variants to dispatch between.
void mod_eltwise_mul(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    mod_eltwise_mul_gen(out, in1, in2, n, proc);
}

void mod_eltwise_mul_addto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    mod_eltwise_mul_addto_gen(out, in1, in2, n, proc);
}

void mod_eltwise_mul_subto(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    mod_eltwise_mul_subto_gen(out, in1, in2, n, proc);
}

void mod_eltwise_scale(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    mod_eltwise_scale_gen(out, in, scale, n, proc);
}

void mod_eltwise_fma(uint64_t *out, uint64_t *in, uint64_t scale, uint64_t n, NTT_proc proc)
{
    mod_eltwise_fma_gen(out, in, scale, n, proc);
}

void mod_eltwise_add(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    mod_eltwise_add_gen(out, in1, in2, n, proc);
}

void mod_eltwise_sub(uint64_t *out, uint64_t *in1, uint64_t *in2, uint64_t n, NTT_proc proc)
{
    mod_eltwise_sub_gen(out, in1, in2, n, proc);
}

void mod_eltwise_negate(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    mod_eltwise_negate_gen(out, in, n, proc);
}

void mod_eltwise_reduce(uint64_t *out, uint64_t *in, uint64_t n, NTT_proc proc)
{
    mod_eltwise_reduce_gen(out, in, n, proc);
}

void mod_eltwise_reduce_signed(uint64_t *out, int64_t *in, uint64_t n, NTT_proc proc)
{
    mod_eltwise_reduce_signed_gen(out, in, n, proc);
}

void mod_eltwise_add_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n, NTT_proc proc)
{
    mod_eltwise_add_scalar_gen(out, in, scalar, n, proc);
}

void mod_eltwise_sub_scalar(uint64_t *out, uint64_t *in, uint64_t scalar, uint64_t n, NTT_proc proc)
{
    mod_eltwise_sub_scalar_gen(out, in, scalar, n, proc);
}

void mod_reduce_array_mp(uint64_t *out, uint64_t *in_high, uint64_t *in_low, uint64_t n,
                         NTT_proc proc)
{
    mod_reduce_array_mp_gen(out, in_high, in_low, n, proc);
}

#endif
